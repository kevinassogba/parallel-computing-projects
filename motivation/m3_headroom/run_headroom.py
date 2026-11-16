"""M3 (pilot for H2/H4): what limits how many agent sessions can keep their warm
GPU tool state resident on ONE GPU — memory capacity, compute, or bandwidth?

K closed-loop session processes share the GPU (MPS if enabled by the caller).
Each holds its own tool state (tool_state.py kinds) and loops:
    idle for a think time sampled from public traces  ->  one tool call (probe).
Measured per K: tool-call latency (p50/p99) vs K=1, sessions that failed to
start (CUDA OOM), device memory, and NVML utilization (sampler run separately).
HBM capacity is emulated by a ballast allocation (--hbm-cap-gb), anchored at
real SKU sizes (80 / 96 / 141 GB); the 80 GB point is validated on a real H100.

  python run_headroom.py --kind mace --k 1 2 4 8 16 32 64 --hbm-cap-gb 80 \
      --windows ../results/public/window_samples_freeinference.csv --duration 300 \
      --repeats 3 --out results/m3/h200_cap80.jsonl
"""

import argparse
import csv
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL_STATE = HERE.parent / "m2_park_restore" / "tool_state.py"

WORKER = r'''
import json, os, random, sys, time, importlib.util
spec = importlib.util.spec_from_file_location("ts", sys.argv[1]); ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)
cfg = json.loads(sys.argv[2]); out = cfg["out"]
import torch
class A: pass
a = A(); a.__dict__.update(cfg["state_args"])
rec = {"worker": cfg["worker"], "event": "start", "t": time.time()}
try:
    torch.zeros(1, device="cuda"); probe, info = ts.build(a); c0 = probe(); torch.cuda.synchronize()
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    if "out of memory" not in str(e).lower(): raise
    rec.update(event="oom", error=str(e)[:300]); open(out, "a").write(json.dumps(rec) + "\n"); sys.exit(3)
if a.kind == "synthetic": torch.cuda.empty_cache()
rec.update(event="ready", mem_reserved_mb=torch.cuda.memory_reserved() / 2**20,
           mem_allocated_mb=torch.cuda.memory_allocated() / 2**20)
open(out, "a").write(json.dumps(rec) + "\n")
win = [float(x) for x in open(cfg["windows"]).read().split()[1:]]
rng = random.Random(cfg["seed"]); t_stop = cfg["t_stop"]
while time.time() < t_stop:
    time.sleep(min(rng.choice(win), cfg["max_window_s"]))
    if time.time() >= t_stop: break
    t = time.time(); c = probe(); torch.cuda.synchronize()
    open(out, "a").write(json.dumps({"worker": cfg["worker"], "event": "call", "t": t,
        "latency_s": time.time() - t, "match": c == c0}) + "\n")
'''

BALLAST = r'''
import sys, time, torch
gb = float(sys.argv[1]); bufs = []
left = int(gb * 2**30)
while left > 0:
    n = min(left, 2**30); bufs.append(torch.empty(n, dtype=torch.uint8, device="cuda")); left -= n
torch.cuda.synchronize(); print("BALLAST_READY", flush=True)
while True: time.sleep(3600)
'''


def total_hbm_gb():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip()) / 1024


def run_point(args, k, rep):
    work = Path(args.work) / f"k{k}-r{rep}"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "events.jsonl"
    log.unlink(missing_ok=True)
    procs = []
    ballast = None
    if args.hbm_cap_gb:
        gb = total_hbm_gb() - args.hbm_cap_gb
        if gb > 0:
            ballast = subprocess.Popen([sys.executable, "-c", BALLAST, str(gb)], stdout=subprocess.PIPE, text=True)
            assert "BALLAST_READY" in ballast.stdout.readline()
    state_args = {"kind": args.kind, "gb": args.gb, "mace_model": args.mace_model,
                  "supercell": args.supercell, "ref": args.ref}
    t_stop = time.time() + args.warmup_s + args.duration
    for w in range(k):
        cfg = {"worker": w, "out": str(log), "state_args": state_args, "windows": args.windows,
               "seed": 1000 * rep + w, "t_stop": t_stop, "max_window_s": args.max_window_s}
        procs.append(subprocess.Popen([sys.executable, "-c", WORKER, str(TOOL_STATE), json.dumps(cfg)],
                                      stdout=subprocess.DEVNULL, stderr=open(work / f"w{w}.err", "w")))
    for p in procs:
        p.wait()
    if ballast:
        ballast.send_signal(signal.SIGTERM); ballast.wait()
    ev = [json.loads(l) for l in open(log)] if log.exists() else []
    t_meas = t_stop - args.duration
    lat = sorted(e["latency_s"] for e in ev if e["event"] == "call" and e["t"] >= t_meas)
    q = lambda f: lat[min(len(lat) - 1, int(f * len(lat)))] if lat else None
    return {"host": socket.gethostname(), "kind": args.kind, "k": k, "repeat": rep, "hbm_cap_gb": args.hbm_cap_gb,
            "mps_pipe_dir_set": "CUDA_MPS_PIPE_DIRECTORY" in os.environ, "mps_note": args.mps_note,
            "ready": sum(e["event"] == "ready" for e in ev), "oom": sum(e["event"] == "oom" for e in ev),
            "calls": len(lat), "p50_s": q(0.5), "p99_s": q(0.99),
            "mismatch": sum(1 for e in ev if e["event"] == "call" and not e["match"]),
            "mem_per_session_mb": [e["mem_reserved_mb"] for e in ev if e["event"] == "ready"][:1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["synthetic", "mace", "kernelbench"])
    p.add_argument("--gb", type=float, default=1.0)
    p.add_argument("--mace-model", default="medium-mpa-0")
    p.add_argument("--supercell", type=int, default=4)
    p.add_argument("--ref", default="")
    p.add_argument("--k", type=int, nargs="+", required=True)
    p.add_argument("--hbm-cap-gb", type=float, default=0.0, help="0 = no cap")
    p.add_argument("--windows", required=True, help="CSV of think-time samples (s), header row first")
    p.add_argument("--max-window-s", type=float, default=300.0)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--warmup-s", type=float, default=60.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--mps-note", default="", help="free text recorded with results, e.g. 'mps on'")
    p.add_argument("--work", default="/tmp/agentfn-m3")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for rep in range(1, args.repeats + 1):
        for k in args.k:
            rec = run_point(args, k, rep)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if rec["oom"] and rec["oom"] == k:
                break  # nothing fits beyond this point


if __name__ == "__main__":
    main()
