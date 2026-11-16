"""M2: how long does it take to park an agent tool's warm GPU state in a lower
memory tier and bring it back, compared with keeping it resident or cold-starting?

Per trial (one tool_state.py process per trial):
  resident : probe latency with the state kept in HBM (reference)
  driver   : cuda-checkpoint lock+checkpoint (device memory -> host memory owned by
             the driver; on GH200 that host memory is LPDDR5X over NVLink-C2C,
             on x86 it is DDR over PCIe), then restore+unlock, then probe
  criu:DIR : as `driver`, then CRIU dumps the process image to DIR (a Gen5 NVMe
             mount, or /dev/shm for a DRAM baseline) and restores it; page cache
             dropped before restore when --drop-caches (root)
  cold     : kill and relaunch the tool process (what job-per-call pays)
Every restored process must reproduce its pre-park checksum (correctness gate);
trials that fail are kept but flagged and excluded from summaries.

  python bench_park_restore.py --kind synthetic --gb 1 2 4 8 16 32 --methods driver cold \
      --repeats 5 --out results/m2/h100.jsonl
  sudo -E python bench_park_restore.py --kind mace --methods driver criu:/mnt/nvme/agentfn criu:/dev/shm/agentfn \
      --drop-caches --repeats 5 --out results/m2/gh200.jsonl
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = shutil.which("cuda-checkpoint") or os.path.expanduser("~/agentfn/bin/cuda-checkpoint")


def sh(cmd, check=True):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> {r.returncode}: {r.stderr.strip()[:500]}")
    return dt, r


def gpu_used_mb(pid):
    _, r = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], check=False)
    for line in r.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return float(parts[1])
    return 0.0


def rss_mb(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return None


def wait_file(path, timeout):
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(str(path))
        time.sleep(0.002)
    return json.loads(path.read_text())


def spawn(args, sd):
    if sd.exists():
        shutil.rmtree(sd)
    sd.mkdir(parents=True)
    cmd = [sys.executable, str(HERE / "tool_state.py"), "--kind", args.kind, "--state-dir", str(sd)]
    if args.kind == "synthetic":
        cmd += ["--gb", str(args.cur_gb)]
    elif args.kind == "mace":
        cmd += ["--mace-model", args.mace_model, "--supercell", str(args.supercell)]
    else:
        cmd += ["--ref", args.ref]
    t0 = time.time()
    log = open(sd / "proc.log", "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    ready = wait_file(sd / "ready.json", args.timeout)
    ready["spawn_to_ready_s"] = time.time() - t0
    return proc, ready


def probe(pid, sd, n):
    os.kill(pid, signal.SIGUSR1)
    return wait_file(sd / f"probe_{n}.json", 600)


def criu(args, *a):
    base = ["criu"] if os.geteuid() == 0 else ["sudo", "-n", "criu"]
    return sh(base + list(a))


def trial(args, method, rep, out):
    sd = Path(args.work) / f"{args.kind}-{os.getpid()}"
    proc, ready = spawn(args, sd)
    pid = proc.pid
    rec = {"host": socket.gethostname(), "kind": args.kind, "gb": args.cur_gb if args.kind == "synthetic" else None,
           "method": method, "repeat": rep, "pid": pid, **{f"ready_{k}": v for k, v in ready.items() if k != "pid"}}
    n = 0
    try:
        for _ in range(args.warm_probes):
            n += 1
            rec.setdefault("resident_probe_s", []).append(probe(pid, sd, n)["latency_s"])
        rec["gpu_used_mb_before"] = gpu_used_mb(pid)
        rec["rss_mb_before"] = rss_mb(pid)
        if method == "cold":
            proc.terminate(); proc.wait()
            proc, ready2 = spawn(args, sd)
            pid = proc.pid
            rec.update(cold_spawn_to_ready_s=ready2["spawn_to_ready_s"], cold_ctx_init_s=ready2["ctx_init_s"],
                       cold_build_s=ready2["build_s"], cold_first_probe_s=ready2["first_probe_s"],
                       correct=ready2["checksum"] == ready["checksum"])
            return rec
        rec["lock_s"], _ = sh([CKPT, "--action", "lock", "--pid", str(pid), "--timeout", "30000"])
        rec["checkpoint_s"], _ = sh([CKPT, "--action", "checkpoint", "--pid", str(pid)])
        rec["gpu_used_mb_parked"] = gpu_used_mb(pid)
        rec["rss_mb_parked"] = rss_mb(pid)
        if method.startswith("criu:"):
            d = Path(method.split(":", 1)[1]) / f"img-{pid}-{rep}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            rec["criu_dump_s"], _ = criu(args, "dump", "-t", str(pid), "-D", str(d), "--shell-job",
                                         "--libdir", "/nonexistent", "-o", "dump.log")
            proc.wait()
            rec["image_mb"] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**20
            time.sleep(args.park_s)
            if args.drop_caches:
                subprocess.run(["sync"]); Path("/proc/sys/vm/drop_caches").write_text("3\n")
            rec["criu_restore_s"], _ = criu(args, "restore", "-D", str(d), "--shell-job", "--restore-detached",
                                            "--libdir", "/nonexistent", "-o", "restore.log")
            if not args.keep_images:
                shutil.rmtree(d, ignore_errors=True)
        else:
            time.sleep(args.park_s)
        rec["restore_s"], _ = sh([CKPT, "--action", "restore", "--pid", str(pid)])
        rec["unlock_s"], _ = sh([CKPT, "--action", "unlock", "--pid", str(pid)])
        n += 1
        pr = probe(pid, sd, n)
        rec["post_restore_probe_s"] = pr["latency_s"]
        rec["correct"] = bool(pr["match"])
        rec["gpu_used_mb_after"] = gpu_used_mb(pid)
        return rec
    except Exception as e:  # keep failed trials, flagged
        rec["error"] = repr(e)
        rec["correct"] = False
        return rec
    finally:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["synthetic", "mace", "kernelbench"])
    p.add_argument("--gb", type=float, nargs="*", default=[1.0])
    p.add_argument("--mace-model", default="medium-mpa-0")
    p.add_argument("--supercell", type=int, default=4)
    p.add_argument("--ref", default="")
    p.add_argument("--methods", nargs="+", default=["driver", "cold"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warm-probes", type=int, default=3)
    p.add_argument("--park-s", type=float, default=1.0)
    p.add_argument("--drop-caches", action="store_true")
    p.add_argument("--keep-images", action="store_true")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--work", default="/tmp/agentfn-m2")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sizes = args.gb if args.kind == "synthetic" else [None]
    for gb in sizes:
        args.cur_gb = gb
        for rep in range(1, args.repeats + 1):
            for m in args.methods:  # interleave methods within a repeat
                rec = trial(args, m, rep, args.out)
                print(json.dumps({k: rec.get(k) for k in ("kind", "gb", "method", "repeat", "correct",
                                  "checkpoint_s", "restore_s", "criu_dump_s", "criu_restore_s",
                                  "cold_spawn_to_ready_s", "error")}), flush=True)


if __name__ == "__main__":
    main()
