"""M4 (H5): does sharing a GPU with other agents' tool calls distort kernel timing
enough to mislead a kernel-optimizing agent?

Candidates are the kernels agents produced in M1 (kb_eval snapshots, correct ones
only) plus the eager reference of each problem. Each candidate is timed with
KernelBench's own timing (eval_kernel_against_ref, cuda_event, 100 trials):
  exclusive : nothing else on the GPU
  shared-K  : K co-runner sessions (M3 workers with public-trace think times)
Repeats are interleaved across conditions. --analyze reports, per kernel-time
bucket (>10 / >1 / >0.1 ms, as in KREX), the p95 timing inflation vs exclusive,
and per problem the Kendall tau between exclusive and shared rankings.

  python run_fidelity.py collect --sessions ~/agentfn/sessions --out cands.jsonl
  python run_fidelity.py run --cands cands.jsonl --k 0 4 8 16 --corunner-kind mace \
      --windows ../results/public/window_samples_freeinference.csv --repeats 5 --out results/m4/h100.jsonl
  python run_fidelity.py analyze --results results/m4/h100.jsonl
"""

import argparse
import itertools
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "m3_headroom"))


def collect(a):
    """Candidate list from M1 kb_eval logs: every correct snapshot + one eager reference per problem."""
    seen_ref = set()
    with open(a.out, "w") as f:
        for log in Path(a.sessions).expanduser().glob("*/kb_eval.jsonl"):
            for line in open(log):
                r = json.loads(line)
                if not r.get("correct") or not r.get("kernel_snapshot"):
                    continue
                ref = r.get("ref_abs") or str(log.parent / "ref.py")
                f.write(json.dumps({"problem": ref, "kernel": r["kernel_snapshot"], "kind": "agent"}) + "\n")
                if ref not in seen_ref:
                    seen_ref.add(ref)
                    eager = Path(ref).with_name("eager_as_new.py")
                    eager.write_text(Path(ref).read_text() + "\n\nModelNew = Model\n")
                    f.write(json.dumps({"problem": ref, "kernel": str(eager), "kind": "eager"}) + "\n")
    print("wrote", a.out)


def time_candidate(ref, kernel, trials, arch):
    import torch
    from kernelbench import eval as ke
    from kernelbench import utils as ku
    ku.set_gpu_arch([arch])
    res = ke.eval_kernel_against_ref(Path(ref).read_text(), Path(kernel).read_text(), num_correct_trials=1,
                                     num_perf_trials=trials, measure_performance=True, timing_method="cuda_event",
                                     device=torch.device("cuda:0"))
    return bool(res.correctness), res.runtime, res.runtime_stats


def start_corunners(a, k, seed):
    from run_headroom import WORKER, TOOL_STATE
    procs = []
    log = Path(a.work) / f"corunners-k{k}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    state_args = {"kind": a.corunner_kind, "gb": a.corunner_gb, "mace_model": "medium-mpa-0", "supercell": 4, "ref": ""}
    for w in range(k):
        cfg = {"worker": w, "out": str(log), "state_args": state_args, "windows": a.windows, "seed": seed + w,
               "t_stop": time.time() + 10 ** 7, "max_window_s": a.max_window_s}
        procs.append(subprocess.Popen([sys.executable, "-c", WORKER, str(TOOL_STATE), json.dumps(cfg)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(a.corunner_warmup_s)
    return procs


def run(a):
    cands = [json.loads(l) for l in open(a.cands)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for rep in range(1, a.repeats + 1):
        for k in a.k:  # conditions interleaved within each repeat
            procs = start_corunners(a, k, 1000 * rep) if k > 0 else []
            try:
                for c in cands:
                    t = time.time()
                    try:
                        ok, ms, stats = time_candidate(c["problem"], c["kernel"], a.trials, a.arch)
                        err = None
                    except Exception as e:  # keep, flagged
                        ok, ms, stats, err = False, None, None, repr(e)[:300]
                    rec = {"host": socket.gethostname(), **c, "k": k, "repeat": rep, "correct": ok,
                           "mean_ms": ms, "stats": stats, "error": err, "t": t}
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            finally:
                for p in procs:
                    p.terminate()
                for p in procs:
                    p.wait()


def kendall_tau(x, y):
    pairs = list(itertools.combinations(range(len(x)), 2))
    s = sum((1 if (x[i] - x[j]) * (y[i] - y[j]) > 0 else -1) for i, j in pairs if x[i] != x[j] and y[i] != y[j])
    return s / len(pairs) if pairs else None


def analyze(a):
    rows = [json.loads(l) for l in open(a.results)]
    rows = [r for r in rows if r["correct"] and r["mean_ms"]]
    med = defaultdict(list)
    for r in rows:
        med[(r["problem"], r["kernel"], r["k"])].append(r["mean_ms"])
    m = {key: sorted(v)[len(v) // 2] for key, v in med.items()}
    ks = sorted({r["k"] for r in rows} - {0})
    out = {}
    for k in ks:
        infl = defaultdict(list)
        for r in rows:
            if r["k"] != k:
                continue
            base = m.get((r["problem"], r["kernel"], 0))
            if not base:
                continue
            bucket = ">10ms" if base > 10 else ">1ms" if base > 1 else ">0.1ms" if base > 0.1 else "<=0.1ms"
            infl[bucket].append(r["mean_ms"] / base - 1)
        p95 = {b: sorted(v)[int(0.95 * (len(v) - 1))] for b, v in infl.items() if v}
        taus = []
        for prob in {p for p, _, kk in m if kk == 0}:
            ker = [kk for p, kk, kc in m if p == prob and kc == 0 and (p, kk, k) in m]
            if len(ker) >= 3:
                taus.append(kendall_tau([m[(prob, x, 0)] for x in ker], [m[(prob, x, k)] for x in ker]))
        out[f"k={k}"] = {"p95_inflation_by_bucket": p95, "n_problems_tau": len(taus),
                         "median_kendall_tau": sorted(taus)[len(taus) // 2] if taus else None}
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--sessions", required=True); c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cands", required=True); r.add_argument("--k", type=int, nargs="+", default=[0, 4, 8, 16])
    r.add_argument("--corunner-kind", default="mace", choices=["synthetic", "mace"])
    r.add_argument("--corunner-gb", type=float, default=1.0)
    r.add_argument("--corunner-warmup-s", type=float, default=60.0)
    r.add_argument("--windows", required=True); r.add_argument("--max-window-s", type=float, default=300.0)
    r.add_argument("--trials", type=int, default=100); r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--arch", default=os.environ.get("KB_GPU_ARCH", "Hopper"))
    r.add_argument("--work", default="/tmp/agentfn-m4"); r.add_argument("--out", required=True)
    an = sub.add_parser("analyze"); an.add_argument("--results", required=True)
    a = p.parse_args()
    {"collect": collect, "run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
