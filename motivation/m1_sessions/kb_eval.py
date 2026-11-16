"""KernelBench evaluation tool, run ON THE TOOL HOST by the agent (via kb_eval.sh).

Mirrors KernelBench's scripts/run_and_check.py local path (same defaults:
5 correctness trials, 100 perf trials, cuda_event timing, eager + torch.compile
reference timings), and additionally appends one JSON record per call to
$AGENTFN_TOOL_LOG with phase timings, so M1 can attribute GPU time to phases
and M4 can reuse agent-generated candidates.

    python kb_eval.py --ref ref.py --kernel model_new.py [--arch Hopper]
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
from kernelbench import eval as kernel_eval
from kernelbench import utils as kernel_utils
from kernelbench.timing import measure_ref_program_time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--arch", default=os.environ.get("KB_GPU_ARCH", "Hopper"))
    p.add_argument("--num-correct-trials", type=int, default=5)
    p.add_argument("--num-perf-trials", type=int, default=100)
    p.add_argument("--timing-method", default="cuda_event")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--backend", default="cuda")
    p.add_argument("--no-compile-ref", action="store_true")
    a = p.parse_args()

    ref_src = Path(a.ref).read_text()
    kernel_src = Path(a.kernel).read_text()
    device = torch.device("cuda:0")
    kernel_utils.set_gpu_arch([a.arch])
    rec = {"tool": "kb_eval", "host": socket.gethostname(), "pid": os.getpid(),
           "session": os.environ.get("AGENTFN_SESSION", ""), "ref": a.ref, "kernel": a.kernel,
           "t_start": time.time()}
    # keep every evaluated candidate (the agent overwrites model_new.py), for M4
    snap_dir = Path(a.kernel).resolve().parent / "candidates"
    snap_dir.mkdir(exist_ok=True)
    snap = snap_dir / f"{int(rec['t_start'] * 1000)}_{Path(a.kernel).name}"
    snap.write_text(kernel_src)
    rec["kernel_snapshot"] = str(snap)
    rec["ref_abs"] = str(Path(a.ref).resolve())

    t0 = time.time()
    torch.cuda.init()
    rec["ctx_init_s"] = time.time() - t0

    t0 = time.time()
    res = kernel_eval.eval_kernel_against_ref(
        ref_src, kernel_src, num_correct_trials=a.num_correct_trials,
        num_perf_trials=a.num_perf_trials, measure_performance=True,
        timing_method=a.timing_method, verbose=False, device=device,
        backend=a.backend, precision=kernel_eval.get_torch_dtype_from_string(a.precision))
    rec["eval_s"] = time.time() - t0
    rec["compiled"] = bool(getattr(res, "compiled", False))
    rec["correct"] = bool(getattr(res, "correctness", False))
    rec["kernel_ms"] = getattr(res, "runtime", None)
    rec["kernel_stats"] = getattr(res, "runtime_stats", None)

    t0 = time.time()
    eager = measure_ref_program_time("ref", ref_src, num_trials=a.num_perf_trials,
                                     use_torch_compile=False, timing_method=a.timing_method,
                                     device=device, precision=a.precision)
    rec["ref_eager_s"] = time.time() - t0
    rec["ref_eager_ms"] = eager.get("mean")
    if not a.no_compile_ref:
        t0 = time.time()
        comp = measure_ref_program_time("ref", ref_src, num_trials=a.num_perf_trials,
                                        use_torch_compile=True, timing_method=a.timing_method,
                                        device=device, precision=a.precision)
        rec["ref_compile_s"] = time.time() - t0
        rec["ref_compile_ms"] = comp.get("mean")
    rec["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    rec["t_end"] = time.time()

    log = os.environ.get("AGENTFN_TOOL_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    print(f"compiled={rec['compiled']} correct={rec['correct']}")
    print(f"custom kernel time: {rec['kernel_ms']} ms")
    print(f"reference eager time: {rec['ref_eager_ms']} ms")
    if "ref_compile_ms" in rec:
        print(f"reference torch.compile time: {rec['ref_compile_ms']} ms")
    if rec["correct"] and rec["kernel_ms"]:
        print(f"speedup over eager: {rec['ref_eager_ms'] / rec['kernel_ms']:.2f}x")
        if rec.get("ref_compile_ms"):
            print(f"speedup over torch.compile: {rec['ref_compile_ms'] / rec['kernel_ms']:.2f}x")
    elif getattr(res, "metadata", None):
        print("details:", json.dumps(res.metadata, default=str)[:4000])


if __name__ == "__main__":
    main()
