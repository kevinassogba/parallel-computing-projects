"""M5 ground truth for the kernel workload (run ON THE TOOL HOST, idle GPU, after a run).

What an agent *believes* it achieved comes from timings taken under the run's policy.
The true outcome is re-measured here under exclusive conditions: for each session,
the candidate the agent itself ranked best (correct, highest perceived speedup) is
re-timed together with the reference, `--repeats` times, with nothing else on the GPU.

    ~/agentfn/kb/bin/python remeasure_kb.py --run-prefix m5-<run_id>- --gpu 0 --repeats 3 \
        --out ~/agentfn/logs/m5/<run_id>/remeasure.jsonl
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path.home() / "agentfn"


def perceived_best(log):
    best = None
    for line in open(log):
        r = json.loads(line)
        if r.get("correct") and r.get("kernel_ms") and r.get("ref_eager_ms"):
            s = r["ref_eager_ms"] / r["kernel_ms"]
            if best is None or s > best[0]:
                best = (s, r)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-prefix", required=True, help="session-id prefix of the run, e.g. m5-kb_shared_k4_s1-")
    p.add_argument("--gpu", default="0")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
    for d in sorted((ROOT / "sessions").glob(a.run_prefix + "*")):
        log = d / "kb_eval.jsonl"
        rec = {"session": d.name}
        best = perceived_best(log) if log.exists() else None
        rec["n_evals"] = sum(1 for _ in open(log)) if log.exists() else 0
        if best is None:
            rec.update(solved_correct=False)
        else:
            rec.update(solved_correct=True, perceived_speedup_eager=best[0],
                       perceived_speedup_compile=(best[1]["ref_compile_ms"] / best[1]["kernel_ms"])
                       if best[1].get("ref_compile_ms") else None)
            tlog = d / "remeasure_tmp.jsonl"
            tlog.unlink(missing_ok=True)
            for _ in range(a.repeats):
                subprocess.run([str(ROOT / "kb" / "bin" / "python"), str(ROOT / "kb_eval.py"),
                                "--ref", best[1]["ref_abs"], "--kernel", best[1]["kernel_snapshot"]],
                               cwd=d, env=dict(env, AGENTFN_TOOL_LOG=str(tlog), AGENTFN_SESSION="remeasure"),
                               capture_output=True, text=True)
            rs = [json.loads(l) for l in open(tlog)] if tlog.exists() else []
            ok = [r for r in rs if r.get("correct") and r.get("kernel_ms")]
            if ok:
                se = sorted(r["ref_eager_ms"] / r["kernel_ms"] for r in ok)
                sc = sorted(r["ref_compile_ms"] / r["kernel_ms"] for r in ok if r.get("ref_compile_ms"))
                rec.update(true_speedup_eager=se[len(se) // 2], true_speedup_compile=sc[len(sc) // 2] if sc else None,
                           remeasure_n=len(ok), still_correct=True)
            else:
                rec.update(still_correct=False, remeasure_n=0)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))


if __name__ == "__main__":
    main()
