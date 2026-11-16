"""M5 analysis: per run (one policy, one seed) and per policy (mean ± sd over seeds).

Per run, from the files run_m5.py collects:
  outcome    kernel: re-measured true speedup of the agent's own best candidate
             (remeasure.jsonl from remeasure_kb.py); solved = correct and true speedup
             over eager > 1. chemgraph: cg_scores.jsonl (ChemGraph judge); solved = score 1.
  behaviour  GPU tool calls, all tool calls, LLM calls per session; share of sessions whose
             observations contain OOM / CUDA errors; perceived-vs-true speedup gap.
  demand     makespan, GPU-hours allocated (G x makespan), GPU energy (NVML power summed
             over the G GPUs during the run), session duration.
  efficiency solved tasks per GPU-hour and GPU joules per solved task (C2), next to the
             classic per-call metrics (tool calls per GPU-hour), so the two can disagree.
  confound   median LLM latency per run; if it drifts across policies, it is reported.

    python analyze_m5.py --runs results/m5/* --out results/m5/summary
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import pandas as pd

ERR = re.compile(r"out of memory|CUDA error|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|illegal memory access", re.I)


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def gpu_energy_j(run, g, t0, t1, off):
    f = next(iter(Path(run, "tool_host_logs").glob("gpu_devices.csv")), None)
    if f is None:
        return None
    d = pd.read_csv(f)
    d = d[(d.gpu < g) & (d.t - off >= t0) & (d.t - off <= t1)].copy()
    d["power_w"] = pd.to_numeric(d.power_w, errors="coerce")
    d = d.sort_values(["gpu", "t"])
    d["dt"] = d.groupby("gpu").t.diff().fillna(0).clip(upper=1.0)
    return float((d.power_w * d.dt).sum())


def one_run(run):
    m = json.load(open(Path(run, "manifest.json")))
    a = m["args"]
    ses = [s for s in jl(Path(run, "sessions.jsonl")) if "session" in s]
    if not ses:
        return None
    sids = {s["session"] for s in ses}
    g = a["gpus"]
    t0, t1 = m["t_start"], m["t_end"]
    off = (m["clock_offset_start_s"] + m["clock_offset_end_s"]) / 2
    llm = pd.DataFrame([r for f in Path(run).glob("llm_requests_slot*.jsonl") for r in jl(f) if r.get("session") in sids])
    rec = {"run": m["run"], "policy": a["policy"], "workload": a["workload"], "seed": a["seed"],
           "sessions_per_gpu": 1 if a["policy"] == "exclusive" else a["sessions_per_gpu"],
           "cap_gb": a["cap_gb"] if a["policy"] == "capped" else None, "mps": a["mps"],
           "sessions": len(ses), "gpus": g, "makespan_s": t1 - t0, "gpu_hours": g * (t1 - t0) / 3600,
           "session_duration_s_median": median(s["t_end"] - s["t_start"] for s in ses),
           "llm_calls_per_session": (len(llm) / len(ses)) if len(llm) else None,
           "llm_latency_s_median": float(llm.latency_s.median()) if len(llm) else None}
    errs = 0
    if a["workload"] == "kernelbench":
        outc = {r["session"]: r for r in jl(Path(run, "tool_host_logs", "remeasure.jsonl"))}
        tool = pd.DataFrame(jl(Path(run, "tool_calls.jsonl")))
        tool = tool[tool.tool == "bash"] if len(tool) else tool
        gpu_calls = sum(len(jl(f)) for f in Path(run, "traj").glob("*.kb_eval.jsonl"))
        solved, gap, true_sp = 0, [], []
        for s in ses:
            o = outc.get(s["session"], {})
            ts = o.get("true_speedup_eager")
            if ts is not None:
                true_sp.append(ts)
                if o.get("perceived_speedup_eager"):
                    gap.append(o["perceived_speedup_eager"] / ts - 1)
            solved += bool(o.get("still_correct") and ts and ts > 1)
            tj = Path(run, "traj", f"{s['session']}.json")
            errs += bool(tj.exists() and ERR.search(tj.read_text(errors="replace")))
        rec.update(solved=solved, tool_calls_per_session=len(tool) / len(ses) if len(tool) else None,
                   gpu_tool_calls_per_session=gpu_calls / len(ses),
                   true_speedup_eager_mean=mean(true_sp) if true_sp else None,
                   perceived_over_true_median=median(gap) if gap else None,
                   outcome_coverage=len(outc) / len(ses))
        gpu_calls_total = gpu_calls
    else:
        sc = {r["session"]: r for r in jl(Path(run, "cg_scores.jsonl"))}
        tools = [r for f in Path(run, "tool_host_logs").glob("tools_slot*.jsonl") for r in jl(f)]
        runase = [r for r in tools if r.get("event") == "run_ase"]
        park = [r for r in jl(Path(run, "tool_host_logs", "parker.jsonl")) if r.get("op") == "restore" and r.get("was_parked")]
        solved = sum(1 for s in ses if sc.get(s["session"], {}).get("correct"))
        for s in ses:
            lg = Path(run, "traj", f"{s['session']}.log")
            errs += bool(lg.exists() and ERR.search(lg.read_text(errors="replace")))
        rec.update(solved=solved, gpu_tool_calls_per_session=len(runase) / len(ses),
                   restore_s_median=median(r["restore_s"] + r["unlock_s"] for r in park) if park else None,
                   outcome_coverage=len(sc) / len(ses))
        gpu_calls_total = len(runase)
    e = gpu_energy_j(run, g, t0, t1, off)
    rec.update(success_rate=rec["solved"] / len(ses), sessions_with_gpu_errors=errs / len(ses),
               gpu_energy_j=e, solved_per_gpu_hour=rec["solved"] / rec["gpu_hours"],
               gpu_j_per_solved=(e / rec["solved"]) if (e and rec["solved"]) else None,
               gpu_tool_calls_per_gpu_hour=gpu_calls_total / rec["gpu_hours"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in (one_run(x) for x in a.runs if Path(x, "manifest.json").exists()) if r]
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    keys = ["workload", "policy", "sessions_per_gpu", "cap_gb", "mps"]
    metrics = ["success_rate", "solved_per_gpu_hour", "gpu_j_per_solved", "gpu_tool_calls_per_gpu_hour",
               "gpu_tool_calls_per_session", "sessions_with_gpu_errors", "llm_latency_s_median", "makespan_s"]
    agg = []
    for k, grp in df.groupby(keys, dropna=False):
        r = dict(zip(keys, k))
        r["n_seeds"] = len(grp)
        for mname in metrics:
            v = [x for x in grp[mname] if x is not None and x == x]
            r[f"{mname}_mean"] = mean(v) if v else None
            r[f"{mname}_sd"] = stdev(v) if len(v) >= 2 else None
        r["provisional"] = len(grp) < 3
        agg.append(r)
    pd.DataFrame(agg).to_csv(out / "policies.csv", index=False)
    print(pd.DataFrame(agg).to_string())


if __name__ == "__main__":
    main()
