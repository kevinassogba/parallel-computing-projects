"""Analyze M1 runs. Inputs (one directory per workload run, copied to one place):
  sessions.jsonl            run_kernelbench.sh / run_chemgraph.sh (harness clock)
  llm_requests.jsonl        llm_proxy.py (harness clock)
  tool_calls.jsonl          ssh_env.py, kernel workload only (harness clock)
  traj/*.kb_eval.jsonl      kb_eval.py on the tool host (tool-host clock)
  chemgraph_tools.jsonl     chemgraph_mcp_gpu.py on the tool host (tool-host clock)
  gpu_devices.csv, gpu_procs.csv   gpu_sampler.py on the tool host (tool-host clock)
Tool-host times are mapped to the harness clock with the per-session offset
measured by the run scripts (mean of the before/after midpoint probes).

Outputs (JSON + CSV under --out): per workload, distributions of LLM latency,
GPU-tool duration, windows between consecutive GPU tool calls (the parking
window for GPU state), per-session GPU duty cycle under session-exclusive use,
per-call GPU state size S, and task correctness.

  python analyze_m1.py --run results/m1/kernelbench --run results/m1/chemgraph --out results/m1/summary
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def jl(path):
    p = Path(path)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def q(s, fs=(0.1, 0.5, 0.9, 0.99)):
    s = pd.Series(s).dropna()
    return {"n": int(len(s)), **{f"p{int(f * 100)}": float(s.quantile(f)) for f in fs}} if len(s) else {"n": 0}


def gpu_tool_intervals(run, ses):
    """GPU tool calls as (session, start, end, pid) on the harness clock."""
    off = {s["session"]: sum(s["clock_offset_s"]) / 2 for s in ses}
    rows = []
    for f in Path(run, "traj").glob("*.kb_eval.jsonl"):
        for r in jl(f):
            o = off.get(r["session"], 0.0)
            rows.append((r["session"], r["t_start"] - o, r["t_end"] - o, r["pid"], r.get("correct")))
    for r in jl(Path(run, "chemgraph_tools.jsonl")):
        if r.get("event") != "run_ase":
            continue
        # chemgraph MCP calls carry no session id: assign by time to the session that spans them
        for s in ses:
            o = off[s["session"]]
            if s["t_start"] <= r["t_start"] - o <= s["t_end"]:
                rows.append((s["session"], r["t_start"] - o, r["t_end"] - o, r["pid"], None))
                break
    return pd.DataFrame(rows, columns=["session", "t0", "t1", "pid", "correct"])


def duty_cycle(run, ses, gpu=0, thresh=0):
    p = Path(run, "gpu_devices.csv")
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d = d[d.gpu == gpu]
    out = []
    for s in ses:
        o = sum(s["clock_offset_s"]) / 2
        w = d[(d.t - o >= s["t_start"]) & (d.t - o <= s["t_end"])]
        if len(w):
            out.append({"session": s["session"], "busy_frac": float((w.util_gpu_pct > thresh).mean()),
                        "mean_util_pct": float(w.util_gpu_pct.mean()), "span_s": s["t_end"] - s["t_start"],
                        "mean_power_w": float(pd.to_numeric(w.power_w, errors="coerce").mean())})
    return pd.DataFrame(out)


def state_size(run, iv):
    p = Path(run, "gpu_procs.csv")
    if not p.exists() or iv.empty:
        return pd.Series(dtype=float)
    pr = pd.read_csv(p)
    pr["used_mb"] = pd.to_numeric(pr.used_mb, errors="coerce")
    return pr[pr.pid.isin(set(iv.pid))].groupby("pid").used_mb.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for run in a.run:
        ses = jl(Path(run, "sessions.jsonl"))
        if not ses:
            continue
        wl = ses[0]["workload"]
        sids = {s["session"] for s in ses}
        llm = pd.DataFrame([r for r in jl(Path(run, "llm_requests.jsonl")) if r.get("session") in sids])
        iv = gpu_tool_intervals(run, ses).sort_values(["session", "t0"])
        iv["dur_s"] = iv.t1 - iv.t0
        iv["next_t0"] = iv.groupby("session").t0.shift(-1)
        windows = (iv.next_t0 - iv.t1).dropna()
        dc = duty_cycle(run, ses)
        s_mb = state_size(run, iv)
        rec = {
            "sessions": len(ses),
            "llm_latency_s": q(llm.latency_s) if not llm.empty else {"n": 0},
            "llm_calls_per_session": q(llm.groupby("session").size()) if not llm.empty else {"n": 0},
            "gpu_tool_calls_per_session": q(iv.groupby("session").size()),
            "gpu_tool_duration_s": q(iv.dur_s),
            "gpu_tool_window_s": q(windows),
            "share_windows_ge": {f"{t}s": float((windows >= t).mean()) for t in (0.5, 1, 2, 5)} if len(windows) else {},
            "session_gpu_busy_frac": q(dc.busy_frac) if not dc.empty else {"n": 0},
            "session_mean_gpu_util_pct": q(dc.mean_util_pct) if not dc.empty else {"n": 0},
            "gpu_state_mb_per_process": q(s_mb),
        }
        if wl == "kernelbench":
            rec["kb_eval_correct_share"] = float(iv.correct.dropna().astype(bool).mean()) if iv.correct.notna().any() else None
        summary[wl] = rec
        iv.to_csv(out / f"{wl}_gpu_tool_calls.csv", index=False)
        windows.to_csv(out / f"{wl}_gpu_tool_windows.csv", index=False, header=["win_s"])
        if not dc.empty:
            dc.to_csv(out / f"{wl}_duty_cycle.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
