"""Derive, from public agent traces, the quantities M1 would otherwise measure:

  1. LLM call latency as seen by the harness (FreeInference: gateway timing of
     real remote providers).
  2. Tool-call latency by tool category (FreeInference latency_ms; TraceLab
     tool_wall_latency_ms).
  3. Parking windows: for a tool environment, the idle gap between the end of one
     tool call and the start of the next tool call in the same session. This is
     the time available to park and restore tool state (H3). Windows that contain
     a human turn are separated from autonomous ones.
  4. Tool busy fraction per session: share of session wall time with any tool
     executing — the upper bound on how busy a session-exclusive tool resource is
     (cf. KREX: GPU commands = 19.3% of session time, GPU busy = 3.4%).
  5. How often coding-agent tool calls touch the GPU at all (programs such as
     nvcc, nvidia-smi, ncu, nsys, torchrun, srun/sbatch).

Usage: python analyze.py --data ../data --out ../results/public
"""

import argparse
import json
from pathlib import Path

import duckdb

GPU_PROGRAMS = ["nvcc", "nvidia-smi", "ncu", "nsys", "nvprof", "cuda-gdb", "cuobjdump",
                "torchrun", "deepspeed", "accelerate", "srun", "sbatch", "salloc", "mpirun"]
WINDOW_BINS = [0, 1, 2, 5, 10, 30, 60, 300, 1e12]  # seconds
BUSY_BINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1e12]  # last bin: >= 0.75 (span estimates can exceed 1 slightly)


def bin_sql(col, edges):
    cases = " ".join(f"WHEN {col} < {hi} THEN '{lo:g}-{hi:g}'" if hi < 1e11 else f"WHEN TRUE THEN '>{lo:g}'"
                     for lo, hi in zip(edges[:-1], edges[1:]))
    return f"CASE {cases} END"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "public"))
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("SET threads TO 8")
    fi = data / "freeinference"
    con.sql(f"""CREATE VIEW fi_req AS SELECT session_id, step, start_ms, end_ms, ttft_ms, step_trigger,
                 prompt_tokens, completion_tokens, model FROM read_parquet('{fi}/*/requests.parquet')""")
    con.sql(f"CREATE VIEW fi_tc AS SELECT * FROM read_parquet('{fi}/*/tool_calls.parquet')")
    con.sql(f"CREATE VIEW fi_ses AS SELECT * FROM read_parquet('{fi}/*/sessions.parquet')")
    con.sql(f"ATTACH '{data / 'tracelab' / 'syfi_coding_trace.duckdb'}' AS tl (READ_ONLY)")
    summary = {}

    # 1. LLM latency (FreeInference)
    con.sql("""CREATE TABLE llm AS SELECT session_id, step, (end_ms - start_ms) / 1000.0 AS lat_s,
               ttft_ms / 1000.0 AS ttft_s, completion_tokens, model FROM fi_req
               WHERE end_ms > start_ms""")
    summary["fi_llm_latency_s"] = dict(zip(
        ["n", "p10", "p50", "p90", "p99", "mean"],
        con.execute("""SELECT count(*), quantile_cont(lat_s,0.1), median(lat_s), quantile_cont(lat_s,0.9),
                   quantile_cont(lat_s,0.99), avg(lat_s) FROM llm""").fetchone()))

    # 2. Tool latency by category
    con.sql("""COPY (SELECT 'freeinference' AS trace, category, count(*) AS n,
                 median(latency_ms)/1000.0 AS p50_s, quantile_cont(latency_ms,0.9)/1000.0 AS p90_s,
                 avg(latency_ms)/1000.0 AS mean_s FROM fi_tc WHERE latency_ms IS NOT NULL
                 GROUP BY 1,2 ORDER BY n DESC) TO '{}' (HEADER)""".format(out / "tool_latency_by_category.csv"))
    summary["tl_tool_latency_s"] = dict(zip(
        ["n", "p50", "p90", "p99"],
        con.execute("""SELECT count(*), median(tool_wall_latency_ms)/1000.0, quantile_cont(tool_wall_latency_ms,0.9)/1000.0,
                   quantile_cont(tool_wall_latency_ms,0.99)/1000.0 FROM tl.tool_calls
                   WHERE tool_wall_latency_ms IS NOT NULL""").fetchone()))

    # 3a. Parking windows, TraceLab (exact timestamps)
    con.sql("""CREATE TABLE tl_tc AS SELECT r.session_id, r.round_index, t.tool_index, t.emitted_at, t.result_at,
               t.tool_wall_latency_ms, r.provider FROM tl.tool_calls t JOIN tl.rounds r USING (round_pk)
               WHERE t.emitted_at IS NOT NULL AND t.result_at IS NOT NULL AND t.result_at >= t.emitted_at""")
    con.sql("""CREATE TABLE tl_user AS SELECT r.session_id, e.timestamp AS ts FROM tl.timing_events e
               JOIN tl.rounds r USING (round_pk) WHERE e.event_type = 'user_message'""")
    con.sql("""CREATE TABLE tl_win AS
               WITH ordered AS (
                 SELECT session_id, provider, emitted_at, result_at,
                        lead(emitted_at) OVER (PARTITION BY session_id ORDER BY emitted_at, tool_index) AS next_start,
                        max(result_at) OVER (PARTITION BY session_id ORDER BY emitted_at, tool_index
                                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS busy_until
                 FROM tl_tc)
               SELECT o.session_id, o.provider, epoch(o.next_start) - epoch(o.busy_until) AS win_s,
                      EXISTS (SELECT 1 FROM tl_user u WHERE u.session_id = o.session_id
                              AND u.ts > o.busy_until AND u.ts < o.next_start) AS has_user
               FROM ordered o WHERE o.next_start IS NOT NULL AND o.next_start > o.busy_until""")

    # 3b. Parking windows, FreeInference (tool start ~ end of the request that emitted it)
    con.sql("""CREATE TABLE fi_steps AS
               SELECT r.session_id, r.step, r.end_ms AS t0, r.end_ms + max(c.latency_ms) AS t1
               FROM fi_req r JOIN fi_tc c USING (session_id, step)
               WHERE c.latency_ms IS NOT NULL AND c.category NOT IN ('Subagent', 'Ask User')
               GROUP BY 1, 2, 3""")
    con.sql("""CREATE TABLE fi_win AS
               WITH o AS (SELECT session_id, step, t0, t1,
                            lead(t0) OVER (PARTITION BY session_id ORDER BY step) AS next_t0,
                            lead(step) OVER (PARTITION BY session_id ORDER BY step) AS next_step
                          FROM fi_steps)
               SELECT o.session_id, (o.next_t0 - o.t1) / 1000.0 AS win_s,
                      EXISTS (SELECT 1 FROM fi_req q WHERE q.session_id = o.session_id AND q.step > o.step
                              AND q.step <= o.next_step AND q.step_trigger = 'user') AS has_user
               FROM o WHERE o.next_t0 IS NOT NULL AND o.next_t0 > o.t1""")

    rows = []
    for name in ("tl_win", "fi_win"):
        trace = "tracelab" if name == "tl_win" else "freeinference"
        q = con.execute(f"""SELECT count(*), median(win_s), quantile_cont(win_s,0.1), quantile_cont(win_s,0.9),
                        avg(CASE WHEN win_s >= 1 THEN 1 ELSE 0 END), avg(CASE WHEN win_s >= 2 THEN 1 ELSE 0 END),
                        avg(CASE WHEN win_s >= 5 THEN 1 ELSE 0 END)
                        FROM {name} WHERE NOT has_user""").fetchone()
        summary[f"{trace}_autonomous_window_s"] = dict(zip(
            ["n", "p50", "p10", "p90", "share_ge_1s", "share_ge_2s", "share_ge_5s"], q))
        summary[f"{trace}_windows_with_user_turn"] = con.execute(f"SELECT count(*) FROM {name} WHERE has_user").fetchone()[0]
        rows += con.sql(f"""SELECT '{trace}' AS trace, {bin_sql('win_s', WINDOW_BINS)} AS bin,
                            count(*) AS n FROM {name} WHERE NOT has_user GROUP BY 1, 2""").fetchall()
    with open(out / "parking_windows_binned.csv", "w") as f:
        f.write("trace,bin,n\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    # 4. Tool busy fraction per session (union of tool intervals / session span), TraceLab
    con.sql("""CREATE TABLE tl_busy AS
               WITH iv AS (SELECT session_id, epoch(emitted_at) s, epoch(result_at) e FROM tl_tc),
               m AS (SELECT session_id, s, e, max(e) OVER (PARTITION BY session_id ORDER BY s, e
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max FROM iv),
               g AS (SELECT session_id, s, e, sum(CASE WHEN prev_max IS NULL OR s > prev_max THEN 1 ELSE 0 END)
                       OVER (PARTITION BY session_id ORDER BY s, e) AS grp FROM m),
               u AS (SELECT session_id, min(s) s, max(e) e FROM g GROUP BY session_id, grp),
               span AS (SELECT r.session_id, min(epoch(ev.timestamp)) s, max(epoch(ev.timestamp)) e
                        FROM tl.timing_events ev JOIN tl.rounds r USING (round_pk) GROUP BY 1)
               SELECT u.session_id, sum(u.e - u.s) AS busy_s, any_value(span.e - span.s) AS span_s,
                      sum(u.e - u.s) / nullif(any_value(span.e - span.s), 0) AS busy_frac
               FROM u JOIN span USING (session_id) GROUP BY 1 HAVING any_value(span.e - span.s) > 60""")
    summary["tl_session_tool_busy_frac"] = dict(zip(
        ["n_sessions", "p10", "p50", "p90", "mean", "time_weighted"],
        con.execute("""SELECT count(*), quantile_cont(busy_frac,0.1), median(busy_frac), quantile_cont(busy_frac,0.9),
                   avg(busy_frac), sum(busy_s)/sum(span_s) FROM tl_busy""").fetchone()))
    con.sql(f"""COPY (SELECT {bin_sql('busy_frac', BUSY_BINS)} AS bin, count(*) AS n FROM tl_busy GROUP BY 1)
                TO '{out / 'tool_busy_fraction_binned.csv'}' (HEADER)""")
    # Same, but over the agent's active time only (tool time + autonomous gaps; human waits removed)
    con.sql("""CREATE TABLE tl_active AS
               SELECT b.session_id, b.busy_s / (b.busy_s + w.auto_s) AS active_busy_frac, b.busy_s + w.auto_s AS active_s
               FROM tl_busy b JOIN (SELECT session_id, sum(win_s) AS auto_s FROM tl_win WHERE NOT has_user
                                    GROUP BY 1) w USING (session_id) WHERE b.busy_s + w.auto_s > 60""")
    summary["tl_session_tool_busy_frac_active_time"] = dict(zip(
        ["n_sessions", "p10", "p50", "p90", "mean", "time_weighted"],
        con.execute("""SELECT count(*), quantile_cont(active_busy_frac,0.1), median(active_busy_frac),
                       quantile_cont(active_busy_frac,0.9), avg(active_busy_frac),
                       sum(active_busy_frac * active_s) / sum(active_s) FROM tl_active""").fetchone()))
    con.sql(f"""COPY (SELECT {bin_sql('active_busy_frac', BUSY_BINS)} AS bin, count(*) AS n FROM tl_active GROUP BY 1)
                TO '{out / 'tool_busy_fraction_active_binned.csv'}' (HEADER)""")

    # 5. GPU-indicative programs
    gp = ", ".join(f"'{x}'" for x in GPU_PROGRAMS)
    summary["fi_gpu_program_calls"] = dict(zip(["calls", "share_of_shell_calls", "sessions", "share_of_sessions"],
        con.execute(f"""WITH sh AS (SELECT session_id, list_has_any(programs, [{gp}]) AS gpu FROM fi_tc
                                WHERE programs IS NOT NULL AND len(programs) > 0)
                    SELECT sum(gpu::INT), avg(gpu::INT), count(DISTINCT CASE WHEN gpu THEN session_id END),
                           count(DISTINCT CASE WHEN gpu THEN session_id END) / count(DISTINCT session_id)
                    FROM sh""").fetchone()))
    summary["tl_gpu_program_calls"] = dict(zip(["calls", "share_of_shell_calls"],
        con.execute(f"""SELECT sum(list_has_any(executables, [{gp}])::INT), avg(list_has_any(executables, [{gp}])::INT)
                    FROM tl.tool_calls WHERE executables IS NOT NULL AND len(executables) > 0""").fetchone()))
    con.sql(f"""COPY (SELECT p AS program, count(*) AS n FROM (SELECT unnest(programs) p FROM fi_tc)
                WHERE p IN ({gp}) GROUP BY 1 ORDER BY 2 DESC) TO '{out / 'gpu_programs_freeinference.csv'}' (HEADER)""")
    con.sql(f"""COPY (SELECT p AS program, count(*) AS n FROM (SELECT unnest(executables) p FROM tl.tool_calls)
                WHERE p IN ({gp}) GROUP BY 1 ORDER BY 2 DESC) TO '{out / 'gpu_programs_tracelab.csv'}' (HEADER)""")

    # Think-time samples for M3 (autonomous windows, FreeInference; fixed seed)
    con.sql(f"""COPY (SELECT win_s FROM (SELECT win_s FROM fi_win WHERE NOT has_user)
                      USING SAMPLE reservoir(100000 ROWS) REPEATABLE (42))
                TO '{out / 'window_samples_freeinference.csv'}' (HEADER)""")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
