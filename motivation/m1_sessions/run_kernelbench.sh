#!/usr/bin/env bash
# M1 (kernel agent): run mini-SWE-agent sessions on the HARNESS HOST; every agent
# command executes on the TOOL HOST over SSH; the LLM is Dynamo on the LLM HOST,
# reached through the timing proxy (common/llm_proxy.py) on this host.
#
#   bash run_kernelbench.sh TOOL_HOST PROBLEMS_FILE OUT_DIR [REPEATS]
#
# PROBLEMS_FILE: one "level problem_id" pair per line (see problems_kb.txt).
# Prerequisites: setup/harness_host.sh install, setup/tool_host.sh install on
# TOOL_HOST, proxy running (setup/harness_host.sh proxy LLM_HOST).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../setup/versions.env"
TOOL_HOST="$1"; PROBLEMS="$2"; OUT="$3"; REPEATS="${4:-3}"
VENV="${VENV:-$HOME/agentfn-harness-venv}"
MAX_EVALS="${MAX_EVALS:-10}"
STEP_LIMIT="${STEP_LIMIT:-60}"
SESSION_FILE="${SESSION_FILE:-/tmp/agentfn_session}"
mkdir -p "$OUT/traj"
cp "$HERE/../setup/versions.env" "$OUT/"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export MSWEA_COST_TRACKING=ignore_errors

clock_offset() {  # tool-host clock minus harness clock, midpoint method (seconds)
  local t0 t1 tr
  t0=$(date +%s.%N); tr=$(ssh -n -o BatchMode=yes "$TOOL_HOST" date +%s.%N); t1=$(date +%s.%N)
  python3 -c "print($tr - ($t0 + $t1) / 2)"
}

task_text=$(sed "s/{max_evals}/$MAX_EVALS/" "$HERE/kb_task.txt")
while read -r level pid <&3; do
  [[ -z "${level:-}" || "$level" == \#* ]] && continue
  for ((r = 1; r <= REPEATS; r++)); do
    sid="kb-L${level}-P${pid}-r${r}-$(date +%s)"
    echo "$sid" > "$SESSION_FILE"
    wd=$(ssh -n -o BatchMode=yes "$TOOL_HOST" "source ~/agentfn/env.sh && agentfn-kb-prepare $level $pid $sid")
    off0=$(clock_offset)
    t0=$(date +%s.%N)
    AGENTFN_SESSION="$sid" "$VENV/bin/mini" -c mini.yaml \
      --environment-class ssh_env.SSHEnvironment \
      -c environment.host="$TOOL_HOST" -c environment.remote_cwd="$wd" \
      -c environment.log="$OUT/tool_calls.jsonl" -c environment.session="$sid" \
      -m "hosted_vllm/$LLM_MODEL" \
      -c model.model_kwargs.api_base="http://127.0.0.1:$PROXY_PORT/v1" \
      -c agent.step_limit="$STEP_LIMIT" -c agent.cost_limit=0 \
      -t "$task_text" -y --exit-immediately -o "$OUT/traj/$sid.json" \
      < /dev/null > "$OUT/traj/$sid.log" 2>&1 || echo "session $sid exited non-zero (kept)"
    t1=$(date +%s.%N)
    off1=$(clock_offset)
    scp -q "$TOOL_HOST:$wd/kb_eval.jsonl" "$OUT/traj/$sid.kb_eval.jsonl" 2>/dev/null || true
    echo "{\"session\":\"$sid\",\"workload\":\"kernelbench\",\"level\":$level,\"problem\":$pid,\"repeat\":$r,\"t_start\":$t0,\"t_end\":$t1,\"clock_offset_s\":[$off0,$off1]}" \
      >> "$OUT/sessions.jsonl"
  done
done 3< "$PROBLEMS"
