#!/usr/bin/env bash
# M1 (science agent): run ChemGraph on the HARNESS HOST over the 40 queries of
# ChemGraph's public ground-truth set. Tools run on the TOOL HOST behind the MCP
# server (chemgraph_mcp_gpu.py, reached through an SSH tunnel); the LLM is Dynamo
# on the LLM HOST, reached through the timing proxy on this host.
#
#   bash run_chemgraph.sh TOOL_HOST OUT_DIR [REPEATS]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../setup/versions.env"
TOOL_HOST="$1"; OUT="$2"; REPEATS="${3:-3}"
VENV="${VENV:-$HOME/agentfn-harness-venv}"
SESSION_FILE="${SESSION_FILE:-/tmp/agentfn_session}"
GT_URL="https://huggingface.co/datasets/Autonomous-Scientific-Agents/groundtruth/resolve/main/output_data.json"
mkdir -p "$OUT/runs"
cp "$HERE/../setup/versions.env" "$OUT/"
[[ -f "$OUT/groundtruth.json" ]] || curl -L --fail -s -o "$OUT/groundtruth.json" "$GT_URL"

# Tunnel to the MCP server on the tool host (server binds 127.0.0.1 there).
ssh -n -o BatchMode=yes -o ExitOnForwardFailure=yes -f -N \
  -L "$CHEMGRAPH_MCP_PORT:127.0.0.1:$CHEMGRAPH_MCP_PORT" "$TOOL_HOST"
trap 'pkill -f "L $CHEMGRAPH_MCP_PORT:127.0.0.1:$CHEMGRAPH_MCP_PORT" || true' EXIT

clock_offset() {
  local t0 t1 tr
  t0=$(date +%s.%N); tr=$(ssh -n -o BatchMode=yes "$TOOL_HOST" date +%s.%N); t1=$(date +%s.%N)
  python3 -c "print($tr - ($t0 + $t1) / 2)"
}

n=$(python3 -c "import json;print(len(json.load(open('$OUT/groundtruth.json'))))")
for ((r = 1; r <= REPEATS; r++)); do
  for ((i = 0; i < n; i++)); do
    qid=$(python3 -c "import json;print(json.load(open('$OUT/groundtruth.json'))[$i]['id'])")
    query=$(python3 -c "import json;print(json.load(open('$OUT/groundtruth.json'))[$i]['query'])")
    sid="cg-q${qid}-r${r}-$(date +%s)"
    echo "$sid" > "$SESSION_FILE"
    off0=$(clock_offset); t0=$(date +%s.%N)
    "$VENV/bin/chemgraph" run --model "$LLM_MODEL" \
      --base-url "http://127.0.0.1:$PROXY_PORT/v1" \
      --mcp-url "http://127.0.0.1:$CHEMGRAPH_MCP_PORT/mcp/" \
      --workflow single_agent -s -o state --output-file "$OUT/runs/$sid.json" \
      -q "$query" < /dev/null > "$OUT/runs/$sid.log" 2>&1 || echo "session $sid exited non-zero (kept)"
    t1=$(date +%s.%N); off1=$(clock_offset)
    echo "{\"session\":\"$sid\",\"workload\":\"chemgraph\",\"query_id\":\"$qid\",\"repeat\":$r,\"t_start\":$t0,\"t_end\":$t1,\"clock_offset_s\":[$off0,$off1]}" \
      >> "$OUT/sessions.jsonl"
  done
done
