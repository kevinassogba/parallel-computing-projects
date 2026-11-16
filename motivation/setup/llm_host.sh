#!/usr/bin/env bash
# LLM host (A100-DGX, 80 GB): serve the agent model with NVIDIA Dynamo.
# This machine only plays the remote LLM provider; it never runs tools.
#
#   bash llm_host.sh install          # one-time: venv + ai-dynamo[vllm]==$DYNAMO_VERSION
#   bash llm_host.sh serve [NGPU]     # frontend + NGPU vLLM workers (default 1), TP=1 each
#   bash llm_host.sh check            # one chat request + one tool-call request
#   bash llm_host.sh stop
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/versions.env"
VENV="${VENV:-$HOME/agentfn-llm-venv}"
LOG="${LOG:-$HOME/agentfn-llm-logs}"
mkdir -p "$LOG"

case "${1:-}" in
install)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" "ai-dynamo[vllm]==${DYNAMO_VERSION}"
  "$VENV/bin/python" - <<'EOF' | tee "$LOG/versions.txt"
import importlib.metadata as m
for p in ("ai-dynamo", "ai-dynamo-runtime", "vllm", "torch"):
    try: print(p, m.version(p))
    except m.PackageNotFoundError: print(p, "not installed")
EOF
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a "$LOG/versions.txt"
  ;;
serve)
  NGPU="${2:-1}"
  source "$VENV/bin/activate"
  nohup python -m dynamo.frontend --http-port "$LLM_HTTP_PORT" --discovery-backend file \
    > "$LOG/frontend.log" 2>&1 &
  echo $! > "$LOG/frontend.pid"
  for ((g = 0; g < NGPU; g++)); do
    CUDA_VISIBLE_DEVICES=$g nohup python -m dynamo.vllm \
      --model "$LLM_MODEL" --discovery-backend file \
      --kv-events-config '{"enable_kv_cache_events": false}' \
      --dyn-tool-call-parser "$LLM_TOOL_PARSER" \
      --max-model-len "$LLM_MAX_MODEL_LEN" \
      > "$LOG/worker$g.log" 2>&1 &
    echo $! > "$LOG/worker$g.pid"
  done
  echo "frontend on :$LLM_HTTP_PORT, $NGPU worker(s); logs in $LOG"
  ;;
check)
  curl -s "localhost:$LLM_HTTP_PORT/v1/models" | python3 -m json.tool
  curl -s "localhost:$LLM_HTTP_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$LLM_MODEL\", \"max_tokens\": 64,
    \"messages\": [{\"role\": \"user\", \"content\": \"List the files in /tmp.\"}],
    \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"bash\",
      \"description\": \"run a shell command\",
      \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}},
      \"required\": [\"command\"]}}}]}" | python3 -m json.tool
  ;;
stop)
  for f in "$LOG"/*.pid; do kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"; done
  ;;
*) sed -n 2,9p "$0"; exit 1 ;;
esac
