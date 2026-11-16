#!/usr/bin/env bash
# HARNESS HOST (x86 server): runs the agent harnesses and the LLM timing proxy.
# No GPU is needed or used here.
#
#   bash harness_host.sh install                 # venv: mini-swe-agent, ChemGraph (client), aiohttp
#   bash harness_host.sh proxy start LLM_HOST LOGFILE | stop
#   bash harness_host.sh check LLM_HOST          # end-to-end: proxy -> Dynamo tool-call request
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/versions.env"
VENV="${VENV:-$HOME/agentfn-harness-venv}"
RUN="$HOME/agentfn-harness-run"; mkdir -p "$RUN"

case "${1:-}" in
install)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" \
    "mini-swe-agent @ git+https://github.com/SWE-agent/mini-swe-agent@${MINI_SWE_AGENT_REF}" \
    "chemgraph @ git+https://github.com/argonne-lcf/ChemGraph@${CHEMGRAPH_REF}" \
    aiohttp
  "$VENV/bin/python" -c "import importlib.metadata as m;[print(p,m.version(p)) for p in ('mini-swe-agent','chemgraph','litellm','aiohttp')]" \
    | tee "$RUN/versions.txt"
  ;;
proxy)
  case "${2:-}" in
  start)
    LLM_HOST="$3"; LOGF="$4"; mkdir -p "$(dirname "$LOGF")"
    nohup "$VENV/bin/python" "$HERE/../common/llm_proxy.py" --upstream "http://$LLM_HOST:$LLM_HTTP_PORT" \
      --port "$PROXY_PORT" --host 127.0.0.1 --log "$LOGF" --session-file /tmp/agentfn_session \
      > "$RUN/proxy.out" 2>&1 &
    echo $! > "$RUN/proxy.pid"; echo "proxy on 127.0.0.1:$PROXY_PORT -> $LLM_HOST:$LLM_HTTP_PORT, log $LOGF" ;;
  stop) kill "$(cat "$RUN/proxy.pid")" 2>/dev/null || true; rm -f "$RUN/proxy.pid" ;;
  esac ;;
check)
  echo check > /tmp/agentfn_session
  curl -s "127.0.0.1:$PROXY_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$LLM_MODEL\", \"max_tokens\": 64, \"stream\": true,
    \"messages\": [{\"role\": \"user\", \"content\": \"Run ls in /tmp.\"}],
    \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"run a shell command\",
      \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}, \"required\": [\"command\"]}}}]}" | tail -3
  ;;
*) sed -n 2,7p "$0"; exit 1 ;;
esac
