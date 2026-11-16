#!/usr/bin/env bash
# Download the two public agent traces used to derive think-time and tool-gap
# distributions without running any agent:
#   * TraceLab (UW, arXiv 2606.30560): Claude Code / Codex sessions, CC BY 4.0.
#   * FreeInference agentic trace (Harvard): 16 weeks of agent sessions behind a
#     gateway; only the flattened Parquet tables (requests, tool_calls, sessions).
# Usage: bash fetch.sh [DATA_DIR]   (default: ../data)
set -euo pipefail
DATA="${1:-$(dirname "$0")/../data}"
mkdir -p "$DATA/tracelab" "$DATA/freeinference"

TL_TAG="v0.0.2"
curl -L --fail -C - -o "$DATA/tracelab/syfi_coding_trace.duckdb" \
  "https://github.com/uw-syfi/TraceLab/releases/download/${TL_TAG}/syfi_coding_trace.duckdb"
echo "$TL_TAG" > "$DATA/tracelab/RELEASE"

HF="https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace"
weeks=$(curl -s "https://huggingface.co/api/datasets/harvardMadsys/freeinference_agentic_trace/tree/main/release" \
  | python3 -c "import sys,json;[print(e['path'].split('/')[-1]) for e in json.load(sys.stdin) if e['type']=='directory']")
for w in $weeks; do
  d="$DATA/freeinference/$w"; mkdir -p "$d"
  for t in requests tool_calls sessions; do
    curl -L --fail -s -C - -o "$d/$t.parquet" "$HF/resolve/main/release/$w/processed/$t.parquet"
  done
  curl -L --fail -s -o "$d/tables.json" "$HF/resolve/main/release/$w/processed/tables.json"
  echo "fetched $w"
done
