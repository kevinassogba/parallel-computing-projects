#!/usr/bin/env bash
# TOOL HOST (H100 / H200 / GH200 / B200): where agent tool calls execute.
#
#   bash tool_host.sh install        # venvs: kb (py3.10, KernelBench) and cg (py3.12, ChemGraph+MACE),
#                                    # sampler deps, cuda-checkpoint, helper commands in ~/agentfn
#   bash tool_host.sh mcp start|stop # ChemGraph MCP server (GPU-default MACE), localhost only
#   bash tool_host.sh sample start OUT_PREFIX | stop     # NVML sampler
#   bash tool_host.sh info           # record versions/topology into ~/agentfn/info.txt
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/versions.env"
ROOT="$HOME/agentfn"; SRC="$ROOT/src"; BIN="$ROOT/bin"
mkdir -p "$SRC" "$BIN" "$ROOT/run" "$ROOT/sessions"
ARCH="$(uname -m)"

gpu_arch_name() {  # KernelBench arch names
  local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  case "$cc" in 8.0|8.6|8.7) echo Ampere ;; 9.0) echo Hopper ;; 10.*|12.*) echo Blackwell ;; *) echo Unknown ;; esac
}

case "${1:-}" in
install)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  git -C "$SRC" clone -q https://github.com/ScalingIntelligence/KernelBench kernelbench 2>/dev/null || true
  git -C "$SRC/kernelbench" checkout -q "$KERNELBENCH_REF"
  git -C "$SRC" clone -q https://github.com/argonne-lcf/ChemGraph chemgraph 2>/dev/null || true
  git -C "$SRC/chemgraph" checkout -q "$CHEMGRAPH_REF"
  git -C "$SRC" clone -q https://github.com/NVIDIA/cuda-checkpoint cuda-checkpoint 2>/dev/null || true
  git -C "$SRC/cuda-checkpoint" checkout -q "$CUDA_CHECKPOINT_REF"

  uv venv "$ROOT/kb" --python 3.10
  uv pip install --python "$ROOT/kb/bin/python" -e "$SRC/kernelbench[gpu]" nvidia-ml-py
  uv venv "$ROOT/cg" --python 3.12
  uv pip install --python "$ROOT/cg/bin/python" -e "$SRC/chemgraph[calculators]" nvidia-ml-py
  cp "$HERE/../m1_sessions/kb_eval.py" "$HERE/../m1_sessions/chemgraph_mcp_gpu.py" "$ROOT/"
  cp -r "$HERE/../common" "$HERE/../m2_park_restore" "$HERE/../m3_headroom" "$HERE/../m4_fidelity" "$HERE/../m5_policy" "$ROOT/"
  cp "$HERE/../results/public/window_samples_freeinference.csv" "$ROOT/" 2>/dev/null \
    || echo "note: run public_traces/analyze.py first to create window_samples_freeinference.csv"

  cpbin="$SRC/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint"
  [[ "$ARCH" == aarch64 ]] && cpbin="$SRC/cuda-checkpoint/bin/aarch64_Linux/cuda-checkpoint"
  install -m 0755 "$cpbin" "$BIN/cuda-checkpoint"

  cat > "$ROOT/env.sh" <<EOF
export PATH="$BIN:\$PATH"
export KB_GPU_ARCH="$(gpu_arch_name)"
export AGENTFN_ROOT="$ROOT"
EOF
  cat > "$BIN/kb_eval" <<EOF
#!/usr/bin/env bash
# kb_eval KERNEL_FILE — evaluate a candidate against ./ref.py (see kb_eval.py)
export AGENTFN_TOOL_LOG="\${AGENTFN_TOOL_LOG:-\$PWD/kb_eval.jsonl}"
exec "$ROOT/kb/bin/python" "$ROOT/kb_eval.py" --ref ref.py --kernel "\$1"
EOF
  cat > "$BIN/agentfn-kb-prepare" <<EOF
#!/usr/bin/env bash
# agentfn-kb-prepare LEVEL PROBLEM_ID SESSION — make a fresh workdir, print its path
set -euo pipefail
wd="$ROOT/sessions/\$3"; mkdir -p "\$wd"
src=\$(ls "$SRC/kernelbench/KernelBench/level\$1/\$2_"*.py | head -1)
cp "\$src" "\$wd/ref.py"
cp "$SRC/kernelbench/src/kernelbench/prompts/model_ex_add.py" "\$wd/example_ref.py"
cp "$SRC/kernelbench/src/kernelbench/prompts/model_new_ex_add.py" "\$wd/example_new.py"
echo "\$wd"
EOF
  chmod +x "$BIN/kb_eval" "$BIN/agentfn-kb-prepare"
  "$0" info
  ;;
mcp)
  case "${2:-}" in
  start)
    mkdir -p "$ROOT/logs"
    AGENTFN_TOOL_LOG="$ROOT/logs/chemgraph_tools.jsonl" CHEMGRAPH_MCP_PORT="$CHEMGRAPH_MCP_PORT" \
      nohup "$ROOT/cg/bin/python" "$ROOT/chemgraph_mcp_gpu.py" --transport streamable_http \
      --host 127.0.0.1 --port "$CHEMGRAPH_MCP_PORT" > "$ROOT/logs/mcp.log" 2>&1 &
    echo $! > "$ROOT/run/mcp.pid"; echo "MCP server pid $(cat "$ROOT/run/mcp.pid")" ;;
  stop) kill "$(cat "$ROOT/run/mcp.pid")" 2>/dev/null || true; rm -f "$ROOT/run/mcp.pid" ;;
  esac ;;
sample)
  case "${2:-}" in
  start)
    nohup "$ROOT/kb/bin/python" "$ROOT/common/gpu_sampler.py" --out "$3" --hz "$SAMPLE_HZ" \
      > "$ROOT/logs/sampler.log" 2>&1 &
    echo $! > "$ROOT/run/sampler.pid" ;;
  stop) kill "$(cat "$ROOT/run/sampler.pid")" 2>/dev/null || true; rm -f "$ROOT/run/sampler.pid" ;;
  esac ;;
info)
  {
    echo "host: $(hostname)  arch: $ARCH  date: $(date -Is)"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv
    nvidia-smi topo -m || true
    echo "cuda-checkpoint: $("$BIN/cuda-checkpoint" --help 2>&1 | grep -m1 Version || echo missing)"
    echo "criu: $(criu --version 2>/dev/null | head -1 || echo missing)"
    lscpu | grep -E "Model name|Socket|NUMA node\(s\)|^CPU\(s\)"
    free -g | head -2
    lsblk -d -o NAME,MODEL,SIZE,TRAN | grep -i nvme || true
    for v in kb cg; do echo "[$v]"; "$ROOT/$v/bin/python" -c "import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda)"; done
    for r in kernelbench chemgraph cuda-checkpoint; do echo "$r $(git -C "$SRC/$r" rev-parse HEAD)"; done
  } | tee "$ROOT/info.txt"
  ;;
*) sed -n 2,8p "$0"; exit 1 ;;
esac
