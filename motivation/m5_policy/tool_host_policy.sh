#!/usr/bin/env bash
# M5 tool-host policy controls. Called over SSH by run_m5.py; also usable by hand.
#
#   tool_host_policy.sh mps on|off
#   tool_host_policy.sh ballast start CAP_GB | stop        # cap usable HBM on every GPU (SKU-anchored)
#   tool_host_policy.sh mcp start RUN NSLOTS NGPU PARK CACHE | stop
#        PARK=1: per-slot park handshake (needs parker); CACHE=1: warm MACE calculators
#   tool_host_policy.sh parker start RUN | stop
#   tool_host_policy.sh sampler start RUN | stop
#   tool_host_policy.sh collect RUN                        # print the log directory for scp
set -euo pipefail
ROOT="$HOME/agentfn"; source "$ROOT/env.sh"
PORT_BASE="${MCP_PORT_BASE:-9100}"
RUNDIR="$ROOT/run/m5"; mkdir -p "$RUNDIR"
logs() { echo "$ROOT/logs/m5/$1"; }

case "${1:-}" in
mps)
  if [[ "$2" == on ]]; then nvidia-cuda-mps-control -d && echo "MPS on"
  else echo quit | nvidia-cuda-mps-control 2>/dev/null || true; echo "MPS off"; fi ;;
ballast)
  if [[ "$2" == start ]]; then
    cap="$3"; n=$(nvidia-smi -L | wc -l)
    for ((g = 0; g < n; g++)); do
      tot=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits)
      gb=$(python3 -c "print(max(0.0, $tot/1024 - $cap))")
      CUDA_VISIBLE_DEVICES=$g nohup "$ROOT/kb/bin/python" -c "
import sys, time, torch
left = int(float(sys.argv[1]) * 2**30); bufs = []
while left > 0:
    n = min(left, 2**30); bufs.append(torch.empty(n, dtype=torch.uint8, device='cuda')); left -= n
torch.cuda.synchronize(); print('BALLAST_READY', flush=True)
while True: time.sleep(3600)" "$gb" > "$RUNDIR/ballast$g.log" 2>&1 &
      echo $! > "$RUNDIR/ballast$g.pid"
    done
    for ((g = 0; g < n; g++)); do
      until grep -q BALLAST_READY "$RUNDIR/ballast$g.log" 2>/dev/null; do sleep 0.5; done
    done
    echo "ballast: usable HBM capped at ${cap} GB on $n GPU(s)"
  else
    for f in "$RUNDIR"/ballast*.pid; do [[ -f "$f" ]] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
  fi ;;
mcp)
  if [[ "$2" == start ]]; then
    run="$3"; nslots="$4"; ngpu="$5"; park="$6"; cache="$7"; L=$(logs "$run"); mkdir -p "$L" "$RUNDIR/park"
    rm -rf "$RUNDIR/park"/*
    for ((s = 0; s < nslots; s++)); do
      g=$((s % ngpu)); d="$RUNDIR/park/slot$s"; mkdir -p "$d"
      env CUDA_VISIBLE_DEVICES=$g AGENTFN_TOOL_LOG="$L/tools_slot$s.jsonl" AGENTFN_CACHE_CALC="$cache" \
          $( [[ "$park" == 1 ]] && echo "AGENTFN_PARK_DIR=$d" ) \
          nohup "$ROOT/cg/bin/python" "$ROOT/chemgraph_mcp_gpu.py" --transport streamable_http \
          --host 127.0.0.1 --port $((PORT_BASE + s)) > "$L/mcp_slot$s.log" 2>&1 &
      echo $! > "$d/pid"
    done
    sleep 5; echo "started $nslots MCP servers on ports $PORT_BASE.. (park=$park cache=$cache)"
  else
    for f in "$RUNDIR"/park/slot*/pid; do [[ -f "$f" ]] && kill "$(cat "$f")" 2>/dev/null || true; done
  fi ;;
parker)
  if [[ "$2" == start ]]; then
    L=$(logs "$3"); mkdir -p "$L"; rm -f "$RUNDIR/park/STOP"
    nohup "$ROOT/kb/bin/python" "$ROOT/m5_policy/parker.py" --slots-dir "$RUNDIR/park" --log "$L/parker.jsonl" \
      > "$L/parker.out" 2>&1 &
    echo $! > "$RUNDIR/parker.pid"
  else
    touch "$RUNDIR/park/STOP"; sleep 1; kill "$(cat "$RUNDIR/parker.pid")" 2>/dev/null || true; rm -f "$RUNDIR/parker.pid"
  fi ;;
sampler)
  if [[ "$2" == start ]]; then
    L=$(logs "$3"); mkdir -p "$L"
    nohup "$ROOT/kb/bin/python" "$ROOT/common/gpu_sampler.py" --out "$L/gpu" --hz 10 > "$L/sampler.out" 2>&1 &
    echo $! > "$RUNDIR/sampler.pid"
  else
    kill "$(cat "$RUNDIR/sampler.pid")" 2>/dev/null || true; rm -f "$RUNDIR/sampler.pid"
  fi ;;
collect) logs "$2" ;;
*) sed -n 2,11p "$0"; exit 1 ;;
esac
