# Experiments — plan and run book

What this tests and why: see `../serverless_agentic_opportunities_report.html` (§4 thesis, §5 hypotheses).
In short: for agent GPU tools, does the platform's resource policy change what agents do, how much GPU
they use, and whether they succeed (H6–H8)? Is sharing capacity bounded by memory (H2–H4)?

**Topology (every run):** LLM host = A100-DGX running NVIDIA Dynamo · harness host = x86 server ·
tool host = H100 / H200 / GH200. The LLM is always on a different machine from the tools.

**Pinned versions** (`setup/versions.env`):
- Dynamo 1.6.2, the last line shipping the KV Block Manager (deprecated in 1.5.0, removal targeted
  for 1.7.0). KVBM is not used here.
- Model: Qwen3-Coder-30B-A3B-Instruct, BF16, TP=1, tool parser `qwen3_coder`.
- Workloads: KernelBench via mini-SWE-agent; ChemGraph (MCP) + MACE-MP `medium-mpa-0`.
- Pin the harness and tool commits in `versions.env` after the first install.

**Rules:**
- the same task population, GPU budget and LLM across policies;
- outcomes re-measured on an idle GPU (kernels) or scored by ChemGraph's judge;
- n ≥ 3 seeds;
- memory settings at real product sizes (80 / 96 / 141 GB);
- versions copied into every result directory;
- laptop smoke runs are not for the paper.

## Studies

| ID | Question | Scripts | Machines |
|---|---|---|---|
| **M5 (core)** | Does policy change behaviour, demand, outcome; do classic and per-solved-task metrics disagree? (H6–H8) | `m5_policy/` | A100 · x86 · H100, then H200 |
| M1 | GPU tool duration, gaps between GPU calls, duty cycle, state size S (H1) | `m1_sessions/` | A100 · x86 · H100 |
| M2 | Park/restore cost per memory tier vs resident and cold start (H3) | `m2_park_restore/` | H100 host, GH200, Gen5 NVMe |
| M3 | Sharing headroom: HBM capacity vs bandwidth vs compute (H2, H4) | `m3_headroom/` | H200 capped, H100 |
| M4 | Timing distortion under sharing (H5) | `m4_fidelity/` | H100 |

Already done from public traces (`public_traces/`, `results/public/`):
- think-time and idle-gap distributions, which M3 and M4 sample;
- the finding that coding traces contain almost no GPU tool calls, which is why M1–M5 are needed.

## Run book

### 0. Install (once per machine)
```bash
bash setup/llm_host.sh install && bash setup/llm_host.sh serve 8 && bash setup/llm_host.sh check   # A100-DGX, 1 replica per GPU
bash setup/tool_host.sh install                                                                   # tool host
bash setup/harness_host.sh install                                                                # x86; passwordless SSH to the tool host
```
Before the first run, run `python public_traces/analyze.py`; `tool_host.sh install` copies its
think-time samples to the tool host.

### 1. M5 kill check (first)
One seed each of exclusive vs 8 sessions per GPU on KernelBench. If behaviour, outcome and cost do not
differ beyond seed noise, stop: the thesis fails (report §4.3).
```bash
python m5_policy/run_m5.py --workload kernelbench --policy exclusive --gpus 2 --seed 7 --tool-host H100 --llm-host A100 --out results/m5/kb_excl_s7
python m5_policy/run_m5.py --workload kernelbench --policy shared --sessions-per-gpu 8 --gpus 2 --seed 7 --tool-host H100 --llm-host A100 --out results/m5/kb_shared_k8_s7
python m5_policy/analyze_m5.py --runs results/m5/kb_* --out results/m5/summary
```

### 2. M1 — session anatomy (25 kernel problems × 3, 40 ChemGraph queries × 3)
```bash
bash setup/harness_host.sh proxy start <LLM_HOST> results/m1/llm_requests.jsonl
bash setup/tool_host.sh sample start ~/agentfn/logs/gpu && bash setup/tool_host.sh mcp start     # tool host
bash m1_sessions/run_kernelbench.sh <TOOL_HOST> m1_sessions/problems_kb.txt results/m1/kernelbench 3
bash m1_sessions/run_chemgraph.sh  <TOOL_HOST> results/m1/chemgraph 3
python m1_sessions/analyze_m1.py --run results/m1/kernelbench --run results/m1/chemgraph --out results/m1/summary
```
Before analysis, copy the tool host's `~/agentfn/logs/{gpu_devices.csv,gpu_procs.csv,chemgraph_tools.jsonl}`
and `results/m1/llm_requests.jsonl` into each run directory. Restart the sampler between workloads.

### 3. M2 — park / restore (tool host; CRIU needs root)
```bash
source ~/agentfn/env.sh; cd ~/agentfn/m2_park_restore; PY=~/agentfn/kb/bin/python
$PY bench_park_restore.py --kind synthetic --gb 1 2 4 8 16 32 --methods driver cold --repeats 5 --out ~/agentfn/results/m2/$(hostname)_synthetic.jsonl
~/agentfn/cg/bin/python bench_park_restore.py --kind mace --methods driver cold --repeats 5 --out ~/agentfn/results/m2/$(hostname)_mace.jsonl
sudo -E $PY bench_park_restore.py --kind synthetic --gb 1 4 16 --methods criu:/mnt/nvme/agentfn criu:/dev/shm/agentfn --drop-caches --repeats 5 --out ~/agentfn/results/m2/$(hostname)_criu.jsonl
```

### 4. M3 — headroom (H200 capped at product sizes; run with and without MPS)
```bash
for cap in 80 96 141; do ~/agentfn/cg/bin/python ~/agentfn/m3_headroom/run_headroom.py --kind mace \
  --k 1 2 4 8 16 32 64 --hbm-cap-gb $cap --windows ~/agentfn/window_samples_freeinference.csv \
  --duration 300 --repeats 3 --out ~/agentfn/results/m3/h200_cap$cap.jsonl; done
```

### 5. M4 — fidelity (needs M1's kernel candidates)
```bash
K=~/agentfn/kb/bin/python; F=~/agentfn/m4_fidelity/run_fidelity.py
$K $F collect --sessions ~/agentfn/sessions --out cands.jsonl
$K $F run --cands cands.jsonl --k 0 4 8 16 --corunner-kind synthetic --windows ~/agentfn/window_samples_freeinference.csv --repeats 5 --out ~/agentfn/results/m4/h100.jsonl
$K $F analyze --results ~/agentfn/results/m4/h100.jsonl
```

### 6. M5 full grid
**Design:**
- GPU budget: G = 2 GPUs in every policy, so 8 sessions per GPU (16 slots) is still fewer than the 25 kernel tasks.
- Sessions per GPU k ∈ {1, 2, 4, 8}, each with and without MPS.
- Seeds 7, 8, 9; the seed also changes the task order. The same seeds are used across policies.
- Capped HBM (80 / 96 / 141 GB at k = 8) only on the H200, with exclusive and uncapped k = 8 on the
  same H200 as references.
- Naive park/restore (ChemGraph only, with `--warm-tools`) at k = 8.
```bash
for s in 7 8 9; do
  python m5_policy/run_m5.py --workload kernelbench --policy exclusive --gpus 2 --seed $s --tool-host H100 --llm-host A100 --out results/m5/kb_excl_s$s
  for k in 2 4 8; do for mps in "" --mps; do
    python m5_policy/run_m5.py --workload kernelbench --policy shared --sessions-per-gpu $k --gpus 2 $mps --seed $s \
      --tool-host H100 --llm-host A100 --out results/m5/kb_shared_k${k}${mps:+_mps}_s$s
  done; done
done
# ChemGraph: same grid with --workload chemgraph --warm-tools, plus --policy park-naive --sessions-per-gpu 8
# H200:      --policy capped --cap-gb 80|96|141 --sessions-per-gpu 8 --tool-host H200 (+ references on the same H200)
python m5_policy/analyze_m5.py --runs results/m5/* --out results/m5/summary
python m5_policy/plot_m5.py --summary results/m5/summary --workload kernelbench --out figures/m5_kernelbench
```
**After each run:**
- Kernel runs re-time every session's own best candidate alone on an idle GPU (`remeasure_kb.py`);
  that re-measurement is the true outcome.
- ChemGraph runs are scored by ChemGraph's judge (`score_cg.py`).
- `analyze_m5.py` reports the median LLM latency per run, so any drift between policies is visible as
  a confound.

## Check on the first real run
- `score_cg.py` was written before seeing a real `chemgraph run --output-file`; check its `unparsed` count.
- The GPU-error metric in `analyze_m5.py` is a regex over trajectories and logs; spot-check it.
- `ssh_env.py` timings include the SSH round trip, which `ssh_probe` records measure; kb_eval timings
  on the tool host are exact.
- M1 assigns ChemGraph tool calls to sessions by time; this is valid because M1 runs one session at
  a time. M5 uses one tool server per slot instead.
- HBM capping by ballast emulates capacity only; the H100 vs capped-H200 pair isolates bandwidth.
- CRIU dumps use `--libdir /nonexistent` so CRIU's CUDA plugin does not repeat the cuda-checkpoint step.
- FreeInference does not define tool `latency_ms`; TraceLab is the primary source for idle gaps.

## Smoke tests done (laptop RTX 3050 Ti; not for the paper)
- M2 driver checkpoint/restore matched its checksums; cold restart works.
- M3 ballast, OOM detection and latency logging work.
- The LLM proxy logs streamed tool calls.
- The M5 parker completed 3 park/restore cycles on a 512 MB GPU process (≈0.3 s restore; results matched).
- The M5 orchestrator dry-run issues the expected commands.
- The analysis and plot scripts run on a synthetic fixture.
