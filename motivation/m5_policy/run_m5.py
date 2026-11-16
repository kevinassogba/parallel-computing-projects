"""M5 orchestrator (harness host): run a fixed population of agent sessions on a
FIXED GPU budget under one platform policy, and record what is needed to measure
the policy -> observation -> behaviour -> demand loop.

Policies (all use the same G GPUs on the tool host):
  exclusive   : 1 session per GPU at a time (G slots); the rest of the population waits
  shared      : k sessions per GPU (G*k slots), no coordination (optionally MPS)
  capped      : as shared, with usable HBM capped at a real SKU size (--cap-gb 80/96/141)
  park-naive  : as shared; each session's tool server is checkpointed after every GPU call
                and restored on the next one, on the critical path (ChemGraph workload only)
The observation-preserving policy (C3) is added once the system exists.

The population is tasks x repeats (default: 1 repeat per run; run each policy with 3 seeds
so population-level metrics such as makespan and energy have n = 3), shuffled with a fixed seed, and identical across
policies. The LLM (Dynamo) and its capacity are the same in every run; per-slot proxies
log every LLM request so that think-time drift between policies is visible.

  python run_m5.py --workload kernelbench --policy shared --sessions-per-gpu 4 --gpus 8 \
      --tool-host H100 --llm-host A100 --tasks ../m1_sessions/problems_kb.txt --repeats 3 \
      --out results/m5/kb_shared_k4 [--mps] [--dry-run]
"""

import argparse
import concurrent.futures as cf
import json
import os
import queue
import random
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
M1 = HERE.parent / "m1_sessions"
GT_URL = "https://huggingface.co/datasets/Autonomous-Scientific-Agents/groundtruth/resolve/main/output_data.json"


def load_env():
    env = {}
    for line in (HERE.parent / "setup" / "versions.env").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


class Runner:
    def __init__(self, a):
        self.a, self.env = a, load_env()
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "traj").mkdir(exist_ok=True)
        self.run_id = self.out.name
        self.lock = threading.Lock()
        self.procs = []

    # ---------- helpers ----------
    def sh(self, cmd, check=True, **kw):
        if self.a.dry_run:
            print("[dry-run]", cmd if isinstance(cmd, str) else " ".join(map(shlex.quote, cmd)))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.run(cmd, check=check, capture_output=True, text=True,
                              shell=isinstance(cmd, str), **kw)

    def remote(self, cmd, check=True):
        return self.sh(["ssh", "-n", "-o", "BatchMode=yes", self.a.tool_host,
                        f"source ~/agentfn/env.sh && {cmd}"], check=check)

    def policy_cmd(self, *args):
        return self.remote("~/agentfn/m5_policy/tool_host_policy.sh " + " ".join(map(str, args)))

    def clock_offset(self):
        if self.a.dry_run:
            return 0.0
        t0 = time.time()
        r = self.remote("date +%s.%N")
        t1 = time.time()
        return float(r.stdout.strip()) - (t0 + t1) / 2

    def record(self, rec):
        with self.lock, open(self.out / "sessions.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------- setup / teardown ----------
    def setup(self, nslots):
        a = self.a
        if a.mps:
            self.policy_cmd("mps", "on")
        if a.policy == "capped":
            self.policy_cmd("ballast", "start", a.cap_gb)
        if a.workload == "chemgraph":
            park = 1 if a.policy == "park-naive" else 0
            self.policy_cmd("mcp", "start", self.run_id, nslots, a.gpus, park, int(a.warm_tools))
            if park:
                self.policy_cmd("parker", "start", self.run_id)
            fw = []
            for s in range(nslots):
                port = a.mcp_port_base + s
                fw += ["-L", f"{port}:127.0.0.1:{port}"]
            if not a.dry_run:
                self.procs.append(subprocess.Popen(["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes"]
                                                   + fw + [a.tool_host]))
        self.policy_cmd("sampler", "start", self.run_id)
        for s in range(nslots):  # one timing proxy per slot, so sessions never need a header
            sf = Path(f"/tmp/agentfn_m5_{self.run_id}_slot{s}")
            sf.write_text("")
            cmd = [str(Path(a.venv) / "bin" / "python"), str(HERE.parent / "common" / "llm_proxy.py"),
                   "--upstream", f"http://{a.llm_host}:{self.env['LLM_HTTP_PORT']}",
                   "--port", str(a.proxy_port_base + s), "--host", "127.0.0.1",
                   "--log", str(self.out / f"llm_requests_slot{s}.jsonl"), "--session-file", str(sf)]
            if a.dry_run:
                print("[dry-run]", " ".join(cmd))
            else:
                self.procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if not a.dry_run:
            time.sleep(3)

    def teardown(self):
        a = self.a
        self.policy_cmd("sampler", "stop")
        if a.workload == "chemgraph":
            if a.policy == "park-naive":
                self.policy_cmd("parker", "stop")
            self.policy_cmd("mcp", "stop")
        if a.policy == "capped":
            self.policy_cmd("ballast", "stop")
        if a.mps:
            self.policy_cmd("mps", "off")
        for p in self.procs:
            p.terminate()
        logdir = f"~/agentfn/logs/m5/{self.run_id}"
        if a.workload == "kernelbench" and not a.no_remeasure:
            # ground truth: the agent's own best candidate, re-timed alone on an idle GPU
            self.remote(f"~/agentfn/kb/bin/python ~/agentfn/m5_policy/remeasure_kb.py "
                        f"--run-prefix m5-{self.run_id}-kb- --gpu {a.remeasure_gpu} --repeats 3 "
                        f"--out {logdir}/remeasure.jsonl", check=False)
        self.sh(["scp", "-q", "-r", f"{a.tool_host}:{logdir}", str(self.out / "tool_host_logs")], check=False)
        if a.workload == "chemgraph":
            self.sh([str(Path(a.venv) / "bin" / "python"), str(HERE / "score_cg.py"), "--run", str(self.out)], check=False)

    # ---------- one session ----------
    def session(self, slot, task, rep, idx):
        a, env = self.a, self.env
        gpu = slot % a.gpus
        sf = Path(f"/tmp/agentfn_m5_{self.run_id}_slot{slot}")
        api = f"http://127.0.0.1:{a.proxy_port_base + slot}/v1"
        rec = {"run": self.run_id, "policy": a.policy, "workload": a.workload, "slot": slot, "gpu": gpu,
               "sessions_per_gpu": 1 if a.policy == "exclusive" else a.sessions_per_gpu,
               "cap_gb": a.cap_gb if a.policy == "capped" else None,
               "mps": a.mps, "warm_tools": a.warm_tools, "repeat": rep, "order": idx}
        if a.workload == "kernelbench":
            level, pid = task
            sid = f"m5-{self.run_id}-kb-L{level}-P{pid}-r{rep}"
            rec.update(session=sid, level=level, problem=pid)
            if not a.dry_run:
                sf.write_text(sid)
            wd = self.remote(f"agentfn-kb-prepare {level} {pid} {sid}").stdout.strip() or f"~/agentfn/sessions/{sid}"
            task_text = (M1 / "kb_task.txt").read_text().replace("{max_evals}", str(a.max_evals))
            cmd = [str(Path(a.venv) / "bin" / "mini"), "-c", "mini.yaml",
                   "--environment-class", "ssh_env.SSHEnvironment",
                   "-c", f"environment.host={a.tool_host}", "-c", f"environment.remote_cwd={wd}",
                   "-c", f"environment.log={self.out / 'tool_calls.jsonl'}", "-c", f"environment.session={sid}",
                   "-c", f"environment.gpu={gpu}",
                   "-m", f"hosted_vllm/{env['LLM_MODEL']}", "-c", f"model.model_kwargs.api_base={api}",
                   "-c", f"agent.step_limit={a.step_limit}", "-c", "agent.cost_limit=0",
                   "-t", task_text, "-y", "--exit-immediately", "-o", str(self.out / "traj" / f"{sid}.json")]
            envv = dict(os.environ, PYTHONPATH=f"{M1}:{os.environ.get('PYTHONPATH', '')}",
                        MSWEA_COST_TRACKING="ignore_errors", AGENTFN_SESSION=sid)
            log = self.out / "traj" / f"{sid}.log"
        else:
            qid, query = task
            sid = f"m5-{self.run_id}-cg-q{qid}-r{rep}"
            rec.update(session=sid, query_id=qid)
            if not a.dry_run:
                sf.write_text(sid)
            cmd = [str(Path(a.venv) / "bin" / "chemgraph"), "run", "--model", env["LLM_MODEL"], "--base-url", api,
                   "--mcp-url", f"http://127.0.0.1:{a.mcp_port_base + slot}/mcp/", "--workflow", "single_agent",
                   "-s", "-o", "state", "--output-file", str(self.out / "traj" / f"{sid}.json"), "-q", query]
            envv = dict(os.environ)
            log = self.out / "traj" / f"{sid}.log"
        rec["t_start"] = time.time()
        if a.dry_run:
            print(f"[dry-run] slot={slot} gpu={gpu}", " ".join(map(shlex.quote, cmd[:6])), "...")
            rc = 0
        else:
            with open(log, "w") as lf:
                rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                    env=envv, timeout=a.session_timeout_s).returncode
        rec["t_end"] = time.time()
        rec["exit_code"] = rc
        if a.workload == "kernelbench" and not a.dry_run:
            self.sh(["scp", "-q", f"{a.tool_host}:{wd}/kb_eval.jsonl", str(self.out / "traj" / f"{sid}.kb_eval.jsonl")],
                    check=False)
        self.record(rec)

    # ---------- population ----------
    def tasks(self):
        a = self.a
        if a.workload == "kernelbench":
            ts = [tuple(int(x) for x in l.split()[:2]) for l in open(a.tasks) if l.strip() and not l.startswith("#")]
        else:
            gt = self.out / "groundtruth.json"
            if not gt.exists():
                self.sh(["curl", "-L", "--fail", "-s", "-o", str(gt), GT_URL])
            data = json.load(open(gt)) if gt.exists() else [{"id": "dry", "query": "dry run"}]
            ts = [(d["id"], d["query"]) for d in data]
        pop = [(t, r) for r in range(1, a.repeats + 1) for t in ts]
        random.Random(a.seed).shuffle(pop)
        return pop

    def run(self):
        a = self.a
        nslots = a.gpus * (1 if a.policy == "exclusive" else a.sessions_per_gpu)
        pop = self.tasks()
        manifest = {"run": self.run_id, "args": vars(a), "nslots": nslots, "population": len(pop),
                    "versions": self.env, "t_start": time.time(), "clock_offset_start_s": self.clock_offset()}
        self.setup(nslots)
        q = queue.Queue()
        for i, (t, r) in enumerate(pop):
            q.put((i, t, r))

        def worker(slot):
            while True:
                try:
                    i, t, r = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    self.session(slot, t, r, i)
                except Exception as e:  # keep going; record the failure
                    self.record({"run": self.run_id, "slot": slot, "order": i, "error": repr(e)[:500]})

        try:
            with cf.ThreadPoolExecutor(max_workers=nslots) as ex:
                list(ex.map(worker, range(nslots)))
        finally:
            manifest.update(t_end=time.time(), clock_offset_end_s=self.clock_offset())
            self.teardown()
            (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(f"done: {len(pop)} sessions, {nslots} slots, makespan {manifest['t_end'] - manifest['t_start']:.0f} s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True, choices=["kernelbench", "chemgraph"])
    p.add_argument("--policy", required=True, choices=["exclusive", "shared", "capped", "park-naive"])
    p.add_argument("--gpus", type=int, required=True, help="GPU budget G on the tool host (same for all policies)")
    p.add_argument("--sessions-per-gpu", type=int, default=1, help="k; ignored for exclusive")
    p.add_argument("--cap-gb", type=float, default=0.0, help="capped: usable HBM per GPU (80/96/141)")
    p.add_argument("--mps", action="store_true")
    p.add_argument("--warm-tools", action="store_true", help="chemgraph: keep MACE calculators resident")
    p.add_argument("--tool-host", required=True)
    p.add_argument("--llm-host", required=True)
    p.add_argument("--tasks", default=str(M1 / "problems_kb.txt"))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-evals", type=int, default=10)
    p.add_argument("--step-limit", type=int, default=60)
    p.add_argument("--session-timeout-s", type=int, default=3600)
    p.add_argument("--proxy-port-base", type=int, default=8101)
    p.add_argument("--mcp-port-base", type=int, default=9100)
    p.add_argument("--venv", default=os.path.expanduser("~/agentfn-harness-venv"))
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-remeasure", action="store_true", help="kernelbench: skip exclusive re-timing")
    p.add_argument("--remeasure-gpu", default="0")
    a = p.parse_args()
    if a.policy == "park-naive" and a.workload != "chemgraph":
        sys.exit("park-naive needs a persistent per-session tool server (chemgraph workload)")
    if a.policy == "capped" and not a.cap_gb:
        sys.exit("--cap-gb is required for the capped policy")
    Runner(a).run()


if __name__ == "__main__":
    main()
