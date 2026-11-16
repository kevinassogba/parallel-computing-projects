"""Naive park/restore daemon for M5 (tool host).

Each per-session tool server (chemgraph_mcp_gpu.py with AGENTFN_PARK_DIR=<slot dir>)
handshakes through files in its slot directory:
  need  -> the server is about to use the GPU; restore it if parked, then write `ready`
  done  -> the server finished a GPU call; park it (cuda-checkpoint lock + checkpoint)
The restore is on the critical path (that is what makes this policy "naive"): its
latency is visible to the agent as part of the tool call. Every operation is logged.

    python parker.py --slots-dir ~/agentfn/run/m5/park --log ~/agentfn/logs/m5/<run>/parker.jsonl
Slot directories must contain a `pid` file with the server's PID.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

CKPT = shutil.which("cuda-checkpoint") or os.path.expanduser("~/agentfn/bin/cuda-checkpoint")


def ck(action, pid):
    t = time.time()
    r = subprocess.run([CKPT, "--action", action, "--pid", str(pid)] + (["--timeout", "30000"] if action == "lock" else []),
                       capture_output=True, text=True)
    return time.time() - t, r.returncode == 0, r.stderr.strip()[:300]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slots-dir", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--poll-s", type=float, default=0.0005)
    a = p.parse_args()
    root = Path(a.slots_dir).expanduser()
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    parked = {}  # slot -> bool
    log = open(a.log, "a")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    while not (root / "STOP").exists():
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            pidf = d / "pid"
            if not pidf.exists():
                continue
            try:
                pid = int(pidf.read_text().strip())
            except ValueError:  # pid file still being written
                continue
            slot = d.name
            if (d / "need").exists():
                t0 = time.time()
                rec = {"slot": slot, "pid": pid, "op": "restore", "t": t0, "was_parked": parked.get(slot, False)}
                if parked.get(slot):
                    rec["restore_s"], ok1, e1 = ck("restore", pid)
                    rec["unlock_s"], ok2, e2 = ck("unlock", pid)
                    rec["ok"] = ok1 and ok2
                    rec["err"] = e1 or e2
                    parked[slot] = False
                (d / "need").unlink(missing_ok=True)
                (d / "ready").write_text(str(time.time()))
                rec["handshake_s"] = time.time() - t0
                emit(rec)
            if (d / "done").exists():
                (d / "done").unlink(missing_ok=True)
                t0 = time.time()
                rec = {"slot": slot, "pid": pid, "op": "park", "t": t0}
                rec["lock_s"], ok1, e1 = ck("lock", pid)
                rec["checkpoint_s"], ok2, e2 = ck("checkpoint", pid) if ok1 else (0.0, False, "lock failed")
                rec["ok"] = ok1 and ok2
                rec["err"] = e1 or e2
                if ok1 and not ok2:
                    ck("unlock", pid)  # leave it running rather than locked
                parked[slot] = rec["ok"]
                emit(rec)
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
