"""mini-SWE-agent environment that executes every agent command on the GPU tool
host over SSH, as on an HPC cluster (harness on a login node, tools on compute
nodes). Each command is logged with harness-side start/end times.

Use with:  mini --environment-class ssh_env.SSHEnvironment \
               -c environment.host=TOOL_HOST -c environment.remote_cwd=/path \
               -c environment.log=results/m1/tool_calls.jsonl ...
(PYTHONPATH must contain this directory.)

An SSH ControlMaster socket is used so per-command overhead stays small; the
overhead itself is measured by `ssh true` probes and logged as tool "ssh_probe".
"""

import json
import os
import shlex
import time
from typing import Any

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig, _run


class SSHEnvironmentConfig(LocalEnvironmentConfig):
    host: str = ""
    remote_cwd: str = ""
    remote_env: dict[str, str] = {}
    log: str = ""
    session: str = ""
    control_path: str = "/tmp/agentfn-ssh-%r@%h:%p"
    timeout: int = 900
    gpu: str = ""  # M5 placement policy: CUDA_VISIBLE_DEVICES for every command of this session


class SSHEnvironment(LocalEnvironment):
    def __init__(self, **kwargs):
        super().__init__(config_class=SSHEnvironmentConfig, **kwargs)
        if not self.config.host:
            raise ValueError("environment.host is required")
        self.session = self.config.session or os.environ.get("AGENTFN_SESSION", "")
        self._ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
                     "-o", f"ControlPath={self.config.control_path}", "-o", "ControlPersist=600",
                     self.config.host]
        self._log_probe()

    def _remote(self, command: str, cwd: str) -> str:
        env = {"AGENTFN_SESSION": self.session, **self.config.remote_env}
        if self.config.gpu != "":
            env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        inner = (f"source ~/agentfn/env.sh 2>/dev/null; cd {shlex.quote(cwd)} && "
                 f"{exports} bash -c {shlex.quote(command)}")
        return " ".join(shlex.quote(x) for x in self._ssh) + " " + shlex.quote(inner)

    def _append(self, rec: dict):
        if self.config.log:
            with open(self.config.log, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def _log_probe(self):
        t0 = time.time()
        _run(" ".join(shlex.quote(x) for x in self._ssh) + " true", os.getcwd(), dict(os.environ), 60)
        self._append({"session": self.session, "tool": "ssh_probe", "t_start": t0, "t_end": time.time()})

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        rcwd = cwd or self.config.remote_cwd or "~"
        t0 = time.time()
        try:
            res = _run(self._remote(command, rcwd), os.getcwd(), dict(os.environ), timeout or self.config.timeout)
            output = {"output": res.stdout, "returncode": res.returncode, "exception_info": ""}
        except Exception as e:  # same contract as LocalEnvironment
            raw = getattr(e, "output", None)
            raw = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
            output = {"output": raw, "returncode": -1,
                      "exception_info": f"An error occurred while executing the command: {e}",
                      "extra": {"exception_type": type(e).__name__, "exception": str(e)}}
        t1 = time.time()
        self._append({"session": self.session, "tool": "bash", "t_start": t0, "t_end": t1,
                      "returncode": output["returncode"], "uses_kb_eval": "kb_eval" in command,
                      "cmd_head": command.strip().split("\n")[0][:200]})
        self._check_finished(output)
        return output
