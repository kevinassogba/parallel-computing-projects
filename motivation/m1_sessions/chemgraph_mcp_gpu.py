"""ChemGraph general-tools MCP server for the TOOL HOST, unchanged except for two
things a GPU deployment needs:
  1. MACE calculators default to device="cuda" (ChemGraph's default is "cpu");
  2. every run_ase call and every calculator construction is timed and appended
     to $AGENTFN_TOOL_LOG (JSONL), so M1 can split a tool call into model
     load/construction vs. compute.

    AGENTFN_TOOL_LOG=results/m1/chemgraph_tools.jsonl \
    python chemgraph_mcp_gpu.py --transport streamable_http --host 127.0.0.1 --port 9003

Bind to 127.0.0.1 and reach it from the harness host through an SSH tunnel
(ssh -N -L 9003:127.0.0.1:9003 TOOL_HOST); ChemGraph's server is unauthenticated.
"""

import json
import os
import socket
import time

from chemgraph.schemas.calculators import mace_calc
from chemgraph.schemas import ase_input
import chemgraph.tools.ase_core as ase_core

DEVICE = os.environ.get("AGENTFN_MACE_DEVICE", "cuda")
LOG = os.environ.get("AGENTFN_TOOL_LOG", "")
# M5 options (both off = stock ChemGraph behaviour):
#   AGENTFN_CACHE_CALC=1  keep constructed MACE calculators resident (a warm tool server);
#                         stock ChemGraph rebuilds the calculator on every run_ase call
#   AGENTFN_PARK_DIR=dir  handshake with m5_policy/parker.py: before GPU work write dir/need and
#                         wait for dir/ready (parker restores the process); after, write dir/done
CACHE_CALC = os.environ.get("AGENTFN_CACHE_CALC") == "1"
PARK_DIR = os.environ.get("AGENTFN_PARK_DIR", "")
_calc_cache = {}

mace_calc.MaceCalc.model_fields["device"].default = DEVICE
mace_calc.MaceCalc.model_rebuild(force=True)
ase_input.ASEInputSchema.model_rebuild(force=True)
assert mace_calc.MaceCalc().device == DEVICE, "MACE default device patch did not apply"


def log(rec: dict):
    if LOG:
        rec.update(host=socket.gethostname(), pid=os.getpid())
        with open(LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


_orig_get_calc = mace_calc.MaceCalc.get_calculator


def timed_get_calculator(self):
    key = (self.calculator_type, str(self.model), self.device, getattr(self, "default_dtype", None))
    if CACHE_CALC and key in _calc_cache:
        log({"event": "mace_cache_hit", "t_start": time.time(), "t_end": time.time()})
        return _calc_cache[key]
    t0 = time.time()
    calc = _orig_get_calc(self)
    log({"event": "mace_construct", "t_start": t0, "t_end": time.time(),
         "calculator_type": self.calculator_type, "model": str(self.model), "device": self.device})
    if CACHE_CALC:
        _calc_cache[key] = calc
    return calc


def _park_handshake(kind: str):
    """need -> wait for ready (restored); done -> parker may park us."""
    if not PARK_DIR:
        return
    d = os.path.abspath(PARK_DIR)
    if kind == "need":
        t0 = time.time()
        open(os.path.join(d, "need"), "w").write(str(t0))
        ready = os.path.join(d, "ready")
        while not os.path.exists(ready):
            time.sleep(0.0005)
        os.remove(ready)
        log({"event": "park_wait", "t_start": t0, "t_end": time.time()})
    else:
        open(os.path.join(d, "done"), "w").write(str(time.time()))


mace_calc.MaceCalc.get_calculator = timed_get_calculator

_orig_run_ase = ase_core.run_ase_core


def timed_run_ase_core(params):
    _park_handshake("need")
    t0 = time.time()
    try:
        return _orig_run_ase(params)
    finally:
        calc = getattr(params, "calculator", None)
        log({"event": "run_ase", "t_start": t0, "t_end": time.time(),
             "driver": getattr(params, "driver", None),
             "calculator": type(calc).__name__ if calc is not None else None})
        _park_handshake("done")


ase_core.run_ase_core = timed_run_ase_core

# Import after patching: mcp_tools binds run_ase_core by name at import time.
from chemgraph.mcp.mcp_tools import mcp  # noqa: E402
from chemgraph.mcp.server_utils import run_mcp_server  # noqa: E402

if __name__ == "__main__":
    run_mcp_server(mcp, default_port=int(os.environ.get("CHEMGRAPH_MCP_PORT", "9003")))
