"""A process that holds one agent tool's warm GPU state, for park/restore tests.

Kinds:
  synthetic  --gb X           X GB of device tensors (sweeps state size S)
  mace       --mace-model M   MACE-MP model resident on the GPU + one structure
  kernelbench --level L --problem P   KernelBench reference model + inputs on the GPU

Control is by signals and files only, so CRIU can dump/restore the process:
  SIGUSR1 -> run one probe step, write <state_dir>/probe_<n>.json
  SIGTERM -> exit
On start it writes <state_dir>/ready.json (pid, setup timings, checksum of the
first probe). Probe results carry a checksum so a restored process is checked
for correctness against the pre-park result.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch

STATE = {"n": 0, "pending": 0, "stop": False}


def build(args):
    """Return (probe_fn, info). probe_fn() -> float checksum."""
    dev = torch.device("cuda:0")
    if args.kind == "synthetic":
        n = int(args.gb * 2**30 // 4)
        chunk = 2**28  # 1 GiB of fp32 per tensor
        bufs = []
        left = n
        g = torch.Generator(device=dev).manual_seed(0)
        while left > 0:
            m = min(chunk, left)
            bufs.append(torch.rand(m, device=dev, generator=g))
            left -= m

        def probe():
            # touches every byte: a restore that lost data changes the sum
            # per-row fp32 partial sums (tiny), so the probe adds no large temporaries to S
            return float(sum(b.view(-1, 1 << 20).sum(dim=1).double().sum().item() for b in bufs))

        return probe, {"bytes_tensors": n * 4}

    if args.kind == "mace":
        from ase.build import bulk
        from mace.calculators import mace_mp
        calc = mace_mp(model=args.mace_model, device="cuda", default_dtype="float64")
        atoms = bulk("Si", "diamond", a=5.43).repeat((args.supercell,) * 3)
        atoms.rattle(stdev=0.01, seed=0)
        atoms.calc = calc

        def probe():
            atoms.calc.results.clear()
            return float(atoms.get_potential_energy())

        return probe, {"n_atoms": len(atoms), "model": args.mace_model}

    if args.kind == "kernelbench":
        from kernelbench.eval import load_original_model_and_inputs, set_seed
        src = Path(args.ref).read_text()
        ctx = {}
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(src, ctx)
        set_seed(0)
        model = Model(*[x.to(dev) if torch.is_tensor(x) else x for x in get_init_inputs()]).to(dev)
        set_seed(1)
        inputs = [x.to(dev) if torch.is_tensor(x) else x for x in get_inputs()]

        @torch.no_grad()
        def probe():
            out = model(*inputs)
            out = out if torch.is_tensor(out) else out[0]
            return float(out.double().sum().item())

        return probe, {"ref": args.ref}

    raise SystemExit(f"unknown kind {args.kind}")


def on_usr1(*_):
    STATE["pending"] += 1


def on_term(*_):
    STATE["stop"] = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["synthetic", "mace", "kernelbench"])
    p.add_argument("--state-dir", required=True)
    p.add_argument("--gb", type=float, default=1.0)
    p.add_argument("--mace-model", default="medium-mpa-0")
    p.add_argument("--supercell", type=int, default=4)
    p.add_argument("--ref", default="")
    args = p.parse_args()
    sd = Path(args.state_dir)
    sd.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    t_ctx = time.time()
    probe, info = build(args)
    torch.cuda.synchronize()
    t_build = time.time()
    c0 = probe()
    if args.kind == "synthetic":
        torch.cuda.empty_cache()  # keep S equal to the requested tensors, not probe temporaries
    torch.cuda.synchronize()
    t_probe = time.time()

    signal.signal(signal.SIGUSR1, on_usr1)
    signal.signal(signal.SIGTERM, on_term)
    ready = {"pid": os.getpid(), "kind": args.kind, **info,
             "ctx_init_s": t_ctx - t0, "build_s": t_build - t_ctx, "first_probe_s": t_probe - t_build,
             "checksum": c0, "mem_allocated_mb": torch.cuda.memory_allocated() / 2**20,
             "mem_reserved_mb": torch.cuda.memory_reserved() / 2**20, "t_ready": t_probe}
    (sd / "ready.json").write_text(json.dumps(ready))

    while not STATE["stop"]:
        if STATE["pending"] > 0:
            STATE["pending"] -= 1
            STATE["n"] += 1
            t = time.time()
            c = probe()
            torch.cuda.synchronize()
            res = {"n": STATE["n"], "t_start": t, "latency_s": time.time() - t,
                   "checksum": c, "match": c == c0}
            tmp = sd / f".probe_{STATE['n']}.json"
            tmp.write_text(json.dumps(res))
            tmp.rename(sd / f"probe_{STATE['n']}.json")
        else:
            time.sleep(0.001)
    sys.exit(0)


if __name__ == "__main__":
    main()
