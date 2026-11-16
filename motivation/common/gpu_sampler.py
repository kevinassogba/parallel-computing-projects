"""NVML sampler for the tool host: device utilization, device memory, and the GPU
memory held by each compute process. Writes two CSVs (devices, processes).

    python gpu_sampler.py --out results/m1/gpu --hz 10 [--duration SECONDS]

Stop with SIGINT/SIGTERM. Requires nvidia-ml-py (import name pynvml).
"""

import argparse
import csv
import signal
import time
from pathlib import Path

import pynvml

RUN = True


def stop(*_):
    global RUN
    RUN = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output prefix; writes <out>_devices.csv and <out>_procs.csv")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=0.0, help="0 = until signalled")
    args = p.parse_args()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / args.hz
    t_end = time.time() + args.duration if args.duration > 0 else float("inf")

    with open(f"{out}_devices.csv", "w", newline="") as fd, open(f"{out}_procs.csv", "w", newline="") as fp:
        wd, wp = csv.writer(fd), csv.writer(fp)
        wd.writerow(["t", "gpu", "util_gpu_pct", "util_mem_pct", "mem_used_mb", "mem_total_mb", "power_w", "sm_clock_mhz"])
        wp.writerow(["t", "gpu", "pid", "used_mb"])
        while RUN and time.time() < t_end:
            t0 = time.time()
            for i, h in enumerate(handles):
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                try:
                    pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except pynvml.NVMLError:
                    pw = ""
                clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                wd.writerow([f"{t0:.3f}", i, u.gpu, u.memory, m.used // 2**20, m.total // 2**20, pw, clk])
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                except pynvml.NVMLError:
                    procs = []
                for pr in procs:
                    used = pr.usedGpuMemory // 2**20 if pr.usedGpuMemory else ""
                    wp.writerow([f"{t0:.3f}", i, pr.pid, used])
            fd.flush()
            fp.flush()
            time.sleep(max(0.0, period - (time.time() - t0)))
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
