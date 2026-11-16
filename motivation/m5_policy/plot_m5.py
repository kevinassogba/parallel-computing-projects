"""M5 figure from results/m5/summary/policies.csv (no recomputation):
(a) solved tasks per GPU-hour and (b) GPU joules per solved task, per policy, at a
fixed GPU budget, with the classic per-call metric printed for contrast."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
from matplotlib.patches import Patch

FIG_W, INK = 7.16, "#000000"
TICK_PT, LABEL_PT, TITLE_PT, NOTE_PT = 5.8, 6.8, 7.5, 4.9
SERIES = {"exclusive": ("#004488", "////"), "shared": ("#117733", "\\\\\\\\"),
          "capped": ("#882255", "xxx"), "park-naive": ("#999933", "||||")}


def label(r):
    if r.policy == "exclusive":
        return "exclusive"
    s = f"{r.policy}\nk={int(r.sessions_per_gpu)}"
    if r.policy == "capped":
        s += f", {int(r.cap_gb)} GB"
    if r.mps:
        s += ", MPS"
    return s


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#b8b8b8", lw=0.45)
    ax.grid(axis="x", visible=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.7); s.set_color(INK)
    ax.tick_params(labelsize=TICK_PT, colors=INK)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", which="both", right=False)


def panel(ax, df, col, ylabel, title, fmt):
    for i, r in enumerate(df.itertuples()):
        v, sd = getattr(r, f"{col}_mean"), getattr(r, f"{col}_sd")
        c, h = SERIES[r.policy]
        if v is None or v != v:
            ax.text(i, 0, "not yet run", rotation=90, fontsize=NOTE_PT, ha="center", va="bottom")
            continue
        ax.bar(i, v, 0.6, color=c, edgecolor=INK, hatch=h, linewidth=0.6, zorder=2)
        if sd == sd and sd is not None:
            ax.errorbar(i, v, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=4)
        ax.text(i, v * 1.02 + (sd or 0), fmt.format(v) + ("*" if r.provisional else ""), rotation=90,
                fontsize=NOTE_PT, ha="center", va="bottom")
    tops = [getattr(r, f"{col}_mean") + (getattr(r, f"{col}_sd") if getattr(r, f"{col}_sd") == getattr(r, f"{col}_sd") else 0)
            for r in df.itertuples() if getattr(r, f"{col}_mean") == getattr(r, f"{col}_mean")]
    if tops:
        ax.set_ylim(0, max(tops) * 1.3)
    ax.set_xticks(range(len(df)), [label(r) for r in df.itertuples()])
    ax.set_ylabel(ylabel, fontsize=LABEL_PT)
    ax.set_title(title, fontsize=TITLE_PT, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(Path(a.summary) / "policies.csv")
    df = df[df.workload == a.workload].sort_values(["policy", "sessions_per_gpu", "cap_gb"], na_position="first")
    plt.style.use(["science", "high-contrast", "no-latex"])
    plt.rcParams["hatch.linewidth"] = 0.45
    plt.rcParams["hatch.color"] = "white"
    fig, (x, y) = plt.subplots(1, 2, figsize=(FIG_W, 2.4))
    for ax in (x, y):
        style(ax)
    panel(x, df, "solved_per_gpu_hour", "solved tasks per GPU-hour", "(a) Outcome per GPU-hour", "{:.2f}")
    for s in ("mean", "sd"):
        df[f"gpu_kj_per_solved_{s}"] = df[f"gpu_j_per_solved_{s}"] / 1000.0
    panel(y, df, "gpu_kj_per_solved", "GPU energy per solved task (kJ)", "(b) Energy per solved task", "{:.0f}")
    handles = [Patch(facecolor=c, edgecolor=INK, hatch=h, linewidth=0.6, label=p) for p, (c, h) in SERIES.items()
               if p in set(df.policy)]
    for hd in handles:
        hd.set_hatchcolor("white") if hasattr(hd, "set_hatchcolor") else None
    if df.provisional.any():
        handles.append(Patch(facecolor="white", edgecolor="white", label="* n < 3: provisional"))
    fig.legend(handles=handles, fontsize=6.2, frameon=False, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 1.0, f"{a.workload}: same GPU budget, same LLM, different platform policies",
             fontsize=TITLE_PT, ha="center", va="bottom", fontweight="bold")
    fig.tight_layout(w_pad=1.6, rect=(0, 0.12, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
