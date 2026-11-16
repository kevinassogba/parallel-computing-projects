"""Figure from results/public (no recomputation): (a) idle windows between
consecutive tool calls of a session (no human turn inside), (b) share of a
session's active time with a tool executing."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.patches import Patch

RES = Path(__file__).resolve().parent.parent / "results" / "public"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG_W, INK = 7.16, "#000000"
TICK_PT, LABEL_PT, TITLE_PT, NOTE_PT = 5.8, 6.8, 7.5, 4.9
SERIES = {"tracelab": ("#004488", "////"), "freeinference": ("#117733", "\\\\\\\\")}
NAMES = {"tracelab": "TraceLab (Claude Code / Codex, 52 developers, 8.1k sessions)",
         "freeinference": "FreeInference (267 accounts, 14 harnesses, 14k sessions)"}
WIN_ORDER = ["0-1", "1-2", "2-5", "5-10", "10-30", "30-60", "60-300", ">300"]
WIN_TICKS = ["<1", "1–2", "2–5", "5–10", "10–30", "30–60", "60–300", ">300"]
BUSY_ORDER = ["0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.75", ">0.75"]
BUSY_TICKS = ["<5", "5–10", "10–20", "20–30", "30–50", "50–75", ">75"]


def read(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#b8b8b8", lw=0.45)
    ax.grid(axis="x", visible=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.7); s.set_color(INK)
    ax.tick_params(labelsize=TICK_PT, colors=INK)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", which="both", right=False)
    ax.xaxis.set_minor_locator(plt.NullLocator())


def bars(ax, x, h, w, series):
    col, hatch = SERIES[series]
    ax.bar(x, h, w, color=col, edgecolor=INK, hatch=hatch, linewidth=0.6, zorder=2)
    for xi, hi in zip(x, h):
        ax.text(xi, hi + 0.6, f"{hi:.0f}", ha="center", va="bottom", rotation=90, fontsize=NOTE_PT, color=INK)


def main():
    plt.style.use(["science", "high-contrast", "no-latex"])
    plt.rcParams["hatch.linewidth"] = 0.45
    plt.rcParams["hatch.color"] = "white"
    fig, (a, b) = plt.subplots(1, 2, figsize=(FIG_W, 2.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax in (a, b):
        style(ax)

    win = read("parking_windows_binned.csv")
    w = 0.38
    for k, tr in enumerate(["tracelab", "freeinference"]):
        cnt = {r["bin"]: int(r["n"]) for r in win if r["trace"] == tr}
        tot = sum(cnt.values())
        h = [100 * cnt.get(bn, 0) / tot for bn in WIN_ORDER]
        bars(a, [i + (k - 0.5) * w for i in range(len(WIN_ORDER))], h, w, tr)
    a.set_xticks(range(len(WIN_ORDER)), WIN_TICKS)
    a.set_xlabel("idle window between consecutive tool calls (s)", fontsize=LABEL_PT)
    a.set_ylabel("share of windows (%)", fontsize=LABEL_PT)
    a.set_ylim(0, 42)
    a.set_title("(a) Time a tool environment sits idle", fontsize=TITLE_PT, loc="left")

    busy = {r["bin"]: int(r["n"]) for r in read("tool_busy_fraction_active_binned.csv")}
    tot = sum(busy.values())
    bars(b, list(range(len(BUSY_ORDER))), [100 * busy.get(bn, 0) / tot for bn in BUSY_ORDER], 0.6, "tracelab")
    b.set_xticks(range(len(BUSY_ORDER)), BUSY_TICKS)
    b.set_xlabel("share of a session's active time with a tool running (%)", fontsize=LABEL_PT)
    b.set_ylabel("share of sessions (%)", fontsize=LABEL_PT)
    b.set_ylim(0, 42)
    b.set_title("(b) Tool busy time per session", fontsize=TITLE_PT, loc="left")

    handles = []
    for tr in ("tracelab", "freeinference"):
        col, hatch = SERIES[tr]
        p = Patch(facecolor=col, edgecolor=INK, hatch=hatch, linewidth=0.6, label=NAMES[tr])
        handles.append(p)
    fig.legend(handles=handles, fontsize=6.2, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6, handleheight=1.1, columnspacing=1.4)
    fig.text(0.5, 1.0, "Public agent traces: idle windows and busy time of the tool side (human turns excluded)",
             fontsize=TITLE_PT, ha="center", va="bottom", fontweight="bold")
    fig.tight_layout(w_pad=1.6, rect=(0, 0.1, 1, 1))
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"public_traces_idle.{ext}", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
