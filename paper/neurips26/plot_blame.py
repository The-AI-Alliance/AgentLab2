#!/usr/bin/env python3
"""
Generate blame-distribution figures for the CUBE position paper.

Each bar spans 100 % of tasks: a pale "success" segment on the left (scaled
to the benchmark's pass-rate) followed by failure-attribution segments that
fill the remaining (1 - pass_rate) portion proportionally to the blame counts.

Reads  data/blame_counts.csv  and writes two PDFs:
  figures/blame_distribution.pdf        — 3-row, aggregated by modality
  figures/blame_distribution_bench.pdf  — 8-row, one bar per benchmark

Usage
-----
    python plot_blame.py            # real data
    python plot_blame.py --fake     # hardcoded placeholders
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")                    # vector PDF backend — no rasterisation
matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   9,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "figure.dpi":        150,
})

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# ── Colours ───────────────────────────────────────────────────────────────────
SUCCESS_COLOR = "#F0EDE8"   # pale beige, almost white

COLORS = {
    "model_capability":         "#385878",   # powder-blue dark
    "agent_scaffolding":        "#6890B8",   # powder-blue mid
    "submission_format":        "#A8C4D8",   # powder-blue pale
    "tool_failure":             "#386050",   # sage-green  dark
    "action_space_limited":     "#689878",   # sage-green  mid
    "insufficient_observation": "#A8C8B0",   # sage-green  pale
    "env_failure":              "#783858",   # dusty-rose  dark
    "task_unclear":             "#B07898",   # dusty-rose  mid
    "eval_brittle":             "#D8B8CC",   # dusty-rose  pale
}

# Blame categories only (success is drawn separately)
CATEGORIES = list(COLORS)

GROUPS = [
    ("Success", ["success"]),
    ("Agent",   ["model_capability", "agent_scaffolding", "submission_format"]),
    ("Tool",    ["tool_failure", "action_space_limited", "insufficient_observation"]),
    ("Bench",   ["env_failure", "task_unclear", "eval_brittle"]),
]

MODALITIES = ["Web", "CUA", "SWE"]

# Benchmark rows ordered by modality (Web → CUA → SWE)
BENCHMARK_ROWS = [
    ("WorkArena L2",       "Web", "WorkArena",          "L2"),
    ("WebArena-Verified",  "Web", "WebArena-Verified",   "full"),
    ("MiniWoB",            "Web", "MiniWoB",             "full"),
    ("OSWorld (Comp13)",   "CUA", "OSWorld-Computer13",  "test_nogdrive"),
    ("Win. Agent Arena",   "CUA", "Windows Agent Arena", "full"),
    ("SWE-bench Verified", "SWE", "SWE-bench Verified",  "full"),
    ("SWE-bench Live",     "SWE", "SWE-bench Live",      "lite-gold-226"),
    ("TerminalBench",      "SWE", "TerminalBench",       "full"),
]

# ── Pass rates ────────────────────────────────────────────────────────────────
# Average of GPT-5.4 and Sonnet 4.6 pass-rates for the matching config.
# Blame counts are the sum of GPT-5.4 (table 1) + Sonnet 4.6 (table 2) experiments.
# These scale the blame fractions so bars represent 100 % of all tasks.
PASS_RATES_BENCH = {
    "WorkArena L2":       0.324,   # avg(Sonnet 29.7+35.5, GPT-5.4 28.9+35.7) cfg A+B
    "WebArena-Verified":  0.119,   # avg(Sonnet 14.85, GPT-5.4 8.99)
    "MiniWoB":            0.560,   # avg(Sonnet 59.4, GPT-5.4 52.5)
    "OSWorld (Comp13)":   0.411,   # avg(Sonnet 51.6, GPT-5.4 30.7)
    "Win. Agent Arena":   0.395,   # avg of PyAutoGUI + Comp-13 configs
    "SWE-bench Verified": 0.617,   # avg(Sonnet 65.1, GPT-5.4 58.2)
    "SWE-bench Live":     0.370,   # avg(Sonnet 38.6, GPT-5.4 35.4)
    "TerminalBench":      0.205,   # avg(Sonnet 29.5, GPT-5.4 11.4)
}

# Weighted modality averages (Sonnet 4.6 + GPT-5.4, weighted by n_tasks × n_seeds)
PASS_RATES_MOD = {
    "Web": 0.412,
    "CUA": 0.399,
    "SWE": 0.504,
}

# ── Placeholder data ──────────────────────────────────────────────────────────
_RAW_MOD = {
    "Web": dict(zip(CATEGORIES, [0.38, 0.22, 0.08, 0.04, 0.02, 0.05, 0.03, 0.06, 0.07])),
    "CUA": dict(zip(CATEGORIES, [0.32, 0.28, 0.06, 0.05, 0.03, 0.11, 0.06, 0.03, 0.03])),
    "SWE": dict(zip(CATEGORIES, [0.787, 0.025, 0.000, 0.007, 0.004, 0.000, 0.101, 0.000, 0.076])),
}
FAKE_FRACTIONS_MOD = {
    mod: {cat: v / sum(d.values()) for cat, v in d.items()}
    for mod, d in _RAW_MOD.items()
}
FAKE_FRACTIONS_BENCH = {
    label: FAKE_FRACTIONS_MOD[mod]
    for label, mod, _, _ in BENCHMARK_ROWS
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv_modality(csv_path):
    df = pd.read_csv(csv_path)
    fracs = {}
    for mod in MODALITIES:
        sub = df[df["modality"] == mod]
        counts = {cat: float(sub[cat].fillna(0).sum()) for cat in CATEGORIES}
        total = sum(counts.values())
        if total == 0:
            print(f"[warn] no data for modality {mod!r}", file=sys.stderr)
            continue
        fracs[mod] = {cat: counts[cat] / total for cat in CATEGORIES}
    return fracs


def load_csv_benchmark(csv_path):
    df = pd.read_csv(csv_path)
    fracs = {}
    for label, _mod, cube, subset in BENCHMARK_ROWS:
        sub = df[(df["cube"] == cube) & (df["subset"] == subset)]
        if sub.empty:
            continue
        counts = {cat: float(sub[cat].fillna(0).sum()) for cat in CATEGORIES}
        total = sum(counts.values())
        if total == 0:
            continue
        fracs[label] = {cat: counts[cat] / total for cat in CATEGORIES}
    return fracs

# ── Axes geometry (figure-fraction coords) ────────────────────────────────────
BAR_AXES = (0.08, 0.13, 0.53, 0.81)
LEG_AXES = (0.64, 0.03, 0.35, 0.94)


def _bar_center_fig(bar_idx, n_bars):
    _, bot, _, h = BAR_AXES
    axes_frac = 1.0 - (bar_idx + 0.5) / n_bars
    return bot + axes_frac * h


def _fig_to_leg(fig_y):
    _, bot, _, h = LEG_AXES
    return (fig_y - bot) / h

# ── Bar panel ─────────────────────────────────────────────────────────────────

def draw_bars(ax, fracs, labels, bar_h, pass_rates=None):
    for yi, label in enumerate(labels):
        left = 0.0

        # Success segment — spans the pass-rate fraction of the bar
        pr = (pass_rates or {}).get(label, 0.0)
        if pr > 0:
            ax.barh(yi, pr, left=left, height=bar_h,
                    color=SUCCESS_COLOR, linewidth=0.4, edgecolor="#D8D4CE")
            left += pr
        failure_rate = 1.0 - pr

        # Blame segments — each fraction is scaled to the failure portion
        for cat in CATEGORIES:
            w = fracs.get(label, {}).get(cat, 0.0) * failure_rate
            if w == 0:
                continue
            ax.barh(yi, w, left=left, height=bar_h, color=COLORS[cat], linewidth=0)
            left += w

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", length=3, width=0.6)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.invert_yaxis()

# ── Legend panel ──────────────────────────────────────────────────────────────

def draw_legend(ax, group_centers_leg, spacing=0.078, sw_h=0.028):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_title_r = 0.28
    x_div     = 0.32
    x_sw_l    = 0.36
    x_sw_r    = 0.52
    x_lbl     = 0.57

    # Build item positions for each category in each group
    item_y     = {}
    grp_extent = {}
    for grp, cats in GROUPS:
        ctr = group_centers_leg[grp]
        n   = len(cats)
        ys  = [ctr + ((n - 1) / 2 - i) * spacing for i in range(n)]
        for cat, y in zip(cats, ys):
            item_y[cat] = y
        grp_extent[grp] = (max(ys), min(ys))

    # Box background + divider column
    ax.add_patch(mpatches.Rectangle(
        (0.01, 0.01), 0.98, 0.98,
        linewidth=0.75, edgecolor="#999999", facecolor="white",
        transform=ax.transAxes, zorder=1, clip_on=False,
    ))
    ax.add_patch(mpatches.Rectangle(
        (0.01, 0.01), x_div - 0.01, 0.98,
        linewidth=0, facecolor="#F4F4F5",
        transform=ax.transAxes, zorder=2, clip_on=False,
    ))
    ax.add_patch(mpatches.Rectangle(
        (0.01, 0.01), 0.98, 0.98,
        linewidth=0.75, edgecolor="#999999", facecolor="none",
        transform=ax.transAxes, zorder=4, clip_on=False,
    ))
    ax.plot([x_div, x_div], [0.01, 0.99], color="#CCCCCC",
            linewidth=0.55, zorder=3, transform=ax.transAxes, clip_on=False)

    # Group separators
    grp_names = [g for g, _ in GROUPS]
    for i in range(len(grp_names) - 1):
        bot_y = grp_extent[grp_names[i]][1]
        top_y = grp_extent[grp_names[i + 1]][0]
        sep_y = (bot_y + top_y) / 2
        ax.plot([0.01, 0.99], [sep_y, sep_y], color="#DDDDDD",
                linewidth=0.5, zorder=3)

    # Group titles and swatches
    for grp, cats in GROUPS:
        ax.text(x_title_r, group_centers_leg[grp], grp,
                ha="right", va="center", fontsize=7.5, fontweight="bold", zorder=5)
        for cat in cats:
            y = item_y[cat]
            if cat == "success":
                # Thin border so the near-white swatch is visible
                ax.add_patch(mpatches.Rectangle(
                    (x_sw_l, y - sw_h / 2), x_sw_r - x_sw_l, sw_h,
                    facecolor=SUCCESS_COLOR, linewidth=0.6,
                    edgecolor="#BBBBBB", zorder=5,
                ))
                ax.text(x_lbl, y, "success (passed)",
                        ha="left", va="center", fontsize=6.2, zorder=5)
            else:
                ax.add_patch(mpatches.Rectangle(
                    (x_sw_l, y - sw_h / 2), x_sw_r - x_sw_l, sw_h,
                    color=COLORS[cat], linewidth=0, zorder=5,
                ))
                ax.text(x_lbl, y, cat,
                        ha="left", va="center", fontsize=6.2, zorder=5)

# ── Figure builders ───────────────────────────────────────────────────────────

def make_modality_figure(fracs):
    fig = plt.figure(figsize=(7.4, 2.4))
    ax_bar = fig.add_axes(BAR_AXES)
    ax_leg = fig.add_axes(LEG_AXES)

    draw_bars(ax_bar, fracs, MODALITIES, bar_h=0.55, pass_rates=PASS_RATES_MOD)

    group_centers_leg = {
        "Success": 0.88,
        "Agent":   0.63,
        "Tool":    0.38,
        "Bench":   0.13,
    }
    draw_legend(ax_leg, group_centers_leg, spacing=0.088, sw_h=0.034)
    return fig


def make_benchmark_figure(fracs):
    labels = [row[0] for row in BENCHMARK_ROWS]
    fig = plt.figure(figsize=(7.4, 2.4))
    ax_bar = fig.add_axes(BAR_AXES)
    ax_leg = fig.add_axes(LEG_AXES)

    draw_bars(ax_bar, fracs, labels, bar_h=0.62, pass_rates=PASS_RATES_BENCH)

    group_centers_leg = {
        "Success": 0.88,
        "Agent":   0.63,
        "Tool":    0.38,
        "Bench":   0.13,
    }
    draw_legend(ax_leg, group_centers_leg, spacing=0.078, sw_h=0.028)
    return fig

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true",
                        help="use hardcoded placeholder data instead of CSV")
    args = parser.parse_args()

    tag      = "_fake" if args.fake else ""
    csv_path = HERE / "data" / "blame_counts.csv"

    if args.fake:
        fracs_mod   = FAKE_FRACTIONS_MOD
        fracs_bench = FAKE_FRACTIONS_BENCH
    else:
        fracs_mod   = load_csv_modality(csv_path)
        fracs_bench = load_csv_benchmark(csv_path)

    for fig, stem in [
        (make_modality_figure(fracs_mod),    f"blame_distribution{tag}"),
        (make_benchmark_figure(fracs_bench), f"blame_distribution_bench{tag}"),
    ]:
        out = HERE / "figures" / f"{stem}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
