#!/usr/bin/env python3
"""
Paper-ready benchmark timeline — white background, print-friendly colors.

Usage:
    python3 plot_timeline.py
"""

import calendar
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

matplotlib.use("Agg")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parents[2] / "code_for_gtc_slides" / "benchmarks_dates.csv"
OUT_PNG  = Path(__file__).parent / "figures" / "benchmarks_timeline.png"

FIGURE_W_IN = 18.0
FIGURE_H_IN = 8.0

CATEGORY_ORDER = [
    "Web / UI", "Desktop", "Mobile", "Code / SWE", "Tool Use / API",
    "Embodied 3D", "Robotics", "Navigation", "Game / RL", "Multi-Agent",
    "Social Simulation", "Scientific Discovery", "Healthcare", "Education",
    "Data Science", "Cybersecurity", "Network", "Workflow", "Planning",
    "RL / Multi-Turn", "Agent Safety", "Finance", "Legal",
    "Autonomous Driving", "Niche",
]

CATEGORY_MAP: dict[str, str] = {}

CATEGORY_COLORS: dict[str, str] = {
    "Web / UI":             "#D42B2B",
    "Desktop":              "#E05C3A",
    "Mobile":               "#E07A20",
    "Code / SWE":           "#8844DD",
    "Tool Use / API":       "#B89000",
    "Embodied 3D":          "#1450CC",
    "Robotics":             "#3B7FCC",
    "Navigation":           "#5599CC",
    "Game / RL":            "#1A8C2A",
    "Multi-Agent":          "#4E9933",
    "Social Simulation":    "#6BAA5A",
    "Scientific Discovery": "#009AAA",
    "Healthcare":           "#2AAA99",
    "Education":            "#CC8800",
    "Data Science":         "#3388AA",
    "Cybersecurity":        "#AA4422",
    "Network":              "#446699",
    "Workflow":             "#778800",
    "Planning":             "#556600",
    "RL / Multi-Turn":      "#996600",
    "Agent Safety":         "#CC7700",
    "Finance":              "#2D8877",
    "Legal":                "#556688",
    "Autonomous Driving":   "#4A6678",
    "Niche":                "#666677",
}

_LABEL_COLUMN = "Category"
_FONT_CANDIDATES = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

# Linear trend fit window: all complete months from this date forward.
_FIT_START = (2023, 1)


def _best_font() -> str:
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for font in _FONT_CANDIDATES:
        if font in available:
            return font
    return "sans-serif"


_FONT = _best_font()


def _quarter_label(year: int, month: int) -> str:
    return f"{year:04d}-Q{(month - 1) // 3 + 1}"


def _quarter_fraction(label: str) -> float:
    today = date.today()
    year = int(label[:4])
    q = int(label[6])
    q_start = date(year, (q - 1) * 3 + 1, 1)
    end_month = q * 3
    end_day = calendar.monthrange(year, end_month)[1]
    q_end = date(year, end_month, end_day)
    if today >= q_end:
        return 1.0
    if today < q_start:
        return 0.0
    return (today - q_start).days / (q_end - q_start).days


def load_data(csv_path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row["First Published"].strip()
            category = CATEGORY_MAP.get(row[_LABEL_COLUMN].strip(), row[_LABEL_COLUMN].strip())
            if not date_str or date_str == "?":
                continue
            parts = date_str.split("-")
            year = int(parts[0])
            month = int(parts[1]) if len(parts) >= 2 else 1
            counts[_quarter_label(year, month)][category] += 1
    return counts


def build_timeline(
    counts: dict[str, dict[str, int]],
) -> tuple[list[str], dict[str, list[int]]]:
    all_quarters = sorted(counts.keys())
    first_y, first_q = int(all_quarters[0][:4]), int(all_quarters[0][6])
    last_y, last_q = int(all_quarters[-1][:4]), int(all_quarters[-1][6])

    labels: list[str] = []
    y, q = first_y, first_q
    while (y, q) <= (last_y, last_q):
        labels.append(f"{y:04d}-Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1

    cat_series: dict[str, list[int]] = {cat: [0] * len(labels) for cat in CATEGORY_ORDER}
    for i, label in enumerate(labels):
        for cat, n in counts.get(label, {}).items():
            if cat in cat_series:
                cat_series[cat][i] += n

    return labels, cat_series


def _fit_monthly_trend(csv_path: Path) -> tuple[float, float]:
    """
    Fit a linear model to monthly benchmark counts from _FIT_START to the last
    complete month (current partial month excluded).

    Returns (slope, intercept) where the model is:
        predicted_count(t) = slope * t + intercept
    with t = months offset from today (t=0 → current month, t=1 → one month ahead).
    A positive slope means the field is still accelerating.
    """
    today = date.today()
    current_ym = (today.year, today.month)

    monthly: dict[tuple[int, int], int] = defaultdict(int)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row["First Published"].strip()
            if not date_str or date_str == "?":
                continue
            parts = date_str.split("-")
            year = int(parts[0])
            month = int(parts[1]) if len(parts) >= 2 else 1
            monthly[(year, month)] += 1

    # Collect complete months in [_FIT_START, current_ym)
    fit_y, fit_m = _FIT_START
    xs, ys = [], []
    y, m = fit_y, fit_m
    while (y, m) < current_ym:
        # t in months from today (negative = past)
        t = (y - today.year) * 12 + (m - today.month)
        xs.append(float(t))
        ys.append(float(monthly.get((y, m), 0)))
        m += 1
        if m > 12:
            m = 1
            y += 1

    if len(xs) < 2:
        return 0.0, float(np.mean(ys) if ys else 0.0)

    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _predict_months(slope: float, intercept: float, t1: float, t2: float) -> float:
    """
    Expected benchmark count over [t1, t2] months from today,
    integrating the linear rate model rate(t) = slope*t + intercept.
    """
    val = slope * (t2**2 - t1**2) / 2.0 + intercept * (t2 - t1)
    return max(0.0, val)


def _months_from_today(d: date) -> float:
    today = date.today()
    return (d - today).days / 30.4375


def _load_individual_dates(csv_path: Path) -> list[date]:
    """Return all benchmark publication dates, sorted ascending."""
    dates: list[date] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row["First Published"].strip()
            if not date_str or date_str == "?":
                continue
            parts = date_str.split("-")
            year = int(parts[0])
            month = int(parts[1]) if len(parts) >= 2 else 1
            day = int(parts[2]) if len(parts) >= 3 else 1
            try:
                dates.append(date(year, month, day))
            except ValueError:
                pass
    return sorted(dates)


def _date_to_xpos(
    d: date,
    labels: list[str],
    fore_labels: list[str] | None = None,
) -> float | None:
    """Map a date to a continuous x-axis position aligned with the bar centers.

    Bar for quarter index i spans x = [i-0.5, i+0.5].
    Past quarters use indices 0..len(labels)-1.
    Future quarters (fore_labels) use indices len(labels)..len(labels)+len(fore_labels)-1.
    Returns None if the date falls outside all known quarters.
    """
    fore_labels = fore_labels or []
    label = _quarter_label(d.year, d.month)
    if label in labels:
        bar_idx = labels.index(label)
    elif label in fore_labels:
        bar_idx = len(labels) + fore_labels.index(label)
    else:
        return None
    q = (d.month - 1) // 3 + 1
    q_start = date(d.year, (q - 1) * 3 + 1, 1)
    q_end_m = q * 3
    q_end = date(d.year, q_end_m, calendar.monthrange(d.year, q_end_m)[1])
    fraction = (d - q_start).days / ((q_end - q_start).days + 1)
    return bar_idx - 0.5 + fraction


def _forecast_quarters(
    labels: list[str],
    slope: float,
    intercept: float,
) -> tuple[list[str], np.ndarray]:
    """Project future full quarters using the linear monthly trend model."""
    last_y, last_q = int(labels[-1][:4]), int(labels[-1][6])
    fore_labels: list[str] = []
    y, q = last_y, last_q
    while True:
        q += 1
        if q > 4:
            q = 1
            y += 1
        if y > 2026:
            break
        fore_labels.append(f"{y:04d}-Q{q}")

    if not fore_labels:
        return [], np.array([])

    fore_counts = []
    for lbl in fore_labels:
        qy, qq = int(lbl[:4]), int(lbl[6])
        q_start = date(qy, (qq - 1) * 3 + 1, 1)
        end_m = qq * 3
        q_end = date(qy, end_m, calendar.monthrange(qy, end_m)[1])
        t1 = _months_from_today(q_start)
        t2 = _months_from_today(q_end)
        fore_counts.append(_predict_months(slope, intercept, t1, t2))

    return fore_labels, np.array(fore_counts)


def plot_timeline(
    labels: list[str],
    cat_series: dict[str, list[int]],
    total: int,
    all_dates: list[date],
) -> None:
    plt.rcParams["font.family"] = _FONT

    fig, ax = plt.subplots(figsize=(FIGURE_W_IN, FIGURE_H_IN))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    TEXT_COLOR     = "#222222"
    SPINE_COLOR    = "#AAAAAA"
    YEAR_SEP_COLOR = "#CCCCCC"
    ACCENT_COLOR   = "#1A6BB5"

    x = np.arange(len(labels))
    bar_width = 0.82

    # Raw counts — no extrapolation for partial quarters
    totals = np.array([
        sum(cat_series[cat][i] for cat in CATEGORY_ORDER)
        for i in range(len(labels))
    ])

    last_label = labels[-1]
    last_frac  = _quarter_fraction(last_label)
    is_partial = 0 < last_frac < 1.0

    # Fit linear trend on monthly point-level data
    slope, intercept = _fit_monthly_trend(CSV_PATH)
    print(f"Monthly trend model: rate(t) = {slope:+.2f}·t + {intercept:.1f}  "
          f"(t=0 → today, rate at today ≈ {intercept:.1f}/month)")

    # Estimate remaining benchmarks for the partial quarter (gray bar only)
    estimated_remaining = 0.0
    if is_partial:
        qy, qq = int(last_label[:4]), int(last_label[6])
        q_end_m = qq * 3
        q_end = date(qy, q_end_m, calendar.monthrange(qy, q_end_m)[1])
        t2 = _months_from_today(q_end)
        estimated_remaining = _predict_months(slope, intercept, 0.0, t2)
        pct = int(round(last_frac * 100))
        print(f"{last_label} is {pct}% complete — actual: {int(totals[-1])}, "
              f"estimated remaining: {estimated_remaining:.1f}")

    fore_labels, fore_counts = _forecast_quarters(labels, slope, intercept)
    x_fore = np.arange(len(labels), len(labels) + len(fore_labels))

    # Cumulative projection uses actual count at today, then model forward
    today = date.today()
    past_dates = [d for d in all_dates if d <= today]
    n_today = len(past_dates)
    eoy_total = int(n_today + _predict_months(
        slope, intercept, 0.0,
        _months_from_today(date(2026, 12, 31)),
    ))

    if len(fore_labels):
        for lbl, fc in zip(fore_labels, fore_counts):
            print(f"  Forecast {lbl}: {fc:.1f}")
    print(f"  Cumulative at today: {n_today},  projected EOY 2026: {eoy_total}")

    year_ticks = [i for i, lbl in enumerate(labels) if lbl.endswith("Q1")]
    year_labels_txt = [lbl[:4] for lbl in labels if lbl.endswith("Q1")]

    # Alternating year band shading
    for j, yt in enumerate(year_ticks):
        next_yt = year_ticks[j + 1] if j + 1 < len(year_ticks) else len(labels)
        if j % 2 == 0:
            ax.axvspan(yt - 0.5, next_yt - 0.5, color="#000000", alpha=0.03, zorder=0)

    # Stacked bars (actual data)
    bottom = np.zeros(len(labels))
    for cat in CATEGORY_ORDER:
        values = np.array(cat_series[cat], dtype=float)
        if values.sum() == 0:
            continue
        ax.bar(x, values, width=bar_width, bottom=bottom,
               color=CATEGORY_COLORS[cat], label=cat, linewidth=0)
        bottom += values

    # Gray hatched completion bar on top of the partial quarter
    if is_partial:
        last_i = len(labels) - 1
        ax.bar(x[last_i], estimated_remaining, width=bar_width,
               bottom=totals[last_i],
               facecolor="#CCCCCC", alpha=0.55,
               edgecolor="#888888", linewidth=1.0,
               hatch="///", zorder=4)

    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels_txt, fontsize=22, color=TEXT_COLOR, fontfamily=_FONT)
    ax.tick_params(axis="x", which="major", length=6, color=SPINE_COLOR)
    ax.tick_params(axis="x", which="minor", length=3, color=SPINE_COLOR)

    # Forecast bars (future full quarters)
    if len(fore_labels):
        fore_bars = ax.bar(
            x_fore, fore_counts, width=bar_width,
            facecolor="#EEEEEE", alpha=0.6,
            edgecolor="#888888", linewidth=1.2, zorder=3,
        )
        for patch in fore_bars.patches:
            patch.set_linestyle(":")

    # Y-axis ceiling: use actual + estimated Q2 total for last bar height
    last_bar_height = (totals[-1] + estimated_remaining) if is_partial else totals[-1]
    max_bar = max(float(last_bar_height), float(totals[:-1].max() if is_partial else totals.max()),
                  float(fore_counts.max()) if len(fore_counts) else 0.0)
    y_ceil = max_bar * 1.15
    ax.set_ylim(0, y_ceil)
    n_total = len(labels) + len(fore_labels)
    ax.set_xlim(-0.5, n_total - 0.5)
    ax.set_xticks(list(range(n_total)), minor=True)
    ax.set_ylabel("Agentic Benchmarks Published", fontsize=22, color=TEXT_COLOR,
                  labelpad=12, fontfamily=_FONT)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=18, colors=TEXT_COLOR)

    # Cumulative line on secondary Y-axis
    ax2 = ax.twinx()
    ax2.set_facecolor("none")

    # Smooth solid line: one point per actual benchmark up to today
    cum_xs = [
        xp for d in past_dates
        if (xp := _date_to_xpos(d, labels, fore_labels)) is not None
    ]
    cum_ys = list(range(1, len(cum_xs) + 1))
    if cum_xs:
        ax2.plot(cum_xs, cum_ys, color=ACCENT_COLOR, linewidth=2.5, zorder=5,
                 solid_capstyle="round")

    # Today's position — start of dotted projection
    x_today = _date_to_xpos(today, labels, fore_labels)

    # Smooth dotted projection from today to EOY 2026, sampled weekly
    eoy = date(2026, 12, 31)
    proj_dates = []
    d = today
    import datetime as _dt
    while d <= eoy:
        proj_dates.append(d)
        d += _dt.timedelta(weeks=1)
    if proj_dates[-1] < eoy:
        proj_dates.append(eoy)

    proj_xs = [_date_to_xpos(d, labels, fore_labels) for d in proj_dates]
    proj_ys = [
        n_today + _predict_months(slope, intercept, 0.0, _months_from_today(d))
        for d in proj_dates
    ]
    # Filter out None (dates outside known quarters shouldn't happen here)
    proj_xy = [(xp, yp) for xp, yp in zip(proj_xs, proj_ys) if xp is not None]
    if proj_xy:
        pxs, pys = zip(*proj_xy)
        ax2.plot(list(pxs), list(pys),
                 color="#888888", linewidth=1.8, linestyle=":", zorder=5, alpha=0.85,
                 solid_capstyle="round")
        _bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9)
        ax2.text(
            pxs[-1] - 0.1, pys[-1],
            f"~{eoy_total} by EOY 2026",
            color="#888888", fontsize=13, fontfamily=_FONT,
            fontweight="bold", ha="right", va="center", bbox=_bbox,
        )

    cum_max = max(n_today, eoy_total)
    ax2.set_ylim(0, cum_max * 1.22)
    ax2.set_ylabel("Cumulative", fontsize=20,
                   color=ACCENT_COLOR, labelpad=12, fontfamily=_FONT)
    ax2.tick_params(axis="y", labelsize=16, colors=ACCENT_COLOR)
    ax2.spines["right"].set_color(ACCENT_COLOR)
    ax2.spines["right"].set_linewidth(1.0)
    for sp in ["top", "left", "bottom"]:
        ax2.spines[sp].set_visible(False)

    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(SPINE_COLOR)
        ax.spines[sp].set_linewidth(0.8)

    for xt in year_ticks[1:]:
        ax.axvline(xt - 0.5, color=YEAR_SEP_COLOR, linewidth=0.8, zorder=1)

    # ChatGPT annotation
    chatgpt_q = "2022-Q4"
    if chatgpt_q in labels:
        xi = labels.index(chatgpt_q)
        chatgpt_frac = (date(2022, 11, 30) - date(2022, 10, 1)).days / (date(2022, 12, 31) - date(2022, 10, 1)).days
        x_chatgpt = xi - 0.5 + chatgpt_frac
        ax.axvline(x_chatgpt, color="#888888", linewidth=1.8, linestyle="--", zorder=4, alpha=0.7)
        ax.text(x_chatgpt + 0.4, y_ceil * 0.68, "ChatGPT\nNov 2022",
                color="#444444", fontsize=18, fontfamily=_FONT, va="top", linespacing=1.5,
                fontweight="bold")

    # Headline annotation
    _bbox_hl = dict(boxstyle="round,pad=0.35", facecolor="white",
                    edgecolor="#CCCCCC", linewidth=0.8, alpha=0.92)
    ax.text(0.5, 0.976,
            f"{total:,} published agentic benchmarks",
            transform=ax.transAxes,
            fontsize=26, fontfamily=_FONT, fontweight="bold",
            ha="center", va="top", color=TEXT_COLOR,
            bbox=_bbox_hl, zorder=7)

    # Legend
    handles = [
        Patch(facecolor=CATEGORY_COLORS[cat], label=cat, linewidth=0)
        for cat in CATEGORY_ORDER
        if sum(cat_series[cat]) > 0
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.88),
        bbox_transform=ax.transAxes,
        frameon=True,
        fontsize=17,
        title="Category",
        title_fontsize=18,
        labelcolor=TEXT_COLOR,
        ncols=3,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#CCCCCC")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.92)
    legend.get_title().set_color(TEXT_COLOR)
    legend.get_title().set_fontfamily(_FONT)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {OUT_PNG}")


def main() -> None:
    counts = load_data(CSV_PATH)
    labels, cat_series = build_timeline(counts)
    total = sum(sum(v.values()) for v in counts.values())
    all_dates = _load_individual_dates(CSV_PATH)
    plot_timeline(labels, cat_series, total, all_dates)


if __name__ == "__main__":
    main()
