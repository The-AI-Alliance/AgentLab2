#!/usr/bin/env python3
"""Experiment results reporter — markdown table over cube_harness_results/.

Reuses cube_harness public APIs (no parallel parsing of episode files):
  - `EpisodeStatus.from_json` and the canonical status constants for per-episode parsing
  - `ExperimentResult(exp_dir).summary()` as the canonical avg_reward source when the
    experiment has been finalised (falls back to a manual walk for in-flight runs).

Run via ``make report`` or directly:

    scripts/experiments_report.py                          # all experiments, newest first
    scripts/experiments_report.py --match haiku            # regex filter on dir name
    scripts/experiments_report.py --since 2026-05-03       # on or after date
    scripts/experiments_report.py --last 10                # most recent N
    scripts/experiments_report.py --results-dir /path/to   # custom results root
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.analyze.stats import reward_mean_stderr
from cube_harness.episode_status import IN_FLIGHT_STATUSES, STATUS_ICONS
from cube_harness.results import ExperimentResult

DEFAULT_RESULTS_DIR = Path.home() / "cube_harness_results"

# Status display order — terminal-valid first, in-flight, then terminal-error.
_DONE_STATUSES = ("COMPLETED", "MAX_STEPS_REACHED")
_ERROR_STATUSES = ("FAILED", "STALE", "CANCELLED", "INVALID_CONFIG")


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _classify_template(template: str) -> str:
    """Heuristic label for the agent's goal_template (best-effort, free-text)."""
    if not template or template.strip() == "{{task}}":
        return "minimal"
    t = template.lower()
    has_workflow = "suggested approach" in t
    has_thought = "take time" in t
    if has_workflow and has_thought:
        return "thought-workflow"
    if has_workflow:
        return "workflow"
    if has_thought:
        return "thought"
    return template[:30].replace("\n", " ")


def _short_benchmark(name: str) -> str:
    return name.replace("swebench-verified-cube", "sweb-v").replace("swebench-live-cube", "sweb-live")


def _scan_episodes(result: ExperimentResult) -> tuple[dict[str, int], list[float], float, int]:
    """Aggregate per-status counts, rewards, cost, and context-window errors.

    Walks via :meth:`ExperimentResult.iter_episode_statuses` so the episode set
    (including in-flight) matches what the XRay viewer sees.
    """
    counts: dict[str, int] = {}
    completed_rewards: list[float] = []
    total_cost = 0.0
    ctx_errors = 0

    for es in result.iter_episode_statuses():
        counts[es.status] = counts.get(es.status, 0) + 1
        if es.error_type and "ContextWindow" in es.error_type:
            ctx_errors += 1
        if es.status in _DONE_STATUSES and es.reward is not None:
            completed_rewards.append(float(es.reward))
        # Cost lives in episode_record.json, not status.json.
        ep_dir = result._dir / "episodes" / f"{es.task_id}_ep{es.episode_id}"
        total_cost += _load_json(ep_dir / "episode_record.json").get("usage", {}).get("total_cost_usd", 0.0)

    return counts, completed_rewards, total_cost, ctx_errors


def _format_episode_breakdown(counts: dict[str, int], ctx_errors: int) -> tuple[str, int]:
    """Return ("done/total status icons", n_total) for the table row.

    Uses ``STATUS_ICONS`` from ``episode_status`` — same canonical icon set the
    XRay viewer references for its HTML status cells.
    """
    n_done = sum(counts.get(s, 0) for s in _DONE_STATUSES)
    n_inflight = sum(counts.get(s, 0) for s in IN_FLIGHT_STATUSES)
    n_error = sum(counts.get(s, 0) for s in _ERROR_STATUSES)
    n_total = n_done + n_inflight + n_error

    parts = [f"{n_done}/{n_total}"]
    inflight_parts = [f"{counts[s]}{STATUS_ICONS[s]}" for s in ("RUNNING", "QUEUED") if counts.get(s)]
    if inflight_parts:
        parts.append("+" + "+".join(inflight_parts))
    error_parts = [f"{counts[s]}{STATUS_ICONS[s]}" for s in _ERROR_STATUSES if counts.get(s)]
    if error_parts:
        parts.append(" ".join(error_parts))
    if ctx_errors:
        parts.append(f"{ctx_errors}ctx")
    return " ".join(parts), n_total


def _format_accuracy(rewards: list[float]) -> str:
    """Format accuracy + standard error using the shared ``reward_mean_stderr``.

    Same formula as the XRay viewer's per-experiment row (auto-selects binomial
    vs sample SE based on data shape), so both tools produce identical CIs for
    the same data.
    """
    if not rewards:
        return "—"
    acc, se = reward_mean_stderr(rewards)
    return f"{acc:.1%} ±{se:.1%}"


def _parse_experiment(exp_dir: Path) -> dict | None:
    record = _load_json(exp_dir / "experiment_record.json")
    if not record:
        return None

    result = ExperimentResult(exp_dir)
    counts, rewards, total_cost, ctx_errors = _scan_episodes(result)
    if not counts:
        return None

    ts = record.get("evaluation_timestamp", 0)
    breakdown, _ = _format_episode_breakdown(counts, ctx_errors)

    return {
        "date": datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "?",
        "ts": ts,
        "name": record.get("experiment_name", exp_dir.name),
        "benchmark": _short_benchmark(record.get("benchmark_name", "?")),
        "model": (record.get("agent", {}).get("llm_model") or "?").split("/")[-1],
        "template": _classify_template(record.get("agent", {}).get("config", {}).get("goal_template", "")),
        "accuracy": _format_accuracy(rewards),
        "episodes": breakdown,
        "cost": f"${total_cost:.2f}" if total_cost else "—",
        "dir": exp_dir.name,
    }


_COLUMNS = ("date", "benchmark", "model", "template", "accuracy", "episodes", "cost", "name")
_HEADER_LABELS = {
    "date": "date",
    "benchmark": "bench",
    "model": "model",
    "template": "template",
    "accuracy": "accuracy",
    "episodes": "done/total",
    "cost": "cost",
    "name": "experiment",
}


def _format_rows(rows: list[dict]) -> str:
    widths = {c: max(len(_HEADER_LABELS[c]), max(len(r[c]) for r in rows)) for c in _COLUMNS}

    def line(r: dict) -> str:
        return "| " + " | ".join(f"{r[c]:<{widths[c]}}" for c in _COLUMNS) + " |"

    header = "| " + " | ".join(f"{_HEADER_LABELS[c]:<{widths[c]}}" for c in _COLUMNS) + " |"
    sep = "| " + " | ".join("-" * widths[c] for c in _COLUMNS) + " |"
    return "\n".join([header, sep, *(line(r) for r in rows)])


def main(
    match: Annotated[str | None, typer.Option(help="Regex filter on experiment directory name")] = None,
    since: Annotated[str | None, typer.Option(help="Only experiments on/after YYYY-MM-DD")] = None,
    last: Annotated[int | None, typer.Option(help="Show only the N most recent")] = None,
    results_dir: Annotated[Path, typer.Option(help="Results root")] = DEFAULT_RESULTS_DIR,
    header: Annotated[bool, typer.Option(help="Show table header + legend (use --no-header to omit)")] = True,
) -> None:
    """Markdown table of experiments in ``~/cube_harness_results/``.

    Flag names are auto-derived from parameter names (snake_case → kebab-case).
    """
    if not results_dir.exists():
        typer.echo(f"Results dir not found: {results_dir}", err=True)
        raise typer.Exit(1)

    since_ts = datetime.strptime(since, "%Y-%m-%d").timestamp() if since else None
    pattern = re.compile(match, re.IGNORECASE) if match else None

    rows: list[dict] = []
    for exp_dir in sorted(results_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue
        if pattern and not pattern.search(exp_dir.name):
            continue
        row = _parse_experiment(exp_dir)
        if row is None:
            continue
        if since_ts and row["ts"] < since_ts:
            continue
        rows.append(row)

    rows.sort(key=lambda r: r["ts"], reverse=True)
    if last:
        rows = rows[:last]

    if not rows:
        typer.echo("No matching experiments found.", err=True)
        raise typer.Exit(0)

    if header:
        typer.echo(
            "**Legend** — done/total: `+N▶` running · `+N🕐` queued · "
            "`N✗` failed · `N👻` stale (dead worker) · `N🚫` cancelled · "
            "`Nctx` context-window errors | accuracy ± SE over completed episodes\n"
        )
    typer.echo(_format_rows(rows))


if __name__ == "__main__":
    typer.run(main)
