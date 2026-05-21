"""Joint CSV across a sweep directory.

Walks experiment subdirectories of `sweep_dir`, reads each
`experiment_investigation_report.csv` (the per-experiment artefact produced by
`investigate_experiment`), joins with `cross_investigation_agreement.csv` if present, and
writes one row per `(experiment, episode)` to
`<sweep_dir>/joint_investigation_report.csv`.

The output is intentionally flat — meant for grep, awk, pandas, and
Auto-CUBE's per-round notes. The original per-experiment files are not
modified.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from cube_harness.analyze.cross_experiment.cross_investigation_agreement import (
    AGREEMENT_COLUMNS,
    AGREEMENT_REPORT_FILENAME,
)

logger = logging.getLogger(__name__)

JOINT_REPORT_FILENAME = "joint_investigation_report.csv"

# Per-experiment context columns — prepended to each row.
_PREFIX_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "family_id",
    "agent_dotted",
    "benchmark_dotted",
    "driver",
    "recipe",
    "litellm_proxy_url",
)

# Columns lifted from each `experiment_investigation_report.csv` (mirrors
# core._write_csv_report). Joint rows keep all of them.
_PER_EPISODE_COLUMNS: tuple[str, ...] = (
    "trajectory_id",
    "episode_record",
    "reward",
    "n_steps",
    "outcome",
    "primary_blame",
    "primary_blame_confidence",
    "other_blames",
    "hypothesis_confidence",
    "summary",
    "hypothesis",
    "cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "duration_s",
)

# Cross-investigator agreement columns we join in (primary_key = (trajectory_id, recipe)).
# `trajectory_id` and `recipe` are already in _PREFIX/_PER_EPISODE — skip them
# from the join projection to avoid duplicates.
_JOIN_COLUMNS: tuple[str, ...] = tuple(c for c in AGREEMENT_COLUMNS if c not in ("trajectory_id", "recipe"))

JOINT_REPORT_COLUMNS: tuple[str, ...] = _PREFIX_COLUMNS + _PER_EPISODE_COLUMNS + _JOIN_COLUMNS


def _load_summary(experiment_dir: Path) -> dict[str, Any]:
    """Read `experiment_investigation_summary.json` if present — a few prefix columns
    come from here."""
    path = experiment_dir / "experiment_investigation_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.warning("Could not parse %s: %s", path, e)
        return {}


def _load_experiment_config(experiment_dir: Path) -> dict[str, Any]:
    """Read `experiment_config.json` — provides agent / benchmark dotted names."""
    path = experiment_dir / "experiment_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.warning("Could not parse %s: %s", path, e)
        return {}


def _resolve_prefix(experiment_dir: Path) -> dict[str, str]:
    """Build the prefix-column dict for one experiment."""
    summary = _load_summary(experiment_dir)
    config = _load_experiment_config(experiment_dir)

    agent_dotted = ""
    benchmark_dotted = ""
    if isinstance(config, dict):
        agent_cfg = config.get("agent_config", {})
        benchmark_cfg = config.get("benchmark_config", {})
        if isinstance(agent_cfg, dict):
            agent_dotted = agent_cfg.get("_type", "")
        if isinstance(benchmark_cfg, dict):
            benchmark_dotted = benchmark_cfg.get("_type", "")

    return {
        "experiment_id": experiment_dir.name,
        # `family_id` is left blank by default — sweeps that group experiments
        # by family populate the JSON summary's `family_id` field; we forward it
        # here when present.
        "family_id": str(summary.get("family_id", "")),
        "agent_dotted": agent_dotted,
        "benchmark_dotted": benchmark_dotted,
        "driver": str(summary.get("driver", "")),
        "recipe": str(summary.get("recipe", "")),
        "litellm_proxy_url": str(summary.get("litellm_proxy_url", "") or ""),
    }


def _load_agreement_rows(experiment_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read `cross_investigation_agreement.csv` keyed by (trajectory_id, recipe)."""
    path = experiment_dir / AGREEMENT_REPORT_FILENAME
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row.get("trajectory_id", ""), row.get("recipe", ""))
            out[key] = row
    return out


def _discover_experiments(sweep_dir: Path) -> list[Path]:
    """Direct children of `sweep_dir` that contain an `experiment_investigation_report.csv`."""
    out: list[Path] = []
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "experiment_investigation_report.csv").exists():
            out.append(child)
    return out


def write_joint_csv(
    sweep_dir: Path,
    *,
    experiment_dirs: Sequence[Path] | None = None,
) -> Path:
    """Walk `sweep_dir`, read per-experiment CSVs, write `joint_investigation_report.csv`.

    `experiment_dirs` overrides the auto-walk — useful for sweeps where the
    layout isn't a flat list of children.

    Atomic write: writes to `<name>.tmp` then renames.
    """
    sweep_dir = Path(sweep_dir).resolve()
    out = sweep_dir / JOINT_REPORT_FILENAME
    tmp = out.with_suffix(out.suffix + ".tmp")

    dirs = list(experiment_dirs) if experiment_dirs is not None else _discover_experiments(sweep_dir)
    if not dirs:
        logger.warning("write_joint_csv: no experiments found under %s", sweep_dir)
        # Still write a header-only file so consumers can detect the empty case.
        with tmp.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(JOINT_REPORT_COLUMNS)).writeheader()
        tmp.replace(out)
        return out

    rows_written = 0
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(JOINT_REPORT_COLUMNS))
        w.writeheader()
        for exp_dir in dirs:
            prefix = _resolve_prefix(exp_dir)
            agreement = _load_agreement_rows(exp_dir)
            csv_path = exp_dir / "experiment_investigation_report.csv"
            with csv_path.open() as cf:
                for episode_row in csv.DictReader(cf):
                    joint_row: dict[str, str] = {col: "" for col in JOINT_REPORT_COLUMNS}
                    joint_row.update(prefix)
                    for col in _PER_EPISODE_COLUMNS:
                        joint_row[col] = episode_row.get(col, "")
                    join_key = (joint_row["trajectory_id"], joint_row["recipe"])
                    join_row = agreement.get(join_key, {})
                    for col in _JOIN_COLUMNS:
                        joint_row[col] = join_row.get(col, "")
                    w.writerow(joint_row)
                    rows_written += 1

    tmp.replace(out)
    logger.info("Wrote %s (%d rows from %d experiments)", out, rows_written, len(dirs))
    return out


__all__ = [
    "JOINT_REPORT_FILENAME",
    "JOINT_REPORT_COLUMNS",
    "write_joint_csv",
]
