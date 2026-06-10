"""Convert an experiment directory into a cube-registry community-journal record.

The output dict conforms to ``results-schema.json`` in cube-registry; the
companion ``scripts/submit_to_journal.py`` writes it to a file (and optionally
opens a PR against cube-registry). The schema is documented in
``cube-registry/openspec/changes/results-journal/design.md``.

This module is pure — no I/O beyond reading the experiment directory.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cube_harness.episode_status import EpisodeStatus, Status
from cube_harness.eval_log import EvalLog
from cube_harness.reproducibility import samples
from cube_harness.results import ExperimentResult

JOURNAL_SCHEMA_VERSION = "1.0"

# Slash → __ when sanitizing evaluation_id into a filename stem (matches
# cube-registry's results_check.py).
_FILENAME_REPLACE = str.maketrans({"/": "__"})

# Mirrors results-schema.json's evaluation_id pattern. Checked locally so a bad
# submitter handle (e.g. a git user.name with spaces) fails before a PR is
# opened, not in registry CI after.
_EVALUATION_ID_RE = re.compile(r"^[A-Za-z0-9_./-]{1,128}$")


@dataclass
class Outcomes:
    """Mutually-exclusive, exhaustive outcome counts over ``n_tasks``.

    Maps cube-harness :data:`EpisodeStatus.Status` values into the five
    buckets the cube-registry schema requires:

    ===========================  ================
    Source status                 Bucket
    ===========================  ================
    COMPLETED, reward > 0        n_success
    COMPLETED, reward == 0       n_failure
    MAX_STEPS_REACHED            n_max_steps
    FAILED / STALE / INVALID     n_system_error
    CANCELLED / in-flight / —    n_missing
    ===========================  ================
    """

    n_success: int = 0
    n_failure: int = 0
    n_max_steps: int = 0
    n_system_error: int = 0
    n_missing: int = 0

    def total(self) -> int:
        return self.n_success + self.n_failure + self.n_max_steps + self.n_system_error + self.n_missing

    def to_dict(self) -> dict[str, int]:
        return {
            "n_success": self.n_success,
            "n_failure": self.n_failure,
            "n_max_steps": self.n_max_steps,
            "n_system_error": self.n_system_error,
            "n_missing": self.n_missing,
        }


_SYSTEM_ERROR_STATUSES: frozenset[Status] = frozenset({"FAILED", "STALE", "INVALID_CONFIG"})
_MISSING_STATUSES: frozenset[Status] = frozenset({"CANCELLED", "QUEUED", "RUNNING"})


def classify(status: EpisodeStatus) -> str:
    """Return the outcome bucket name (``n_success``/…) for one episode status."""
    s: Status = status.status
    if s == "COMPLETED":
        return "n_success" if (status.reward or 0) > 0 else "n_failure"
    if s == "MAX_STEPS_REACHED":
        return "n_max_steps"
    if s in _SYSTEM_ERROR_STATUSES:
        return "n_system_error"
    if s in _MISSING_STATUSES:
        return "n_missing"
    raise ValueError(f"unknown EpisodeStatus: {s!r}")


def sanitize_filename(evaluation_id: str) -> str:
    """``alacoste/run1`` → ``alacoste__run1`` (matches cube-registry's results_check)."""
    return evaluation_id.translate(_FILENAME_REPLACE)


def _read_max_steps(experiment_dir: Path) -> int | None:
    """Read ``experiment_config.json`` and return the experiment's max_steps cap.

    Returns ``None`` when the file is missing or unparseable — non-fatal,
    just leaves the field out of the journal record.
    """
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return None
    value = data.get("max_steps")
    return value if isinstance(value, int) else None


def build_journal_record(
    experiment_dir: Path,
    *,
    submitter: str,
    cube_id: str | None = None,
) -> dict[str, Any]:
    """Build one cube-registry journal record from a completed experiment.

    Args:
        experiment_dir: Path to the experiment output directory (the dir holding
            ``experiment_record.json`` and ``episodes/``).
        submitter: GitHub handle of the submitter. Used to namespace the
            ``evaluation_id`` (``<submitter>/<exp_dir.name>``).
        cube_id: Override the cube id when the benchmark name in
            ``ExperimentRecord`` does not match the cube's registry id
            (e.g. ``"miniwob[level=all]"`` → ``"miniwob"``). Defaults to the
            benchmark name with any ``[…]`` suffix stripped.

    Returns:
        A dict matching cube-registry's ``results-schema.json`` 1.0.
    """
    experiment_dir = Path(experiment_dir)
    eval_log = EvalLog.load(experiment_dir)
    exp = eval_log.experiment
    episodes = eval_log.episodes

    bench_name = exp.benchmark_subset.name
    derived_cube_id = cube_id or bench_name.split("[", 1)[0]

    # ── outcome breakdown from per-episode status.json files ───────────────
    # `iter_episode_statuses` already dedups by (task_id, episode_id), so
    # the per-status sum equals the number of attempted tasks. Tasks in the
    # subset that produced no status file at all roll into n_missing.
    outcomes = Outcomes()
    for status in ExperimentResult(experiment_dir).iter_episode_statuses():
        bucket = classify(status)
        setattr(outcomes, bucket, getattr(outcomes, bucket) + 1)
    n_total = exp.benchmark_subset.n_tasks
    outcomes.n_missing += max(0, n_total - outcomes.total())

    # ── aggregate score + uncertainty from per-episode records ─────────────
    scores = [ep.score for ep in episodes]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    std_err = 0.0
    if len(scores) > 1:
        std_err = statistics.stdev(scores) / (len(scores) ** 0.5)

    # ── cost + wall time ───────────────────────────────────────────────────
    total_cost_usd = sum(ep.usage.total_cost_usd for ep in episodes)
    wall_times = [ep.wall_time_s for ep in episodes if ep.wall_time_s is not None]
    avg_wall_time_s = sum(wall_times) / len(wall_times) if wall_times else None

    max_steps = _read_max_steps(experiment_dir)

    agent_dict: dict[str, Any] = {
        "agent_id": exp.agent.agent_id,
        "config_type": exp.agent.config_type,
        "llm_model": exp.agent.llm_model or "",
        "framework_version": exp.agent.framework_version,
        "git_commit": exp.agent.git_commit or "",
    }
    # Optional fields — only include when non-empty / non-None so the
    # journal-schema's additionalProperties: false stays happy without
    # sending null sentinels we don't actually have.
    if exp.agent.dependency_versions:
        agent_dict["dependency_versions"] = exp.agent.dependency_versions
    if exp.agent.primary_dependencies:
        agent_dict["primary_dependencies"] = exp.agent.primary_dependencies
    if exp.agent.git_remote_url:
        agent_dict["git_remote_url"] = exp.agent.git_remote_url
    if exp.agent.git_is_dirty is not None:
        agent_dict["git_is_dirty"] = exp.agent.git_is_dirty
    if exp.agent.cube_standard_git_commit:
        agent_dict["cube_standard_git_commit"] = exp.agent.cube_standard_git_commit
    if exp.agent.cube_standard_git_is_dirty is not None:
        agent_dict["cube_standard_git_is_dirty"] = exp.agent.cube_standard_git_is_dirty
    if exp.agent.config:
        agent_dict["config"] = exp.agent.config

    subset_dict: dict[str, Any] = {
        "name": bench_name,
        "n_tasks": n_total,
    }
    if exp.benchmark_subset.filter:
        subset_dict["filter"] = exp.benchmark_subset.filter

    results_dict: dict[str, Any] = {
        "avg_score": round(avg_score, 6),
        "std_err": round(std_err, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        "outcomes": outcomes.to_dict(),
    }
    if avg_wall_time_s is not None:
        results_dict["avg_wall_time_s"] = round(avg_wall_time_s, 3)
    if max_steps is not None:
        results_dict["max_steps_per_episode"] = max_steps

    evaluation_id = f"{submitter}/{exp.evaluation_id}"
    if not _EVALUATION_ID_RE.match(evaluation_id):
        raise ValueError(
            f"evaluation_id {evaluation_id!r} does not match the registry's "
            f"{_EVALUATION_ID_RE.pattern!r} pattern — pass --submitter with a plain "
            "GitHub handle (no spaces or special characters)"
        )

    record: dict[str, Any] = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "evaluation_timestamp": exp.evaluation_timestamp,
        "eval_library": {
            "name": exp.eval_library.name,
            "version": exp.eval_library.version,
        },
        "agent": agent_dict,
        "benchmark_name": derived_cube_id,
        "benchmark_version": exp.benchmark_version or "unknown",
        "benchmark_subset": subset_dict,
        "results": results_dict,
    }
    return record


@dataclass
class JournalSubmission:
    """A registry submission: the small summary record + its companion bundle.

    Both files land under ``results/<cube-id>/`` — ``summary_filename`` (the
    human-reviewable, table-driving record, with a ``detailed_results`` pointer)
    and ``bundle_filename`` (the gzipped per-task samples the verifier re-derives
    the summary from).
    """

    record: dict[str, Any]
    bundle: bytes
    summary_filename: str
    bundle_filename: str


def build_journal_submission(
    experiment_dir: Path,
    *,
    submitter: str,
    cube_id: str | None = None,
) -> JournalSubmission:
    """Build the full registry submission: summary record + per-task bundle.

    The summary gains a ``detailed_results`` pointer (file / format / sha256 /
    n_samples). Fails fast if the summary's aggregate doesn't match the bundle —
    the same consistency invariant the registry verifier re-checks in CI, caught
    locally before anything is published.
    """
    record = build_journal_record(experiment_dir, submitter=submitter, cube_id=cube_id)
    bundle = samples.build_samples_bundle(experiment_dir)
    rows = samples.iter_samples(bundle)
    agg = samples.aggregate_from_samples(rows)

    # A task must contribute exactly one sample. Duplicates mean stale data made
    # it into the episode records (e.g. an archived retry attempt) — averaging
    # them silently misreports the score, and no downstream gate can catch it
    # because the summary and the bundle would be consistently wrong together.
    sample_ids = [row.get("sample_id") for row in rows]
    duplicates = sorted({sid for sid in sample_ids if sample_ids.count(sid) > 1})
    if duplicates:
        raise ValueError(
            f"samples bundle has {len(duplicates)} duplicate sample_id(s) "
            f"({duplicates[:5]}{'…' if len(duplicates) > 5 else ''}) — "
            "each task must contribute exactly one episode record"
        )
    n_tasks = record["benchmark_subset"]["n_tasks"]
    if len(rows) > n_tasks:
        raise ValueError(
            f"samples bundle has {len(rows)} rows but the subset only has {n_tasks} tasks — "
            "episode records outnumber tasks"
        )

    summary_avg = record["results"]["avg_score"]
    if agg["n_scored"] and agg["avg_score"] != summary_avg:
        raise ValueError(
            f"summary/bundle mismatch: results.avg_score={summary_avg} but "
            f"bundle recomputes {agg['avg_score']} over {agg['n_scored']} scored samples"
        )

    stem = sanitize_filename(record["evaluation_id"])
    bundle_filename = f"{stem}{samples.SAMPLES_SUFFIX}"
    record["detailed_results"] = {
        "file": bundle_filename,
        "format": samples.SAMPLES_FORMAT,
        "sha256": samples.sha256_hex(bundle),
        "n_samples": agg["n_samples"],
    }
    return JournalSubmission(
        record=record,
        bundle=bundle,
        summary_filename=f"{stem}.json",
        bundle_filename=bundle_filename,
    )
