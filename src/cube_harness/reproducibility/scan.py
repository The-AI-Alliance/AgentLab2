"""Classify experiment directories by reproducibility-journal eligibility.

Used by ``scripts/scan_experiments.py``. Lives in the package so it can be
unit-tested without subprocess invocation.

Categories (more restrictive than the original sketch — the philosophy is
"refuse rather than ship a borderline submission"):

  • ``already_submitted``  — ``submissions.json`` has a ``journal`` decision
    (either ``submitted`` or ``rejected``). Either way, skip — re-decision
    is an explicit ``--force`` operation handled by ``submit_to_journal.py``.
  • ``broken``             — the experiment cannot produce a meaningful score.
    Missing/corrupt ``experiment_record.json``, OR > N% system errors, OR
    every episode missing. Permanently marked via ``record_rejected``.
  • ``unfinished``         — some episode is still ``QUEUED`` / ``RUNNING``,
    OR no terminal-status episodes exist yet. State will change; don't
    write to ``submissions.json``.
  • ``subset_review``      — passed the integrity checks but the subset shape
    isn't a "complete named subset" (debug_limit applied, hand-picked task
    list via ``subset_from_list``, or filter set but n_missing > 0). The
    operator can submit with ``--yes`` after eyeballing the diagnosis.
  • ``submittable``        — clean run of a complete named subset or the full
    benchmark. Hand off to ``submit_to_journal.py``.

The ``is_official`` field on ``ExperimentRecord`` is an explicit operator override
of the subset_review/submittable inference (it never overrides the integrity
checks): ``True`` ⇒ submittable when complete and clean, ``False`` ⇒ never
submittable, ``None`` (default) ⇒ infer from debug_limit + subset shape. Editing
that one field in ``experiment_record.json`` and re-scanning reclassifies a run
without re-running it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from cube_harness.episode_status import IN_FLIGHT_STATUSES
from cube_harness.eval_log import EXPERIMENT_RECORD_FILENAME, ExperimentRecord
from cube_harness.exp_runner import DEFAULT_CANCEL_GRACE_S, DEFAULT_STEP_TIMEOUT_S
from cube_harness.experiment import DEFAULT_ORPHAN_THRESHOLD_S, sweep_stale_statuses
from cube_harness.reproducibility import submissions
from cube_harness.results import ExperimentResult
from cube_harness.storage import ARCHIVED_MARKER, FileStorage

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_ERROR_THRESHOLD = 0.10
"""Episodes that hit FAILED / STALE / INVALID_CONFIG count as system errors.

Above this fraction the recorded ``avg_score`` says more about your infra
than about the model — the run is marked BROKEN. 10% is intentionally tight
("be restrictive about what gets pushed"). Tunable per scan invocation.
"""


def sweep_stale_in_dir(experiment_dir: Path) -> list[str]:
    """Mark dead RUNNING/QUEUED episodes as STALE in *experiment_dir*.

    Pure delegation to :func:`cube_harness.experiment.sweep_stale_statuses` —
    the same code path the runner uses online. Reuses the runner's defaults
    (``DEFAULT_STEP_TIMEOUT_S`` / ``DEFAULT_CANCEL_GRACE_S`` /
    ``DEFAULT_ORPHAN_THRESHOLD_S``) so heartbeat semantics stay identical
    whether sweep is invoked online or offline. Returns the list of swept
    trajectory_ids. Non-fatal: any storage error is logged and swallowed so
    the scan can proceed with whatever statuses are currently on disk.
    """
    try:
        storage = FileStorage(experiment_dir)
        return sweep_stale_statuses(
            storage,
            step_timeout_s=DEFAULT_STEP_TIMEOUT_S,
            cancel_grace_s=DEFAULT_CANCEL_GRACE_S,
            orphan_threshold_s=DEFAULT_ORPHAN_THRESHOLD_S,
        )
    except Exception as e:
        logger.warning("stale-status sweep failed for %s: %s", experiment_dir, e)
        return []


def archive_experiment_dir(experiment_dir: Path) -> Path:
    """Rename *experiment_dir* to ``<name>.archived_<ts>`` so :func:`walk` skips it.

    Mirrors the per-episode archive convention used by FileStorage — same
    ``ARCHIVED_MARKER`` constant + timestamp suffix, just applied one level up.
    Returns the new path. No-op if the dir is already archived.
    """
    experiment_dir = Path(experiment_dir)
    if ARCHIVED_MARKER in experiment_dir.name:
        return experiment_dir
    target = experiment_dir.parent / f"{experiment_dir.name}{ARCHIVED_MARKER}{time.time()}"
    experiment_dir.rename(target)
    logger.info("archived %s -> %s", experiment_dir.name, target.name)
    return target


class ScanCategory(str, Enum):
    already_submitted = "already_submitted"
    broken = "broken"
    unfinished = "unfinished"  # episodes still QUEUED/RUNNING — state may change
    subset_review = "subset_review"
    submittable = "submittable"


@dataclass(frozen=True)
class ScanResult:
    """One experiment dir + its eligibility classification.

    *reasons* explains the classification — empty when ``submittable``; one
    or more diagnoses otherwise. For ``broken``, the first reason is also
    persisted into ``submissions.json`` as the rejection note.
    """

    experiment_dir: Path
    category: ScanCategory
    reasons: tuple[str, ...] = ()
    # Diagnostics for the operator. Always populated when possible so a
    # `--verbose` listing shows the underlying numbers.
    n_tasks: int | None = None
    n_terminal: int = 0
    n_in_flight: int = 0
    n_system_error: int = 0
    n_missing: int = 0
    debug_limit: int | None = None
    benchmark_subset_name: str = ""
    benchmark_subset_filter: str | None = None
    has_explicit_task_list: bool = False
    is_official: bool | None = None
    # Raw per-episode signal collected during the single status pass, so display
    # consumers (XRay's experiment table) don't have to re-read the status files.
    # `status_counts` maps raw EpisodeStatus.status → count; `rewards` are the
    # scored episode rewards (COMPLETED / MAX_STEPS_REACHED).
    status_counts: dict[str, int] = field(default_factory=dict)
    rewards: tuple[float, ...] = ()


def _load_experiment_record(experiment_dir: Path) -> ExperimentRecord | None:
    """Parse ``experiment_record.json`` or return None on missing/corrupt."""
    record_path = experiment_dir / EXPERIMENT_RECORD_FILENAME
    if not record_path.exists():
        return None
    try:
        return ExperimentRecord.model_validate_json(record_path.read_text())
    except Exception:
        return None


def _read_debug_limit_fallback(experiment_dir: Path) -> int | None:
    """Old experiment records predate ExperimentRecord.debug_limit; fall back to
    experiment_config.json so the scan classifier still has the signal."""
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        return None
    try:
        value = json.loads(config_path.read_text()).get("debug_limit")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, int) else None


def classify(
    experiment_dir: Path,
    *,
    system_error_threshold: float = DEFAULT_SYSTEM_ERROR_THRESHOLD,
    sweep_stale: bool = True,
) -> ScanResult:
    """Decide which bucket *experiment_dir* falls into.

    Pure with respect to ``submissions.json`` and the result classification.
    When *sweep_stale* is True (the default), runs
    :func:`sweep_stale_in_dir` first so dead RUNNING/QUEUED episodes (from a
    crashed Ray driver or killed worker) are reclassified as STALE before the
    eligibility logic runs. Without the sweep, those experiments would be
    permanently flagged as ``unfinished`` even when no worker is alive to
    finish them. Pass ``sweep_stale=False`` for a strictly read-only scan.
    """
    experiment_dir = Path(experiment_dir)
    if sweep_stale:
        sweep_stale_in_dir(experiment_dir)

    # ── Already-decided takes priority over every other signal ────────────
    if submissions.has_decision(experiment_dir, "journal"):
        prior = submissions.read(experiment_dir).get("journal", {})
        return ScanResult(
            experiment_dir,
            ScanCategory.already_submitted,
            (f"prior journal decision: {prior.get('status')} — {prior.get('reason') or prior.get('evaluation_id')}",),
        )

    # ── Integrity: must have a parseable ExperimentRecord ─────────────────
    record = _load_experiment_record(experiment_dir)
    if record is None:
        return ScanResult(
            experiment_dir,
            ScanCategory.broken,
            (f"missing or unparseable {EXPERIMENT_RECORD_FILENAME}",),
        )

    bench_subset = record.benchmark_subset
    n_tasks = bench_subset.n_tasks
    debug_limit = record.debug_limit
    if debug_limit is None:
        debug_limit = _read_debug_limit_fallback(experiment_dir)

    # ── Outcome breakdown from the per-episode status files ──────────────
    n_terminal = 0
    n_in_flight = 0
    n_system_error = 0
    seen_task_ids: set[str] = set()
    status_counts: dict[str, int] = {}
    rewards: list[float] = []
    try:
        for status in ExperimentResult(experiment_dir).iter_episode_statuses():
            seen_task_ids.add(status.task_id)
            status_counts[status.status] = status_counts.get(status.status, 0) + 1
            if status.status in IN_FLIGHT_STATUSES:
                n_in_flight += 1
                continue
            n_terminal += 1
            if status.status in {"FAILED", "STALE", "INVALID_CONFIG"}:
                n_system_error += 1
            elif status.status in {"COMPLETED", "MAX_STEPS_REACHED"} and status.reward is not None:
                rewards.append(float(status.reward))
    except Exception as e:
        return ScanResult(
            experiment_dir,
            ScanCategory.broken,
            (f"failed to read episode statuses: {e}",),
            n_tasks=n_tasks,
        )

    n_missing = max(0, n_tasks - len(seen_task_ids))
    has_explicit_task_list = bool(bench_subset.task_ids)

    base = dict(
        experiment_dir=experiment_dir,
        n_tasks=n_tasks,
        n_terminal=n_terminal,
        n_in_flight=n_in_flight,
        n_system_error=n_system_error,
        n_missing=n_missing,
        debug_limit=debug_limit,
        status_counts=status_counts,
        rewards=tuple(rewards),
        benchmark_subset_name=bench_subset.name,
        benchmark_subset_filter=bench_subset.filter,
        has_explicit_task_list=has_explicit_task_list,
        is_official=record.is_official,
    )

    # ── In-flight episodes mean the experiment is still running ──────────
    # "In-flight" is determined per-episode after the heartbeat sweep: a dead
    # worker/driver leaves STALE (terminal) episodes, so a truly abandoned run
    # reaches zero in-flight and falls through to the broken/err-rate rules
    # below. A run that still has a *live* episode heartbeat stays unfinished —
    # even in ray mode where workers can outlive a crashed driver.
    if n_in_flight > 0:
        return ScanResult(
            **base,
            category=ScanCategory.unfinished,
            reasons=(f"{n_in_flight} episode(s) still QUEUED/RUNNING",),
        )

    # ── No terminal episodes at all ⇒ nothing to submit ─────────────────
    if n_terminal == 0:
        return ScanResult(
            **base,
            category=ScanCategory.broken,
            reasons=("no terminal-status episodes recorded — experiment never produced any data",),
        )

    # ── System-error rate is permanent broken-ness ──────────────────────
    err_rate = n_system_error / n_terminal if n_terminal else 0.0
    if err_rate > system_error_threshold:
        return ScanResult(
            **base,
            category=ScanCategory.broken,
            reasons=(
                f"{n_system_error}/{n_terminal} episodes errored "
                f"({err_rate * 100:.0f}%, threshold {system_error_threshold * 100:.0f}%)",
            ),
        )

    # ── Missing tasks ⇒ unfinished (could resume) ───────────────────────
    if n_missing > 0:
        return ScanResult(
            **base,
            category=ScanCategory.unfinished,
            reasons=(f"{n_missing}/{n_tasks} task(s) have no status file — experiment may have stopped early",),
        )

    # ── Explicit run-intent override ─────────────────────────────────────
    # is_official is the operator's stated intent; it overrides the subset-shape
    # inference below but never the integrity checks above (a run must still be
    # complete and non-broken). None ⇒ fall through to inference.
    if record.is_official is False:
        return ScanResult(
            **base,
            category=ScanCategory.subset_review,
            reasons=("is_official=False — run marked debug, not for submission",),
        )
    if record.is_official is True:
        return ScanResult(**base, category=ScanCategory.submittable)

    # ── Subset-shape gate (is_official is None ⇒ infer) ──────────────────
    review_reasons: list[str] = []
    if debug_limit:
        review_reasons.append(f"debug_limit={debug_limit} was applied — not a full subset")
    if has_explicit_task_list:
        # subset_from_list submissions are not reproducibility-reference-able
        # without external documentation of why those specific tasks were
        # chosen. Default to requiring operator acknowledgement.
        review_reasons.append(
            f"benchmark_subset.task_ids is set ({len(bench_subset.task_ids or [])} explicit tasks) — "
            "hand-picked subsets need --yes to submit"
        )

    if review_reasons:
        return ScanResult(**base, category=ScanCategory.subset_review, reasons=tuple(review_reasons))

    return ScanResult(**base, category=ScanCategory.submittable)


def walk(
    root: Path,
    *,
    system_error_threshold: float = DEFAULT_SYSTEM_ERROR_THRESHOLD,
    sweep_stale: bool = True,
) -> list[ScanResult]:
    """Classify every direct subdirectory of *root* that contains an
    ``experiment_record.json``. Dirs without that file are silently skipped
    (they're not experiment dirs at all). Archived dirs — those carrying the
    standard ``ARCHIVED_MARKER`` suffix — are also skipped, so re-scanning a
    root after ``--archive-broken`` doesn't re-surface what was just moved."""
    root = Path(root)
    results: list[ScanResult] = []
    if not root.is_dir():
        return results
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if ARCHIVED_MARKER in entry.name:
            continue
        if not (entry / EXPERIMENT_RECORD_FILENAME).exists():
            continue
        results.append(
            classify(
                entry,
                system_error_threshold=system_error_threshold,
                sweep_stale=sweep_stale,
            )
        )
    return results
