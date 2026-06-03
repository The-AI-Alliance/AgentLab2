"""Read / write ``submissions.json`` inside an experiment directory.

Each entry records a per-destination outcome — either a successful submission
(so the scan script skips the dir on the next run) or an explicit rejection
(so a broken experiment is permanently marked non-submission-worthy with the
diagnosis preserved). Schema is dict-keyed by destination name so the same
file covers both the cube-registry journal and EEE.

File path: ``<experiment_dir>/submissions.json``.

File format (additive — both shapes round-trip cleanly through Python):

    {
      "journal": {
        "status": "submitted",
        "evaluation_id": "alacoste/20260404_195953_genny_miniwob",
        "schema_version": "1.0",
        "submitted_at": "2026-06-03T12:00:00Z",
        "submitted_by": "alacoste",
        "pr_url": "https://github.com/The-AI-Alliance/cube-registry/pull/124",
        "local_path": "./journal-out/results/miniwob/…json"
      },
      "eee": {
        "status": "submitted",
        "evaluation_id": "miniwob/azure/gpt-5.4-mini/20260404_…",
        "schema_version": "0.2.2",
        "submitted_at": "2026-06-03T12:05:00Z",
        "local_path": "./eee-out/data/miniwob/Azure/…json"
      }
    }

Or for a broken experiment:

    {
      "journal": {
        "status": "rejected",
        "reason": "broken: 18% of episodes errored, threshold is 10%",
        "decided_at": "2026-06-03T12:00:00Z"
      }
    }

The scan script treats both ``"status": "submitted"`` and
``"status": "rejected"`` as "do not consider for re-submission." Only the
absence of a destination key triggers a fresh eligibility check.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SUBMISSIONS_FILENAME = "submissions.json"

# Destination keys used in submissions.json. Extend the literal as new
# destinations are added.
SubmissionDestination = Literal["journal", "eee"]


def _path(experiment_dir: Path) -> Path:
    return Path(experiment_dir) / SUBMISSIONS_FILENAME


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def read(experiment_dir: Path) -> dict[str, dict[str, Any]]:
    """Return the parsed submissions.json, or an empty dict when the file is missing."""
    path = _path(experiment_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def has_decision(experiment_dir: Path, destination: SubmissionDestination) -> bool:
    """True iff *destination* already has a submitted-or-rejected entry."""
    entry = read(experiment_dir).get(destination)
    return isinstance(entry, dict) and entry.get("status") in {"submitted", "rejected"}


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """tmp-file + os.replace so a crash mid-write never leaves a half-file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def record_submitted(
    experiment_dir: Path,
    destination: SubmissionDestination,
    *,
    evaluation_id: str,
    schema_version: str,
    submitted_by: str | None = None,
    pr_url: str | None = None,
    local_path: str | None = None,
) -> None:
    """Record a successful submission for *destination*.

    Idempotent: overwrites a prior entry for the same destination. The scan
    script considers any present entry as "do not re-submit", so a re-run is
    safe but pointless.
    """
    payload = read(experiment_dir)
    entry: dict[str, Any] = {
        "status": "submitted",
        "evaluation_id": evaluation_id,
        "schema_version": schema_version,
        "submitted_at": _now_iso(),
    }
    if submitted_by:
        entry["submitted_by"] = submitted_by
    if pr_url:
        entry["pr_url"] = pr_url
    if local_path:
        entry["local_path"] = local_path
    payload[destination] = entry
    _write_atomic(_path(experiment_dir), payload)


def record_rejected(
    experiment_dir: Path,
    destination: SubmissionDestination,
    *,
    reason: str,
) -> None:
    """Permanently mark *destination* as non-submission-worthy with *reason*.

    Used by the scan script when an experiment fails an eligibility check that
    can never resolve favorably (e.g. > N% system errors). Idempotent: keeps
    the original rejection timestamp if one was already recorded, so the
    diagnosis history isn't lost.
    """
    payload = read(experiment_dir)
    existing = payload.get(destination, {})
    decided_at = existing.get("decided_at") if existing.get("status") == "rejected" else _now_iso()
    payload[destination] = {
        "status": "rejected",
        "reason": reason,
        "decided_at": decided_at,
    }
    _write_atomic(_path(experiment_dir), payload)
