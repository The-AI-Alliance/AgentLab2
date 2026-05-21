"""Per-episode status file used to drive resume/retry decisions.

`status.json` is the control-plane sibling of the trajectory data. It is read
and written without deserialising trajectory steps, so the driver can make
retry decisions in O(N) file reads rather than O(N×steps) deserialisations.

See `openspec/changes/episode-status/proposal.md` for the full design.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

Status = Literal[
    "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "STALE", "MAX_STEPS_REACHED", "INVALID_CONFIG"
]

# In-flight: episode hasn't reached a terminal state yet. Driver poll should
# leave these alone (subject to orphan/heartbeat sweeps for dead-worker cleanup).
IN_FLIGHT_STATUSES: frozenset[Status] = frozenset({"QUEUED", "RUNNING"})

TERMINAL_STATUSES: frozenset[Status] = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "STALE", "MAX_STEPS_REACHED", "INVALID_CONFIG"}
)
# Two terminal statuses are deliberately NOT retriable:
# - MAX_STEPS_REACHED: the agent legitimately ran out of its step budget; retrying
#   would just truncate again from a fresh initial state.
# - INVALID_CONFIG: a permanent provider error (e.g. typo'd model name, bad API key,
#   malformed request). The identical request will fail identically on retry, so
#   replaying the episode only burns the retry budget. See is_permanent_llm_error.
RETRIABLE_STATUSES: frozenset[Status] = frozenset({"FAILED", "CANCELLED", "STALE"})

STATUS_FILENAME = "status.json"

# Canonical icons by raw Status. One-character/one-grapheme per entry so terminal
# tools (e.g. ``scripts/experiments_report.py``) can lay them out cleanly. The XRay viewer
# wraps these in ``<span title='…'>`` for hover tooltips and additionally splits
# COMPLETED into success/fail using the recorded reward; that split lives in
# ``analyze/xray_utils.py`` since it depends on reward semantics, not status alone.
STATUS_ICONS: dict[Status, str] = {
    "QUEUED": "🕐",
    "RUNNING": "▶",
    "COMPLETED": "✓",
    "MAX_STEPS_REACHED": "⏱",
    "FAILED": "✗",
    "STALE": "👻",
    "CANCELLED": "🚫",
    "INVALID_CONFIG": "⛔",
}


@dataclass
class EpisodeStatus:
    """Lifecycle snapshot of an episode.

    Written by both the worker (real progress) and the driver (pre-claim,
    cancellation). `last_heartbeat_at` is `None` until the worker enters
    `_run_loop`; this is the "queued in Ray" signal for the driver poll.
    """

    status: Status
    task_id: str
    episode_id: int
    started_at: float
    ended_at: float | None = None
    last_heartbeat_at: float | None = None
    current_step: int = 0
    reward: float | None = None
    had_step_errors: bool = False
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "EpisodeStatus":
        """Parse a status.json blob, ignoring unknown keys for forward compat.

        A status file written by a future version may carry fields this version
        doesn't know about. Dropping them silently lets older readers continue
        functioning instead of treating the file as corrupt.
        """
        data = json.loads(raw)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def read(cls, path: Path) -> "EpisodeStatus | None":
        if not path.exists():
            return None
        try:
            return cls.from_json(path.read_text())
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def write(self, path: Path) -> None:
        """Atomic write: tmp sibling + os.replace() so partial files are never observed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json())
        os.replace(tmp, path)


def should_sweep_running_to_stale(
    status: EpisodeStatus,
    *,
    now: float,
    step_timeout_s: float,
    cancel_grace_s: float,
    process_start_s: float | None = None,
) -> bool:
    """Return True iff a RUNNING episode's worker appears dead and should become STALE.

    Two cases:
    - heartbeat predates `process_start_s` — worker is provably from a dead prior process.
    - heartbeat age > `step_timeout_s + cancel_grace_s` — worker died without writing a
      terminal status.

    Returns False when `last_heartbeat_at` is None (episode never left QUEUED; no worker).
    """
    if status.last_heartbeat_at is None:
        return False
    if process_start_s is not None and status.last_heartbeat_at < process_start_s:
        return True
    return now - status.last_heartbeat_at > step_timeout_s + cancel_grace_s


def next_retry_count(prior: "EpisodeStatus | None") -> int:
    """Return the retry_count for a new attempt, given the prior status (if any).

    - No prior: 0 (original attempt).
    - Prior in-flight (QUEUED / RUNNING): same retry_count (idempotent re-pre-claim).
    - Prior terminal (FAILED/CANCELLED/STALE/COMPLETED/MAX_STEPS_REACHED/INVALID_CONFIG): prior + 1.
    """
    if prior is None:
        return 0
    if prior.status in IN_FLIGHT_STATUSES:
        return prior.retry_count
    return prior.retry_count + 1
