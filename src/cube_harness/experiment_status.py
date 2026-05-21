"""Experiment-level status file, heartbeated by the runner driver process.

`experiment_status.json` sits at the root of `output_dir` alongside
`experiment_config.json`. It is written by the runner (not by Ray workers)
so the XRay viewer can determine whether the experiment driver is still alive
— and, if it is not, promote orphaned QUEUED episodes to STALE.

See also: `episode_status.py` for the per-episode equivalent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus

logger = logging.getLogger(__name__)

EXPERIMENT_STATUS_FILENAME = "experiment_status.json"


@dataclass
class ExperimentStatus:
    """Lifecycle snapshot of an entire experiment run, written by the driver.

    Fields
    ------
    status
        Terminal statuses are written once at shutdown; RUNNING is heartbeated.
    mode
        "ray"        — driver runs a Ray cluster; heartbeat comes from the
                       poll loop in _poll_ray (every ~30 s).
        "sequential" — driver IS the worker; heartbeat is updated before each
                       episode.run() call, so it may lag by one episode's
                       wall-clock time. XRay must fall back to episode-level
                       heartbeats before declaring the driver dead.
    pid / host
        Used for informational display and to detect cross-machine restarts.
    ray_dashboard_url
        Populated in "ray" mode when the dashboard URL is available;
        None otherwise.
    completed / failed / stale
        Running counters, updated alongside the heartbeat.
    """

    status: Literal["RUNNING", "COMPLETED", "INTERRUPTED"]
    mode: Literal["ray", "sequential"]
    pid: int
    host: str
    started_at: float
    last_heartbeat_at: float
    total_episodes: int
    completed: int = 0
    failed: int = 0
    stale: int = 0
    ray_dashboard_url: str | None = None
    ended_at: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "ExperimentStatus":
        """Parse, dropping unknown keys for forward compatibility."""
        data = json.loads(raw)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def read(cls, path: Path) -> "ExperimentStatus | None":
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

    def heartbeat(
        self,
        path: Path,
        *,
        completed: int | None = None,
        failed: int | None = None,
    ) -> None:
        """Update `last_heartbeat_at` (and optionally counters) and write best-effort.

        Heartbeat writes are best-effort — a transient I/O error must not abort the
        run. XRay tolerates a missed write because it reads the file on demand and
        the next heartbeat will overwrite atomically.
        """
        self.last_heartbeat_at = time.time()
        if completed is not None:
            self.completed = completed
        if failed is not None:
            self.failed = failed
        try:
            self.write(path)
        except Exception:
            logger.debug("Heartbeat write failed for %s", path, exc_info=True)


def is_driver_alive(
    exp_status: ExperimentStatus | None,
    exp_dir: Path,
    *,
    timeout_s: float,
) -> bool:
    """Heuristic check: does the experiment driver appear to be alive?

    Takes the already-read `ExperimentStatus` (None = file absent → assume alive,
    covers pre-heartbeat experiments and the initial migration window). Mode-aware:

    - "ray": trusts the experiment heartbeat alone (driver polls every ~30 s, so
      a heartbeat older than `timeout_s` means the driver crashed).
    - "sequential": falls back to per-episode heartbeats when the experiment
      heartbeat is stale, because the driver heartbeats *between* episodes — a
      long-running episode makes the experiment heartbeat look stale even though
      the driver is alive inside `episode.run()`.

    `timeout_s` is the staleness threshold for the heartbeat. Callers should pass
    a value at least an order of magnitude larger than `_EXP_HEARTBEAT_INTERVAL_S`
    in `exp_runner.py` to avoid false positives during short network/disk hiccups.
    """
    if exp_status is None:
        return True  # no status file → assume alive (pre-heartbeat experiment)
    if exp_status.status in ("COMPLETED", "INTERRUPTED"):
        return False
    now = time.time()
    if now - exp_status.last_heartbeat_at <= timeout_s:
        return True  # fresh experiment heartbeat
    if exp_status.mode == "sequential":
        # Stale experiment heartbeat in sequential mode: check per-episode heartbeats.
        # A live RUNNING episode means the driver is mid-episode and alive.
        episodes_dir = exp_dir / "episodes"
        if episodes_dir.exists():
            for ep_dir in episodes_dir.iterdir():
                if not ep_dir.is_dir() or ".archived_" in ep_dir.name:
                    continue
                ep_status = EpisodeStatus.read(ep_dir / STATUS_FILENAME)
                if ep_status is None or ep_status.status != "RUNNING":
                    continue
                ep_hb = ep_status.last_heartbeat_at or ep_status.started_at
                if now - ep_hb <= timeout_s:
                    return True
    return False
