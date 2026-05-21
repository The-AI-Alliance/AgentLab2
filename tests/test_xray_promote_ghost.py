"""Unit tests for _promote_ghost_episodes (xray_utils).

Covers:
- PR #372: QUEUED episodes must not be promoted without a dead-driver signal.
- PR #365: QUEUED promoted when driver is confirmed dead via experiment_status.json;
           RUNNING promoted immediately in sequential mode when driver is dead.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cube_harness.analyze.xray_utils import _promote_ghost_episodes
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus
from cube_harness.experiment_status import EXPERIMENT_STATUS_FILENAME, ExperimentStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_ep_status(ep_dir: Path, status: str, age_seconds: float) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    s = EpisodeStatus(
        status=status,
        task_id="test-task",
        episode_id=0,
        started_at=now - age_seconds,
        last_heartbeat_at=now - age_seconds,
    )
    s.write(ep_dir / STATUS_FILENAME)


def _write_exp_status(exp_dir: Path, status: str, mode: str, hb_age_seconds: float = 0) -> None:
    now = time.time()
    es = ExperimentStatus(
        status=status,
        mode=mode,
        pid=12345,
        host="test-host",
        started_at=now - hb_age_seconds - 1,
        last_heartbeat_at=now - hb_age_seconds,
        total_episodes=1,
    )
    es.write(exp_dir / EXPERIMENT_STATUS_FILENAME)


# ---------------------------------------------------------------------------
# QUEUED promotion
# ---------------------------------------------------------------------------


def test_queued_not_promoted_without_experiment_status(tmp_path: Path) -> None:
    """No experiment_status.json → driver assumed alive → QUEUED never promoted.

    Backward-compat: old experiments without a status file must not have their
    QUEUED episodes swept to STALE by the viewer.
    """
    ep = tmp_path / "episodes" / "ep_000"
    _write_ep_status(ep, "QUEUED", age_seconds=99_999)

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == "QUEUED"


@pytest.mark.parametrize("mode", ["ray", "sequential"])
def test_queued_promoted_when_driver_dead(tmp_path: Path, mode: str) -> None:
    """QUEUED → STALE when experiment_status.json reports a dead driver."""
    ep = tmp_path / "episodes" / "ep_000"
    _write_ep_status(ep, "QUEUED", age_seconds=99_999)
    _write_exp_status(tmp_path, status="INTERRUPTED", mode=mode)

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == "STALE"


def test_queued_not_promoted_when_driver_alive(tmp_path: Path) -> None:
    """QUEUED preserved when the driver heartbeat is fresh."""
    ep = tmp_path / "episodes" / "ep_000"
    _write_ep_status(ep, "QUEUED", age_seconds=99_999)
    _write_exp_status(tmp_path, status="RUNNING", mode="ray", hb_age_seconds=10)

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == "QUEUED"


# ---------------------------------------------------------------------------
# RUNNING promotion — ray mode
# ---------------------------------------------------------------------------


def test_old_running_episode_promoted_to_stale(tmp_path: Path) -> None:
    """RUNNING episode whose heartbeat is beyond GHOST_TIMEOUT becomes STALE."""
    ep = tmp_path / "episodes" / "ep_001"
    _write_ep_status(ep, "RUNNING", age_seconds=99_999)

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == "STALE"


def test_fresh_running_episode_not_promoted(tmp_path: Path) -> None:
    """RUNNING episode with a recent heartbeat is left alone."""
    ep = tmp_path / "episodes" / "ep_002"
    _write_ep_status(ep, "RUNNING", age_seconds=30)

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == "RUNNING"


def test_running_ray_dead_driver_still_waits_ghost_timeout(tmp_path: Path) -> None:
    """Ray workers run independently of the driver.

    A dead driver must NOT immediately promote a fresh RUNNING episode — the
    worker may still be alive and will write its own terminal status.
    """
    ep = tmp_path / "episodes" / "ep_000"
    _write_ep_status(ep, "RUNNING", age_seconds=30)
    _write_exp_status(tmp_path, status="INTERRUPTED", mode="ray")

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == "RUNNING"


# ---------------------------------------------------------------------------
# RUNNING promotion — sequential mode (driver == worker)
# ---------------------------------------------------------------------------


def test_running_sequential_promoted_immediately_when_driver_dead(tmp_path: Path) -> None:
    """In sequential mode the driver IS the worker.

    A dead driver means the episode process also died, so we promote RUNNING →
    STALE immediately without waiting for GHOST_TIMEOUT.
    """
    ep = tmp_path / "episodes" / "ep_000"
    _write_ep_status(ep, "RUNNING", age_seconds=30)  # fresh heartbeat
    _write_exp_status(tmp_path, status="INTERRUPTED", mode="sequential")

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == "STALE"


def test_running_sequential_not_promoted_when_driver_alive(tmp_path: Path) -> None:
    """Sequential RUNNING with a live driver and fresh heartbeat is left alone."""
    ep = tmp_path / "episodes" / "ep_000"
    _write_ep_status(ep, "RUNNING", age_seconds=30)
    _write_exp_status(tmp_path, status="RUNNING", mode="sequential", hb_age_seconds=10)

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == "RUNNING"


# ---------------------------------------------------------------------------
# Terminal episodes — never re-promoted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal_status", ["COMPLETED", "FAILED", "CANCELLED", "STALE"])
def test_terminal_episodes_not_touched(tmp_path: Path, terminal_status: str) -> None:
    """Already-terminal episodes are never re-promoted, even with a dead driver."""
    ep = tmp_path / "episodes" / "ep_003"
    _write_ep_status(ep, terminal_status, age_seconds=99_999)
    _write_exp_status(tmp_path, status="INTERRUPTED", mode="ray")

    _promote_ghost_episodes(tmp_path)

    assert EpisodeStatus.read(ep / STATUS_FILENAME).status == terminal_status
