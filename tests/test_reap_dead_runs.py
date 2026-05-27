"""Unit tests for the startup GC `_reap_dead_runs` (auto-fix #206).

Pins the identity + liveness reaping rules without a cluster: reap a prior run's
resources iff its experiment_status.json is RUNNING with a stale heartbeat, by
`run_id` — never a live run, a terminal run, the current run, or a run with no id.
"""

from __future__ import annotations

import time
from pathlib import Path

from cube_harness.exp_runner import _DEAD_RUN_STALE_AFTER_S, _reap_dead_runs
from cube_harness.experiment_status import ExperimentStatus


class _RecordingInfra:
    """Minimal InfraConfig stand-in that records cleanup(run_id) calls."""

    def __init__(self) -> None:
        self.reaped: list[str] = []

    def cleanup(self, run_id: str) -> None:
        self.reaped.append(run_id)


def _write_status(
    root: Path,
    dir_name: str,
    *,
    status: str,
    run_id: str,
    heartbeat_age_s: float,
) -> Path:
    now = time.time()
    st = ExperimentStatus(
        status=status,  # type: ignore[arg-type]
        mode="ray",
        pid=123,
        host="h",
        started_at=now - heartbeat_age_s - 1,
        last_heartbeat_at=now - heartbeat_age_s,
        total_episodes=1,
        run_id=run_id,
    )
    exp_dir = root / dir_name
    st.write(exp_dir / "experiment_status.json")
    return exp_dir


def test_reaps_only_dead_runs_by_id(tmp_path: Path) -> None:
    stale = _DEAD_RUN_STALE_AFTER_S + 60
    _write_status(tmp_path, "dead", status="RUNNING", run_id="rid-dead", heartbeat_age_s=stale)
    _write_status(tmp_path, "alive", status="RUNNING", run_id="rid-alive", heartbeat_age_s=5)
    _write_status(tmp_path, "done", status="COMPLETED", run_id="rid-done", heartbeat_age_s=stale)
    _write_status(tmp_path, "no_id", status="RUNNING", run_id="", heartbeat_age_s=stale)
    current = _write_status(tmp_path, "current", status="RUNNING", run_id="rid-cur", heartbeat_age_s=0)

    infra = _RecordingInfra()
    _reap_dead_runs(infra, current, current_run_id="rid-cur")

    # Only the stale RUNNING sibling with a run_id is reaped.
    assert infra.reaped == ["rid-dead"]


def test_no_siblings_is_noop(tmp_path: Path) -> None:
    current = _write_status(tmp_path, "current", status="RUNNING", run_id="rid-cur", heartbeat_age_s=0)
    infra = _RecordingInfra()
    _reap_dead_runs(infra, current, current_run_id="rid-cur")
    assert infra.reaped == []


def test_cleanup_failure_is_swallowed(tmp_path: Path) -> None:
    """A backend cleanup error for one dead run must not abort the GC / the launch."""
    stale = _DEAD_RUN_STALE_AFTER_S + 60
    _write_status(tmp_path, "dead", status="RUNNING", run_id="rid-dead", heartbeat_age_s=stale)
    current = _write_status(tmp_path, "current", status="RUNNING", run_id="rid-cur", heartbeat_age_s=0)

    class _Boom(_RecordingInfra):
        def cleanup(self, run_id: str) -> None:
            super().cleanup(run_id)
            raise RuntimeError("backend down")

    infra = _Boom()
    _reap_dead_runs(infra, current, current_run_id="rid-cur")  # must not raise
    assert infra.reaped == ["rid-dead"]
