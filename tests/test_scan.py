"""Tests for cube_harness.reproducibility.scan + submissions."""

from __future__ import annotations

import json
from pathlib import Path

from cube_harness.episode_status import EpisodeStatus, Status
from cube_harness.eval_log import EvalLog
from cube_harness.reproducibility import submissions
from cube_harness.reproducibility.scan import (
    ScanCategory,
    classify,
    walk,
)

# Reuse the helpers from the reproducibility test module.
from tests.test_reproducibility import _ep_record, _exp_record, _write_status

# ──────────────────────────────────────────────────────────────────────────
# Submissions helpers
# ──────────────────────────────────────────────────────────────────────────


class TestSubmissions:
    def test_read_missing_returns_empty(self, tmp_path: Path) -> None:
        assert submissions.read(tmp_path) == {}

    def test_has_decision_false_when_no_file(self, tmp_path: Path) -> None:
        assert not submissions.has_decision(tmp_path, "journal")

    def test_record_submitted_roundtrip(self, tmp_path: Path) -> None:
        submissions.record_submitted(
            tmp_path,
            "journal",
            evaluation_id="alacoste/run_1",
            schema_version="1.0",
            submitted_by="alacoste",
            pr_url="https://example/pr/1",
            local_path="/tmp/x.json",
        )
        assert submissions.has_decision(tmp_path, "journal")
        data = submissions.read(tmp_path)
        assert data["journal"]["status"] == "submitted"
        assert data["journal"]["evaluation_id"] == "alacoste/run_1"
        assert "submitted_at" in data["journal"]

    def test_record_rejected_preserves_initial_timestamp(self, tmp_path: Path) -> None:
        submissions.record_rejected(tmp_path, "journal", reason="broken: errors")
        first = submissions.read(tmp_path)["journal"]["decided_at"]
        # Re-rejecting must keep the original timestamp so history isn't lost.
        submissions.record_rejected(tmp_path, "journal", reason="broken: errors")
        second = submissions.read(tmp_path)["journal"]["decided_at"]
        assert first == second

    def test_pending_is_not_a_decision(self, tmp_path: Path) -> None:
        submissions.record_pending(tmp_path, "journal")
        assert submissions.read(tmp_path)["journal"]["status"] == "pending"
        # Transient — must not block a later (re)submission.
        assert not submissions.has_decision(tmp_path, "journal")

    def test_failed_is_not_a_decision(self, tmp_path: Path) -> None:
        submissions.record_failed(tmp_path, "journal", reason="push timed out")
        entry = submissions.read(tmp_path)["journal"]
        assert entry["status"] == "failed" and entry["reason"] == "push timed out"
        # Retryable, unlike rejected — stays eligible.
        assert not submissions.has_decision(tmp_path, "journal")

    def test_submitted_overwrites_pending(self, tmp_path: Path) -> None:
        submissions.record_pending(tmp_path, "journal")
        submissions.record_submitted(tmp_path, "journal", evaluation_id="j/1", schema_version="1.0")
        assert submissions.has_decision(tmp_path, "journal")
        assert submissions.read(tmp_path)["journal"]["status"] == "submitted"

    def test_journal_and_eee_coexist(self, tmp_path: Path) -> None:
        submissions.record_submitted(
            tmp_path,
            "journal",
            evaluation_id="j/1",
            schema_version="1.0",
        )
        submissions.record_submitted(
            tmp_path,
            "eee",
            evaluation_id="e/1",
            schema_version="0.2.2",
        )
        data = submissions.read(tmp_path)
        assert set(data) == {"journal", "eee"}


# ──────────────────────────────────────────────────────────────────────────
# Scan fixtures
# ──────────────────────────────────────────────────────────────────────────


def _populate_clean_run(exp_dir: Path, n_tasks: int = 3, n_success: int = 3) -> None:
    """An experiment dir with n_tasks tasks, all COMPLETED with reward 1/0."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp = _exp_record(n_tasks=n_tasks)
    EvalLog(experiment=exp, episodes=[_ep_record(f"t{i}", 1.0) for i in range(n_tasks)]).save(exp_dir)
    for i in range(n_tasks):
        reward = 1.0 if i < n_success else 0.0
        _write_status(exp_dir, f"t{i}", "COMPLETED", reward=reward)


def _set_record_field(exp_dir: Path, **kwargs) -> None:
    """Patch experiment_record.json with the supplied top-level fields."""
    path = exp_dir / "experiment_record.json"
    data = json.loads(path.read_text())
    data.update(kwargs)
    path.write_text(json.dumps(data))


def _set_subset_field(exp_dir: Path, **kwargs) -> None:
    path = exp_dir / "experiment_record.json"
    data = json.loads(path.read_text())
    data["benchmark_subset"].update(kwargs)
    path.write_text(json.dumps(data))


def _add_status(exp_dir: Path, task_id: str, status: Status, reward: float | None = None) -> None:
    _write_status(exp_dir, task_id, status, reward=reward)


# ──────────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────────


class TestClassifierHappyPath:
    def test_clean_full_subset_is_submittable(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "clean"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        result = classify(exp_dir)
        assert result.category is ScanCategory.submittable
        assert result.reasons == ()


class TestClassifierBroken:
    def test_missing_record_is_broken(self, tmp_path: Path) -> None:
        # No experiment_record.json at all.
        d = tmp_path / "ghost"
        d.mkdir()
        result = classify(d)
        assert result.category is ScanCategory.broken
        assert "missing" in result.reasons[0] or "unparseable" in result.reasons[0]

    def test_corrupt_record_is_broken(self, tmp_path: Path) -> None:
        d = tmp_path / "corrupt"
        d.mkdir()
        (d / "experiment_record.json").write_text("{not valid json")
        result = classify(d)
        assert result.category is ScanCategory.broken

    def test_high_system_error_rate_is_broken(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "errored"
        _populate_clean_run(exp_dir, n_tasks=4, n_success=0)
        # Replace 2 of 4 COMPLETED statuses with FAILED.
        for i in (0, 1):
            _add_status(exp_dir, f"t{i}", "FAILED")
        result = classify(exp_dir, system_error_threshold=0.10)
        assert result.category is ScanCategory.broken
        assert "errored" in result.reasons[0]

    def test_no_terminal_episodes_is_broken(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "empty"
        exp_dir.mkdir()
        exp = _exp_record(n_tasks=3)
        EvalLog(experiment=exp, episodes=[]).save(exp_dir)
        # No status.json files written at all.
        result = classify(exp_dir)
        assert result.category is ScanCategory.broken


class TestClassifierUnfinished:
    def test_in_flight_episode_is_unfinished(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "running"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        # Add a 4th status that's still RUNNING.
        _set_subset_field(exp_dir, n_tasks=4)
        _add_status(exp_dir, "t3", "RUNNING")
        result = classify(exp_dir)
        assert result.category is ScanCategory.unfinished
        assert "QUEUED/RUNNING" in result.reasons[0]

    def test_missing_status_files_is_unfinished(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "partial"
        _populate_clean_run(exp_dir, n_tasks=5, n_success=3)
        # Bump n_tasks to 10 — 5 tasks have no status file at all.
        _set_subset_field(exp_dir, n_tasks=10)
        result = classify(exp_dir)
        assert result.category is ScanCategory.unfinished
        assert "have no status file" in result.reasons[0]


class TestClassifierLiveStragglerStaysUnfinished:
    """A run with a *live* in-flight episode stays unfinished even if heavily
    stale — the per-episode heartbeat is the liveness signal, and in ray mode a
    worker can outlive a crashed driver. (We deliberately do NOT auto-archive on
    stale fraction.) Once the sweep promotes the straggler, err-rate marks it
    broken; see test_high_system_error_rate_is_broken."""

    def test_stale_heavy_run_with_in_flight_is_unfinished(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "running_straggler"
        _populate_clean_run(exp_dir, n_tasks=4, n_success=4)
        _set_subset_field(exp_dir, n_tasks=4)
        # 3 STALE + 1 RUNNING → 75% stale, but the RUNNING episode is still live.
        _add_status(exp_dir, "t0", "STALE")
        _add_status(exp_dir, "t1", "STALE")
        _add_status(exp_dir, "t2", "STALE")
        _add_status(exp_dir, "t3", "RUNNING")
        result = classify(exp_dir, sweep_stale=False)
        assert result.category is ScanCategory.unfinished

    def test_stale_heavy_run_no_in_flight_is_broken(self, tmp_path: Path) -> None:
        # Same run once the straggler is swept/finished: no in-flight, stale
        # dominates → broken (archivable) via the system-error rate.
        exp_dir = tmp_path / "abandoned"
        _populate_clean_run(exp_dir, n_tasks=4, n_success=4)
        _set_subset_field(exp_dir, n_tasks=4)
        _add_status(exp_dir, "t0", "STALE")
        _add_status(exp_dir, "t1", "STALE")
        _add_status(exp_dir, "t2", "STALE")  # 3/4 = 75% stale, nothing in-flight
        result = classify(exp_dir, sweep_stale=False)
        assert result.category is ScanCategory.broken


class TestClassifierSubsetReview:
    def test_debug_limit_flagged(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "debug"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_record_field(exp_dir, debug_limit=3)
        result = classify(exp_dir)
        assert result.category is ScanCategory.subset_review
        assert any("debug_limit" in r for r in result.reasons)

    def test_explicit_task_list_flagged(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "handpicked"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_subset_field(exp_dir, task_ids=["t0", "t1", "t2"])
        result = classify(exp_dir)
        assert result.category is ScanCategory.subset_review
        assert any("hand-picked" in r for r in result.reasons)

    def test_is_official_false_blocks_clean_full_run(self, tmp_path: Path) -> None:
        # A clean full run that would be submittable is held back when marked debug.
        exp_dir = tmp_path / "marked_debug"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_record_field(exp_dir, is_official=False)
        result = classify(exp_dir)
        assert result.category is ScanCategory.subset_review
        assert any("is_official=False" in r for r in result.reasons)

    def test_is_official_true_promotes_hand_picked_subset(self, tmp_path: Path) -> None:
        # is_official=True overrides the subset-shape gate: a hand-picked list becomes
        # submittable (the operator vouches it's an official eval).
        exp_dir = tmp_path / "vouched"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_subset_field(exp_dir, task_ids=["t0", "t1", "t2"])
        _set_record_field(exp_dir, is_official=True)
        result = classify(exp_dir)
        assert result.category is ScanCategory.submittable

    def test_is_official_true_does_not_override_unfinished(self, tmp_path: Path) -> None:
        # Intent never overrides integrity: a run still missing tasks stays unfinished.
        exp_dir = tmp_path / "vouched_incomplete"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_subset_field(exp_dir, n_tasks=5)  # 2 tasks have no status file
        _set_record_field(exp_dir, is_official=True)
        result = classify(exp_dir)
        assert result.category is ScanCategory.unfinished

    def test_is_official_true_does_not_override_unfinished_with_debug_limit(self, tmp_path: Path) -> None:
        # The integrity gate fires before the is_official override even when a debug_limit
        # truncated the run: a partial run stays unfinished, never submittable.
        exp_dir = tmp_path / "vouched_truncated"
        _populate_clean_run(exp_dir, n_tasks=1, n_success=1)
        _set_subset_field(exp_dir, n_tasks=3)  # 2 tasks never ran
        _set_record_field(exp_dir, debug_limit=1, is_official=True)
        result = classify(exp_dir)
        assert result.category is ScanCategory.unfinished

    def test_is_official_none_falls_back_to_inference(self, tmp_path: Path) -> None:
        # Default None → existing inference: clean full run is submittable.
        exp_dir = tmp_path / "inferred"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_record_field(exp_dir, is_official=None)
        result = classify(exp_dir)
        assert result.category is ScanCategory.submittable


class TestClassifierIdempotency:
    def test_already_submitted_takes_priority(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "submitted"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        submissions.record_submitted(
            exp_dir,
            "journal",
            evaluation_id="alice/run1",
            schema_version="1.0",
        )
        result = classify(exp_dir)
        assert result.category is ScanCategory.already_submitted
        assert "alice/run1" in result.reasons[0]

    def test_rejected_also_blocks(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "rejected"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        submissions.record_rejected(exp_dir, "journal", reason="broken: test")
        result = classify(exp_dir)
        assert result.category is ScanCategory.already_submitted


class TestWalk:
    def test_walk_skips_non_experiment_dirs(self, tmp_path: Path) -> None:
        # Real experiment dir
        _populate_clean_run(tmp_path / "real")
        # Random non-experiment dir
        (tmp_path / "junk").mkdir()
        (tmp_path / "junk" / "readme.txt").write_text("hi")
        # Stray file at root — must be ignored
        (tmp_path / "loose_file.json").write_text("{}")

        results = walk(tmp_path)
        assert {r.experiment_dir.name for r in results} == {"real"}

    def test_walk_returns_empty_for_missing_root(self, tmp_path: Path) -> None:
        assert walk(tmp_path / "does_not_exist") == []


class TestSweepIntegration:
    def _write_dead_worker_episode(self, exp_dir: Path, task_id: str = "t3") -> str:
        """Plant a RUNNING episode whose heartbeat is way past the sweep threshold.

        FileStorage.list_episode_statuses only surfaces episode dirs that
        carry an episode_config.json (planned-or-run marker), so the stub
        below mirrors what Experiment.prepare_episodes writes at startup.
        """
        traj_id = f"{task_id}_ep0"
        ep_dir = exp_dir / "episodes" / traj_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "episode_config.json").write_text("{}")
        EpisodeStatus(
            status="RUNNING",
            task_id=task_id,
            episode_id=0,
            started_at=0.0,
            last_heartbeat_at=0.0,  # ancient — well past step_timeout
        ).write(ep_dir / "status.json")
        return traj_id

    def test_sweep_converts_dead_running_to_stale(self, tmp_path: Path) -> None:
        """A RUNNING episode with a stale heartbeat must roll into n_system_error."""
        from cube_harness.reproducibility.scan import sweep_stale_in_dir

        exp_dir = tmp_path / "sweep"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_subset_field(exp_dir, n_tasks=4)
        traj_id = self._write_dead_worker_episode(exp_dir)

        swept = sweep_stale_in_dir(exp_dir)
        assert traj_id in swept, f"sweep should have caught the dead worker: {swept}"

    def test_classify_with_sweep_disabled_keeps_in_flight_signal(self, tmp_path: Path) -> None:
        """--no-sweep-stale leaves the on-disk status as-is."""
        exp_dir = tmp_path / "no_sweep"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_subset_field(exp_dir, n_tasks=4)
        _add_status(exp_dir, "t3", "RUNNING")  # stays RUNNING when sweep skipped
        result = classify(exp_dir, sweep_stale=False)
        assert result.category is ScanCategory.unfinished

    def test_classify_with_sweep_promotes_dead_to_broken(self, tmp_path: Path) -> None:
        """With sweep on, a dead RUNNING episode becomes STALE and (when its rate
        crosses the threshold) the experiment goes BROKEN."""
        exp_dir = tmp_path / "swept"
        _populate_clean_run(exp_dir, n_tasks=3, n_success=3)
        _set_subset_field(exp_dir, n_tasks=4)
        self._write_dead_worker_episode(exp_dir)
        # With 1 swept STALE among 4 terminal = 25% system error → > 10% threshold.
        result = classify(exp_dir, sweep_stale=True, system_error_threshold=0.10)
        assert result.category is ScanCategory.broken
        assert result.n_system_error >= 1


class TestArchive:
    def test_archive_renames_with_marker(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility.scan import archive_experiment_dir
        from cube_harness.storage import ARCHIVED_MARKER

        exp_dir = tmp_path / "to_archive"
        exp_dir.mkdir()
        new_path = archive_experiment_dir(exp_dir)
        assert ARCHIVED_MARKER in new_path.name
        assert new_path.exists()
        assert not exp_dir.exists()

    def test_archive_is_idempotent_on_already_archived(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility.scan import archive_experiment_dir
        from cube_harness.storage import ARCHIVED_MARKER

        archived = tmp_path / f"already{ARCHIVED_MARKER}1234.5"
        archived.mkdir()
        new_path = archive_experiment_dir(archived)
        # No double-archive: returned path is the same.
        assert new_path == archived

    def test_walk_skips_archived_dirs(self, tmp_path: Path) -> None:
        from cube_harness.storage import ARCHIVED_MARKER

        _populate_clean_run(tmp_path / "live")
        _populate_clean_run(tmp_path / f"old{ARCHIVED_MARKER}9999.0")
        results = walk(tmp_path)
        names = {r.experiment_dir.name for r in results}
        assert names == {"live"}, f"archived dir should be skipped: {names}"
