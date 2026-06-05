"""Tests for cube_harness.analyze.xray_utils module."""

import json
import os
import time
from pathlib import Path

import pytest

from cube_harness.analyze import xray_utils
from cube_harness.core import Trajectory
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus

# ---------------------------------------------------------------------------
# Additional fixtures (complement conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_agent_trajectories() -> list[Trajectory]:
    """Multiple trajectories with different agents and tasks."""
    trajs = []
    ep = 0
    for agent in ["agent_a", "agent_b"]:
        for task in ["task_1", "task_2"]:
            for seed in range(2):
                traj = Trajectory(
                    id=f"{task}_ep{ep}",
                    metadata={"agent_name": agent, "task_id": task, "seed": seed},
                    start_time=float(seed),
                    end_time=float(seed + 1),
                    reward_info={"reward": 1.0 if seed == 0 else 0.0},
                )
                trajs.append(traj)
                ep += 1
    return trajs


# ---------------------------------------------------------------------------
# TestFormatDuration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_milliseconds(self) -> None:
        result = xray_utils.format_duration(0.5)
        assert result == "500ms"

    def test_seconds(self) -> None:
        result = xray_utils.format_duration(4.2)
        assert result == "4.2s"

    def test_minutes_and_seconds(self) -> None:
        result = xray_utils.format_duration(3 * 60 + 12)
        assert result == "3m 12s"

    def test_hours_and_minutes(self) -> None:
        result = xray_utils.format_duration(3600 + 5 * 60)
        assert result == "1h 5m"

    def test_boundary_at_one_second(self) -> None:
        # Exactly 1 second goes to the seconds branch, not ms
        result = xray_utils.format_duration(1.0)
        assert result == "1.0s"

    def test_just_below_one_second(self) -> None:
        result = xray_utils.format_duration(0.999)
        assert result == "999ms"


class TestGetExperimentsTableRows:
    def test_flat_layout_status_cell_shows_total(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "flat_exp"
        exp_dir.mkdir()
        (exp_dir / "a.metadata.json").write_text("{}")
        (exp_dir / "b.metadata.json").write_text("{}")
        rows = xray_utils.get_experiments_table_rows(tmp_path)
        flat = next(r for r in rows if r["experiment"] == "flat_exp")
        assert "status" in flat
        # flat layout has no episodes/ dir — falls back to "? = N" format
        assert "2" in flat["status"]

    def test_legacy_trajectories_subdir_has_status(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "legacy_exp"
        traj_dir = exp_dir / "trajectories"
        traj_dir.mkdir(parents=True)
        (traj_dir / "x.metadata.json").write_text("{}")
        rows = xray_utils.get_experiments_table_rows(tmp_path)
        leg = next(r for r in rows if r["experiment"] == "legacy_exp")
        assert "status" in leg

    def test_agent_populated_from_episode_metadata(self, tmp_path: Path) -> None:
        ep_dir = tmp_path / "exp_a" / "episodes" / "task_1_ep0"
        ep_dir.mkdir(parents=True)
        (ep_dir / "episode.metadata.json").write_text('{"metadata": {"agent_name": "react_agent"}}')
        rows = xray_utils.get_experiments_table_rows(tmp_path)
        row = next(r for r in rows if r["experiment"] == "exp_a")
        assert row["agent"] == "react_agent"

    def test_agent_uses_agent_name_property_not_class(self, tmp_path: Path) -> None:
        """`AgentConfig.agent_name` is a @property — it isn't in the JSON dump.

        The experiments table must still surface it (e.g. "ReactAgent-gpt-4o") rather
        than the class short name ("ReactAgentConfig"). Same fix powers the agent tab
        identifier so multi-experiment loads stay distinct.
        """
        from cube_harness.agents.react import ReactAgentConfig
        from cube_harness.llm import LLMConfig

        cfg = ReactAgentConfig(llm_config=LLMConfig(model_name="gpt-4o"))
        exp_dir = tmp_path / "exp_react"
        (exp_dir / "episodes").mkdir(parents=True)
        (exp_dir / "experiment_config.json").write_text(
            json.dumps({"agent_config": json.loads(cfg.model_dump_json(serialize_as_any=True))})
        )
        rows = xray_utils.get_experiments_table_rows(tmp_path)
        row = next(r for r in rows if r["experiment"] == "exp_react")
        assert row["agent"] == "ReactAgent-gpt-4o"

    def test_status_cell_from_status_json(self, tmp_path: Path) -> None:
        now = time.time()
        status_data = [
            ("ep0", {"status": "COMPLETED", "task_id": "t0", "episode_id": 0, "started_at": now, "ended_at": now}),
            (
                "ep1",
                {"status": "RUNNING", "task_id": "t1", "episode_id": 1, "started_at": now, "last_heartbeat_at": now},
            ),
        ]
        for ep_name, data in status_data:
            ep_dir = tmp_path / "exp_b" / "episodes" / ep_name
            ep_dir.mkdir(parents=True)
            (ep_dir / "status.json").write_text(json.dumps(data))
        rows = xray_utils.get_experiments_table_rows(tmp_path)
        row = next(r for r in rows if r["experiment"] == "exp_b")
        assert "✅" in row["status"]
        assert "▶️" in row["status"]
        assert "/ 2" in row["status"]

    def test_cache_written_when_all_terminal(self, tmp_path: Path) -> None:
        now = time.time()
        ep_dir = tmp_path / "exp_c" / "episodes" / "ep0"
        ep_dir.mkdir(parents=True)
        (ep_dir / "status.json").write_text(
            json.dumps({"status": "COMPLETED", "task_id": "t0", "episode_id": 0, "started_at": now, "ended_at": now})
        )
        xray_utils.get_experiments_table_rows(tmp_path)
        cache = tmp_path / "exp_c" / xray_utils._XRAY_CACHE_FILENAME
        assert cache.exists()

    def test_cache_not_written_when_running(self, tmp_path: Path) -> None:
        now = time.time()
        ep_dir = tmp_path / "exp_d" / "episodes" / "ep0"
        ep_dir.mkdir(parents=True)
        (ep_dir / "status.json").write_text(
            json.dumps(
                {"status": "RUNNING", "task_id": "t0", "episode_id": 0, "started_at": now, "last_heartbeat_at": now}
            )
        )
        xray_utils.get_experiments_table_rows(tmp_path)
        cache = tmp_path / "exp_d" / xray_utils._XRAY_CACHE_FILENAME
        assert not cache.exists()

    def test_cache_invalidated_on_episode_dir_mtime_change(self, tmp_path: Path) -> None:
        now = time.time()
        ep_dir = tmp_path / "exp_e" / "episodes" / "ep0"
        ep_dir.mkdir(parents=True)
        (ep_dir / "status.json").write_text(
            json.dumps({"status": "COMPLETED", "task_id": "t0", "episode_id": 0, "started_at": now, "ended_at": now})
        )
        xray_utils.get_experiments_table_rows(tmp_path)
        cache = tmp_path / "exp_e" / xray_utils._XRAY_CACHE_FILENAME
        assert cache.exists()
        # Touch an episode dir to simulate a new write
        future = now + 100
        os.utime(ep_dir, (future, future))
        assert not xray_utils._is_cache_valid(tmp_path / "exp_e", cache.stat().st_mtime)

    def test_cache_invalidated_when_submission_recorded_then_cleared(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        now = time.time()
        exp = tmp_path / "exp_subs"
        ep_dir = exp / "episodes" / "ep0"
        ep_dir.mkdir(parents=True)
        (ep_dir / "status.json").write_text(
            json.dumps({"status": "COMPLETED", "task_id": "t0", "episode_id": 0, "started_at": now, "ended_at": now})
        )
        # No experiment_record.json → classifies broken; that's fine, we only care
        # that _category tracks the submission state across cache reads.
        cat = lambda: xray_utils.get_experiments_table_rows(tmp_path)[0]["_category"]  # noqa: E731
        assert cat() == "broken"
        # Record a submission: a journal decision short-circuits classify.
        submissions.record_submitted(exp, "journal", evaluation_id="a", schema_version="1.0")
        assert cat() == "already_submitted"  # cache must NOT serve the stale 'broken'
        # Roll back: deleting submissions.json must invalidate the cache too
        # (the bug — an mtime-vs-cache check misses deletion).
        (exp / submissions.SUBMISSIONS_FILENAME).unlink()
        assert cat() == "broken"

    def test_ghost_episode_promoted_to_stale(self, tmp_path: Path) -> None:
        old_ts = time.time() - xray_utils.GHOST_TIMEOUT - 100
        ep_dir = tmp_path / "exp_f" / "episodes" / "ep0"
        ep_dir.mkdir(parents=True)
        (ep_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "task_id": "t0",
                    "episode_id": 0,
                    "started_at": old_ts,
                    "last_heartbeat_at": old_ts,
                }
            )
        )
        xray_utils._promote_ghost_episodes(tmp_path / "exp_f")
        updated = EpisodeStatus.read(ep_dir / STATUS_FILENAME)
        assert updated is not None
        assert updated.status == "STALE"


# ---------------------------------------------------------------------------
# TestTrajectoryStatusLegacy — heuristic fallback (no _episode_status in metadata)
# ---------------------------------------------------------------------------


class TestTrajectoryStatusLegacy:
    """Tests for _infer_status_legacy(), exercised through trajectory_status() when
    no _episode_status key is present (pre-PR#315 experiments without status.json)."""

    def test_missing_no_failure_is_queued(self) -> None:
        stub = Trajectory(id="t", metadata={"_missing": True})
        assert xray_utils.trajectory_status(stub) == "queued"

    def test_missing_with_failure_is_system_error(self) -> None:
        stub = Trajectory(id="t", metadata={"_missing": True, "_failure_text": "Traceback..."})
        assert xray_utils.trajectory_status(stub) == "system_error"

    def test_running_no_failure_is_running(self) -> None:
        traj = Trajectory(id="t", start_time=1.0)
        assert xray_utils.trajectory_status(traj) == "running"

    def test_running_with_failure_is_system_error(self) -> None:
        """A real trajectory that started but has failure.txt injected is system_error."""
        traj = Trajectory(id="t", start_time=1.0, metadata={"_failure_text": "Ray actor died"})
        assert xray_utils.trajectory_status(traj) == "system_error"

    def test_completed_with_reward_is_success(self) -> None:
        traj = Trajectory(id="t", start_time=1.0, end_time=2.0, reward_info={"reward": 1.0})
        assert xray_utils.trajectory_status(traj) == "success"

    def test_completed_no_reward_is_fail(self) -> None:
        traj = Trajectory(id="t", start_time=1.0, end_time=2.0, reward_info={"reward": 0.0})
        assert xray_utils.trajectory_status(traj) == "fail"

    def test_completed_no_reward_info_is_fail(self) -> None:
        traj = Trajectory(id="t", start_time=1.0, end_time=2.0)
        assert xray_utils.trajectory_status(traj) == "fail"

    def test_failure_text_ignored_when_end_time_set(self) -> None:
        """A completed trajectory with a stale failure.txt should not be system_error."""
        traj = Trajectory(
            id="t", start_time=1.0, end_time=2.0, reward_info={"reward": 1.0}, metadata={"_failure_text": "old error"}
        )
        assert xray_utils.trajectory_status(traj) == "success"


# ---------------------------------------------------------------------------
# TestTrajectoryStatusFromEpisodeStatus — canonical path (status.json present)
# ---------------------------------------------------------------------------


class TestTrajectoryStatusFromEpisodeStatus:
    """trajectory_status() reads _episode_status injected by FileStorage from status.json."""

    def test_queued(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "QUEUED"})
        assert xray_utils.trajectory_status(traj) == "queued"

    def test_running(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "RUNNING"})
        assert xray_utils.trajectory_status(traj) == "running"

    def test_completed_with_reward_is_success(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "COMPLETED"}, reward_info={"reward": 1.0})
        assert xray_utils.trajectory_status(traj) == "success"

    def test_completed_no_reward_is_fail(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "COMPLETED"}, reward_info={"reward": 0.0})
        assert xray_utils.trajectory_status(traj) == "fail"

    def test_completed_no_reward_info_is_fail(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "COMPLETED"})
        assert xray_utils.trajectory_status(traj) == "fail"

    def test_max_steps_reached(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "MAX_STEPS_REACHED"})
        assert xray_utils.trajectory_status(traj) == "max_steps"

    def test_failed(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "FAILED"})
        assert xray_utils.trajectory_status(traj) == "failed"

    def test_stale(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "STALE"})
        assert xray_utils.trajectory_status(traj) == "stale"

    def test_cancelled(self) -> None:
        traj = Trajectory(id="t", metadata={"_episode_status": "CANCELLED"})
        assert xray_utils.trajectory_status(traj) == "cancelled"

    def test_episode_status_takes_priority_over_heuristic(self) -> None:
        """_episode_status wins even when heuristics would say something different."""
        traj = Trajectory(
            id="t",
            metadata={"_episode_status": "STALE", "_failure_text": "crash"},
            start_time=1.0,
            end_time=2.0,
            reward_info={"reward": 1.0},
        )
        assert xray_utils.trajectory_status(traj) == "stale"

    def test_unknown_status_falls_back_to_legacy(self) -> None:
        """An unrecognised status string doesn't crash — falls back to legacy heuristic."""
        traj = Trajectory(
            id="t",
            metadata={"_episode_status": "FUTURE_UNKNOWN_STATUS"},
            start_time=1.0,
            end_time=2.0,
            reward_info={"reward": 1.0},
        )
        assert xray_utils.trajectory_status(traj) == "success"


# ---------------------------------------------------------------------------
# TestComputeTrajectoryStats
# ---------------------------------------------------------------------------


def _traj_with_summary_stats() -> Trajectory:
    """A terminal trajectory carrying summary_stats (as EventStreamer persists it)."""
    return Trajectory(
        id="summarised",
        metadata={"task_id": "task_1", "agent_name": "agent_a", "_episode_status": "COMPLETED"},
        reward_info={"reward": 1.0},
        summary_stats={
            "n_env_steps": 2,
            "n_agent_steps": 1,
            "total_actions": 1,
            "total_llm_calls": 1,
            "duration": 5.0,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "cost": 0.001,
            "final_reward": 1.0,
        },
    )


class TestComputeTrajectoryStats:
    def test_returns_summary_stats_verbatim(self) -> None:
        traj = _traj_with_summary_stats()
        assert xray_utils.compute_trajectory_stats(traj) == traj.summary_stats

    def test_zeroed_dict_when_no_summary_stats(self) -> None:
        stats = xray_utils.compute_trajectory_stats(Trajectory(id="stub"))
        assert stats["n_env_steps"] == 0
        assert stats["total_actions"] == 0
        assert stats["final_reward"] == 0.0
        assert stats["cost"] == 0.0
        assert stats["duration"] is None

    def test_empty_summary_stats_falls_back_to_zeroed(self) -> None:
        traj = Trajectory(id="stub", summary_stats={})
        assert xray_utils.compute_trajectory_stats(traj)["n_env_steps"] == 0

    def test_token_and_step_counts_from_summary_stats(self) -> None:
        stats = xray_utils.compute_trajectory_stats(_traj_with_summary_stats())
        assert stats["n_env_steps"] == 2
        assert stats["n_agent_steps"] == 1
        assert stats["prompt_tokens"] == 100
        assert stats["completion_tokens"] == 20
        assert stats["total_llm_calls"] == 1
        assert stats["duration"] == pytest.approx(5.0)
        assert stats["final_reward"] == 1.0


# ---------------------------------------------------------------------------
# TestComputeExperimentStats
# ---------------------------------------------------------------------------


class TestComputeExperimentStats:
    def test_empty_list_returns_empty_string(self) -> None:
        assert xray_utils.compute_experiment_stats([]) == ""

    def test_single_finished_trajectory(self) -> None:
        result = xray_utils.compute_experiment_stats([_traj_with_summary_stats()])
        assert "1" in result
        assert "completed" in result.lower()

    def test_counts_system_error_trajectories(self) -> None:
        # A stub with _missing=True and _failure_text is a system_error → counted as errored
        crashed_stub = Trajectory(id="crash", metadata={"_missing": True, "_failure_text": "Traceback: ..."})
        result = xray_utils.compute_experiment_stats([crashed_stub])
        assert "Failed" in result

    def test_counts_running_trajectories(self) -> None:
        # A trajectory with start_time but no end_time and no error steps is "running"
        running_traj = Trajectory(id="running", start_time=1.0)
        result = xray_utils.compute_experiment_stats([running_traj])
        assert "Running" in result
        assert "Failed" not in result

    def test_computes_success_rate(self) -> None:
        result = xray_utils.compute_experiment_stats([_traj_with_summary_stats()])
        assert "Success Rate" in result

    def test_shows_token_totals(self) -> None:
        result = xray_utils.compute_experiment_stats([_traj_with_summary_stats()])
        assert "prompt" in result


class TestRewardMeanStderr:
    def test_empty_returns_zeros(self) -> None:
        assert xray_utils._reward_mean_stderr([]) == (0.0, 0.0)

    def test_binary_uses_binomial_formula(self) -> None:
        # 3 successes / 4 trials → p=0.75, binomial stderr = sqrt(p*(1-p)/n).
        # _reward_mean_stderr now delegates to analyze.stats.reward_mean_stderr,
        # which auto-selects binomial SE for binary data (same as scripts/experiments_report.py).
        rewards = [1.0, 1.0, 1.0, 0.0]
        mean, stderr = xray_utils._reward_mean_stderr(rewards)
        n = len(rewards)
        assert mean == pytest.approx(0.75)
        assert stderr == pytest.approx((mean * (1 - mean) / n) ** 0.5)

    def test_continuous_uses_sample_formula(self) -> None:
        rewards = [0.2, 0.4, 0.6, 0.8]
        mean, stderr = xray_utils._reward_mean_stderr(rewards)
        n = len(rewards)
        expected_var = sum((r - mean) ** 2 for r in rewards) / (n - 1)
        assert mean == pytest.approx(0.5)
        assert stderr == pytest.approx((expected_var / n) ** 0.5)

    def test_single_value_returns_zero_stderr(self) -> None:
        assert xray_utils._reward_mean_stderr([0.5]) == (0.5, 0.0)


# ---------------------------------------------------------------------------
# TestBuildAgentTable
# ---------------------------------------------------------------------------


class TestBuildAgentTable:
    def test_empty_list(self) -> None:
        assert xray_utils.build_agent_table([]) == []

    def test_single_agent(self) -> None:
        traj = Trajectory(id="t", metadata={"agent_name": "agent_a", "task_id": "task_1"})
        rows = xray_utils.build_agent_table([traj])
        assert len(rows) == 1
        assert rows[0]["agent_name"] == "agent_a"

    def test_groups_by_agent_name(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        names = [r["agent_name"] for r in rows]
        assert "agent_a" in names
        assert "agent_b" in names
        assert len(rows) == 2

    def test_unknown_fallback_for_missing_agent_name(self) -> None:
        traj = Trajectory(id="no_agent", metadata={"task_id": "t1"})
        rows = xray_utils.build_agent_table([traj])
        assert rows[0]["agent_name"] == "unknown"

    def test_no_n_trajs_column(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        assert "n_trajs" not in rows[0]

    def test_avg_reward_before_status(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        keys = list(rows[0].keys())
        assert keys.index("avg_reward") < keys.index("status")

    def test_avg_reward_includes_stderr(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """avg_reward cell is formatted as 'mean ± stderr' with 3 decimals each."""
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        for row in rows:
            assert "±" in row["avg_reward"]
            mean_part, stderr_part = row["avg_reward"].split(" ± ")
            assert len(mean_part.split(".")[1]) == 3
            assert len(stderr_part.split(".")[1]) == 3

    def test_has_status_column_not_n_err_n_running(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """n_err and n_running replaced by the unified status cell."""
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        assert len(rows) > 0
        assert "status" in rows[0]
        assert "n_err" not in rows[0]
        assert "n_running" not in rows[0]

    def test_status_cell_contains_total(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """Status cell shows '/ N' total trajectory count."""
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        agent_a_row = next(r for r in rows if r["agent_name"] == "agent_a")
        assert "/ 4" in agent_a_row["status"]

    def test_status_cell_collapses_success_and_fail(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """Success and fail both show as ✓ in the agent table (avg_reward has the breakdown)."""
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        agent_a_row = next(r for r in rows if r["agent_name"] == "agent_a")
        # fixture has 2 success + 2 fail per agent — collapsed to one ✓ count
        assert "✅" in agent_a_row["status"]
        assert "🟢" not in agent_a_row["status"]
        assert "⚫" not in agent_a_row["status"]

    def test_status_cell_shows_error_symbol_for_crashed_traj(self) -> None:
        """⛔ appears in the status cell when a trajectory has FAILED status."""
        traj = Trajectory(id="t", metadata={"agent_name": "agent_a", "_episode_status": "FAILED"})
        rows = xray_utils.build_agent_table([traj])
        assert "⛔" in rows[0]["status"]

    def test_total_cost_dash_for_unloaded_stubs(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """total_cost shows '-' when no cost data is available (metadata stubs have steps=[])."""
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        agent_a_row = next(r for r in rows if r["agent_name"] == "agent_a")
        assert agent_a_row["total_cost"] == "-"

    def test_no_success_rate_column(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """success_rate was removed from the agent table."""
        rows = xray_utils.build_agent_table(multi_agent_trajectories)
        assert len(rows) > 0
        assert "success_rate" not in rows[0]


# ---------------------------------------------------------------------------
# TestBuildTrajectoryTable
# ---------------------------------------------------------------------------


class TestBuildTrajectoryTable:
    def test_filters_by_agent_key(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_trajectory_table(multi_agent_trajectories, "agent_a")
        # agent_a has task_1 (×2) and task_2 (×2) = 4 trajectories
        assert len(rows) == 4
        task_ids = [r["task_id"] for r in rows]
        assert "task_1" in task_ids
        assert "task_2" in task_ids

    def test_one_row_per_trajectory(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """No aggregation — every trajectory gets its own row."""
        rows = xray_utils.build_trajectory_table(multi_agent_trajectories, "agent_b")
        assert len(rows) == 4

    def test_returns_empty_for_unknown_agent(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_trajectory_table(multi_agent_trajectories, "nonexistent_agent")
        assert rows == []

    def test_has_task_id_and_seed_columns(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_trajectory_table(multi_agent_trajectories, "agent_a")
        assert len(rows) > 0
        assert "task_id" in rows[0]
        assert "seed" in rows[0]
        assert "_traj_id" in rows[0]
        assert "status" in rows[0]

    def test_seed_column_omitted_when_all_none(self) -> None:
        trajs = [Trajectory(id=f"task_1_ep{i}", metadata={"agent_name": "a", "task_id": "task_1"}) for i in range(3)]
        rows = xray_utils.build_trajectory_table(trajs, "a")
        assert "seed" not in rows[0]

    def test_traj_id_values_match_trajectory_ids(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_trajectory_table(multi_agent_trajectories, "agent_a")
        traj_ids = [r["_traj_id"] for r in rows]
        assert "task_1_ep0" in traj_ids
        assert "task_1_ep1" in traj_ids

    def test_no_aggregation_columns(self, multi_agent_trajectories: list[Trajectory]) -> None:
        """Removed aggregate columns: n_seeds, n_success, avg_steps, etc."""
        rows = xray_utils.build_trajectory_table(multi_agent_trajectories, "agent_a")
        assert "n_seeds" not in rows[0]
        assert "n_success" not in rows[0]
        assert "avg_steps" not in rows[0]

    def test_sorted_by_task_id_then_start_time(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_trajectory_table(multi_agent_trajectories, "agent_a")
        task_ids = [r["task_id"] for r in rows]
        # task_1 rows come before task_2 (lexicographic sort)
        last_task_1 = max(i for i, t in enumerate(task_ids) if t == "task_1")
        first_task_2 = min(i for i, t in enumerate(task_ids) if t == "task_2")
        assert last_task_1 < first_task_2

    def test_no_reward_column(self, multi_agent_trajectories: list[Trajectory]) -> None:
        rows = xray_utils.build_trajectory_table(multi_agent_trajectories, "agent_a")
        assert "reward" not in rows[0]

    def test_duration_shows_dash_when_no_timing(self) -> None:
        traj = Trajectory(id="t1", metadata={"agent_name": "agent_a", "task_id": "task_x"})
        rows = xray_utils.build_trajectory_table([traj], "agent_a")
        assert rows[0]["duration"] == "-"

    def test_retry_badge_shown_when_retry_count_gt_0(self) -> None:
        traj = Trajectory(
            id="t1_ep0",
            metadata={"agent_name": "a", "task_id": "t1", "_episode_status": "COMPLETED", "_retry_count": 2},
            start_time=0.0,
            end_time=1.0,
            reward_info={"reward": 1.0},
        )
        rows = xray_utils.build_trajectory_table([traj], "a")
        assert "×2" in rows[0]["status"]

    def test_no_retry_badge_when_retry_count_is_0(self) -> None:
        traj = Trajectory(
            id="t1_ep0",
            metadata={"agent_name": "a", "task_id": "t1", "_episode_status": "COMPLETED", "_retry_count": 0},
            start_time=0.0,
            end_time=1.0,
            reward_info={"reward": 1.0},
        )
        rows = xray_utils.build_trajectory_table([traj], "a")
        assert "×" not in rows[0]["status"]


# ---------------------------------------------------------------------------
# TestBuildStatusCell
# ---------------------------------------------------------------------------


class TestBuildStatusCell:
    def test_all_completed_shows_check_and_total(self) -> None:
        cell = xray_utils._build_status_cell(["success", "success", "fail"])
        assert "✅" in cell
        assert "/ 3" in cell

    def test_mixed_statuses_shows_each_symbol(self) -> None:
        cell = xray_utils._build_status_cell(["success", "running", "failed", "stale"])
        assert "▶️" in cell
        assert "⛔" in cell
        assert "👻" in cell
        assert "/ 4" in cell

    def test_success_and_fail_collapse_to_one_count(self) -> None:
        cell = xray_utils._build_status_cell(["success", "success", "fail"])
        # Should show "3✓" not "2✓ + 1⚫"
        assert "3" in cell
        assert "⚫" not in cell

    def test_zero_counts_omitted(self) -> None:
        cell = xray_utils._build_status_cell(["success"])
        assert "▶️" not in cell
        assert "⛔" not in cell

    def test_max_steps_folds_into_completed(self) -> None:
        # max_steps is a terminal outcome — collapses to ✓ at agent level like success/fail
        cell = xray_utils._build_status_cell(["max_steps"])
        assert "✅" in cell
        assert "🎬" not in cell

    def test_cancelled_symbol(self) -> None:
        cell = xray_utils._build_status_cell(["cancelled"])
        assert "⏹️" in cell


# ---------------------------------------------------------------------------
# TestBuildTaskTableStatusPriority
# ---------------------------------------------------------------------------


class TestBuildTrajectoryTableStatusIcons:
    """Each trajectory row shows its own status icon (no aggregation)."""

    def _make_traj(self, agent: str, task: str, traj_id: str, status: str) -> Trajectory:
        return Trajectory(id=traj_id, metadata={"agent_name": agent, "task_id": task, "_episode_status": status})

    def test_failed_row_shows_failed_icon(self) -> None:
        traj = self._make_traj("a", "t1", "t1_ep0", "FAILED")
        rows = xray_utils.build_trajectory_table([traj], "a")
        assert "⛔" in rows[0]["status"]

    def test_stale_row_shows_stale_icon(self) -> None:
        traj = self._make_traj("a", "t1", "t1_ep0", "STALE")
        rows = xray_utils.build_trajectory_table([traj], "a")
        assert "👻" in rows[0]["status"]

    def test_max_steps_row_shows_max_steps_icon(self) -> None:
        traj = self._make_traj("a", "t1", "t1_ep0", "MAX_STEPS_REACHED")
        rows = xray_utils.build_trajectory_table([traj], "a")
        assert "🎬" in rows[0]["status"]

    def test_success_row_shows_green_icon(self) -> None:
        traj = Trajectory(
            id="t1_ep0",
            metadata={"agent_name": "a", "task_id": "t1", "_episode_status": "COMPLETED"},
            reward_info={"reward": 1.0},
        )
        rows = xray_utils.build_trajectory_table([traj], "a")
        assert "🟢" in rows[0]["status"]


# ---------------------------------------------------------------------------
# TestGetLogsTabMarkdownEpisodeStatus
# ---------------------------------------------------------------------------


class TestGetLogsTabMarkdownEpisodeStatus:
    def test_shows_retry_count_when_gt_0(self) -> None:
        traj = Trajectory(id="t", metadata={"_retry_count": 3})
        result = xray_utils.get_logs_tab_markdown(traj, "")
        assert "Attempt" in result
        assert "3" in result

    def test_shows_error_type_and_message(self) -> None:
        traj = Trajectory(
            id="t",
            metadata={"_error_type": "RuntimeError", "_error_message": "OOM on GPU"},
        )
        result = xray_utils.get_logs_tab_markdown(traj, "")
        assert "RuntimeError" in result
        assert "OOM on GPU" in result

    def test_no_episode_status_section_when_all_absent(self) -> None:
        traj = Trajectory(id="t", metadata={})
        result = xray_utils.get_logs_tab_markdown(traj, "")
        assert "Episode Status" not in result


class TestBuildProgressHtml:
    def test_progress_label_and_bar_width(self) -> None:
        html = xray_utils.build_progress_html(3, 4, 1)
        assert "3/4 episodes completed" in html
        assert "1 running" in html
        assert "width:75.0%" in html

    def test_no_ray_link_when_url_absent(self) -> None:
        html = xray_utils.build_progress_html(1, 2, 0)
        assert "Ray dashboard" not in html

    def test_ray_dashboard_link_is_clickable(self) -> None:
        html = xray_utils.build_progress_html(1, 2, 1, ray_dashboard_urls=[("exp_a", "http://127.0.0.1:8265")])
        assert '<a href="http://127.0.0.1:8265"' in html
        assert 'target="_blank"' in html
        assert "🔗 Ray dashboard" in html

    def test_ray_dashboard_link_prefixes_name_for_multiple_experiments(self) -> None:
        html = xray_utils.build_progress_html(
            0,
            2,
            0,
            exp_names=["exp_a", "exp_b"],
            ray_dashboard_urls=[("exp_a", "http://a:8265"), ("exp_b", "http://b:8265")],
        )
        assert "exp_a: " in html
        assert "exp_b: " in html
        assert html.count("🔗 Ray dashboard") == 2

    def test_ray_dashboard_url_is_escaped(self) -> None:
        html = xray_utils.build_progress_html(
            0, 1, 0, ray_dashboard_urls=[("e", 'http://x"><script>alert(1)</script>')]
        )
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


class TestPickDirectory:
    """The native folder-picker wrapper (mocked — no real dialog)."""

    def test_returns_chosen_dir_on_macos(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        target = tmp_path / "results"
        target.mkdir()
        monkeypatch.setattr(xray_utils.sys, "platform", "darwin")

        class _R:
            returncode = 0
            stdout = f"{target}\n"

        monkeypatch.setattr(xray_utils.subprocess, "run", lambda *a, **k: _R())
        assert xray_utils.pick_directory(tmp_path) == target

    def test_returns_none_on_cancel(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.setattr(xray_utils.sys, "platform", "darwin")

        class _R:
            returncode = 1  # user cancelled
            stdout = ""

        monkeypatch.setattr(xray_utils.subprocess, "run", lambda *a, **k: _R())
        assert xray_utils.pick_directory(tmp_path) is None

    def test_returns_none_when_choice_is_not_a_dir(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.setattr(xray_utils.sys, "platform", "darwin")

        class _R:
            returncode = 0
            stdout = f"{tmp_path / 'nope'}\n"

        monkeypatch.setattr(xray_utils.subprocess, "run", lambda *a, **k: _R())
        assert xray_utils.pick_directory(tmp_path) is None


class TestEligibility:
    """Submission-eligibility badge logic (clean + submit)."""

    def test_scan_category_missing_dir_is_broken(self, tmp_path: Path) -> None:
        # No experiment_record.json → classify returns broken; helper never raises.
        assert xray_utils.scan_category(tmp_path / "nope") == "broken"

    def test_badge_uses_category_when_no_submission(self, tmp_path: Path) -> None:
        badge = xray_utils.eligibility_badge(tmp_path, "submittable")
        assert "submittable" in badge
        assert xray_utils.eligibility_badge(tmp_path, "broken").count("broken")

    def test_submitted_state_overrides_category(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_submitted(
            tmp_path, "journal", evaluation_id="me/exp", schema_version="1.0", pr_url="http://x"
        )
        badge = xray_utils.eligibility_badge(tmp_path, "broken")  # category ignored once submitted
        assert "✅" in badge and "registry" in badge

    def test_eee_and_registry_both_submitted(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_submitted(tmp_path, "journal", evaluation_id="a", schema_version="1.0")
        submissions.record_submitted(tmp_path, "eee", evaluation_id="b", schema_version="0.2")
        badge = xray_utils.eligibility_badge(tmp_path, "submittable")
        assert "registry" in badge and "eee" in badge


def test_rejected_state_shows_rejected_badge(tmp_path: Path) -> None:
    from cube_harness.reproducibility import submissions  # noqa: PLC0415

    submissions.record_rejected(tmp_path, "journal", reason="broken: 58/279 episodes errored")
    badge = xray_utils.eligibility_badge(tmp_path, "already_submitted")
    assert "🚫 rejected" in badge and "58/279" in badge


class TestIsArchivable:
    """Archive auto-select: broken + rejected + explicit-debug (is_official=False),
    but not submittable / submitted / a bare subset_review."""

    def test_broken_is_archivable(self, tmp_path: Path) -> None:
        assert xray_utils.is_archivable(tmp_path, "broken")

    def test_explicit_debug_is_archivable(self, tmp_path: Path) -> None:
        # is_official=False ⇒ operator marked it debug ⇒ archivable, even though
        # the category is subset_review.
        assert xray_utils.is_archivable(tmp_path, "subset_review", is_official=False)

    def test_submittable_and_bare_subset_review_are_not(self, tmp_path: Path) -> None:
        assert not xray_utils.is_archivable(tmp_path, "submittable")
        assert not xray_utils.is_archivable(tmp_path, "subset_review")  # is_official None ⇒ keep
        assert not xray_utils.is_archivable(tmp_path, "subset_review", is_official=True)

    def test_rejected_run_is_archivable(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_rejected(tmp_path, "journal", reason="broken: all episodes ghost")
        # category is already_submitted (a journal decision exists) — but it's a rejection.
        assert xray_utils.is_archivable(tmp_path, "already_submitted")

    def test_official_is_an_absolute_keep(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        # is_official=True pins the run — never archived, even broken or rejected.
        assert not xray_utils.is_archivable(tmp_path, "broken", is_official=True)
        submissions.record_rejected(tmp_path, "journal", reason="broken: errors")
        assert not xray_utils.is_archivable(tmp_path, "already_submitted", is_official=True)

    def test_successfully_submitted_run_is_not_archivable(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_submitted(tmp_path, "journal", evaluation_id="a", schema_version="1.0")
        assert not xray_utils.is_archivable(tmp_path, "already_submitted")


class TestIsSubmittablePick:
    """Submit auto-select: submittable AND not already submitted / mid-submission."""

    def test_clean_submittable_is_picked(self, tmp_path: Path) -> None:
        assert xray_utils.is_submittable_pick(tmp_path, "submittable")

    def test_non_submittable_category_is_not(self, tmp_path: Path) -> None:
        assert not xray_utils.is_submittable_pick(tmp_path, "broken")
        assert not xray_utils.is_submittable_pick(tmp_path, "subset_review")

    def test_already_submitted_is_not_re_picked(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_submitted(tmp_path, "journal", evaluation_id="a", schema_version="1.0")
        # Even if the cached category still says submittable, a submitted run is skipped.
        assert not xray_utils.is_submittable_pick(tmp_path, "submittable")

    def test_pending_is_not_re_picked(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_pending(tmp_path, "journal")
        assert not xray_utils.is_submittable_pick(tmp_path, "submittable")

    def test_failed_is_still_picked_for_retry(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_failed(tmp_path, "journal", reason="transient")
        assert xray_utils.is_submittable_pick(tmp_path, "submittable")


class TestSubmissionBadges:
    def test_pending_badge(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_pending(tmp_path, "journal")
        assert "submitting" in xray_utils.eligibility_badge(tmp_path, "submittable")

    def test_failed_badge_shows_reason(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        submissions.record_failed(tmp_path, "journal", reason="push timed out")
        badge = xray_utils.eligibility_badge(tmp_path, "submittable")
        assert "submit failed" in badge and "push timed out" in badge


class TestPersistBrokenRejection:
    def test_broken_dir_gets_durable_rejection(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        exp = tmp_path / "broken_run"
        exp.mkdir()  # no experiment_record.json → classifies broken
        assert xray_utils.persist_broken_rejection(exp) is True
        assert submissions.read(exp)["journal"]["status"] == "rejected"
        # Idempotent: a second call sees the prior decision and does nothing new.
        assert xray_utils.persist_broken_rejection(exp) is False

    def test_decided_dir_is_left_alone(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility import submissions  # noqa: PLC0415

        exp = tmp_path / "submitted_run"
        exp.mkdir()
        submissions.record_submitted(exp, "journal", evaluation_id="a", schema_version="1.0")
        assert xray_utils.persist_broken_rejection(exp) is False
        assert submissions.read(exp)["journal"]["status"] == "submitted"
