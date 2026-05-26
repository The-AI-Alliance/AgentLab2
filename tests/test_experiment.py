"""Tests for cube_harness.experiment module."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cube.benchmark import BenchmarkConfig as CubeBenchmarkConfig
from cube.benchmark import BenchmarkMetadata
from cube.core import EnvironmentOutput, Observation
from cube.task import TaskMetadata

from cube_harness.core import Trajectory, TrajectoryStep
from cube_harness.episode import Episode
from cube_harness.episode_status import EpisodeStatus
from cube_harness.exp_runner import _kill_stale_workers, run_sequentially
from cube_harness.experiment import _UUID_SUFFIX_RE, Experiment, ExpResult, sweep_stale_statuses
from cube_harness.storage import FileStorage
from tests.conftest import (
    MockAgentConfig,
    MockCubeBenchmark,
    MockCubeBenchmarkConfig,
    MockCubeTask,
    MockCubeTaskConfig,
    MockToolConfig,
)


def _make_failing_benchmark() -> CubeBenchmarkConfig:
    """BenchmarkConfig whose single task raises on the first step() call."""

    class _FailingTask(MockCubeTask):
        def step(self, actions):
            raise RuntimeError("injected step failure")

    class _FailingTaskConfig(MockCubeTaskConfig):
        def make(self, runtime_context=None) -> _FailingTask:
            return _FailingTask(
                metadata=self.metadata,
                tool_config=self.tool_config or MockToolConfig(),
            )

    class _FailingBenchmarkConfig(MockCubeBenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="failing", version="0.1.0", description="test")
        task_metadata = {"fail_task_0": TaskMetadata(id="fail_task_0")}
        task_config_class = _FailingTaskConfig
        benchmark_class = MockCubeBenchmark

    return _FailingBenchmarkConfig()


def _make_neverending_benchmark(max_steps: int) -> CubeBenchmarkConfig:
    """BenchmarkConfig whose single task never sets done=True, triggering MAX_STEPS_REACHED."""

    class _NeverDoneTask(MockCubeTask):
        def step(self, actions):
            return EnvironmentOutput(obs=Observation.from_text("still going"), reward=0.0, done=False)

    class _NeverDoneTaskConfig(MockCubeTaskConfig):
        def make(self, runtime_context=None) -> _NeverDoneTask:
            return _NeverDoneTask(
                metadata=self.metadata,
                tool_config=self.tool_config or MockToolConfig(),
            )

    class _NeverDoneBenchmarkConfig(MockCubeBenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="neverending", version="0.1.0", description="test")
        task_metadata = {"never_task_0": TaskMetadata(id="never_task_0")}
        task_config_class = _NeverDoneTaskConfig
        benchmark_class = MockCubeBenchmark

    return _NeverDoneBenchmarkConfig()


def _make_benchmark(n: int) -> CubeBenchmarkConfig:
    """Create a cube BenchmarkConfig with n tasks for testing.

    Inherits ``benchmark_metadata``/``task_config_class``/``benchmark_class``
    from ``MockCubeBenchmarkConfig`` so all calls share the same cache dir
    and the back-stamp on ``MockCubeTaskConfig._benchmark_cache_dir`` stays
    consistent — cube-standard rejects re-stamping with a different value.
    """
    task_meta = {f"task_{i}": TaskMetadata(id=f"task_{i}") for i in range(n)}

    class _NTaskBenchmarkConfig(MockCubeBenchmarkConfig):
        task_metadata = task_meta

    return _NTaskBenchmarkConfig()


class TestExpResult:
    """Tests for ExpResult class."""

    def test_exp_result_creation(self):
        """Test ExpResult creation."""
        result = ExpResult(exp_id="test_exp_123", tasks_num=10)

        assert result.exp_id == "test_exp_123"
        assert result.tasks_num == 10
        assert result.config == {}
        assert result.trajectories == {}
        assert result.failures == {}

    def test_exp_result_with_trajectories(self):
        """Test ExpResult with trajectories."""
        obs = Observation.from_text("done")
        step = TrajectoryStep(output=EnvironmentOutput(obs=obs, reward=1.0, done=True))
        traj = Trajectory(id="test_traj", metadata={"task_id": "task_1"}, steps=[step])

        result = ExpResult(exp_id="test_exp", tasks_num=1, trajectories={"task_1": traj})

        assert len(result.trajectories) == 1
        assert "task_1" in result.trajectories

    def test_exp_result_with_failures(self):
        """Test ExpResult with failures."""
        result = ExpResult(
            exp_id="test_exp",
            tasks_num=3,
            failures={"task_2": "Connection error", "task_3": "Timeout"},
        )

        assert len(result.failures) == 2
        assert result.failures["task_2"] == "Connection error"

    def test_exp_result_with_config(self):
        """Test ExpResult with config."""
        config = {"model": "gpt-4", "temperature": 0.7}
        result = ExpResult(exp_id="test_exp", tasks_num=5, config=config)

        assert result.config["model"] == "gpt-4"
        assert result.config["temperature"] == 0.7

    def test_exp_result_serialization(self):
        """Test ExpResult JSON serialization."""
        result = ExpResult(exp_id="test_exp", tasks_num=5, config={"key": "value"})
        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["exp_id"] == "test_exp"
        assert data["tasks_num"] == 5
        assert data["config"]["key"] == "value"


class TestExperiment:
    """Tests for Experiment class."""

    def test_experiment_creation(self, tmp_dir, mock_agent_config, mock_cube_benchmark_config):
        """Test Experiment creation."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        assert exp.name == "test_experiment"
        assert exp.output_dir == tmp_dir
        assert exp.agent_config == mock_agent_config
        assert exp.benchmark_config == mock_cube_benchmark_config

    def test_experiment_auto_output_dir(self, mock_agent_config, mock_cube_benchmark_config):
        """output_dir is auto-generated when omitted."""
        exp = Experiment(
            name="my_run",
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        assert exp.output_dir is not None
        assert exp.output_dir.exists()
        assert _UUID_SUFFIX_RE.search(exp.output_dir.name)
        assert "mock-cube" in exp.output_dir.name
        assert mock_agent_config.agent_name in exp.output_dir.name

    def test_experiment_config_property(self, tmp_dir, mock_agent_config, mock_cube_benchmark_config):
        """Test Experiment config property."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        config = exp.config
        assert config["name"] == "test_experiment"
        assert config["output_dir"] == tmp_dir
        assert "agent_config" in config
        assert "benchmark_config" in config

    def test_experiment_create_episodes(self, tmp_dir, mock_agent_config, mock_cube_benchmark_config):
        """Test Experiment creates one episode per task in the benchmark."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        with mock_cube_benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)

        assert len(episodes) == len(mock_cube_benchmark_config.task_metadata)
        for i, episode in enumerate(episodes):
            assert isinstance(episode, Episode)
            assert episode.config.id == i
            assert episode.config.output_dir == tmp_dir
            assert episode.config.task_config is not None

    def test_experiment_create_episodes_multiple_tasks(self, tmp_dir, mock_agent_config):
        """Test Experiment create_episodes with multiple tasks."""
        benchmark_config = _make_benchmark(5)

        exp = Experiment(
            name="multi_task_exp",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )

        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        assert len(episodes) == 5
        task_ids = {e.config.task_config.task_id for e in episodes}
        assert task_ids == {f"task_{i}" for i in range(5)}

    def test_experiment_save_config(self, tmp_dir, mock_agent_config, mock_cube_benchmark_config):
        """Test Experiment save_config."""

        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        exp.save_config()

        config_path = tmp_dir / "experiment_config.json"
        assert config_path.exists()

        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["name"] == "test_experiment"

    def test_experiment_save_config_creates_directory(self, tmp_dir, mock_agent_config, mock_cube_benchmark_config):
        """Test Experiment save_config creates output directory."""

        nested_dir = tmp_dir / "nested" / "output"
        exp = Experiment(
            name="test_experiment",
            output_dir=nested_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        exp.save_config()

        assert exp.output_dir.exists()
        assert (exp.output_dir / "experiment_config.json").exists()

    def test_experiment_serialization(self, tmp_dir, mock_agent_config, mock_cube_benchmark_config):
        """Test Experiment JSON serialization."""

        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        json_str = exp.model_dump_json(serialize_as_any=True)
        data = json.loads(json_str)

        assert data["name"] == "test_experiment"
        assert "agent_config" in data
        assert "benchmark_config" in data

    def test_experiment_episodes_have_tasks_from_benchmark(
        self, tmp_dir, mock_agent_config, mock_cube_benchmark_config
    ):
        """Test that created episodes have tasks from benchmark."""
        exp = Experiment(
            name="test_experiment",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        with mock_cube_benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        expected_task_ids = set(mock_cube_benchmark_config.task_metadata.keys())
        actual_task_ids = {e.config.task_config.task_id for e in episodes}
        assert actual_task_ids == expected_task_ids

    def test_retry_failed_episodes(self, tmp_dir, mock_agent_config):
        """resume=True returns FAILED + missing-status episodes (gated by max_retries)."""
        benchmark_config = _make_benchmark(3)

        exp = Experiment(
            name="test_retry_failed",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )

        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)

            # Episode 0: complete successfully → COMPLETED status
            episodes[0].run()

            # Episode 1: write a FAILED status manually
            storage = FileStorage(tmp_dir)
            traj_id_1 = f"{episodes[1].config.task_config.task_id}_ep{episodes[1].config.id}"
            storage.write_episode_status(
                traj_id_1,
                EpisodeStatus(
                    status="FAILED",
                    task_id=episodes[1].config.task_config.task_id,
                    episode_id=episodes[1].config.id,
                    started_at=time.time() - 10,
                    ended_at=time.time(),
                    last_heartbeat_at=time.time(),
                    error_type="RuntimeError",
                    error_message="boom",
                    retry_count=0,
                ),
            )

            # Episode 2: no status.json (never started)

            exp.resume = True
            failed_episodes = exp.get_episodes_to_run(benchmark)
            # Both episode 1 (FAILED) and episode 2 (missing status) qualify
            ids = {ep.config.id for ep in failed_episodes}
            assert ids == {episodes[1].config.id, episodes[2].config.id}

    def test_retry_respects_max_retries(self, tmp_dir, mock_agent_config):
        """retry_count >= max_retries excludes the episode from retry."""
        benchmark_config = _make_benchmark(2)
        exp = Experiment(
            name="test_max_retries",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
            max_retries=2,
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
            storage = FileStorage(tmp_dir)
            # Episode 0: capped (retry_count == max_retries)
            storage.write_episode_status(
                f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}",
                EpisodeStatus(
                    status="FAILED",
                    task_id=episodes[0].config.task_config.task_id,
                    episode_id=episodes[0].config.id,
                    started_at=time.time(),
                    retry_count=2,
                ),
            )
            # Episode 1: still under cap
            storage.write_episode_status(
                f"{episodes[1].config.task_config.task_id}_ep{episodes[1].config.id}",
                EpisodeStatus(
                    status="FAILED",
                    task_id=episodes[1].config.task_config.task_id,
                    episode_id=episodes[1].config.id,
                    started_at=time.time(),
                    retry_count=1,
                ),
            )
            exp.resume = True
            retried = exp.get_episodes_to_run(benchmark)
            ids = {ep.config.id for ep in retried}
            assert ids == {episodes[1].config.id}

    def test_resume_returns_unstarted(self, tmp_dir, mock_agent_config):
        """Test resume=True returns only unstarted episodes."""
        benchmark_config = _make_benchmark(3)

        exp = Experiment(
            name="test_resume_unstarted",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )

        with benchmark_config.make() as benchmark:
            # First call creates all episodes and saves configs
            episodes = exp.get_episodes_to_run(benchmark)
            assert len(episodes) == 3

            # Run only first episode (leaving 2 and 3 unstarted)
            episodes[0].run()

            # With resume=True, should return only unstarted episodes
            exp.resume = True
            resumed_episodes = exp.get_episodes_to_run(benchmark)
            assert len(resumed_episodes) == 2

    def test_run_sequentially(self, tmp_dir, mock_agent_config, mock_cube_benchmark_config):
        """run_sequentially completes all episodes and returns trajectories keyed by trajectory_id."""
        exp = Experiment(
            name="test_run_sequential",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        result = run_sequentially(exp)

        with mock_cube_benchmark_config.make() as benchmark:
            expected_trajectory_ids = {
                f"{ep.config.task_config.task_id}_ep{ep.config.id}" for ep in exp.get_episodes_to_run(benchmark)
            }
        assert set(result.trajectories.keys()) == expected_trajectory_ids
        assert result.failures == {}

    def test_resume_and_retry_empty_when_all_succeeded(self, tmp_dir, mock_agent_config, mock_cube_benchmark_config):
        """resume returns empty when all episodes succeeded."""
        exp = Experiment(
            name="test_no_relaunch",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=mock_cube_benchmark_config,
        )

        # Create episodes and run all successfully
        with mock_cube_benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
            for episode in episodes:
                episode.run()

            exp.resume = True
            exp.resume = True
            assert exp.get_episodes_to_run(benchmark) == []


class TestStatusBasedSelection:
    """Focused tests for the status-driven selection logic."""

    # ------------------------------------------------------------------
    # Parametrized sweep: direct sweep_stale_statuses behaviour
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "initial_status,started_age,heartbeat_age,step_timeout,cancel_grace,orphan_threshold,expect_swept",
        [
            # RUNNING with stale heartbeat → swept
            ("RUNNING", 3600, 3600, 1.0, 1.0, 10.0, True),
            # RUNNING with fresh heartbeat → not swept
            ("RUNNING", 0, 0, 60.0, 60.0, 3600.0, False),
            # QUEUED older than orphan_threshold → swept
            ("QUEUED", 7200, None, 1.0, 1.0, 10.0, True),
            # QUEUED younger than orphan_threshold → not swept
            ("QUEUED", 0, None, 60.0, 60.0, 3600.0, False),
        ],
        ids=["running-stale", "running-fresh", "queued-orphan", "queued-fresh"],
    )
    def test_sweep_stale_statuses(
        self,
        tmp_dir: "Path",
        mock_agent_config: "MockAgentConfig",
        initial_status: str,
        started_age: float,
        heartbeat_age: float | None,
        step_timeout: float,
        cancel_grace: float,
        orphan_threshold: float,
        expect_swept: bool,
    ) -> None:
        """sweep_stale_statuses marks stale in-flight entries STALE and leaves fresh ones alone."""
        benchmark_config = _make_benchmark(1)
        exp = Experiment(
            name="test_sweep", output_dir=tmp_dir, agent_config=mock_agent_config, benchmark_config=benchmark_config
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)
        now = time.time()
        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status=initial_status,
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=now - started_age,
                last_heartbeat_at=None if heartbeat_age is None else now - heartbeat_age,
                current_step=0,
            ),
        )

        swept = sweep_stale_statuses(
            storage,
            step_timeout_s=step_timeout,
            cancel_grace_s=cancel_grace,
            orphan_threshold_s=orphan_threshold,
        )

        if expect_swept:
            assert swept == [traj_id]
            assert storage.read_episode_status(traj_id).status == "STALE"
        else:
            assert swept == []
            assert storage.read_episode_status(traj_id).status == initial_status

    # ------------------------------------------------------------------
    # Parametrized table: resume=True selection by status
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "status,extra_fields,expect_returned",
        [
            # retriable — resume must include
            ("FAILED", {}, True),
            ("CANCELLED", {}, True),
            ("STALE", {}, True),
            # terminal non-retriable — resume must skip
            ("COMPLETED", {"reward": 1.0, "ended_at": 0}, False),
            ("MAX_STEPS_REACHED", {"reward": 0.0, "ended_at": 0}, False),
            # in-flight — resume must skip (not swept; fresh timestamps)
            ("QUEUED", {}, False),
            ("RUNNING", {"last_heartbeat_at": 0}, False),
        ],
        ids=["failed", "cancelled", "stale", "completed", "max-steps-reached", "queued", "running"],
    )
    def test_resume_selection_by_status(
        self,
        tmp_dir: "Path",
        mock_agent_config: "MockAgentConfig",
        status: str,
        extra_fields: dict,
        expect_returned: bool,
    ) -> None:
        """resume=True returns retriable statuses and skips terminal/in-flight ones."""
        benchmark_config = _make_benchmark(1)
        exp = Experiment(
            name="test_sel", output_dir=tmp_dir, agent_config=mock_agent_config, benchmark_config=benchmark_config
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)

        now = time.time()
        resolved = {k: (now if v == 0 else v) for k, v in extra_fields.items()}
        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status=status,
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=now,
                retry_count=0,
                **resolved,
            ),
        )

        exp.resume = True
        # Use generous timeouts so fresh QUEUED/RUNNING are not swept to STALE.
        with benchmark_config.make() as benchmark:
            result = exp.get_episodes_to_run(
                benchmark, step_timeout_s=3600.0, cancel_grace_s=3600.0, orphan_threshold_s=3600.0
            )

        if expect_returned:
            assert len(result) == 1 and result[0].config.id == episodes[0].config.id
        else:
            assert result == []

    def test_sequential_pre_claims_only_episodes_that_will_run(self, tmp_dir, mock_agent_config) -> None:
        """Sequential mode pre-claims only the episodes that will actually run.

        With debug_limit=1, only 1 of 3 episodes should be pre-claimed. The other 2
        must have no status.json — they were never submitted, so a resume run can pick
        them up without waiting for orphan_threshold_s to expire.
        """
        benchmark_config = _make_benchmark(3)
        exp = Experiment(
            name="test_seq_preclaim",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )
        storage = FileStorage(tmp_dir)

        run_sequentially(exp, debug_limit=1)

        # Only the 1 episode that ran should have a status.json.
        all_statuses = storage.list_episode_statuses()
        assert len(all_statuses) == 1, (
            f"Expected only 1 episode pre-claimed (debug_limit=1), got {len(all_statuses)}: {list(all_statuses)}"
        )

        # That episode should be COMPLETED.
        completed = [tid for tid, s in all_statuses.items() if s.status == "COMPLETED"]
        assert len(completed) == 1

    def test_debug_limit_orphans_do_not_block_resume(self, tmp_dir, mock_agent_config) -> None:
        """Episodes beyond debug_limit must not appear as QUEUED orphans that block resumption.

        Before the fix, all episodes were pre-claimed as QUEUED before debug_limit slicing.
        QUEUED is not in RETRIABLE_STATUSES, so those orphans would be silently skipped on
        a resume=True run until orphan_threshold_s elapsed.
        """
        benchmark_config = _make_benchmark(3)
        exp = Experiment(
            name="test_debug_orphan",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )
        storage = FileStorage(tmp_dir)

        run_sequentially(exp, debug_limit=1)

        # Confirm no QUEUED orphans exist for the un-run episodes.
        all_statuses = storage.list_episode_statuses()
        queued = [tid for tid, s in all_statuses.items() if s.status == "QUEUED"]
        assert queued == [], f"Found orphaned QUEUED episodes that would block resume: {queued}"

        # resume=True should find 2 episodes to run (the ones that weren't debug-limited).
        exp.resume = True
        with benchmark_config.make() as benchmark:
            episodes_to_resume = exp.get_episodes_to_run(benchmark)
        assert len(episodes_to_resume) == 2, f"Expected 2 episodes available for resume, got {len(episodes_to_resume)}"

    def test_heartbeat_advances_current_step_and_timestamp(self, tmp_dir, mock_agent_config):
        """RUNNING → RUNNING: each turn updates last_heartbeat_at and current_step without changing status."""
        benchmark_config = _make_benchmark(1)
        exp = Experiment(
            name="test_heartbeat",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)

        t0 = time.time() - 10
        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="RUNNING",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=t0,
                last_heartbeat_at=t0,
                current_step=0,
                retry_count=0,
            ),
        )

        # Simulate two heartbeat writes (what the loop does at the start of each turn).
        for turn in range(1, 3):
            s = storage.read_episode_status(traj_id)
            s.last_heartbeat_at = time.time()
            s.current_step = turn
            storage.write_episode_status(traj_id, s)

        final = storage.read_episode_status(traj_id)
        assert final.status == "RUNNING"  # status unchanged
        assert final.current_step == 2  # advanced through two turns
        assert final.last_heartbeat_at > t0  # heartbeat is newer than start

    def test_cancelled_at_max_retries_cap_is_terminal(self, tmp_dir, mock_agent_config):
        """CANCELLED → terminal: retry_count >= max_retries excludes from retry selection."""
        benchmark_config = _make_benchmark(1)
        exp = Experiment(
            name="test_cancelled_cap",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
            max_retries=2,
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        storage = FileStorage(tmp_dir)
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"

        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="CANCELLED",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time(),
                ended_at=time.time(),
                error_type="StepTimeout",
                error_message="Step 3 exceeded 1800s",
                retry_count=2,  # == max_retries → capped
            ),
        )

        exp.resume = True
        with benchmark_config.make() as benchmark:
            assert exp.get_episodes_to_run(benchmark) == []

    def test_stale_at_max_retries_cap_is_terminal(self, tmp_dir, mock_agent_config):
        """STALE → terminal: retry_count >= max_retries excludes from retry selection."""
        benchmark_config = _make_benchmark(1)
        exp = Experiment(
            name="test_stale_cap",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
            max_retries=2,
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        storage = FileStorage(tmp_dir)
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"

        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="STALE",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time(),
                ended_at=time.time(),
                error_type="WorkerDied",
                retry_count=2,  # == max_retries → capped
            ),
        )

        exp.resume = True
        with benchmark_config.make() as benchmark:
            assert exp.get_episodes_to_run(benchmark) == []

    def test_resume_auto_sweeps_stale_running_and_returns_for_retry(self, tmp_dir, mock_agent_config) -> None:
        """resume=True sweeps a stale RUNNING entry to STALE and returns it for retry in one call."""
        benchmark_config = _make_benchmark(1)
        exp = Experiment(
            name="test_resume_sweep",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)

        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="RUNNING",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time() - 3600,
                last_heartbeat_at=time.time() - 3600,
                current_step=3,
                retry_count=0,
            ),
        )

        exp.resume = True
        with benchmark_config.make() as benchmark:
            retried = exp.get_episodes_to_run(
                benchmark, step_timeout_s=1.0, cancel_grace_s=1.0, orphan_threshold_s=10.0
            )

        # The sweep ran implicitly — status is now STALE.
        assert storage.read_episode_status(traj_id).status == "STALE"
        # And the episode was returned for retry.
        assert len(retried) == 1
        assert retried[0].config.id == episodes[0].config.id

    def test_resume_auto_sweeps_orphaned_queued_and_returns_for_retry(self, tmp_dir, mock_agent_config) -> None:
        """resume=True sweeps an orphaned QUEUED entry to STALE and returns it for retry in one call."""
        benchmark_config = _make_benchmark(1)
        exp = Experiment(
            name="test_resume_sweep_queued",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        storage = FileStorage(tmp_dir)

        storage.write_episode_status(
            traj_id,
            EpisodeStatus(
                status="QUEUED",
                task_id=episodes[0].config.task_config.task_id,
                episode_id=episodes[0].config.id,
                started_at=time.time() - 7200,
                last_heartbeat_at=None,
                current_step=0,
                retry_count=0,
            ),
        )

        exp.resume = True
        with benchmark_config.make() as benchmark:
            retried = exp.get_episodes_to_run(
                benchmark, step_timeout_s=1.0, cancel_grace_s=1.0, orphan_threshold_s=10.0
            )

        assert storage.read_episode_status(traj_id).status == "STALE"
        assert len(retried) == 1
        assert retried[0].config.id == episodes[0].config.id

    def test_worker_writes_failed_on_unhandled_exception(self, tmp_dir, mock_agent_config) -> None:
        """RUNNING → FAILED: unhandled exception in step() causes the worker to write FAILED with error fields."""
        benchmark_config = _make_failing_benchmark()
        exp = Experiment(
            name="test_failed_status",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        storage = FileStorage(tmp_dir)

        with pytest.raises(RuntimeError, match="injected step failure"):
            episodes[0].run()

        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        status = storage.read_episode_status(traj_id)
        assert status is not None
        assert status.status == "FAILED"
        assert status.error_type == "RuntimeError"
        assert "injected step failure" in (status.error_message or "")
        assert status.ended_at is not None

    def test_worker_writes_max_steps_reached_when_loop_exhausted(self, tmp_dir, mock_agent_config) -> None:
        """RUNNING → MAX_STEPS_REACHED: loop exhausts max_steps without done=True."""
        max_steps = 2
        benchmark_config = _make_neverending_benchmark(max_steps)
        exp = Experiment(
            name="test_max_steps_status",
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            benchmark_config=benchmark_config,
            max_steps=max_steps,
        )
        with benchmark_config.make() as benchmark:
            episodes = exp.get_episodes_to_run(benchmark)
        storage = FileStorage(tmp_dir)

        episodes[0].run()

        traj_id = f"{episodes[0].config.task_config.task_id}_ep{episodes[0].config.id}"
        status = storage.read_episode_status(traj_id)
        assert status is not None
        assert status.status == "MAX_STEPS_REACHED"
        assert status.ended_at is not None
        # Not retriable — resume must skip it.
        exp.resume = True
        with benchmark_config.make() as benchmark:
            assert exp.get_episodes_to_run(benchmark) == []


class TestKillStaleWorkersRaceGuard:
    """Unit tests for the _kill_stale_workers race guard.

    The guard prevents the driver from stamping CANCELLED after the worker has
    already written a terminal status (COMPLETED or FAILED) between the staleness
    check and the ray.cancel call.
    """

    def _stale_running_status(self, task_id: str, episode_id: int) -> EpisodeStatus:
        return EpisodeStatus(
            status="RUNNING",
            task_id=task_id,
            episode_id=episode_id,
            started_at=time.time() - 7200,
            last_heartbeat_at=time.time() - 7200,  # ancient — triggers kill
            current_step=5,
            retry_count=0,
        )

    def test_driver_does_not_clobber_completed_written_by_worker(self, tmp_dir) -> None:
        """RUNNING (stale) → worker races to COMPLETED → driver skips CANCELLED."""
        storage = FileStorage(tmp_dir)
        traj_id = "task_race_ep0"
        task_id = "task_race"

        stale_status = self._stale_running_status(task_id, 0)
        completed_status = EpisodeStatus(
            status="COMPLETED",
            task_id=task_id,
            episode_id=0,
            started_at=time.time() - 7200,
            ended_at=time.time(),
            last_heartbeat_at=time.time(),
            reward=1.0,
            retry_count=0,
        )

        fake_ref = MagicMock()
        ref_to_traj_id = {fake_ref: traj_id}
        results = ExpResult(exp_id="test", tasks_num=1)
        episodes_in_progress = [fake_ref]

        # First read: stale RUNNING. Second read (after ray.cancel): COMPLETED.
        with (
            patch("cube_harness.exp_runner.ray.cancel") as mock_cancel,
            patch.object(storage, "read_episode_status", side_effect=[stale_status, completed_status]),
            patch.object(storage, "write_episode_status") as mock_write,
        ):
            _kill_stale_workers(
                episodes_in_progress,
                ref_to_traj_id,
                storage,
                results,
                step_timeout_s=1.0,
                setup_timeout_s=1.0,
                cancel_grace_s=1.0,
            )

        # ray.cancel was still called (driver didn't know worker had finished).
        mock_cancel.assert_called_once_with(fake_ref, force=True)
        # write_episode_status was NOT called — COMPLETED is preserved, not clobbered.
        mock_write.assert_not_called()
        # Ref removed from in-progress (driver moves on regardless).
        assert fake_ref not in episodes_in_progress
        # CANCELLED was NOT recorded as a failure.
        assert traj_id not in results.failures

    def test_driver_stamps_cancelled_when_still_running_after_cancel(self, tmp_dir) -> None:
        """RUNNING (stale) → worker does not race → driver writes CANCELLED."""
        storage = FileStorage(tmp_dir)
        traj_id = "task_norace_ep0"
        task_id = "task_norace"

        stale_status = self._stale_running_status(task_id, 0)

        fake_ref = MagicMock()
        ref_to_traj_id = {fake_ref: traj_id}
        results = ExpResult(exp_id="test", tasks_num=1)
        episodes_in_progress = [fake_ref]

        # Both reads return RUNNING — worker never wrote a terminal status.
        with (
            patch("cube_harness.exp_runner.ray.cancel"),
            patch.object(storage, "read_episode_status", side_effect=[stale_status, stale_status]),
            patch.object(storage, "write_episode_status") as mock_write,
        ):
            _kill_stale_workers(
                episodes_in_progress,
                ref_to_traj_id,
                storage,
                results,
                step_timeout_s=1.0,
                setup_timeout_s=1.0,
                cancel_grace_s=1.0,
            )

        # CANCELLED was written.
        assert mock_write.call_count == 1
        written: EpisodeStatus = mock_write.call_args[0][1]
        assert written.status == "CANCELLED"
        assert written.error_type == "StepTimeout"
        assert "step 5 exceeded" in written.error_message
        # Recorded as a failure.
        assert traj_id in results.failures
        assert fake_ref not in episodes_in_progress
