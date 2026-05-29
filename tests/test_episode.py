"""Tests for cube_harness.episode module."""

import json
from pathlib import Path

import pytest
from cube.core import Action, EnvironmentOutput, Observation
from cube.task import TaskConfig, TaskMetadata

from cube_harness.agent import AgentConfig
from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep
from cube_harness.episode import Episode
from cube_harness.storage import _read_step_file
from tests.conftest import MockAgent, MockAgentConfig, MockCubeTask, MockCubeTaskConfig, MockToolConfig


def _make_test_episode(
    id: int, output_dir: Path, agent_config: AgentConfig, task_config: TaskConfig, max_steps: int = 5
) -> Episode:
    return Episode(
        id=id,
        output_dir=output_dir,
        agent_config=agent_config,
        task_config=task_config,
        exp_name="test-episode",
        max_steps=max_steps,
        runtime_context=None,
        storage=None,
    )


class TestEpisode:
    """Tests for Episode class."""

    def test_episode_creation(self, mock_episode, tmp_dir):
        """Test Episode creation."""
        assert mock_episode.config.id == 0
        assert mock_episode.config.output_dir == tmp_dir
        assert mock_episode.config.max_steps == 5

    def test_episode_custom_max_steps(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Test Episode with custom max_steps."""
        episode = _make_test_episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            max_steps=10,
        )

        assert episode.config.max_steps == 10

    def test_episode_run_completes(self, mock_episode):
        """Test Episode run completes successfully."""
        trajectory = mock_episode.run()

        assert isinstance(trajectory, Trajectory)
        assert "task_id" in trajectory.metadata
        # Steps stream to disk; the returned trajectory carries metadata + summary only.
        loaded = mock_episode.storage.load_trajectory(trajectory.id)
        assert len(loaded.steps) >= 2  # initial env output + agent output + final env output

    def test_episode_run_saves_trajectory(self, mock_episode, tmp_dir):
        """Test Episode run saves trajectory files."""
        mock_episode.run()

        episodes_dir = tmp_dir / "episodes"
        assert episodes_dir.exists()

        ep_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
        assert len(ep_dirs) >= 1
        assert (ep_dirs[0] / "episode.metadata.json").exists()
        assert (ep_dirs[0] / "episode_config.json").exists()
        assert (ep_dirs[0] / "steps").exists()

    def test_episode_run_metadata_file_content(self, mock_episode, tmp_dir):
        """Test Episode run creates correct metadata file."""
        mock_episode.run()

        episodes_dir = tmp_dir / "episodes"
        ep_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
        assert len(ep_dirs) > 0, "No episode directory found"

        with open(ep_dirs[0] / "episode.metadata.json") as f:
            metadata = json.load(f)["metadata"]

        assert "task_id" in metadata

    def test_episode_run_step_files(self, mock_episode, tmp_dir):
        """Test Episode run creates per-step files."""
        mock_episode.run()

        episodes_dir = tmp_dir / "episodes"
        ep_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
        assert len(ep_dirs) > 0, "No episode directory found"

        steps_dir = ep_dirs[0] / "steps"
        step_files = sorted(steps_dir.iterdir())
        assert len(step_files) >= 1

        for step_file in step_files:
            data = _read_step_file(step_file)
            assert isinstance(data, dict)

    def test_episode_run_respects_max_steps(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Test Episode run respects max_steps limit."""

        # Create an agent that never stops
        class NeverStopsAgent(MockAgent):
            def step(self, obs):
                _ = obs
                self.step_count += 1
                # Return non-stop action
                return AgentOutput(actions=[Action(name="click", arguments={"element_id": "btn"})])

        class NeverStopsConfig(type(mock_agent_config)):
            def make(self, *args, **kwargs):
                _ = args, kwargs
                agent = NeverStopsAgent(config=self)
                return agent

        config = NeverStopsConfig()

        episode = _make_test_episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=config,
            task_config=mock_cube_task_config,
            max_steps=3,
        )

        trajectory = episode.run()

        # Should have stopped at max_steps (steps stream to disk; read the streamed count)
        assert trajectory.summary_stats["n_agent_steps"] <= 3

    def test_episode_run_stops_on_done(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Test Episode run stops when done=True."""
        episode = _make_test_episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            max_steps=100,  # High limit
        )

        trajectory = episode.run()

        # Should stop before max_steps because agent returns final_step
        assert trajectory.reward_info["done"] is True

    def test_storage_save_trajectory_creates_directory(self, mock_episode, tmp_dir):
        """Test save_trajectory creates episode directory."""
        trajectory = Trajectory(id="test_traj", metadata={"task_id": "test"})
        mock_episode.storage.save_trajectory(trajectory)

        episodes_dir = tmp_dir / "episodes"
        assert episodes_dir.exists()

    def test_storage_save_step_without_trajectory(self, mock_episode):
        """Test save_step raises error if called before save_trajectory."""
        obs = Observation.from_text("test")
        step = TrajectoryStep(output=EnvironmentOutput(obs=obs))

        with pytest.raises(ValueError, match="Episode directory does not exist"):
            mock_episode.storage.save_step(step, "nonexistent_traj", 0)

    def test_storage_save_step_creates_files(self, mock_episode, tmp_dir):
        """Test save_step creates per-step files."""
        trajectory = Trajectory(id="test_traj", metadata={"task_id": "test"})
        mock_episode.storage.save_trajectory(trajectory)

        for i in range(3):
            obs = Observation.from_text(f"step {i}")
            step = TrajectoryStep(output=EnvironmentOutput(obs=obs))
            mock_episode.storage.save_step(step, trajectory.id, i)

        episodes_dir = tmp_dir / "episodes"
        ep_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
        assert len(ep_dirs) > 0
        steps_dir = ep_dirs[0] / "steps"
        step_files = list(steps_dir.iterdir())
        assert len(step_files) == 3

    def test_episode_closes_env_on_completion(self, tmp_dir, mock_agent_config):
        """Test Episode closes environment after run."""
        close_calls: list[bool] = []

        class TrackCloseTask(MockCubeTask):
            def close(self):
                close_calls.append(True)
                super().close()

        class TrackCloseConfig(MockCubeTaskConfig):
            def make(self, runtime_context=None):
                _ = runtime_context
                return TrackCloseTask(
                    metadata=TaskMetadata(id=self.task_id),
                    tool_config=MockToolConfig(),
                )

        episode = _make_test_episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=TrackCloseConfig(metadata=TaskMetadata(id="track_close_task")),
        )
        episode.run()

        assert close_calls, "task.close() was not called"

    def test_episode_closes_env_on_error(self, tmp_dir):
        """Test Episode closes environment even when error occurs."""
        close_calls: list[bool] = []

        class TrackCloseTask(MockCubeTask):
            def close(self):
                close_calls.append(True)
                super().close()

        class TrackCloseConfig(MockCubeTaskConfig):
            def make(self, runtime_context=None):
                _ = runtime_context
                return TrackCloseTask(
                    metadata=TaskMetadata(id=self.task_id),
                    tool_config=MockToolConfig(),
                )

        class ErrorAgent(MockAgent):
            def step(self, obs):
                _ = obs
                raise RuntimeError("Test error")

        class ErrorConfig(MockAgentConfig):
            def make(self, *args, **kwargs) -> "ErrorAgent":
                _ = args, kwargs
                return ErrorAgent(config=self)

        config = ErrorConfig()

        episode = _make_test_episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=config,
            task_config=TrackCloseConfig(metadata=TaskMetadata(id="track_close_error_task")),
        )

        with pytest.raises(RuntimeError, match="Test error"):
            episode.run()

        assert close_calls, "task.close() was not called on error"

    def test_episode_output_filename(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Test Episode generates correct output directory name."""
        episode = _make_test_episode(
            id=42,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
        )

        episode.run()

        episodes_dir = tmp_dir / "episodes"
        ep_dirs = [d.name for d in episodes_dir.iterdir() if d.is_dir()]
        assert any("_ep42" in d for d in ep_dirs)

    def test_episode_captures_agent_error(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Test Episode captures agent errors correctly in trajectory."""

        class ErrorAgent(MockAgent):
            def step(self, obs):
                _ = obs
                raise RuntimeError("Agent step failed")

        class ErrorConfig(type(mock_agent_config)):
            def make(self, *args, **kwargs) -> "ErrorAgent":
                _ = args, kwargs
                return ErrorAgent(config=self)

        config = ErrorConfig()

        episode = _make_test_episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=config,
            task_config=mock_cube_task_config,
        )

        # Episode should raise the error
        with pytest.raises(RuntimeError, match="Agent step failed"):
            episode.run()

        # But error should be saved in trajectory before raising
        from cube_harness.storage import FileStorage

        storage = FileStorage(tmp_dir)
        traj_id = f"{episode.config.task_config.task_id}_ep{episode.config.id}"
        trajectory = storage.load_trajectory(traj_id)

        # Find the agent output step with error
        agent_steps = [s for s in trajectory.steps if isinstance(s.output, AgentOutput)]
        assert len(agent_steps) > 0, "No agent steps found in trajectory"

        error_step = next((s for s in agent_steps if s.output.error is not None), None)
        assert error_step is not None, "No error found in agent steps"
        assert error_step.output.error is not None
        assert error_step.output.error.error_type == "RuntimeError"
        assert "Agent step failed" in error_step.output.error.exception_str

    def test_episode_captures_env_error(self, tmp_dir, mock_agent_config):
        """Test Episode captures environment errors correctly in trajectory."""

        class ErrorEvalTask(MockCubeTask):
            def evaluate(self, obs=None):
                _ = obs
                raise ValueError("Environment validation failed")

        class ErrorEvalConfig(MockCubeTaskConfig):
            def make(self, runtime_context=None):
                _ = runtime_context
                return ErrorEvalTask(
                    metadata=TaskMetadata(id=self.task_id),
                    tool_config=MockToolConfig(),
                )

        episode = _make_test_episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=ErrorEvalConfig(metadata=TaskMetadata(id="error_eval_task")),
        )

        # Episode should raise the error (evaluate() is called when done=True via final_step)
        with pytest.raises(ValueError, match="Environment validation failed"):
            episode.run()

        # But error should be saved in trajectory before raising
        from cube_harness.storage import FileStorage

        storage = FileStorage(tmp_dir)
        traj_id = f"{episode.config.task_config.task_id}_ep{episode.config.id}"
        trajectory = storage.load_trajectory(traj_id)

        # Find the environment output step with error
        env_steps = [s for s in trajectory.steps if isinstance(s.output, EnvironmentOutput)]
        assert len(env_steps) > 0, "No env steps found in trajectory"

        error_step = next((s for s in env_steps if s.output.error is not None), None)
        assert error_step is not None, "No error found in env steps"
        assert error_step.output.error is not None
        assert error_step.output.error.error_type == "ValueError"
        assert "Environment validation failed" in error_step.output.error.exception_str

    def test_episode_run_raises_on_duplicate_trajectory(
        self, tmp_dir, mock_agent_config, mock_cube_task_config
    ) -> None:
        """Running the same episode twice raises FileExistsError (prevents accidental overwrites)."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="test-episode",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        episode.run()

        # Second run with a fresh Episode (same ID, new storage session)
        episode2 = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="test-episode",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        with pytest.raises(FileExistsError):
            episode2.run()

    def test_episode_relaunch_archives_old_trajectory(self, tmp_dir, mock_agent_config, mock_cube_task_config) -> None:
        """An episode loaded from config (_allow_overwrite=True) archives the old trajectory."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="test-episode",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        episode.run()

        episode2 = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="test-episode",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        episode2.allow_overwrite = True
        episode2.run()

        episodes_dir = tmp_dir / "episodes"
        archived = [d for d in episodes_dir.iterdir() if ".archived_" in d.name]
        assert len(archived) == 1
        current_dirs = [d for d in episodes_dir.iterdir() if d.is_dir() and ".archived_" not in d.name]
        assert len(current_dirs) == 1


# ---------------------------------------------------------------------------
# Episode invocation of Agent.finalize() — multi-episode-rollouts RFC.
# ---------------------------------------------------------------------------


class _FinalizeTrackingAgent(MockAgent):
    """Records every finalize() call so tests can assert it fired with the right reward.

    Class-level state (the lists) so tests can inspect after Episode.run() finishes —
    Episode constructs a fresh agent per run, so instance state from the test setup
    isn't visible to the running episode unless we use class state.
    """

    rewards_seen: list[float] = []
    finalize_call_count: int = 0
    finalize_return: AgentOutput | None = None  # what finalize() returns; tests set this

    def finalize(self, reward: float) -> AgentOutput | None:
        # Reference the base class explicitly (not type(self)) so subclasses still
        # update the parent's counters — `type(self).x += 1` would create a new
        # attribute on the subclass and leave the parent's count unchanged.
        _FinalizeTrackingAgent.rewards_seen.append(reward)
        _FinalizeTrackingAgent.finalize_call_count += 1
        return _FinalizeTrackingAgent.finalize_return


class _FinalizeTrackingAgentConfig(MockAgentConfig):
    name: str = "finalize_tracking"

    def make(self, action_set=None, **kwargs):
        _ = action_set, kwargs
        return _FinalizeTrackingAgent(config=self)


def _reset_finalize_tracker() -> None:
    """Reset the class-level state on _FinalizeTrackingAgent — call at the start of each test."""
    _FinalizeTrackingAgent.rewards_seen = []
    _FinalizeTrackingAgent.finalize_call_count = 0
    _FinalizeTrackingAgent.finalize_return = None


class TestEpisodeInvokesAgentFinalize:
    """Episode calls agent.finalize(reward) after the per-turn loop exits.

    Spec: openspec/changes/multi-episode-rollouts/ — `finalize` is invoked exactly once
    per Episode.run() call, with the final EnvironmentOutput.reward. Non-None returns
    are appended as synthetic trajectory steps. Called in a finally block so cleanup
    (e.g. memory persistence) runs even on exception.
    """

    def test_finalize_called_exactly_once_per_run(self, tmp_dir, mock_cube_task_config) -> None:
        """One Episode.run() → one finalize() invocation."""
        _reset_finalize_tracker()
        Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=_FinalizeTrackingAgentConfig(),
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=5,
            runtime_context=None,
            storage=None,
        ).run()
        assert _FinalizeTrackingAgent.finalize_call_count == 1

    def test_finalize_receives_final_reward(self, tmp_dir, mock_cube_task_config) -> None:
        """The reward passed to finalize() is the final EnvironmentOutput.reward."""
        _reset_finalize_tracker()
        trajectory = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=_FinalizeTrackingAgentConfig(),
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=5,
            runtime_context=None,
            storage=None,
        ).run()
        # The reward finalize saw must match the trajectory's recorded final reward.
        assert len(_FinalizeTrackingAgent.rewards_seen) == 1
        assert _FinalizeTrackingAgent.rewards_seen[0] == trajectory.reward_info["reward"]

    def test_non_none_return_appended_as_trajectory_step(self, tmp_dir, mock_cube_task_config) -> None:
        """When finalize returns an AgentOutput, Episode appends it as a synthetic step."""
        _reset_finalize_tracker()
        sentinel = AgentOutput(actions=[], thoughts="end-of-episode reflection")
        _FinalizeTrackingAgent.finalize_return = sentinel
        ep = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=_FinalizeTrackingAgentConfig(),
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        trajectory = ep.run()
        loaded = ep.storage.load_trajectory(trajectory.id)
        # Last step in the trajectory should be the finalize AgentOutput we returned.
        last_step = loaded.steps[-1]
        assert isinstance(last_step.output, AgentOutput)
        assert last_step.output.thoughts == "end-of-episode reflection"

    def test_none_return_does_not_append_a_step(self, tmp_dir, mock_cube_task_config) -> None:
        """A None return from finalize must NOT add a synthetic step."""
        _reset_finalize_tracker()
        _FinalizeTrackingAgent.finalize_return = None
        ep = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=_FinalizeTrackingAgentConfig(),
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        # Baseline: same setup but with the default MockAgent (no finalize override).
        # MockAgent's parent (Agent) has a default-no-op finalize returning None.
        baseline_ep = Episode(
            id=1,
            output_dir=tmp_dir,
            agent_config=MockAgentConfig(),
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        traj_tracking = ep.run()
        traj_baseline = baseline_ep.run()
        # Step counts should match: finalize returning None added nothing.
        loaded_tracking = ep.storage.load_trajectory(traj_tracking.id)
        loaded_baseline = baseline_ep.storage.load_trajectory(traj_baseline.id)
        assert len(loaded_tracking.steps) == len(loaded_baseline.steps)

    def test_default_finalize_does_not_add_step_to_existing_agents(
        self, tmp_dir, mock_agent_config, mock_cube_task_config
    ) -> None:
        """Existing agents (using the default no-op finalize) get no extra trajectory step.

        Critical for backward compatibility: ReAct/Genny/legacy agents that inherit the
        default finalize must not see any change to their trajectory shape.
        """
        ep = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        trajectory = ep.run()
        loaded = ep.storage.load_trajectory(trajectory.id)
        # The trajectory's last step should be an EnvironmentOutput (the final env state),
        # NOT an AgentOutput from a no-op finalize call.
        last_step = loaded.steps[-1]
        assert isinstance(last_step.output, EnvironmentOutput)

    def test_finalize_called_even_when_max_steps_reached(self, tmp_dir, mock_cube_task_config) -> None:
        """Episode terminating via max_steps must still call finalize."""
        _reset_finalize_tracker()

        class _NeverDoneTaskConfig(type(mock_cube_task_config)):
            pass

        # Use a finalize-tracking config; max_steps=1 forces early termination.
        Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=_FinalizeTrackingAgentConfig(),
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=1,
            runtime_context=None,
            storage=None,
        ).run()
        assert _FinalizeTrackingAgent.finalize_call_count == 1

    def test_finalize_called_on_exception(self, tmp_dir, mock_cube_task_config) -> None:
        """If agent.step() raises, finalize() should still run as cleanup."""
        _reset_finalize_tracker()

        class _RaisingAgent(_FinalizeTrackingAgent):
            def step(self, obs: Observation) -> AgentOutput:
                raise RuntimeError("synthetic step failure")

        class _RaisingAgentConfig(MockAgentConfig):
            name: str = "raising"

            def make(self, action_set=None, **kwargs):
                _ = action_set, kwargs
                return _RaisingAgent(config=self)

        ep = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=_RaisingAgentConfig(),
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        with pytest.raises(RuntimeError, match="synthetic step failure"):
            ep.run()
        # Cleanup still fired finalize even though step() raised.
        assert _FinalizeTrackingAgent.finalize_call_count == 1

    def test_finalize_exception_does_not_mask_episode(self, tmp_dir, mock_cube_task_config) -> None:
        """If finalize() itself raises, Episode logs it and continues (doesn't crash)."""
        _reset_finalize_tracker()

        class _BadFinalizeAgent(MockAgent):
            def finalize(self, reward: float) -> AgentOutput | None:
                raise RuntimeError("finalize blew up")

        class _BadFinalizeConfig(MockAgentConfig):
            name: str = "bad_finalize"

            def make(self, action_set=None, **kwargs):
                _ = action_set, kwargs
                return _BadFinalizeAgent(config=self)

        ep = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=_BadFinalizeConfig(),
            task_config=mock_cube_task_config,
            exp_name="finalize-test",
            max_steps=5,
            runtime_context=None,
            storage=None,
        )
        # Should not propagate — Episode logs and continues to finalize the trajectory.
        trajectory = ep.run()
        assert trajectory is not None
