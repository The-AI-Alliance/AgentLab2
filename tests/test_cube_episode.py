"""Tests for Episode with the cube path (task_config=...)."""

import warnings

import pytest
from cube.core import EnvironmentOutput, Observation

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.episode import Episode
from cube_harness.storage import FileStorage


class _FailingAgentConfig(AgentConfig):
    """Agent config whose agent raises on the first step — for failure-path tests."""

    def make(self, action_set: object = None, **kwargs: object) -> "Agent":
        _ = action_set, kwargs
        return _FailingAgent(config=self)


class _FailingAgent(Agent):
    name = "FailingAgent"
    description = "Raises on step()."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        raise RuntimeError("boom")


class TestCubeEpisode:
    """Tests for Episode with the cube path (task_config=...)."""

    def test_episode_requires_task_config(self, tmp_dir, mock_agent_config):
        """Episode raises ValueError when task_config is not provided."""
        with pytest.raises((ValueError, TypeError)):
            Episode(id=0, output_dir=tmp_dir, agent_config=mock_agent_config)

    def test_episode_accepts_task_config(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Episode created with task_config= stores it correctly."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="cube_test",
            max_steps=5,
            storage=None,
            runtime_context=None,
        )

        assert episode.config.task_config == mock_cube_task_config

    def test_episode_run_no_deprecation_warning(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Episode.run() uses the cube path: no DeprecationWarning, trajectory is correct.

        MockAgent sends final_step immediately, so the trajectory is fully deterministic:
          step[0]  EnvironmentOutput — initial obs from reset(), done=False
          step[1]  AgentOutput       — final_step action
          step[2]  EnvironmentOutput — task.step() intercepts final_step, calls evaluate(),
                                       done=True, reward=1.0, info={"success": True}
        """
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="cube_test",
            max_steps=5,
            storage=None,
            runtime_context=None,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            trajectory = episode.run()

        assert trajectory.metadata["task_id"] == mock_cube_task_config.task_id
        # Steps stream to disk; load the persisted trajectory to inspect step structure.
        loaded = episode.storage.load_trajectory(trajectory.id)
        assert len(loaded.steps) == 3

        initial_env_step = loaded.steps[0].output
        assert isinstance(initial_env_step, EnvironmentOutput)
        assert initial_env_step.done is False

        agent_step = loaded.steps[1].output
        assert isinstance(agent_step, AgentOutput)
        assert agent_step.actions[0].name == "final_step"

        final_env_step = loaded.last_env_step()
        assert final_env_step.done is True
        assert final_env_step.reward == 1.0

        assert "profiling" in trajectory.reward_info
        trajectory.reward_info.pop("profiling")  # ignore profiling info for this test
        assert trajectory.reward_info == {"reward": 1.0, "done": True, "success": True}

    def test_run_streams_steps_to_disk_and_returns_step_less(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Contract: the loop streams steps to disk; the returned Trajectory carries
        metadata + summary_stats + reward_info but NO steps (loaded lazily from disk).
        This is what keeps driver/worker RAM flat on image-heavy benchmarks."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="cube_test",
            max_steps=5,
            storage=None,
            runtime_context=None,
        )
        trajectory = episode.run()

        # Returned trajectory is step-less but fully summarised.
        assert trajectory.steps == []
        assert trajectory.summary_stats["n_env_steps"] >= 1
        assert trajectory.reward_info["reward"] == 1.0

        # Steps are fully persisted and reload from disk; summary survives the round-trip.
        loaded = episode.storage.load_trajectory(trajectory.id)
        assert len(loaded.steps) == 3
        assert loaded.summary_stats == trajectory.summary_stats

    def test_failed_episode_persists_summary_stats(self, tmp_dir, mock_cube_task_config):
        """A FAILED episode must persist summary_stats to its metadata stub, so the XRay
        tables render correct stats without loading steps (no background bulk-loader)."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=_FailingAgentConfig(),
            task_config=mock_cube_task_config,
            exp_name="cube_test",
            max_steps=5,
            storage=None,
            runtime_context=None,
        )
        with pytest.raises(RuntimeError):
            episode.run()

        trajs = FileStorage(tmp_dir).load_all_trajectory_metadata()
        assert len(trajs) == 1
        # Metadata loaded with steps=[] — stats must come from persisted summary_stats.
        assert not trajs[0].steps
        assert trajs[0].summary_stats
        assert "n_env_steps" in trajs[0].summary_stats

    def test_episode_load_from_config_round_trip(self, tmp_dir, mock_agent_config, mock_cube_task_config):
        """Save EpisodeConfig to disk; reload via load_episode_from_config() without benchmark arg."""
        episode = Episode(
            id=0,
            output_dir=tmp_dir,
            agent_config=mock_agent_config,
            task_config=mock_cube_task_config,
            exp_name="cube_test",
            max_steps=5,
            storage=None,
            runtime_context=None,
        )
        episode.storage.save_episode_config(episode.config)

        config_path = tmp_dir / "episodes" / f"{mock_cube_task_config.task_id}_ep0" / "episode_config.json"
        reloaded = Episode.load_episode_from_config(config_path)  # no benchmark arg

        assert reloaded.config == episode.config
