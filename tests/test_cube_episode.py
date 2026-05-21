"""Tests for Episode with the cube path (task_config=...)."""

import warnings

import pytest
from cube.core import EnvironmentOutput

from cube_harness.core import AgentOutput
from cube_harness.episode import Episode


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
        assert len(trajectory.steps) == 3

        initial_env_step = trajectory.steps[0].output
        assert isinstance(initial_env_step, EnvironmentOutput)
        assert initial_env_step.done is False

        agent_step = trajectory.steps[1].output
        assert isinstance(agent_step, AgentOutput)
        assert agent_step.actions[0].name == "final_step"

        final_env_step = trajectory.last_env_step()
        assert final_env_step.done is True
        assert final_env_step.reward == 1.0

        assert "profiling" in trajectory.reward_info
        trajectory.reward_info.pop("profiling")  # ignore profiling info for this test
        assert trajectory.reward_info == {"reward": 1.0, "done": True, "success": True}

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
