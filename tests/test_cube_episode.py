"""Tests for Episode with the cube path (task_config=...)."""

import warnings
from pathlib import Path

import pytest
from cube.core import Observation
from cube.task import TaskConfig

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, EvaluationEvent, ToolCallEvent
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

        Under RFC agent-owns-loop, MockAgent sends final_step immediately
        and the event stream is:
          events[0]  ToolCallEvent — synthetic reset event (initial obs)
          events[1]  LLMCallEvent    — agent.step() output with final_step action
          events[2]  ToolCallEvent — task.step intercepts final_step,
                                     evaluate() runs, reward=1.0, done=True
          events[3]  EvaluationEvent — terminal recorder.record_evaluation
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
            view = episode.run()

        assert view.metadata["task_id"] == mock_cube_task_config.task_id
        kinds = [type(e.output).__name__ for e in view]
        # MockAgent has no LLM → no LLMCallEvent (LLM auto-emit is the
        # only producer in the auto-recorder design). ToolCallEvent +
        # EvaluationEvent are still emitted by MonitoredTool + Episode.
        assert "ToolCallEvent" in kinds
        assert "EvaluationEvent" in kinds

        # `final_step` IS STOP_ACTION (cube-standard's sentinel) —
        # MonitoredTool short-circuits it BEFORE dispatch and raises
        # TaskDone, so no ToolCallEvent for the final_step action is
        # emitted. MockAgent has no LLM, so no LLMCallEvent carries it
        # either. The agent stop is observable via the absence of any
        # ToolCallEvent past the reset, plus the TaskDone path
        # finalizing cleanly (asserted above by `view.metadata` and
        # `kinds` containing EvaluationEvent).
        # All ToolCallEvents that DID dispatch should be the reset.
        tool_events = [e.output for e in view if isinstance(e.output, ToolCallEvent)]
        assert all(e.action_id == "reset" for e in tool_events)

        # The terminal EvaluationEvent reports the final reward.
        eval_event = next(e.output for e in view if isinstance(e.output, EvaluationEvent))
        assert eval_event.reward == 1.0

        # reward_info carries the terminal eval payload.
        assert view.reward_info["reward"] == 1.0
        assert view.reward_info["done"] is True

    def test_run_streams_events_to_disk(
        self, tmp_dir: Path, mock_agent_config: AgentConfig, mock_cube_task_config: TaskConfig
    ) -> None:
        """RFC agent-owns-loop scope expansion: events stream to disk;
        the returned `TrajectoryView` is a lazy reader (no in-memory event
        list). Keeps driver/worker RAM flat on image-heavy benchmarks."""
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
        view = episode.run()

        # The view is empty in RAM at construction — its index is built
        # from the directory listing; per-event payloads decode on
        # demand. Summary fields come from the eager metadata.json read.
        assert view.summary_stats["n_env_steps"] >= 1
        assert view.reward_info["reward"] == 1.0
        # Cache empty until something iterates / indexes.
        assert view._cache == {}

        # Re-opening the same episode dir gives an equivalent view.
        reopened = episode.storage.load_episode(view.id)
        assert reopened.summary_stats == view.summary_stats
        # Reset ToolCallEvent + dispatch ToolCallEvent + terminal
        # EvaluationEvent = 3 events minimum. MockAgent has no LLM so
        # no LLMCallEvent shows up here; an LLM-driven agent would see 4+.
        assert len(reopened) >= 2

    def test_failed_episode_persists_summary_stats(self, tmp_dir, mock_cube_task_config):
        """A FAILED episode must persist summary_stats to its metadata, so the XRay
        tables render correct stats without loading any events."""
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

        episodes = FileStorage(tmp_dir).list_episodes()
        assert len(episodes) == 1
        # Stats persisted on the failed-path metadata stub.
        assert episodes[0].summary_stats
        assert "n_env_steps" in episodes[0].summary_stats

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
