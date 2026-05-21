"""Tests for cube_harness.analyze.xray_utils module."""

import json
import os
import time
from pathlib import Path

import pytest
from cube.core import Action, Content, EnvironmentOutput, Observation, StepError
from PIL import Image

from cube_harness.analyze import xray_utils
from cube_harness.core import (
    AgentOutput,
    Trajectory,
    TrajectoryStep,
)
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus
from cube_harness.llm import LLMCall, LLMConfig, Message, Prompt, Usage

# ---------------------------------------------------------------------------
# Additional fixtures (complement conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def red_image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def env_step_with_screenshot(red_image: Image.Image) -> EnvironmentOutput:
    obs = Observation(contents=[Content.from_data(red_image, name="screenshot")])
    return EnvironmentOutput(obs=obs, reward=0.0, done=False)


@pytest.fixture
def env_step_with_axtree() -> EnvironmentOutput:
    axtree_text = "RootWebArea 'Example'\n  button 'Submit'"
    obs = Observation(contents=[Content.from_data(axtree_text, name="axtree_txt")])
    return EnvironmentOutput(obs=obs, reward=0.0, done=False)


@pytest.fixture
def env_step_done_success() -> EnvironmentOutput:
    obs = Observation(contents=[Content.from_data("Done!", name="goal")])
    return EnvironmentOutput(obs=obs, reward=1.0, done=True)


@pytest.fixture
def env_step_done_failure() -> EnvironmentOutput:
    obs = Observation(contents=[Content.from_data("Failed", name="goal")])
    return EnvironmentOutput(obs=obs, reward=0.0, done=True)


@pytest.fixture
def sample_llm_call() -> LLMCall:
    config = LLMConfig(model_name="gpt-test")
    prompt = Prompt(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Click the button."},
        ],
        tools=[],
    )
    msg = Message(role="assistant", content="I will click the button.")
    usage = Usage(prompt_tokens=100, completion_tokens=20, cost=0.001)
    return LLMCall(tag="test_call", llm_config=config, prompt=prompt, output=msg, usage=usage)


@pytest.fixture
def agent_step_with_llm_call(sample_llm_call: LLMCall) -> AgentOutput:
    return AgentOutput(
        actions=[Action(id="a1", name="click", arguments={"element_id": "btn"})],
        llm_calls=[sample_llm_call],
    )


@pytest.fixture
def timed_trajectory(env_step_with_screenshot: EnvironmentOutput, agent_step_with_llm_call: AgentOutput) -> Trajectory:
    """Trajectory with timing info and metadata."""
    traj = Trajectory(
        id="test_traj",
        metadata={"task_id": "task_1", "agent_name": "agent_a"},
        start_time=0.0,
        end_time=5.0,
        reward_info={"reward": 1.0},
    )
    traj.steps.append(TrajectoryStep(output=env_step_with_screenshot, start_time=0.0, end_time=1.0))
    traj.steps.append(TrajectoryStep(output=agent_step_with_llm_call, start_time=1.0, end_time=2.0))
    traj.steps.append(
        TrajectoryStep(output=EnvironmentOutput(obs=Observation.from_text("done"), reward=1.0, done=True))
    )
    return traj


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


# ---------------------------------------------------------------------------
# TestGetDirectoryContents
# ---------------------------------------------------------------------------


class TestGetDirectoryContents:
    def test_returns_sentinel_for_missing_dir(self, tmp_path: Path) -> None:
        result = xray_utils.get_directory_contents(tmp_path / "nonexistent")
        assert result == ["Select experiment directory"]

    def test_returns_dirs_with_trajectories_subdir(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "my_exp"
        (exp_dir / "trajectories").mkdir(parents=True)
        (exp_dir / "trajectories" / "run0.metadata.json").write_text("{}")
        result = xray_utils.get_directory_contents(tmp_path)
        assert any("my_exp" in entry for entry in result)

    def test_ignores_dirs_without_trajectories(self, tmp_path: Path) -> None:
        (tmp_path / "no_traj_dir").mkdir()
        result = xray_utils.get_directory_contents(tmp_path)
        assert not any("no_traj_dir" in entry for entry in result)

    def test_count_is_based_on_metadata_files(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        traj_dir = exp_dir / "trajectories"
        traj_dir.mkdir(parents=True)
        (traj_dir / "a.metadata.json").write_text("{}")
        (traj_dir / "b.metadata.json").write_text("{}")
        result = xray_utils.get_directory_contents(tmp_path)
        assert any("2 trajectories" in entry for entry in result)

    def test_sentinel_is_first_entry(self, tmp_path: Path) -> None:
        result = xray_utils.get_directory_contents(tmp_path)
        assert result[0] == "Select experiment directory"

    def test_sorted_reverse_order(self, tmp_path: Path) -> None:
        for name in ["aaa_exp", "zzz_exp"]:
            (tmp_path / name / "trajectories").mkdir(parents=True)
            (tmp_path / name / "trajectories" / "x.metadata.json").write_text("{}")
        result = xray_utils.get_directory_contents(tmp_path)
        names = [e.split(" ")[0] for e in result[1:]]
        assert names == sorted(names, reverse=True)

    def test_returns_dirs_with_flat_layout(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "flat_exp"
        exp_dir.mkdir()
        (exp_dir / "run0_task_foo.metadata.json").write_text("{}")
        (exp_dir / "run1_task_bar.metadata.json").write_text("{}")
        result = xray_utils.get_directory_contents(tmp_path)
        assert any("flat_exp" in entry and "2 trajectories" in entry for entry in result)


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
# TestGetScreenshotFromStep
# ---------------------------------------------------------------------------


class TestGetScreenshotFromStep:
    def test_returns_none_for_none_input(self) -> None:
        assert xray_utils.get_screenshot_from_step(None) is None

    def test_returns_none_for_agent_output(self, agent_step_with_llm_call: AgentOutput) -> None:
        assert xray_utils.get_screenshot_from_step(agent_step_with_llm_call) is None

    def test_extracts_image_from_env_output(self, env_step_with_screenshot: EnvironmentOutput) -> None:
        img = xray_utils.get_screenshot_from_step(env_step_with_screenshot)
        assert isinstance(img, Image.Image)

    def test_returns_none_when_no_image_content(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("no image here"))
        assert xray_utils.get_screenshot_from_step(env_step) is None


# ---------------------------------------------------------------------------
# TestGetCurrentScreenshot
# ---------------------------------------------------------------------------


class TestGetCurrentScreenshot:
    def test_returns_current_screenshot_for_env_step(self, env_step_with_screenshot: EnvironmentOutput) -> None:
        img = xray_utils.get_current_screenshot(env_step_with_screenshot, None)
        assert isinstance(img, Image.Image)

    def test_falls_back_to_prev_env_step_for_agent_step(
        self, agent_step_with_llm_call: AgentOutput, env_step_with_screenshot: EnvironmentOutput
    ) -> None:
        img = xray_utils.get_current_screenshot(agent_step_with_llm_call, env_step_with_screenshot)
        assert isinstance(img, Image.Image)

    def test_returns_none_when_no_screenshots(self, agent_step_with_llm_call: AgentOutput) -> None:
        env_no_img = EnvironmentOutput(obs=Observation.from_text("no image"))
        img = xray_utils.get_current_screenshot(agent_step_with_llm_call, env_no_img)
        assert img is None

    def test_returns_none_for_none_inputs(self) -> None:
        assert xray_utils.get_current_screenshot(None, None) is None


# ---------------------------------------------------------------------------
# TestExtractObsContent
# ---------------------------------------------------------------------------


class TestExtractObsContent:
    def test_finds_axtree_by_name(self, env_step_with_axtree: EnvironmentOutput) -> None:
        result = xray_utils.extract_obs_content(env_step_with_axtree, "axtree")
        assert result == "RootWebArea 'Example'\n  button 'Submit'"

    def test_case_insensitive_match(self, env_step_with_axtree: EnvironmentOutput) -> None:
        result = xray_utils.extract_obs_content(env_step_with_axtree, "AXTREE")
        assert result is not None

    def test_returns_none_for_agent_output(self, agent_step_with_llm_call: AgentOutput) -> None:
        result = xray_utils.extract_obs_content(agent_step_with_llm_call, "axtree")  # type: ignore[arg-type]
        assert result is None

    def test_returns_none_when_no_match(self, env_step_with_axtree: EnvironmentOutput) -> None:
        result = xray_utils.extract_obs_content(env_step_with_axtree, "nonexistent_key")
        assert result is None

    def test_returns_none_for_image_content(self, env_step_with_screenshot: EnvironmentOutput) -> None:
        # screenshot content is not str, should not be returned
        result = xray_utils.extract_obs_content(env_step_with_screenshot, "screenshot")
        assert result is None

    def test_returns_none_for_none_input(self) -> None:
        assert xray_utils.extract_obs_content(None, "axtree") is None


# ---------------------------------------------------------------------------
# TestGetChatBranches
# ---------------------------------------------------------------------------


class TestGetChatBranches:
    def test_returns_empty_for_env_step(self, env_step_with_axtree: EnvironmentOutput) -> None:
        assert xray_utils.get_chat_branches(env_step_with_axtree) == {}

    def test_returns_empty_for_none(self) -> None:
        assert xray_utils.get_chat_branches(None) == {}

    def test_returns_empty_when_no_llm_calls(self) -> None:
        step = AgentOutput(actions=[], llm_calls=[])
        assert xray_utils.get_chat_branches(step) == {}

    def test_one_tab_per_call(self, sample_llm_call: LLMCall) -> None:
        step = AgentOutput(llm_calls=[sample_llm_call])
        assert list(xray_utils.get_chat_branches(step).keys()) == [sample_llm_call.tag]

    def test_tab_name_is_call_tag(self, agent_step_with_llm_call: AgentOutput, sample_llm_call: LLMCall) -> None:
        branches = xray_utils.get_chat_branches(agent_step_with_llm_call)
        assert sample_llm_call.tag in branches

    def test_falls_back_to_id_when_tag_empty(self) -> None:
        config = LLMConfig(model_name="gpt-test")
        prompt = Prompt(messages=[{"role": "user", "content": "x"}], tools=[])
        call = LLMCall(llm_config=config, prompt=prompt, output=Message(role="assistant", content="a"), usage=Usage())
        assert call.tag == ""
        step = AgentOutput(llm_calls=[call])
        assert list(xray_utils.get_chat_branches(step).keys()) == [call.id]

    def test_multiple_calls_get_separate_tabs(self) -> None:
        config = LLMConfig(model_name="gpt-test")
        prompt = Prompt(messages=[{"role": "user", "content": "x"}], tools=[])
        call1 = LLMCall(
            tag="act", llm_config=config, prompt=prompt, output=Message(role="assistant", content="a"), usage=Usage()
        )
        call2 = LLMCall(
            tag="summary",
            llm_config=config,
            prompt=prompt,
            output=Message(role="assistant", content="s"),
            usage=Usage(),
        )
        step = AgentOutput(llm_calls=[call1, call2])
        assert list(xray_utils.get_chat_branches(step).keys()) == ["act", "summary"]

    def test_contains_role_headers(self, agent_step_with_llm_call: AgentOutput, sample_llm_call: LLMCall) -> None:
        html = xray_utils.get_chat_branches(agent_step_with_llm_call)[sample_llm_call.tag]
        assert "system" in html
        assert "user" in html
        assert "assistant" in html

    def test_contains_message_content(self, agent_step_with_llm_call: AgentOutput, sample_llm_call: LLMCall) -> None:
        html = xray_utils.get_chat_branches(agent_step_with_llm_call)[sample_llm_call.tag]
        assert "You are a helpful assistant." in html
        assert "Click the button." in html

    def test_contains_llm_response(self, agent_step_with_llm_call: AgentOutput, sample_llm_call: LLMCall) -> None:
        html = xray_utils.get_chat_branches(agent_step_with_llm_call)[sample_llm_call.tag]
        assert "I will click the button." in html

    def test_renders_tool_calls_in_assistant_response(self, sample_llm_call: LLMCall) -> None:
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        sample_llm_call.output = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="tc1", function=Function(name="browser_click", arguments='{"bid": "42"}'), type="function"
                )
            ],
        )
        step = AgentOutput(llm_calls=[sample_llm_call])
        html = xray_utils.get_chat_branches(step)[sample_llm_call.tag]
        assert "browser_click" in html
        assert "42" in html

    def test_long_content_collapses(self, sample_llm_call: LLMCall) -> None:
        long_content = "x" * (xray_utils._COLLAPSE_THRESHOLD + 1)
        sample_llm_call.prompt.messages[1] = {"role": "user", "content": long_content}
        step = AgentOutput(llm_calls=[sample_llm_call])
        html = xray_utils.get_chat_branches(step)[sample_llm_call.tag]
        assert "<details>" in html  # collapsed block has no "open" attribute
        assert long_content[:50] in html  # content is still present

    def test_handles_list_content_with_image(self, sample_llm_call: LLMCall) -> None:
        sample_llm_call.prompt.messages[1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is a screenshot:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
        step = AgentOutput(llm_calls=[sample_llm_call])
        html = xray_utils.get_chat_branches(step)[sample_llm_call.tag]
        assert "Here is a screenshot:" in html
        assert "data:image/png;base64,abc" in html
        assert "<img" in html


# ---------------------------------------------------------------------------
# TestGetStepErrorMarkdown
# ---------------------------------------------------------------------------


class TestGetStepErrorMarkdown:
    def test_no_error_returns_message(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"))
        result = xray_utils.get_step_error_markdown(env_step)
        assert "No errors" in result

    def test_none_returns_message(self) -> None:
        result = xray_utils.get_step_error_markdown(None)
        assert "No errors" in result

    def test_agent_output_error(self) -> None:
        err = StepError(error_type="ValueError", exception_str="bad value", stack_trace="traceback here")
        step = AgentOutput(error=err)
        result = xray_utils.get_step_error_markdown(step)
        assert "ValueError" in result
        assert "bad value" in result
        assert "traceback here" in result

    def test_env_output_error(self) -> None:
        err = StepError(error_type="TimeoutError", exception_str="timed out", stack_trace="...")
        env_step = EnvironmentOutput(obs=Observation.from_text("failed"), error=err)
        result = xray_utils.get_step_error_markdown(env_step)
        assert "TimeoutError" in result

    def test_env_info_error_fallback(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"), info={"error": "Something went wrong"})
        result = xray_utils.get_step_error_markdown(env_step)
        assert "Something went wrong" in result


# ---------------------------------------------------------------------------
# TestGetStepLogsMarkdown
# ---------------------------------------------------------------------------


class TestGetStepLogsMarkdown:
    def test_returns_no_log_message_when_empty(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"))
        result = xray_utils.get_step_logs_markdown(env_step, None)
        assert "No log information" in result

    def test_shows_info_keys(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"), info={"url": "https://example.com"})
        result = xray_utils.get_step_logs_markdown(env_step, None)
        assert "url" in result
        assert "example.com" in result

    def test_excludes_error_and_message_keys(self) -> None:
        env_step = EnvironmentOutput(
            obs=Observation.from_text("ok"),
            info={"error": "bad", "message": "task done", "url": "https://x.com"},
        )
        result = xray_utils.get_step_logs_markdown(env_step, None)
        assert "error" not in result
        assert "message" not in result or "url" in result

    def test_shows_trajectory_metadata(self) -> None:
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"))
        traj = Trajectory(id="t1", metadata={"agent_name": "test_agent", "task_id": "task_1"})
        result = xray_utils.get_step_logs_markdown(env_step, traj)
        assert "test_agent" in result
        assert "task_1" in result

    def test_agent_output_shows_only_metadata(self, agent_step_with_llm_call: AgentOutput) -> None:
        traj = Trajectory(id="t1", metadata={"agent_name": "agent_a"})
        result = xray_utils.get_step_logs_markdown(agent_step_with_llm_call, traj)
        assert "agent_a" in result

    def test_shows_failure_text_for_real_trajectory(self) -> None:
        """A real (non-missing) trajectory with _failure_text shows System Error section."""
        traj = Trajectory(id="t1", metadata={"task_id": "task_x", "_failure_text": "Ray actor died"})
        result = xray_utils.get_step_logs_markdown(None, traj)
        assert "System Error" in result
        assert "Ray actor died" in result

    def test_failure_text_not_duplicated_in_metadata(self) -> None:
        """_failure_text should not appear in the Trajectory Metadata JSON block."""
        env_step = EnvironmentOutput(obs=Observation.from_text("ok"))
        traj = Trajectory(id="t1", metadata={"task_id": "task_x", "_failure_text": "some trace"})
        result = xray_utils.get_step_logs_markdown(env_step, traj)
        # Appears once (in the System Error block), not twice in the metadata JSON
        assert result.count("some trace") == 1


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


class TestComputeTrajectoryStats:
    def test_empty_trajectory(self) -> None:
        traj = Trajectory(id="empty")
        stats = xray_utils.compute_trajectory_stats(traj)
        assert stats["n_env_steps"] == 0
        assert stats["n_agent_steps"] == 0
        assert stats["total_actions"] == 0
        assert stats["final_reward"] == 0.0
        assert stats["duration"] is None

    def test_counts_env_and_agent_steps(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        assert stats["n_env_steps"] == 2
        assert stats["n_agent_steps"] == 1

    def test_duration_from_timing(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        assert stats["duration"] == pytest.approx(5.0)

    def test_final_reward_from_reward_info(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        assert stats["final_reward"] == 1.0

    def test_aggregates_token_usage(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        # The agent step has usage: prompt=100, completion=20
        assert stats["prompt_tokens"] == 100
        assert stats["completion_tokens"] == 20

    def test_total_llm_calls(self, timed_trajectory: Trajectory) -> None:
        stats = xray_utils.compute_trajectory_stats(timed_trajectory)
        assert stats["total_llm_calls"] == 1


# ---------------------------------------------------------------------------
# TestComputeExperimentStats
# ---------------------------------------------------------------------------


class TestComputeExperimentStats:
    def test_empty_list_returns_empty_string(self) -> None:
        assert xray_utils.compute_experiment_stats([]) == ""

    def test_single_finished_trajectory(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.compute_experiment_stats([timed_trajectory])
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

    def test_computes_success_rate(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.compute_experiment_stats([timed_trajectory])
        assert "Success Rate" in result

    def test_shows_token_totals(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.compute_experiment_stats([timed_trajectory])
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

    def test_single_agent(self, timed_trajectory: Trajectory) -> None:
        rows = xray_utils.build_agent_table([timed_trajectory])
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
# TestComputeStepWidth
# ---------------------------------------------------------------------------


class TestComputeStepWidth:
    def test_returns_min_width_for_none_duration(self) -> None:
        result = xray_utils._compute_step_width(None, 0.0, 1.0)
        assert result == xray_utils._MIN_WIDTH

    def test_returns_min_width_when_min_equals_max(self) -> None:
        result = xray_utils._compute_step_width(0.5, 0.5, 0.5)
        assert result == xray_utils._MIN_WIDTH

    def test_returns_min_width_for_min_duration(self) -> None:
        result = xray_utils._compute_step_width(0.0, 0.0, 10.0)
        assert result == xray_utils._MIN_WIDTH

    def test_returns_max_width_for_max_duration(self) -> None:
        result = xray_utils._compute_step_width(10.0, 0.0, 10.0)
        assert result == xray_utils._MAX_WIDTH

    def test_scales_linearly_between_min_and_max(self) -> None:
        min_w = xray_utils._MIN_WIDTH
        max_w = xray_utils._MAX_WIDTH
        mid = xray_utils._compute_step_width(5.0, 0.0, 10.0)
        assert min_w < mid < max_w
        expected = int(min_w + 0.5 * (max_w - min_w))
        assert mid == expected


# ---------------------------------------------------------------------------
# TestGenerateTimelineHtml
# ---------------------------------------------------------------------------


class TestGenerateTimelineHtml:
    def test_empty_trajectory_returns_placeholder(self) -> None:
        traj = Trajectory(id="empty")
        result = xray_utils.generate_timeline_html(traj, 0)
        assert "No trajectory loaded" in result

    def test_none_trajectory_returns_placeholder(self) -> None:
        result = xray_utils.generate_timeline_html(None, 0)
        assert "No trajectory loaded" in result

    def test_contains_step_numbers(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.generate_timeline_html(timed_trajectory, 0)
        # Step 1, 2, 3 should appear as text in segments
        assert ">1<" in result
        assert ">2<" in result

    def test_current_step_has_gold_border(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.generate_timeline_html(timed_trajectory, 0)
        assert xray_utils._CURRENT_BORDER_COLOR in result

    def test_different_colors_for_env_and_agent(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.generate_timeline_html(timed_trajectory, 0)
        assert xray_utils._ENV_COLOR in result
        assert xray_utils._AGENT_COLOR in result

    def test_done_step_has_success_border(self) -> None:
        env_done = EnvironmentOutput(obs=Observation.from_text("done"), reward=1.0, done=True)
        traj = Trajectory(id="t")
        traj.steps.append(TrajectoryStep(output=env_done))
        result = xray_utils.generate_timeline_html(traj, 0)
        assert xray_utils._SUCCESS_BORDER_COLOR in result

    def test_failed_step_has_failure_border(self) -> None:
        env_failed = EnvironmentOutput(obs=Observation.from_text("failed"), reward=0.0, done=True)
        traj = Trajectory(id="t")
        traj.steps.append(TrajectoryStep(output=env_failed))
        result = xray_utils.generate_timeline_html(traj, 0)
        assert xray_utils._FAILURE_BORDER_COLOR in result

    def test_legend_present(self, timed_trajectory: Trajectory) -> None:
        result = xray_utils.generate_timeline_html(timed_trajectory, 0)
        assert "Env" in result
        assert "Agent" in result
        assert "Current" in result


# ---------------------------------------------------------------------------
# TestGetStepDetailsMarkdown
# ---------------------------------------------------------------------------


class TestGetStepDetailsMarkdown:
    def test_none_step_returns_placeholder(self) -> None:
        result = xray_utils.get_step_details_markdown(None, None)
        assert "No step selected" in result

    def test_env_step_shows_reward(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_step_details_markdown(env_step_done_success, None)
        assert "Reward" in result
        assert "1.00" in result

    def test_env_step_shows_done_status(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_step_details_markdown(env_step_done_success, None)
        assert "Success" in result

    def test_agent_step_shows_actions(self, agent_step_with_llm_call: AgentOutput) -> None:
        result = xray_utils.get_step_details_markdown(agent_step_with_llm_call, None)
        assert "click" in result
        assert "btn" in result

    def test_agent_step_shows_token_usage(self, agent_step_with_llm_call: AgentOutput) -> None:
        result = xray_utils.get_step_details_markdown(agent_step_with_llm_call, None)
        assert "100" in result  # prompt_tokens

    def test_includes_duration_when_timing_available(self, agent_step_with_llm_call: AgentOutput) -> None:
        traj_step = TrajectoryStep(output=agent_step_with_llm_call, start_time=0.0, end_time=2.5)
        result = xray_utils.get_step_details_markdown(agent_step_with_llm_call, traj_step)
        assert "2.5s" in result

    def test_env_step_shows_content_names(self) -> None:
        obs = Observation(
            contents=[
                Content.from_data("goal text", name="goal"),
                Content.from_data("some html", name="html_content"),
            ]
        )
        env_step = EnvironmentOutput(obs=obs)
        result = xray_utils.get_step_details_markdown(env_step, None)
        assert "goal" in result
        assert "html_content" in result


# ---------------------------------------------------------------------------
# TestGetTaskGoal
# ---------------------------------------------------------------------------


class TestGetTaskGoal:
    def test_returns_placeholder_for_none(self) -> None:
        result = xray_utils.get_task_goal(None)
        assert "No trajectory" in result

    def test_returns_placeholder_for_empty_trajectory(self) -> None:
        traj = Trajectory(id="empty")
        result = xray_utils.get_task_goal(traj)
        assert "No goal text found" in result

    def test_extracts_first_text_from_first_env_step(self) -> None:
        obs = Observation(contents=[Content.from_data("Buy 3 apples", name="goal")])
        env_out = EnvironmentOutput(obs=obs)
        traj = Trajectory(id="t1")
        traj.steps.append(TrajectoryStep(output=env_out))
        result = xray_utils.get_task_goal(traj)
        assert result == "Buy 3 apples"

    def test_skips_agent_steps_at_start(self) -> None:
        agent_out = AgentOutput(actions=[])
        obs = Observation(contents=[Content.from_data("Real goal")])
        env_out = EnvironmentOutput(obs=obs)
        traj = Trajectory(id="t1")
        traj.steps.append(TrajectoryStep(output=agent_out))
        traj.steps.append(TrajectoryStep(output=env_out))
        result = xray_utils.get_task_goal(traj)
        assert result == "Real goal"

    def test_skips_image_contents(self, red_image: Image.Image) -> None:
        obs = Observation(
            contents=[
                Content.from_data(red_image, name="screenshot"),
                Content.from_data("Goal text after image"),
            ]
        )
        env_out = EnvironmentOutput(obs=obs)
        traj = Trajectory(id="t1")
        traj.steps.append(TrajectoryStep(output=env_out))
        result = xray_utils.get_task_goal(traj)
        assert result == "Goal text after image"

    def test_skips_whitespace_only_content(self) -> None:
        obs = Observation(
            contents=[
                Content.from_data("   \n  "),
                Content.from_data("Actual goal"),
            ]
        )
        env_out = EnvironmentOutput(obs=obs)
        traj = Trajectory(id="t1")
        traj.steps.append(TrajectoryStep(output=env_out))
        result = xray_utils.get_task_goal(traj)
        assert result == "Actual goal"


# ---------------------------------------------------------------------------
# TestGetAgentActionMarkdown
# ---------------------------------------------------------------------------


class TestGetAgentActionMarkdown:
    def test_returns_terminal_placeholder_for_none(self) -> None:
        result = xray_utils.get_agent_action_markdown(None)
        assert "Terminal" in result or "terminal" in result

    def test_returns_no_actions_placeholder_when_empty(self) -> None:
        agent_out = AgentOutput(actions=[])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "No actions" in result

    def test_formats_as_function_call(self) -> None:
        action = Action(name="browser_click", arguments={"bid": "a42", "button": "left"})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "browser_click" in result
        assert "bid" in result
        assert "a42" in result

    def test_uses_backtick_code_format(self) -> None:
        action = Action(name="noop", arguments={})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "`noop()`" in result

    def test_truncates_long_string_arguments(self) -> None:
        long_text = "x" * 300
        action = Action(name="browser_type", arguments={"text": long_text})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "…" in result
        assert "x" * 300 not in result

    def test_renders_multiple_actions(self) -> None:
        actions = [
            Action(name="browser_click", arguments={"bid": "a1"}),
            Action(name="browser_type", arguments={"bid": "b1", "text": "hello"}),
        ]
        agent_out = AgentOutput(actions=actions)
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "browser_click" in result
        assert "browser_type" in result

    def test_formats_non_string_arguments(self) -> None:
        action = Action(name="scroll", arguments={"delta_x": 0, "delta_y": 100, "relative": True})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "delta_y=100" in result
        assert "relative=True" in result

    def test_empty_arguments_dict(self) -> None:
        action = Action(name="noop", arguments={})
        agent_out = AgentOutput(actions=[action])
        result = xray_utils.get_agent_action_markdown(agent_out)
        assert "noop()" in result


# ---------------------------------------------------------------------------
# TestGetPairedStepDetailsMarkdown
# ---------------------------------------------------------------------------


class TestGetPairedStepDetailsMarkdown:
    def test_returns_placeholder_when_env_none(self) -> None:
        result = xray_utils.get_paired_step_details_markdown(None, None, None, None)
        assert "No step selected" in result

    def test_contains_env_section(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_paired_step_details_markdown(env_step_done_success, None, None, None)
        assert "Environment" in result

    def test_contains_terminal_marker_when_no_agent(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_paired_step_details_markdown(env_step_done_success, None, None, None)
        assert "terminal" in result.lower() or "No agent action" in result

    def test_contains_agent_section_when_agent_present(
        self,
        env_step_done_success: EnvironmentOutput,
        agent_step_with_llm_call: AgentOutput,
    ) -> None:
        result = xray_utils.get_paired_step_details_markdown(
            env_step_done_success, agent_step_with_llm_call, None, None
        )
        assert "Agent" in result
        assert "click" in result

    def test_includes_env_duration_from_traj_step(self, env_step_done_success: EnvironmentOutput) -> None:
        ts = TrajectoryStep(output=env_step_done_success, start_time=0.0, end_time=3.0)
        result = xray_utils.get_paired_step_details_markdown(env_step_done_success, None, ts, None)
        assert "3.0s" in result

    def test_includes_agent_duration_from_traj_step(
        self,
        env_step_done_success: EnvironmentOutput,
        agent_step_with_llm_call: AgentOutput,
    ) -> None:
        env_ts = TrajectoryStep(output=env_step_done_success, start_time=0.0, end_time=1.0)
        agent_ts = TrajectoryStep(output=agent_step_with_llm_call, start_time=1.0, end_time=4.5)
        result = xray_utils.get_paired_step_details_markdown(
            env_step_done_success, agent_step_with_llm_call, env_ts, agent_ts
        )
        assert "3.5s" in result

    def test_shows_reward_for_done_step(self, env_step_done_success: EnvironmentOutput) -> None:
        result = xray_utils.get_paired_step_details_markdown(env_step_done_success, None, None, None)
        assert "1.00" in result


# ---------------------------------------------------------------------------
# TestGetPairedErrorMarkdown
# ---------------------------------------------------------------------------


class TestGetPairedErrorMarkdown:
    def test_no_errors_returns_message(self) -> None:
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"))
        result = xray_utils.get_paired_error_markdown(env_out, None)
        assert "No errors" in result

    def test_shows_env_error(self) -> None:
        err = StepError(error_type="TimeoutError", exception_str="timed out", stack_trace="...")
        env_out = EnvironmentOutput(obs=Observation.from_text("failed"), error=err)
        result = xray_utils.get_paired_error_markdown(env_out, None)
        assert "TimeoutError" in result
        assert "timed out" in result

    def test_shows_agent_error(self) -> None:
        err = StepError(error_type="ValueError", exception_str="bad input", stack_trace="...")
        agent_out = AgentOutput(error=err)
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"))
        result = xray_utils.get_paired_error_markdown(env_out, agent_out)
        assert "ValueError" in result

    def test_shows_both_errors_when_present(self) -> None:
        env_err = StepError(error_type="EnvError", exception_str="env failed", stack_trace="a")
        agent_err = StepError(error_type="AgentError", exception_str="agent failed", stack_trace="b")
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"), error=env_err)
        agent_out = AgentOutput(error=agent_err)
        result = xray_utils.get_paired_error_markdown(env_out, agent_out)
        assert "EnvError" in result
        assert "AgentError" in result

    def test_env_info_error_fallback(self) -> None:
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"), info={"error": "page crashed"})
        result = xray_utils.get_paired_error_markdown(env_out, None)
        assert "page crashed" in result

    def test_no_errors_when_both_none(self) -> None:
        env_out = EnvironmentOutput(obs=Observation.from_text("ok"))
        agent_out = AgentOutput(actions=[])
        result = xray_utils.get_paired_error_markdown(env_out, agent_out)
        assert "No errors" in result


# ---------------------------------------------------------------------------
# TestGetChatBranchesWithMessageObjects
# ---------------------------------------------------------------------------


class TestGetChatBranchesWithMessageObjects:
    """Tests for Message object handling (not just dicts) in get_chat_branches."""

    def test_handles_litellm_message_objects(self) -> None:
        """LLMCall.output is a Message object; prompts can also contain Message objects."""
        config = LLMConfig(model_name="gpt-test")
        prompt = Prompt(messages=[Message(role="user", content="Use a Message object")], tools=[])
        llm_call = LLMCall(
            tag="test", llm_config=config, prompt=prompt, output=Message(role="assistant", content="ok"), usage=Usage()
        )
        step = AgentOutput(llm_calls=[llm_call])
        html = xray_utils.get_chat_branches(step)["test"]
        assert "user" in html
        assert "Use a Message object" in html

    def test_handles_mixed_dict_and_message_in_messages(self) -> None:
        config = LLMConfig(model_name="gpt-test")
        prompt = Prompt(
            messages=[
                Message(role="system", content="System prompt from Message"),
                {"role": "user", "content": "User dict message"},
            ],
            tools=[],
        )
        llm_call = LLMCall(
            tag="test2", llm_config=config, prompt=prompt, output=Message(role="assistant", content="ok"), usage=Usage()
        )
        step = AgentOutput(llm_calls=[llm_call])
        html = xray_utils.get_chat_branches(step)["test2"]
        assert "system" in html
        assert "System prompt from Message" in html
        assert "User dict message" in html


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
        assert "🚫" in cell


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
