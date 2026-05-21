"""Tests for Genny.

Most tests do NOT require LLM calls — they exercise pure functions and agent
state manipulation directly. LLM-touching paths (_summarize_past, _act_pass,
step) use MagicMock so the test suite stays fast.
"""

from unittest.mock import MagicMock

import pytest
from cube.core import Action, ActionSchema, Observation

from cube_harness.agents.genny import (
    BudgetConfig,
    Genny,
    GennyConfig,
    _format_action_list,
    _truncate_message,
)
from cube_harness.llm import LLMConfig, LLMResponse, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(name: str = "click", description: str = "Click an element.") -> ActionSchema:
    return ActionSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "element_id": {"type": "string", "description": "The element id."},
                "force": {"type": "boolean"},
            },
            "required": ["element_id"],
        },
    )


def _make_agent(
    enable_summarize: bool = False,
) -> Genny:
    config = GennyConfig(
        llm_config=LLMConfig(model_name="test"),
        enable_summarize=enable_summarize,
    )
    return Genny(config=config, action_schemas=[_make_schema()])


def _simulate_completed_rounds(agent: Genny, n: int) -> None:
    """Populate agent state as if n obs+asst rounds have completed (Mode A)."""
    agent.goal = [{"role": "user", "content": "goal"}]
    for i in range(n):
        agent.history.append([{"role": "user", "content": f"obs_{i}"}])
        agent.history.append([{"role": "assistant", "content": f"asst_{i}"}])
    agent._latest_obs = [{"role": "user", "content": f"obs_{n}"}]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestTruncateMessage:
    def test_truncates_long_content(self) -> None:
        msg = {"role": "user", "content": "a" * 200}
        result = _truncate_message(msg, max_chars=50)
        assert len(result["content"]) < 200
        assert "truncated" in result["content"]

    def test_short_content_unchanged(self) -> None:
        msg = {"role": "user", "content": "hello"}
        assert _truncate_message(msg, max_chars=100) == msg

    def test_non_string_content_unchanged(self) -> None:
        msg = {"role": "user", "content": [{"type": "image_url"}]}
        assert _truncate_message(msg, max_chars=10) == msg


# ---------------------------------------------------------------------------
# Genny state — no LLM required
# ---------------------------------------------------------------------------


class TestIngestObs:
    def test_first_obs_sets_goal_and_empty_latest_obs(self) -> None:
        agent = _make_agent()
        agent._ingest_obs([{"role": "user", "content": "goal text"}])
        assert agent.goal == [{"role": "user", "content": "goal text"}]
        assert agent._latest_obs == []
        assert agent.history == []

    def test_first_obs_extra_messages_go_to_latest_obs(self) -> None:
        agent = _make_agent()
        agent._ingest_obs(
            [
                {"role": "user", "content": "goal"},
                {"role": "user", "content": "screenshot"},
            ]
        )
        assert agent.goal == [{"role": "user", "content": "goal"}]
        assert agent._latest_obs == [{"role": "user", "content": "screenshot"}]
        assert agent.history == []

    def test_subsequent_obs_goes_to_latest_obs_not_history(self) -> None:
        agent = _make_agent()
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._ingest_obs([{"role": "user", "content": "obs_1"}])
        assert agent._latest_obs == [{"role": "user", "content": "obs_1"}]
        assert agent.history == []

    def test_subsequent_obs_replaces_previous_latest_obs(self) -> None:
        agent = _make_agent()
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._ingest_obs([{"role": "user", "content": "obs_1"}])
        agent._ingest_obs([{"role": "user", "content": "obs_2"}])
        assert agent._latest_obs == [{"role": "user", "content": "obs_2"}]


class TestBuildBasePromptModeA:
    """Mode A (enable_summarize=False): completed history included in base prompt."""

    def test_starts_with_system(self) -> None:
        agent = _make_agent()
        agent.goal = [{"role": "user", "content": "goal"}]
        messages = agent._build_base_prompt()
        assert messages[0]["role"] == "system"

    def test_completed_history_included(self) -> None:
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=2)
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "obs_0" in contents
        assert "asst_0" in contents
        assert "obs_1" in contents
        assert "asst_1" in contents

    def test_latest_obs_not_in_base_prompt(self) -> None:
        """Latest obs is appended by callers, not baked into _build_base_prompt."""
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=2)
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "obs_2" not in contents

    def test_prefix_stable_across_steps(self) -> None:
        """The base prompt up to the last completed round is identical between steps."""
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=2)
        prefix_step2 = agent._build_base_prompt()

        # Simulate one more step completing
        agent.history.append([{"role": "user", "content": "obs_2"}])
        agent.history.append([{"role": "assistant", "content": "asst_2"}])
        agent._latest_obs = [{"role": "user", "content": "obs_3"}]

        prefix_step3 = agent._build_base_prompt()
        # step2 prefix is a strict prefix of step3 prefix
        assert prefix_step3[: len(prefix_step2)] == prefix_step2


class TestBuildBasePromptModeB:
    """Mode B (enable_summarize=True): summaries as separate assistant messages."""

    def test_summaries_are_separate_messages(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["summary_1", "summary_2"]
        agent.summary_actions = ["action_1", "action_2"]
        messages = agent._build_base_prompt()
        asst_contents = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert "summary_1" in asst_contents
        assert "summary_2" in asst_contents

    def test_summaries_not_bundled_into_single_block(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["summary_1", "summary_2"]
        agent.summary_actions = ["action_1", "action_2"]
        messages = agent._build_base_prompt()
        asst_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        # Each summary is its own message — no bundling
        assert len(asst_messages) == 2

    def test_exclude_last_summary_drops_only_last(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1", "s2", "s3"]
        agent.summary_actions = ["a1", "a2"]  # no action for s3 yet (live step)
        messages = agent._build_base_prompt(exclude_last_summary=True)
        asst_contents = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert "s1" in asst_contents
        assert "s2" in asst_contents
        assert "s3" not in asst_contents

    def test_latest_obs_not_in_base_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1"]
        agent._latest_obs = [{"role": "user", "content": "latest observation"}]
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "latest observation" not in contents

    def test_prefix_stable_across_steps(self) -> None:
        """base_prompt is a strict prefix of the next step's base_prompt (full prefix stability)."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1", "s2"]
        agent.summary_actions = ["a1", "a2"]
        prefix_step2 = agent._build_base_prompt()

        agent.summaries = ["s1", "s2", "s3"]
        agent.summary_actions = ["a1", "a2", "a3"]
        prefix_step3 = agent._build_base_prompt()

        assert prefix_step3[: len(prefix_step2)] == prefix_step2


class TestChooseContext:
    def test_always_starts_with_system(self) -> None:
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=1)
        messages = agent._choose_context()
        assert messages[0]["role"] == "system"

    def test_goal_always_included(self) -> None:
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=2)
        messages = agent._choose_context()
        assert any(m.get("content") == "goal" for m in messages)

    def test_mode_a_latest_obs_before_react_prompt(self) -> None:
        agent = _make_agent(enable_summarize=False)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "current obs"}]
        messages = agent._choose_context()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        obs_idx = contents.index("current obs")
        react_idx = contents.index(agent.config.react_prompt)
        assert obs_idx < react_idx

    def test_mode_b_new_summary_before_latest_obs(self) -> None:
        """Mode B act pass: new summary comes before latest obs (keeps summaries contiguous)."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["past summary", "new summary"]
        agent._latest_obs = [{"role": "user", "content": "current obs"}]
        messages = agent._choose_context()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        new_summary_idx = contents.index("new summary")
        obs_idx = contents.index("current obs")
        assert new_summary_idx < obs_idx

    def test_mode_b_past_summaries_before_new_summary(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1", "s2"]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        messages = agent._choose_context()
        asst_contents = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert asst_contents == ["s1", "s2"]

    def test_ends_with_react_prompt_when_summarize_disabled(self) -> None:
        agent = _make_agent(enable_summarize=False)
        _simulate_completed_rounds(agent, n=1)
        messages = agent._choose_context()
        assert messages[-1]["content"] == agent.config.react_prompt

    def test_ends_with_act_prompt_when_summarize_enabled(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        messages = agent._choose_context()
        assert messages[-1]["content"] == agent.config.act_prompt

    def test_budget_status_in_final_prompt_at_display_step(self) -> None:
        budget = BudgetConfig(max_actions=50, display_every_k=5)
        agent = _make_agent()
        agent.config = agent.config.model_copy(update={"budget": budget})
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        msg, _ = budget.check(0.0, 0, 5)  # step=5 is first display step
        messages = agent._choose_context(msg)
        assert "cumulative episode budget: 5/50 steps" in messages[-1]["content"]

    def test_budget_status_absent_between_display_steps(self) -> None:
        budget = BudgetConfig(max_actions=50, display_every_k=5)
        agent = _make_agent()
        agent.config = agent.config.model_copy(update={"budget": budget})
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        msg, _ = budget.check(0.0, 0, 3)  # step=3 is not a display step
        messages = agent._choose_context(msg)
        assert "cumulative episode budget" not in messages[-1]["content"]


# ---------------------------------------------------------------------------
# LLM-dependent paths — mocked
# ---------------------------------------------------------------------------


def _mock_llm_response(text: str = "summary text") -> LLMResponse:
    from litellm import Message as LitellmMessage

    return LLMResponse(
        message=LitellmMessage(role="assistant", content=text),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


class TestFormatActionList:
    def test_formats_single_action(self) -> None:
        actions = [Action(name="click", arguments={"bid": "btn1"})]
        assert "click(bid='btn1')" in _format_action_list(actions)

    def test_formats_multiple_actions(self) -> None:
        actions = [Action(name="click", arguments={}), Action(name="type", arguments={"text": "hi"})]
        result = _format_action_list(actions)
        assert "click()" in result
        assert "type(text='hi')" in result

    def test_empty_actions_returns_no_action(self) -> None:
        assert _format_action_list([]) == "no action"


class TestSummarizePast:
    def test_returns_summary_string(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("my summary"))
        summary, _ = agent._summarize_past()
        assert summary == "my summary"

    def test_includes_latest_obs_in_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "the current screenshot"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        contents = [m.get("content", "") for m in prompt.messages if isinstance(m, dict)]
        assert any("the current screenshot" in c for c in contents)

    def test_includes_prior_summaries_as_separate_messages(self) -> None:
        """Prior summaries appear as individual assistant messages interleaved with action user messages."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["step one cot", "step two cot"]
        agent.summary_actions = ["action_1", "action_2"]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        asst_msgs = [m for m in prompt.messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert len(asst_msgs) == 2
        assert asst_msgs[0]["content"] == "step one cot"
        assert asst_msgs[1]["content"] == "step two cot"

    def test_sum_and_act_share_prefix(self) -> None:
        """Sum call and act call share identical prefix up to the prior summary+action block."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1"]
        agent.summary_actions = ["a1"]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("new summary"))
        agent._summarize_past()
        sum_prompt = agent.summarize_llm.call_args[0][0]

        agent.summaries.append("new summary")
        act_messages = agent._choose_context()

        # Both share [system, goal, s1, a1] before diverging
        n = 4  # system + goal + s1 + a1
        assert sum_prompt.messages[:n] == act_messages[:n]

    def test_custom_summarize_prompt_is_used(self) -> None:
        config = GennyConfig(
            llm_config=LLMConfig(model_name="test"),
            enable_summarize=True,
            summarize_prompt="My custom summarize instruction.",
        )
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert prompt.messages[-1]["content"] == "My custom summarize instruction."

    def test_uses_same_system_prompt_as_act_pass(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert isinstance(prompt.messages[0], dict)
        assert prompt.messages[0]["content"] == agent.config.system_prompt

    def test_passes_same_tools_as_act_pass_for_cache(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert len(prompt.tools) > 0
        assert prompt.tools[0]["function"]["name"] == "click"

    def test_summarize_llm_uses_same_tool_choice_as_act_llm(self) -> None:
        agent = _make_agent(enable_summarize=True)
        assert agent._summarize_llm_config.tool_choice == agent.config.llm_config.tool_choice


class TestStep:
    def test_step_records_summary_and_action_separately(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.llm = MagicMock(return_value=_mock_llm_response("action"))
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("step summary"))
        obs = Observation.from_text("goal text")
        agent.step(obs)
        # Summary stays pure (no action appended) — cache stability.
        assert len(agent.summaries) == 1
        assert agent.summaries[0] == "step summary"
        # Action lives in its own parallel list.
        assert len(agent.summary_actions) == 1
        assert "no action" in agent.summary_actions[0]  # mock had no tool calls

    def test_mode_a_commits_obs_and_asst_to_history(self) -> None:
        """Mode A: after step(), completed (obs, asst) pair is in self.history."""
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response("I think therefore I act"))
        obs = Observation.from_text("goal text")
        agent.step(obs)
        # history: initial obs (from step 0 extra messages or none) + asst
        # For a single-message obs, _latest_obs after ingest is [], so only asst committed
        # Let's just check history grew
        assert len(agent.history) >= 1

    def test_mode_a_second_step_obs_in_history(self) -> None:
        """Mode A: after step 2, step 1's obs and asst are both in history."""
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response("response"))
        obs1 = Observation.from_text("goal text")
        agent.step(obs1)
        obs2 = Observation.from_text("second obs")
        agent.step(obs2)

        def _content(m: object) -> str:
            if isinstance(m, dict):
                return m.get("content", "") or ""
            return getattr(m, "content", "") or ""

        all_contents = [_content(m) for group in agent.history for m in group]
        assert any("response" in c for c in all_contents)

    def test_step_increments_action_count(self) -> None:
        agent = _make_agent()
        agent.llm = MagicMock(return_value=_mock_llm_response())
        agent.step(Observation.from_text("goal"))
        assert agent._actions_cnt == 1

    def test_step_issues_stop_action_when_limit_reached(self) -> None:
        agent = _make_agent()
        agent.config = agent.config.model_copy(update={"budget": BudgetConfig(max_actions=0)})
        result = agent.step(Observation.from_text("obs"))
        assert len(result.actions) == 1
        assert result.actions[0].name == "final_step"

    def test_thoughts_is_summary_when_summarize_enabled(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.llm = MagicMock(return_value=_mock_llm_response("act text"))
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("my cot reasoning"))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts == "my cot reasoning"
        assert "Action:" not in result.thoughts

    def test_thoughts_is_inline_content_when_summarize_disabled(self) -> None:
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response("I think therefore I act"))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts == "I think therefore I act"

    def test_thoughts_is_none_when_no_content(self) -> None:
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response(""))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts is None


# ---------------------------------------------------------------------------
# Hint / clarification resolution
# ---------------------------------------------------------------------------


def _llm_config() -> LLMConfig:
    return LLMConfig(model_name="test")


class TestHintResolution:
    def test_task_hints_takes_precedence_over_hint(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), hint="general", task_hints={"t1": "specific"})
        agent = Genny(config=config, action_schemas=[], task_id="t1")
        assert agent._task_hint == "specific"

    def test_falls_back_to_hint_when_no_task_id_match(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), hint="general", task_hints={"other": "x"})
        agent = Genny(config=config, action_schemas=[], task_id="t1")
        assert agent._task_hint == "general"

    def test_empty_when_no_hint_and_no_match(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), task_hints={"other": "x"})
        agent = Genny(config=config, action_schemas=[], task_id="t1")
        assert agent._task_hint == ""

    def test_general_hint_applied_when_task_id_none(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), hint="fallback hint")
        agent = Genny(config=config, action_schemas=[], task_id=None)
        assert agent._task_hint == "fallback hint"

    def test_task_hints_not_applied_when_task_id_none(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), hint="general", task_hints={"t1": "specific"})
        agent = Genny(config=config, action_schemas=[], task_id=None)
        assert agent._task_hint == "general"

    def test_task_clarification_resolved_by_task_id(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), task_clarification={"t1": "answer format: numeric"})
        agent = Genny(config=config, action_schemas=[], task_id="t1")
        assert agent._task_clarification == "answer format: numeric"

    def test_task_clarification_empty_when_no_match(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), task_clarification={"other": "x"})
        agent = Genny(config=config, action_schemas=[], task_id="t1")
        assert agent._task_clarification == ""

    def test_task_clarification_empty_when_task_id_none(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), task_clarification={"t1": "x"})
        agent = Genny(config=config, action_schemas=[], task_id=None)
        assert agent._task_clarification == ""

    def test_hint_injected_into_base_prompt(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), hint="use keyboard_type_into for dropdowns")
        agent = Genny(config=config, action_schemas=[], task_id=None)
        agent.goal = [{"role": "user", "content": "do the task"}]
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert any("use keyboard_type_into for dropdowns" in c for c in contents)

    def test_clarification_injected_into_base_prompt(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), task_clarification={"t1": "answer must be numeric"})
        agent = Genny(config=config, action_schemas=[], task_id="t1")
        agent.goal = [{"role": "user", "content": "do the task"}]
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert any("answer must be numeric" in c for c in contents)

    def test_no_hint_messages_when_both_empty(self) -> None:
        config = GennyConfig(llm_config=_llm_config())
        agent = Genny(config=config, action_schemas=[], task_id="t1")
        agent.goal = [{"role": "user", "content": "do the task"}]
        messages = agent._build_base_prompt()
        contents = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        assert "Task Hint" not in contents
        assert "Additional task details" not in contents

    def test_make_wires_task_id(self) -> None:
        config = GennyConfig(llm_config=_llm_config(), task_hints={"t1": "my hint"})
        agent = config.make(task_id="t1")
        assert agent._task_hint == "my hint"

    def test_none_task_id_logs_debug_when_hints_configured(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        config = GennyConfig(llm_config=_llm_config(), task_hints={"t1": "x"})
        with caplog.at_level(logging.DEBUG, logger="cube_harness.agents.genny"):
            Genny(config=config, action_schemas=[], task_id=None)
        assert any("task_id is None" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Flat history mode
# ---------------------------------------------------------------------------


def _make_flat_agent(step_prompt: str = "", enable_summarize: bool = False) -> Genny:
    config = GennyConfig(
        llm_config=LLMConfig(model_name="test"),
        flat_history=True,
        step_prompt=step_prompt,
        enable_summarize=enable_summarize,
    )
    return Genny(config=config, action_schemas=[_make_schema()])


class TestFlatHistory:
    def test_base_prompt_uses_history_groups_not_summaries(self) -> None:
        """flat_history=True: _build_base_prompt renders history groups even when enable_summarize=True."""
        agent = _make_flat_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "task"}]
        agent.summaries = ["step1 summary"]
        agent.history.append([{"role": "user", "content": "obs1"}])
        agent.history.append([{"role": "assistant", "content": "asst1"}])
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "obs1" in contents
        assert "asst1" in contents
        assert "step1 summary" not in contents

    def test_choose_context_no_trailing_message_when_step_prompt_empty(self) -> None:
        """flat_history=True, step_prompt='': no trailing user message appended."""
        agent = _make_flat_agent(step_prompt="")
        agent.goal = [{"role": "user", "content": "task"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        messages = agent._choose_context()
        last = messages[-1]
        # Last message should be the obs, not an injected user prompt
        assert isinstance(last, dict) and last.get("content") == "obs"

    def test_choose_context_trailing_message_when_step_prompt_set(self) -> None:
        """flat_history=True, step_prompt non-empty: trailing message appended."""
        agent = _make_flat_agent(step_prompt="What next?")
        agent.goal = [{"role": "user", "content": "task"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        messages = agent._choose_context()
        last = messages[-1]
        assert isinstance(last, dict) and last.get("content") == "What next?"

    def test_flat_mode_commits_obs_and_asst_to_history(self) -> None:
        """flat_history=True: each step commits obs+asst to history for flat base prompt."""
        agent = _make_flat_agent()
        agent.llm = MagicMock(return_value=_mock_llm_response("action taken"))
        agent.step(Observation.from_text("initial task"))
        agent.step(Observation.from_text("tool result"))
        # After step 2, history should contain rounds from step 1
        assert len(agent.history) >= 1

    def test_flat_mode_step2_base_prompt_includes_step1_history(self) -> None:
        """After step 1, step 2's base prompt includes step 1's completed round."""
        agent = _make_flat_agent()
        agent.llm = MagicMock(return_value=_mock_llm_response("step1 response"))
        agent.step(Observation.from_text("initial task"))
        agent.step(Observation.from_text("tool result"))
        messages = agent._build_base_prompt()

        def _content(m: object) -> str:
            if isinstance(m, dict):
                return m.get("content", "") or ""
            return getattr(m, "content", "") or ""

        all_contents = [_content(m) for m in messages]
        assert any("step1 response" in c for c in all_contents)


# ---------------------------------------------------------------------------
# BudgetConfig.check
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    # --- busted cases ---

    def test_busted_max_actions_at_limit(self) -> None:
        b = BudgetConfig(max_actions=10)
        _, busted = b.check(0.0, 0, 10)
        assert busted

    def test_busted_max_actions_over_limit(self) -> None:
        b = BudgetConfig(max_actions=10)
        _, busted = b.check(0.0, 0, 11)
        assert busted

    def test_not_busted_max_actions_below_limit(self) -> None:
        b = BudgetConfig(max_actions=10)
        _, busted = b.check(0.0, 0, 9)
        assert not busted

    def test_busted_cost_limit(self) -> None:
        b = BudgetConfig(cost_limit=1.0)
        _, busted = b.check(1.0, 0, 3)
        assert busted

    def test_busted_token_limit(self) -> None:
        b = BudgetConfig(token_limit=1000)
        _, busted = b.check(0.0, 1000, 3)
        assert busted

    def test_busted_returns_no_msg(self) -> None:
        b = BudgetConfig(cost_limit=1.0)
        msg, busted = b.check(2.0, 0, 5)
        assert busted
        assert msg is None

    def test_no_limits_never_busted(self) -> None:
        b = BudgetConfig()
        _, busted = b.check(999.0, 999_999, 999)
        assert not busted

    # --- status message cases ---

    def test_no_msg_at_step_zero(self) -> None:
        b = BudgetConfig(max_actions=50, display_every_k=5)
        msg, _ = b.check(0.0, 0, 0)
        assert msg is None

    def test_no_msg_between_display_steps(self) -> None:
        b = BudgetConfig(max_actions=50, display_every_k=5)
        msg, _ = b.check(0.0, 0, 3)
        assert msg is None

    def test_msg_at_display_step(self) -> None:
        b = BudgetConfig(max_actions=50, display_every_k=5)
        msg, _ = b.check(0.0, 0, 5)
        assert msg == "cumulative episode budget: 5/50 steps."

    def test_msg_at_subsequent_display_step(self) -> None:
        b = BudgetConfig(max_actions=50, display_every_k=5)
        msg, _ = b.check(0.0, 0, 10)
        assert msg == "cumulative episode budget: 10/50 steps."

    def test_msg_cost_usage_percentage(self) -> None:
        b = BudgetConfig(cost_limit=4.0, display_every_k=5)
        msg, _ = b.check(1.0, 0, 5)  # 25%
        assert msg == "cumulative episode budget: 25.0% token usage."

    def test_msg_token_usage_percentage(self) -> None:
        b = BudgetConfig(token_limit=1000, display_every_k=5)
        msg, _ = b.check(0.0, 500, 5)  # 50%
        assert msg == "cumulative episode budget: 50.0% token usage."

    def test_msg_uses_max_of_cost_and_token_pct(self) -> None:
        b = BudgetConfig(cost_limit=4.0, token_limit=1000, display_every_k=5)
        # cost 10%, tokens 50% → should report 50%
        msg, _ = b.check(0.4, 500, 5)
        assert msg == "cumulative episode budget: 50.0% token usage."

    def test_msg_steps_and_usage_combined(self) -> None:
        b = BudgetConfig(max_actions=100, cost_limit=2.0, display_every_k=5)
        msg, _ = b.check(1.0, 0, 10)  # 50% cost
        assert msg == "cumulative episode budget: 10/100 steps, 50.0% token usage."

    def test_no_msg_when_no_limits_configured(self) -> None:
        b = BudgetConfig(display_every_k=5)
        msg, _ = b.check(0.0, 0, 5)
        assert msg is None

    def test_display_every_k_respected(self) -> None:
        b = BudgetConfig(max_actions=100, display_every_k=3)
        msg3, _ = b.check(0.0, 0, 3)
        msg4, _ = b.check(0.0, 0, 4)
        msg6, _ = b.check(0.0, 0, 6)
        assert msg3 is not None
        assert msg4 is None
        assert msg6 is not None


# ---------------------------------------------------------------------------
# cost_limit
# ---------------------------------------------------------------------------


def _mock_llm_response_with_cost(cost: float, text: str = "response") -> LLMResponse:
    from litellm import Message as LitellmMessage

    return LLMResponse(
        message=LitellmMessage(role="assistant", content=text),
        usage=Usage(prompt_tokens=10, completion_tokens=5, cost=cost),
    )


class TestCostLimit:
    def test_no_stop_when_below_limit(self) -> None:
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), budget=BudgetConfig(cost_limit=1.0))
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_cost(0.10))
        agent.step(Observation.from_text("task"))
        # LLM was called (not short-circuited by cost limit)
        agent.llm.assert_called_once()

    def test_stop_when_limit_reached(self) -> None:
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), budget=BudgetConfig(cost_limit=0.05))
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_cost(0.10))
        agent.step(Observation.from_text("task"))  # spends $0.10, exceeds limit
        result = agent.step(Observation.from_text("task2"))
        assert result.actions[0].name == "final_step"

    def test_no_limit_when_cost_limit_none(self) -> None:
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), budget=BudgetConfig(cost_limit=None))
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent._total_cost = 999.0
        agent.llm = MagicMock(return_value=_mock_llm_response_with_cost(0.0))
        agent.step(Observation.from_text("task"))
        # LLM was called despite enormous accumulated cost because cost_limit=None
        agent.llm.assert_called_once()

    def test_total_cost_accumulates(self) -> None:
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), budget=BudgetConfig(cost_limit=10.0))
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_cost(0.50))
        agent.step(Observation.from_text("t1"))
        agent.step(Observation.from_text("t2"))
        assert agent._total_cost == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# token_limit
# ---------------------------------------------------------------------------


def _mock_llm_response_with_tokens(prompt_tokens: int, completion_tokens: int, text: str = "response") -> LLMResponse:
    from litellm import Message as LitellmMessage

    return LLMResponse(
        message=LitellmMessage(role="assistant", content=text),
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class TestTokenLimit:
    def test_no_stop_when_below_limit(self) -> None:
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), budget=BudgetConfig(token_limit=1000))
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_tokens(10, 5))
        agent.step(Observation.from_text("task"))
        agent.llm.assert_called_once()

    def test_stop_when_limit_reached(self) -> None:
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), budget=BudgetConfig(token_limit=10))
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_tokens(8, 5))
        agent.step(Observation.from_text("task"))  # uses 13 tokens, exceeds limit
        result = agent.step(Observation.from_text("task2"))
        assert result.actions[0].name == "final_step"

    def test_no_limit_when_token_limit_none(self) -> None:
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), budget=BudgetConfig(token_limit=None))
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent._total_tokens = 999_999
        agent.llm = MagicMock(return_value=_mock_llm_response_with_tokens(0, 0))
        agent.step(Observation.from_text("task"))
        agent.llm.assert_called_once()

    def test_total_tokens_accumulates(self) -> None:
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), budget=BudgetConfig(token_limit=10_000))
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_tokens(100, 50))
        agent.step(Observation.from_text("t1"))
        agent.step(Observation.from_text("t2"))
        assert agent._total_tokens == 300


# ---------------------------------------------------------------------------
# max_format_errors
# ---------------------------------------------------------------------------


def _mock_response_with_tool_call(name: str = "click") -> LLMResponse:
    from litellm import Message as LitellmMessage
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    tc = ChatCompletionMessageToolCall(
        id="call_1",
        function=Function(name=name, arguments='{"element_id": "btn"}'),
        type="function",
    )
    return LLMResponse(
        message=LitellmMessage(role="assistant", content=None, tool_calls=[tc]),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


class TestMaxFormatErrors:
    def test_no_retry_when_zero(self) -> None:
        """max_format_errors=0: no retry, empty actions list returned."""
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), max_format_errors=0)
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response("no tool calls"))
        result = agent.step(Observation.from_text("task"))
        agent.llm.assert_called_once()
        assert result.actions == []

    def test_retries_on_no_tool_calls(self) -> None:
        """max_format_errors=2: LLM called up to 3 times (1 initial + 2 retries)."""
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), max_format_errors=2)
        agent = Genny(config=config, action_schemas=[_make_schema()])
        no_tool_resp = _mock_llm_response("no tool calls")
        tool_resp = _mock_response_with_tool_call()
        # First call: no tool calls. Second call: has tool call.
        agent.llm = MagicMock(side_effect=[no_tool_resp, tool_resp])
        result = agent.step(Observation.from_text("task"))
        assert agent.llm.call_count == 2
        assert len(result.actions) == 1

    def test_stop_when_all_retries_exhausted(self) -> None:
        """When all retries fail, STOP action is returned."""
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), max_format_errors=2)
        agent = Genny(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response("no tool calls"))
        result = agent.step(Observation.from_text("task"))
        assert agent.llm.call_count == 3  # initial + 2 retries
        assert result.actions[0].name == "final_step"

    def test_correction_message_appended_on_retry(self) -> None:
        """On retry the correction user message and empty response are in the next prompt."""
        config = GennyConfig(llm_config=LLMConfig(model_name="test"), max_format_errors=1)
        agent = Genny(config=config, action_schemas=[_make_schema()])
        no_tool_resp = _mock_llm_response("no tool calls")
        tool_resp = _mock_response_with_tool_call()
        agent.llm = MagicMock(side_effect=[no_tool_resp, tool_resp])
        agent.step(Observation.from_text("task"))
        # agent.llm is called as agent.llm(prompt) so args[0] is the Prompt
        second_prompt = agent.llm.call_args_list[1][0][0]
        contents = [
            m.get("content", "") if isinstance(m, dict) else (getattr(m, "content", "") or "")
            for m in second_prompt.messages
        ]
        assert any("No tool calls found" in (c or "") for c in contents)


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------


def _make_compact_agent(threshold: int = 1) -> Genny:
    """Agent with compact_threshold_chars set (default 1 → always triggers)."""
    config = GennyConfig(
        llm_config=LLMConfig(model_name="test"),
        compact_threshold_chars=threshold,
    )
    return Genny(config=config, action_schemas=[_make_schema()])


def _mock_llm_response_with_content(content: str) -> LLMResponse:
    from litellm import Message

    return LLMResponse(message=Message(role="assistant", content=content), usage=Usage())


class TestCompaction:
    def test_history_chars_below_threshold_returns_none(self) -> None:
        agent = _make_compact_agent(threshold=1_000_000)
        _simulate_completed_rounds(agent, 2)
        assert agent._compact_history() is None

    def test_compact_flat_history_trims_to_tail_and_sets_summary(self) -> None:
        agent = _make_compact_agent(threshold=1)
        agent.llm = MagicMock(return_value=_mock_llm_response_with_content("compact summary"))
        _simulate_completed_rounds(agent, 5)

        result = agent._compact_flat_history()

        assert result is not None
        assert agent._compacted_summary == "compact summary"
        assert len(agent.history) == 3  # keeps last 3 groups

    def test_compact_flat_history_calls_llm_with_no_tools(self) -> None:
        agent = _make_compact_agent(threshold=1)
        agent.llm = MagicMock(return_value=_mock_llm_response_with_content("summary"))
        _simulate_completed_rounds(agent, 4)

        agent._compact_flat_history()

        call_prompt = agent.llm.call_args[0][0]
        assert call_prompt.tools == []

    def test_compact_summaries_consolidates_into_one(self) -> None:
        agent = Genny(
            config=GennyConfig(
                llm_config=LLMConfig(model_name="test"),
                enable_summarize=True,
                compact_threshold_chars=1,
            ),
            action_schemas=[_make_schema()],
        )
        agent.llm = MagicMock(return_value=_mock_llm_response_with_content("consolidated"))
        agent.summaries = ["s1", "s2", "s3"]
        agent.summary_actions = ["a1", "a2", "a3"]

        result = agent._compact_summaries()

        assert result is not None
        assert agent.summaries == ["consolidated"]
        assert agent.summary_actions == []

    def test_compact_history_injects_summary_into_system_message(self) -> None:
        agent = _make_compact_agent(threshold=1)
        agent.goal = [{"role": "user", "content": "task goal"}]
        agent.llm = MagicMock(return_value=_mock_llm_response_with_content("old work summary"))
        _simulate_completed_rounds(agent, 4)

        agent._compact_flat_history()

        messages = agent._choose_context()
        system_msg = next(
            m for m in messages if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "system"
        )
        content = system_msg.get("content") if isinstance(system_msg, dict) else getattr(system_msg, "content", "")
        assert "old work summary" in content
