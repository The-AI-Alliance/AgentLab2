"""Prompt-structure tests for Genny.

Uses CapturingLLM — a drop-in for LLM that records every Prompt it receives —
to assert on the exact message sequence sent to the model at each step.

This is the only reliable way to verify:
  - prompt structure for each mode
  - cross-step prefix stability (needed for Anthropic prefix caching)
  - within-step cache hit between summarize and act passes (Mode B)
  - flat mode produces mini-swe-agent-equivalent prompts
  - format error retry appends the right correction messages
"""

from cube.core import ActionSchema, Observation
from litellm import Message as LitellmMessage
from litellm.types.utils import ChatCompletionMessageToolCall, Function

from cube_harness.agents.genny import Genny, GennyConfig
from cube_harness.llm import LLMConfig, LLMResponse, Prompt, Usage

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

_SYSTEM = "sys"
_REACT = "react_prompt"
_ACT = "act_prompt"
_SUM_COT = "cot_sum_prompt"


def _llm_cfg(model: str = "test-model") -> LLMConfig:
    return LLMConfig(model_name=model)


class CapturingLLM:
    """Drop-in for LLM. Records every Prompt received; returns configurable responses."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.calls: list[Prompt] = []
        self._queue: list[LLMResponse] = list(responses or [])
        self._default = LLMResponse(
            message=LitellmMessage(role="assistant", content="captured response"),
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    def __call__(self, prompt: Prompt) -> LLMResponse:
        self.calls.append(prompt)
        return self._queue.pop(0) if self._queue else self._default


def _tool_response(tool_name: str = "click") -> LLMResponse:
    tc = ChatCompletionMessageToolCall(
        id="call_1",
        function=Function(name=tool_name, arguments='{"element_id": "btn"}'),
        type="function",
    )
    return LLMResponse(
        message=LitellmMessage(role="assistant", content=None, tool_calls=[tc]),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def _text_response(text: str = "thinking") -> LLMResponse:
    return LLMResponse(
        message=LitellmMessage(role="assistant", content=text),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def _schema() -> ActionSchema:
    return ActionSchema(
        name="click",
        description="Click element.",
        parameters={
            "type": "object",
            "properties": {"element_id": {"type": "string"}},
            "required": ["element_id"],
        },
    )


def _make_agent(
    flat_history: bool = False,
    enable_summarize: bool = False,
    step_prompt: str = "",
    react_prompt: str = _REACT,
    act_prompt: str = _ACT,
    system_prompt: str = _SYSTEM,
    model: str = "test-model",
    **kwargs: object,
) -> Genny:
    config = GennyConfig(
        llm_config=_llm_cfg(model),
        flat_history=flat_history,
        enable_summarize=enable_summarize,
        step_prompt=step_prompt,
        react_prompt=react_prompt,
        act_prompt=act_prompt,
        system_prompt=system_prompt,
        summarize_prompt=_SUM_COT,
        **kwargs,
    )
    return Genny(config=config, action_schemas=[_schema()])


# ---------------------------------------------------------------------------
# Message inspection helpers
# ---------------------------------------------------------------------------


def _role(m: object) -> str:
    if isinstance(m, dict):
        return m.get("role", "?")
    return getattr(m, "role", "?") or "?"


def _content(m: object) -> str:
    if isinstance(m, dict):
        return m.get("content", "") or ""
    return getattr(m, "content", "") or ""


def _roles(messages: list) -> list[str]:
    return [_role(m) for m in messages]


def _contents(messages: list) -> list[str]:
    return [_content(m) for m in messages]


def _sig(m: object) -> tuple[str, str]:
    """(role, content) pair for equality checks across dict/Message types."""
    return (_role(m), _content(m))


def _sigs(messages: list) -> list[tuple[str, str]]:
    return [_sig(m) for m in messages]


# ---------------------------------------------------------------------------
# Mode A — raw history
# ---------------------------------------------------------------------------
# Observation.from_text("x") → single user message {"role":"user","content":"x"}.
# Step 0: obs[0] becomes goal; _latest_obs = [] (obs has only 1 message).
# Step N≥1: _latest_obs = [obs_N_msg].
#
# Expected step 0 act prompt:  [sys, goal, react]   (no latest_obs slot)
# Expected step 1 act prompt:  [sys, goal, asst_0, obs_1, react]
# Expected step 2 act prompt:  [sys, goal, asst_0, obs_1, asst_1, obs_2, react]


class TestModeAPromptStructure:
    def test_step0_role_sequence(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))

        prompt = cap.calls[0]
        assert _roles(prompt.messages) == ["system", "user", "user"]

    def test_step0_contents(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))

        contents = _contents(cap.calls[0].messages)
        assert contents[0] == _SYSTEM
        assert contents[1] == "goal"
        assert contents[2] == _REACT

    def test_step1_role_sequence(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        prompt = cap.calls[1]
        assert _roles(prompt.messages) == ["system", "user", "assistant", "user", "user"]

    def test_step1_contents_order(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        contents = _contents(cap.calls[1].messages)
        assert contents[0] == _SYSTEM
        assert contents[1] == "goal"
        assert contents[2] == "captured response"  # asst from step 0
        assert contents[3] == "obs_1"
        assert contents[4] == _REACT

    def test_step2_role_sequence(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))
        agent.step(Observation.from_text("obs_2"))

        prompt = cap.calls[2]
        assert _roles(prompt.messages) == ["system", "user", "assistant", "user", "assistant", "user", "user"]

    def test_step2_all_history_present(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))
        agent.step(Observation.from_text("obs_2"))

        contents = _contents(cap.calls[2].messages)
        assert "obs_1" in contents
        assert "obs_2" in contents
        assert contents.count("captured response") == 2  # asst from step 0 and step 1

    def test_react_prompt_always_last(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        for i in range(3):
            agent.step(Observation.from_text(f"obs_{i}"))

        for prompt in cap.calls:
            assert _content(prompt.messages[-1]) == _REACT

    def test_system_always_first(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        for i in range(3):
            agent.step(Observation.from_text(f"obs_{i}"))

        for prompt in cap.calls:
            assert _role(prompt.messages[0]) == "system"
            assert _content(prompt.messages[0]) == _SYSTEM


# ---------------------------------------------------------------------------
# Mode A — cross-step prefix stability
# ---------------------------------------------------------------------------
# The "stable prefix" is everything before the latest_obs and react_prompt.
# For step N, messages[:-1] (drop react_prompt) should be a prefix of step N+1.


class TestModeACrossStepPrefixStability:
    def test_step0_prefix_in_step1(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        sigs_0 = _sigs(cap.calls[0].messages[:-1])  # drop react_prompt
        sigs_1 = _sigs(cap.calls[1].messages)
        assert sigs_1[: len(sigs_0)] == sigs_0

    def test_step1_prefix_in_step2(self) -> None:
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))
        agent.step(Observation.from_text("obs_2"))

        sigs_1 = _sigs(cap.calls[1].messages[:-1])  # drop react_prompt
        sigs_2 = _sigs(cap.calls[2].messages)
        assert sigs_2[: len(sigs_1)] == sigs_1

    def test_prefix_stable_over_n_steps(self) -> None:
        """For any two consecutive steps, the earlier step's base is a prefix of the later step."""
        agent = _make_agent()
        cap = CapturingLLM()
        agent.llm = cap

        for i in range(4):
            agent.step(Observation.from_text(f"obs_{i}"))

        for n in range(len(cap.calls) - 1):
            base_n = _sigs(cap.calls[n].messages[:-1])
            next_msgs = _sigs(cap.calls[n + 1].messages)
            assert next_msgs[: len(base_n)] == base_n, f"Prefix stability failed between step {n} and {n + 1}"


# ---------------------------------------------------------------------------
# Mode B — rolling summaries
# ---------------------------------------------------------------------------
# Step 0:
#   sum:  [sys, goal, sum_cot_prompt]
#   act:  [sys, goal, asst:sum_0, act_prompt]
#
# Step 1:
#   sum:  [sys, goal, asst:sum_0_with_action, obs_1, sum_cot_prompt]
#   act:  [sys, goal, asst:sum_0_with_action, asst:sum_1, obs_1, act_prompt]


class TestModeBPromptStructure:
    def test_step0_sum_roles(self) -> None:
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM()
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))

        assert _roles(cap_sum.calls[0].messages) == ["system", "user", "user"]

    def test_step0_sum_instruction_last(self) -> None:
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response("summary_0")])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))

        assert _content(cap_sum.calls[0].messages[-1]) == _SUM_COT

    def test_step0_act_summary_as_separate_assistant_message(self) -> None:
        """Summary appears as a standalone assistant message, not bundled."""
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response("summary_0")])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))

        act_msgs = cap_act.calls[0].messages
        assert _roles(act_msgs) == ["system", "user", "assistant", "user"]
        # The assistant message (index 2) should be the summary
        assert "summary_0" in _content(act_msgs[2])

    def test_step0_act_prompt_last(self) -> None:
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM()
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))

        assert _content(cap_act.calls[0].messages[-1]) == _ACT

    def test_step1_sum_includes_previous_summary(self) -> None:
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM()
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        sum_1_msgs = cap_sum.calls[1].messages
        # [sys, goal, asst:sum_0, user:action_0, obs_1, sum_instruction]
        assert _roles(sum_1_msgs) == ["system", "user", "assistant", "user", "user", "user"]

    def test_step1_act_two_summaries_as_separate_messages(self) -> None:
        """Each step's summary is a separate assistant message, each followed by its action user message."""
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response("sum_0"), _text_response("sum_1")])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        act_1_msgs = cap_act.calls[1].messages
        # [sys, goal, asst:sum_0, user:action_0, asst:sum_1, obs_1, act_prompt]
        assert _roles(act_1_msgs) == ["system", "user", "assistant", "user", "assistant", "user", "user"]

    def test_step1_act_obs_before_act_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response("sum_0"), _text_response("sum_1")])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        act_1_msgs = cap_act.calls[1].messages
        # [sys, goal, asst:sum_0, asst:sum_1, obs_1, act_prompt]
        assert _content(act_1_msgs[-2]) == "obs_1"
        assert _content(act_1_msgs[-1]) == _ACT


# ---------------------------------------------------------------------------
# Mode B — within-step cache hit
# ---------------------------------------------------------------------------
# The summarize-pass base prefix and the act-pass base prefix must be byte-identical
# up to the latest_obs boundary, so there's a cache hit between the two calls.


class TestModeBWithinStepCacheHit:
    def test_step1_sum_and_act_share_same_prefix(self) -> None:
        """
        sum pass:  [sys, goal, asst:sum_0, user:action_0, obs_1, sum_cot]
        act pass:  [sys, goal, asst:sum_0, user:action_0, asst:sum_1, obs_1, act_prompt]

        First 4 messages are identical → within-step cache hit on [sys, goal, sum_0, action_0].
        """
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response("sum_0"), _text_response("sum_1")])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        sum_1 = cap_sum.calls[1].messages
        act_1 = cap_act.calls[1].messages

        # Both start with [sys, goal, asst:sum_0, user:action_0] — 4 identical messages.
        assert _sigs(sum_1[:4]) == _sigs(act_1[:4])

    def test_step2_sum_and_act_share_same_prefix(self) -> None:
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response(f"sum_{i}") for i in range(4)])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        for i in range(3):
            agent.step(Observation.from_text(f"obs_{i}"))

        # step 2: sum has [sys, goal, asst:s0, user:a0, asst:s1, user:a1, obs_2, sum_cot]
        # step 2: act has [sys, goal, asst:s0, user:a0, asst:s1, user:a1, asst:s2, obs_2, act_prompt]
        # shared prefix = first 6 messages
        sum_2 = cap_sum.calls[2].messages
        act_2 = cap_act.calls[2].messages
        prefix_len = 6  # [sys, goal, asst:s0, user:a0, asst:s1, user:a1]
        assert _sigs(sum_2[:prefix_len]) == _sigs(act_2[:prefix_len])


# ---------------------------------------------------------------------------
# Mode B — cross-step prefix stability
# ---------------------------------------------------------------------------
# Step N's act prompt prefix should be a prefix of step N+1's sum prompt.
# This enables cross-step cache hits on the history of summaries.


class TestModeBCrossStepPrefixStability:
    def test_step0_act_prefix_in_step1_sum(self) -> None:
        """
        step 0 act: [sys, goal, asst:sum_0, act_prompt]
        step 1 sum: [sys, goal, asst:sum_0, user:action_0, obs_1, sum_cot]

        sum_0 bytes are IDENTICAL between step 0 act and step 1 sum (no mutation).
        Shared prefix = [sys, goal, asst:sum_0] — full cross-step cache hit on step 0's write.
        """
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response("sum_0"), _text_response("sum_1")])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        act_0 = cap_act.calls[0].messages  # [sys, goal, asst:sum_0, act_prompt]
        sum_1 = cap_sum.calls[1].messages  # [sys, goal, asst:sum_0, user:action_0, obs_1, sum_cot]

        # First 3 messages are byte-identical — full cache hit on step 0's write.
        assert _sigs(act_0[:3]) == _sigs(sum_1[:3])

    def test_cache_breakpoint_prefix_is_prefix_of_next_sum(self) -> None:
        """Anthropic caches the prefix up to the last assistant (breakpoint).
        That cached prefix must be byte-identical to the start of the next step's sum prompt.
        """
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response(f"sum_{i}") for i in range(4)])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        for i in range(4):
            agent.step(Observation.from_text(f"obs_{i}"))

        for n in range(len(cap_act.calls) - 1):
            act_n = cap_act.calls[n].messages
            sum_n1 = cap_sum.calls[n + 1].messages
            # The Anthropic cache breakpoint sits at the last assistant message in act_n.
            last_asst = max(i for i, m in enumerate(act_n) if _role(m) == "assistant")
            cache_prefix = _sigs(act_n[: last_asst + 1])
            assert _sigs(sum_n1[: len(cache_prefix)]) == cache_prefix, (
                f"Cache prefix from step {n} act not a prefix of step {n + 1} sum"
            )

    def test_summaries_grow_by_one_each_step(self) -> None:
        """Each act pass has exactly one more assistant-summary message than the previous."""
        agent = _make_agent(enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response(f"sum_{i}") for i in range(4)])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        for i in range(4):
            agent.step(Observation.from_text(f"obs_{i}"))

        assistant_counts = [sum(1 for r in _roles(p.messages) if r == "assistant") for p in cap_act.calls]
        # step 0 act: 1 summary; step 1 act: 2 summaries; etc.
        for i, count in enumerate(assistant_counts):
            assert count == i + 1, f"Step {i}: expected {i + 1} assistant messages, got {count}"


# ---------------------------------------------------------------------------
# Flat mode prompt structure
# ---------------------------------------------------------------------------
# flat_history=True, step_prompt="" (default):
# Step 0: [sys, goal]  (no trailing user message — _latest_obs is empty for single-msg obs)
# Step 1: [sys, goal, asst_0, obs_1]
# Step 2: [sys, goal, asst_0, obs_1, asst_1, obs_2]
#
# No injected summaries even when enable_summarize=True.


class TestFlatModePromptStructure:
    def test_step0_no_trailing_user_message(self) -> None:
        """With step_prompt='', no trailing user message is appended."""
        agent = _make_agent(flat_history=True, step_prompt="")
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))

        prompt = cap.calls[0]
        assert _roles(prompt.messages) == ["system", "user"]
        assert _content(prompt.messages[0]) == _SYSTEM
        assert _content(prompt.messages[1]) == "goal"

    def test_step1_no_trailing_user_message(self) -> None:
        agent = _make_agent(flat_history=True, step_prompt="")
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))

        prompt = cap.calls[1]
        assert _roles(prompt.messages) == ["system", "user", "assistant", "user"]
        assert _content(prompt.messages[-1]) == "obs_1"

    def test_step2_full_flat_history(self) -> None:
        agent = _make_agent(flat_history=True, step_prompt="")
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))
        agent.step(Observation.from_text("obs_1"))
        agent.step(Observation.from_text("obs_2"))

        prompt = cap.calls[2]
        assert _roles(prompt.messages) == ["system", "user", "assistant", "user", "assistant", "user"]
        contents = _contents(prompt.messages)
        assert contents[3] == "obs_1"
        assert contents[5] == "obs_2"

    def test_step_prompt_non_empty_appends_trailing_message(self) -> None:
        agent = _make_agent(flat_history=True, step_prompt="What next?")
        cap = CapturingLLM()
        agent.llm = cap

        agent.step(Observation.from_text("goal"))

        prompt = cap.calls[0]
        assert _roles(prompt.messages) == ["system", "user", "user"]
        assert _content(prompt.messages[-1]) == "What next?"

    def test_no_react_prompt_injected_in_flat_mode(self) -> None:
        """react_prompt should never appear in flat mode messages."""
        agent = _make_agent(flat_history=True, step_prompt="", react_prompt=_REACT)
        cap = CapturingLLM()
        agent.llm = cap

        for i in range(3):
            agent.step(Observation.from_text(f"obs_{i}"))

        for prompt in cap.calls:
            assert _REACT not in _contents(prompt.messages)

    def test_summaries_not_injected_when_enable_summarize_true(self) -> None:
        """Even with enable_summarize=True, summaries must not appear in flat prompt."""
        agent = _make_agent(flat_history=True, step_prompt="", enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response(f"sum_{i}") for i in range(3)])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        for i in range(3):
            agent.step(Observation.from_text(f"obs_{i}"))

        for prompt in cap_act.calls:
            for m in prompt.messages:
                # No message should contain summary text as assistant-injected content
                if _role(m) == "assistant":
                    # assistant messages in flat mode are from act responses, not summaries
                    assert "sum_" not in _content(m)

    def test_summaries_still_accumulated_for_logging(self) -> None:
        """agent.summaries grows even though summaries aren't in the prompt."""
        agent = _make_agent(flat_history=True, step_prompt="", enable_summarize=True)
        cap_act = CapturingLLM()
        cap_sum = CapturingLLM(responses=[_text_response(f"sum_{i}") for i in range(3)])
        agent.llm = cap_act
        agent.summarize_llm = cap_sum

        for i in range(3):
            agent.step(Observation.from_text(f"obs_{i}"))

        assert len(agent.summaries) == 3


# ---------------------------------------------------------------------------
# Flat mode — prefix stability
# ---------------------------------------------------------------------------
# In flat mode each step's full prompt is a prefix of the next step's prompt
# (no trailing react_prompt to drop).


class TestFlatModePrefixStability:
    def test_each_step_prompt_is_prefix_of_next(self) -> None:
        agent = _make_agent(flat_history=True, step_prompt="")
        cap = CapturingLLM()
        agent.llm = cap

        for i in range(4):
            agent.step(Observation.from_text(f"obs_{i}"))

        for n in range(len(cap.calls) - 1):
            sigs_n = _sigs(cap.calls[n].messages)
            sigs_next = _sigs(cap.calls[n + 1].messages)
            assert sigs_next[: len(sigs_n)] == sigs_n, f"Flat mode prefix stability failed between step {n} and {n + 1}"

    def test_step_prompt_set_prefix_stable(self) -> None:
        """Prefix stability still holds when step_prompt is non-empty."""
        agent = _make_agent(flat_history=True, step_prompt="Go.")
        cap = CapturingLLM()
        agent.llm = cap

        for i in range(3):
            agent.step(Observation.from_text(f"obs_{i}"))

        # In this case the trailing "Go." user message is also part of the prefix
        for n in range(len(cap.calls) - 1):
            sigs_n = _sigs(cap.calls[n].messages[:-1])  # drop trailing "Go."
            sigs_next = _sigs(cap.calls[n + 1].messages)
            assert sigs_next[: len(sigs_n)] == sigs_n, (
                f"Flat+step_prompt prefix stability failed between step {n} and {n + 1}"
            )


# ---------------------------------------------------------------------------
# Format error retry prompt shape
# ---------------------------------------------------------------------------
# When max_format_errors > 0 and response has no tool calls, _act retries by
# appending (empty_response, correction_user_msg) to the prompt.


class TestFormatErrorPromptShape:
    def test_retry_prompt_contains_correction_message(self) -> None:
        """On retry the correction user message is at the end of the prompt."""
        no_tool = _text_response("I will think about it")
        tool = _tool_response()

        agent = _make_agent(max_format_errors=1)
        cap = CapturingLLM(responses=[no_tool, tool])
        agent.llm = cap

        agent.step(Observation.from_text("goal"))

        assert len(cap.calls) == 2
        retry_msgs = cap.calls[1].messages
        assert _role(retry_msgs[-1]) == "user"
        assert "No tool calls found" in _content(retry_msgs[-1])

    def test_retry_prompt_has_original_response_before_correction(self) -> None:
        """The empty response is inserted before the correction message."""
        no_tool = _text_response("I will think about it")
        tool = _tool_response()

        agent = _make_agent(max_format_errors=1)
        cap = CapturingLLM(responses=[no_tool, tool])
        agent.llm = cap

        agent.step(Observation.from_text("goal"))

        retry_msgs = cap.calls[1].messages
        # Second-to-last should be the original assistant response
        assert _role(retry_msgs[-2]) == "assistant"
        assert _content(retry_msgs[-2]) == "I will think about it"

    def test_retry_prompt_is_extension_of_original_prompt(self) -> None:
        """Retry prompt = original prompt + [asst_response, correction_user]."""
        no_tool = _text_response("thinking")
        tool = _tool_response()

        agent = _make_agent(max_format_errors=2)
        cap = CapturingLLM(responses=[no_tool, no_tool, tool])
        agent.llm = cap

        agent.step(Observation.from_text("goal"))

        # call 0: original prompt
        # call 1: prompt + [asst_thinking, correction]
        # call 2: prompt + [asst_thinking, correction, asst_thinking, correction]
        orig = _sigs(cap.calls[0].messages)
        retry_1 = _sigs(cap.calls[1].messages)
        retry_2 = _sigs(cap.calls[2].messages)

        assert retry_1[: len(orig)] == orig
        assert retry_2[: len(orig)] == orig
        assert len(retry_2) == len(retry_1) + 2

    def test_all_retry_calls_in_llm_calls_output(self) -> None:
        """All retry LLMCall objects appear in AgentOutput.llm_calls."""
        no_tool = _text_response("thinking")
        tool = _tool_response()

        agent = _make_agent(max_format_errors=2)
        cap = CapturingLLM(responses=[no_tool, no_tool, tool])
        agent.llm = cap

        result = agent.step(Observation.from_text("goal"))

        act_calls = [c for c in result.llm_calls if c.tag == "act"]
        assert len(act_calls) == 3  # initial + 2 retries


# ---------------------------------------------------------------------------
# Mode A vs flat — structural comparison
# ---------------------------------------------------------------------------
# The only difference between Mode A and flat (with step_prompt="") is:
# Mode A appends react_prompt; flat does not.


class TestModeAVsFlat:
    def test_flat_step0_is_mode_a_minus_react_prompt(self) -> None:
        agent_a = _make_agent(flat_history=False)
        agent_f = _make_agent(flat_history=True, step_prompt="")
        cap_a = CapturingLLM()
        cap_f = CapturingLLM()
        agent_a.llm = cap_a
        agent_f.llm = cap_f

        agent_a.step(Observation.from_text("goal"))
        agent_f.step(Observation.from_text("goal"))

        sigs_a = _sigs(cap_a.calls[0].messages)
        sigs_f = _sigs(cap_f.calls[0].messages)
        # flat = mode A without the final react_prompt message
        assert sigs_a[:-1] == sigs_f

    def test_flat_step1_is_mode_a_minus_react_prompt(self) -> None:
        agent_a = _make_agent(flat_history=False)
        agent_f = _make_agent(flat_history=True, step_prompt="")
        cap_a = CapturingLLM()
        cap_f = CapturingLLM()
        agent_a.llm = cap_a
        agent_f.llm = cap_f

        for agent, cap in [(agent_a, cap_a), (agent_f, cap_f)]:
            agent.step(Observation.from_text("goal"))
            agent.step(Observation.from_text("obs_1"))

        sigs_a = _sigs(cap_a.calls[1].messages)
        sigs_f = _sigs(cap_f.calls[1].messages)
        assert sigs_a[:-1] == sigs_f
