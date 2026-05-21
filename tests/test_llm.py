"""Tests for cube_harness.llm module."""

import json
from unittest.mock import MagicMock, patch

import pytest
import tenacity
from litellm import Message
from litellm.types.utils import ChatCompletionMessageToolCall, Function

from cube_harness.llm import (
    LLM,
    LLMCall,
    LLMConfig,
    LLMResponse,
    Prompt,
    Usage,
    _build_cache_injection_points,
    _is_anthropic_model,
    _mark_last_tool_for_cache,
    get_reasoning,
)


class TestPrompt:
    """Tests for Prompt class."""

    def test_prompt_creation(self, sample_prompt):
        """Test basic Prompt creation."""
        assert len(sample_prompt.messages) == 2
        assert sample_prompt.tools == []

    def test_prompt_with_tools(self):
        """Test Prompt with tools."""
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        prompt = Prompt(messages=[{"role": "user", "content": "Search for X"}], tools=tools)
        assert len(prompt.tools) == 1
        assert prompt.tools[0]["function"]["name"] == "search"

    def test_prompt_empty_defaults(self):
        """Test Prompt with minimal arguments."""
        prompt = Prompt(messages=[])
        assert prompt.messages == []
        assert prompt.tools == []

    def test_prompt_str_representation(self, sample_prompt):
        """Test Prompt string representation."""
        str_repr = str(sample_prompt)
        assert "Messages[2]" in str_repr
        assert "system" in str_repr
        assert "user" in str_repr

    def test_prompt_serialization(self, sample_prompt):
        """Test Prompt JSON serialization."""
        json_str = sample_prompt.model_dump_json()
        data = json.loads(json_str)
        assert len(data["messages"]) == 2
        assert data["tools"] == []


class TestLLMConfig:
    """Tests for LLMConfig class."""

    def test_llm_config_creation(self, sample_llm_config):
        """Test basic LLMConfig creation."""
        assert sample_llm_config.model_name == "gpt-5-nano"
        assert sample_llm_config.temperature == 0.7
        assert sample_llm_config.max_tokens == 4096

    def test_llm_config_defaults(self):
        """Test LLMConfig with default values."""
        config = LLMConfig(model_name="gpt-4")
        assert config.temperature == 1.0
        assert config.max_tokens == 128000
        assert config.max_completion_tokens == 8192
        assert config.reasoning_effort is None
        assert config.tool_choice == "auto"
        assert config.parallel_tool_calls is False
        assert config.num_retries == 5
        assert config.retry_strategy == "exponential_backoff_retry"

    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        # Non-Anthropic model: the temperature/reasoning_effort validator is Anthropic-specific.
        config = LLMConfig(
            model_name="gpt-5-mini",
            temperature=0.5,
            max_tokens=10000,
            max_completion_tokens=2048,
            reasoning_effort="high",
            tool_choice="required",
            parallel_tool_calls=True,
            num_retries=3,
            retry_strategy="constant_retry",
        )
        assert config.reasoning_effort == "high"
        assert config.tool_choice == "required"
        assert config.parallel_tool_calls is True
        assert config.retry_strategy == "constant_retry"

    def test_llm_config_make(self, sample_llm_config):
        """Test creating LLM instance from config."""
        llm = sample_llm_config.make()
        assert isinstance(llm, LLM)
        assert llm.config == sample_llm_config

    def test_llm_config_make_counter(self, sample_llm_config):
        """Test creating token counter from config."""
        counter = sample_llm_config.make_counter()
        assert callable(counter)

    def test_llm_config_serialization(self, sample_llm_config):
        """Test LLMConfig JSON serialization."""
        json_str = sample_llm_config.model_dump_json()
        data = json.loads(json_str)
        assert data["model_name"] == "gpt-5-nano"


class TestLLM:
    """Tests for LLM class."""

    def test_llm_initialization(self, sample_llm_config):
        """Test LLM initialization."""
        llm = LLM(config=sample_llm_config)
        assert llm.config == sample_llm_config

    @patch("cube_harness.llm.litellm.completion")
    def test_llm_call(self, mock_completion, sample_llm_config, sample_prompt):
        """Test LLM call with mocked completion."""
        mock_message = Message(role="assistant", content="Hello! How can I help?")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = None  # Simulate no usage info
        mock_completion.return_value = mock_response

        llm = LLM(config=sample_llm_config)
        result = llm(sample_prompt)

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5-nano"
        assert call_kwargs["temperature"] == 0.7
        assert result.message.content == "Hello! How can I help?"
        assert result.usage.prompt_tokens == 0  # No usage info provided

    @patch("cube_harness.llm.litellm.completion")
    def test_llm_call_with_tools(self, mock_completion, sample_llm_config) -> None:
        """Test LLM call with tools."""
        tool_call = ChatCompletionMessageToolCall(
            id="call_1", function=Function(name="search", arguments='{"query": "test"}'), type="function"
        )
        mock_message = Message(role="assistant", content="Tool call made.", tool_calls=[tool_call])
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = None  # Simulate no usage info
        mock_completion.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        prompt = Prompt(messages=[{"role": "user", "content": "Search"}], tools=tools)

        llm = LLM(config=sample_llm_config)
        result = llm(prompt)

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert result.message.tool_calls is not None
        assert result.message.content == "Tool call made."

    @patch("cube_harness.llm.litellm.completion")
    def test_llm_call_empty_tools_drops_tool_choice(self, mock_completion, sample_llm_config, sample_prompt) -> None:
        """When tools=[], tool_choice and parallel_tool_calls are omitted (some providers reject them)."""
        mock_message = Message(role="assistant", content="no tools")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = None
        mock_completion.return_value = mock_response

        # sample_prompt has tools=[] by default
        llm = LLM(config=sample_llm_config)
        llm(sample_prompt)

        call_kwargs = mock_completion.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs
        assert "parallel_tool_calls" not in call_kwargs

    @patch("cube_harness.llm.litellm.completion")
    def test_llm_call_tool_choice_none_drops_tool_choice(self, mock_completion) -> None:
        """When tool_choice=None, tool_choice and parallel_tool_calls are omitted even with tools present."""
        mock_message = Message(role="assistant", content="ok")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = None
        mock_completion.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        prompt = Prompt(messages=[{"role": "user", "content": "go"}], tools=tools)
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", tool_choice=None))
        llm(prompt)

        call_kwargs = mock_completion.call_args.kwargs
        assert "tools" in call_kwargs
        assert "tool_choice" not in call_kwargs
        assert "parallel_tool_calls" not in call_kwargs


class TestLLMCall:
    """Tests for LLMCall class."""

    def test_llm_call_creation(self, sample_llm_config, sample_prompt):
        """Test basic LLMCall creation."""
        mock_message = Message(role="assistant", content="Test response")
        llm_call = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)

        assert llm_call.llm_config == sample_llm_config
        assert llm_call.prompt == sample_prompt
        assert llm_call.output == mock_message
        assert llm_call.id is not None
        assert llm_call.timestamp is not None

    def test_llm_call_auto_id(self, sample_llm_config, sample_prompt):
        """Test LLMCall auto-generates unique IDs."""
        mock_message = Message(role="assistant", content="Test")
        call1 = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)
        call2 = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)
        assert call1.id != call2.id

    def test_llm_call_timestamp(self, sample_llm_config, sample_prompt):
        """Test LLMCall has timestamp."""
        mock_message = Message(role="assistant", content="Test")
        llm_call = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)
        # Check timestamp is ISO format
        assert "T" in llm_call.timestamp

    def test_llm_call_serialization(self, sample_llm_config, sample_prompt):
        """Test LLMCall JSON serialization."""
        mock_message = Message(role="assistant", content="Test response")
        llm_call = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)
        json_str = llm_call.model_dump_json()
        data = json.loads(json_str)
        assert "id" in data
        assert "timestamp" in data
        assert data["llm_config"]["model_name"] == "gpt-5-nano"


class TestRetryBackoff:
    """Tests for LLM._completion_with_retry tenacity-based retry."""

    def _make_ok_response(self) -> MagicMock:
        return MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="ok"))],
            usage=None,
        )

    @patch("cube_harness.llm.litellm.completion")
    def test_succeeds_on_first_attempt(self, mock_completion: MagicMock) -> None:
        """No retry needed when the first call succeeds."""
        mock_completion.return_value = self._make_ok_response()
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        llm(prompt)
        assert mock_completion.call_count == 1

    @patch("cube_harness.llm.litellm.completion")
    def test_retries_on_transient_error_then_succeeds(self, mock_completion: MagicMock) -> None:
        """Retries when first call raises a retriable error, then succeeds."""
        from litellm.exceptions import InternalServerError

        mock_completion.side_effect = [
            InternalServerError("overloaded", llm_provider="anthropic", model="claude"),
            self._make_ok_response(),
        ]
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with patch("tenacity.wait_exponential", return_value=tenacity.wait_none()):
            result = llm(prompt)
        assert mock_completion.call_count == 2
        assert result.message.content == "ok"

    @patch("cube_harness.llm.litellm.completion")
    def test_does_not_retry_permanent_errors(self, mock_completion: MagicMock) -> None:
        """Non-retriable errors (e.g. NotFoundError) are raised immediately without retry."""
        from litellm.exceptions import NotFoundError

        mock_completion.side_effect = NotFoundError("model not found", llm_provider="openai", model="bad-model")
        llm = LLM(config=LLMConfig(model_name="bad-model", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with pytest.raises(NotFoundError):
            llm(prompt)
        assert mock_completion.call_count == 1

    @patch("cube_harness.llm.litellm.completion")
    def test_reraises_after_exhausting_retries(self, mock_completion: MagicMock) -> None:
        """After num_retries attempts all failing, the exception is re-raised."""
        from litellm.exceptions import RateLimitError

        mock_completion.side_effect = RateLimitError("rate limited", llm_provider="openai", model="gpt-5-nano")
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=2))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with patch("tenacity.wait_exponential", return_value=tenacity.wait_none()):
            with pytest.raises(RateLimitError):
                llm(prompt)
        assert mock_completion.call_count == 2


class TestRetryIntegration:
    """Integration tests for LLM retry — tenacity loop runs for real, only the
    underlying litellm.completion call is stubbed.  We patch time.sleep to keep
    tests fast without bypassing tenacity's own retry/stop/wait machinery."""

    def _ok_response(self) -> MagicMock:
        return MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="ok"))],
            usage=None,
        )

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_full_retry_loop_attempt_count(self, mock_completion: MagicMock, mock_sleep: MagicMock) -> None:
        """Tenacity calls litellm exactly num_retries times when every attempt fails."""
        from litellm.exceptions import ServiceUnavailableError

        mock_completion.side_effect = ServiceUnavailableError("unavailable", llm_provider="anthropic", model="claude")
        llm = LLM(config=LLMConfig(model_name="claude-sonnet", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with pytest.raises(ServiceUnavailableError):
            llm(prompt)
        assert mock_completion.call_count == 3
        # tenacity slept between attempts (num_retries-1 sleeps for num_retries attempts)
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_sleep_duration_is_exponential(self, mock_completion: MagicMock, mock_sleep: MagicMock) -> None:
        """Sleep durations grow exponentially (multiplier=2): ~2s, ~4s."""
        from litellm.exceptions import InternalServerError

        mock_completion.side_effect = [
            InternalServerError("err", llm_provider="anthropic", model="claude"),
            InternalServerError("err", llm_provider="anthropic", model="claude"),
            self._ok_response(),
        ]
        llm = LLM(config=LLMConfig(model_name="claude-sonnet", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        llm(prompt)
        assert mock_completion.call_count == 3
        assert mock_sleep.call_count == 2
        first_sleep, second_sleep = (c.args[0] for c in mock_sleep.call_args_list)
        # multiplier=2: wait = 2 * 2^(attempt-1), capped at 120
        assert 1.9 <= first_sleep <= 2.1, f"expected ~2s, got {first_sleep}"
        assert 3.9 <= second_sleep <= 4.1, f"expected ~4s, got {second_sleep}"

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_num_retries_zero_makes_single_attempt(self, mock_completion: MagicMock, mock_sleep: MagicMock) -> None:
        """num_retries=0 makes exactly one attempt with no sleep — never zero attempts."""
        from litellm.exceptions import RateLimitError

        mock_completion.side_effect = RateLimitError("limited", llm_provider="openai", model="gpt")
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=0))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with pytest.raises(RateLimitError):
            llm(prompt)
        # Exactly 1 attempt — tenacity's minimum even with stop_after_attempt(0)
        assert mock_completion.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_permanent_error_no_sleep(self, mock_completion: MagicMock, mock_sleep: MagicMock) -> None:
        """A non-retriable error raises immediately — tenacity never sleeps."""
        from litellm.exceptions import NotFoundError

        mock_completion.side_effect = NotFoundError("model not found", llm_provider="openai", model="bad")
        llm = LLM(config=LLMConfig(model_name="bad", num_retries=5))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with pytest.raises(NotFoundError):
            llm(prompt)
        assert mock_completion.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_success_after_retries_returns_correct_response(
        self, mock_completion: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """The response from the successful attempt is returned, not a retry artefact."""
        from litellm.exceptions import Timeout

        success = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="final answer"))],
            usage=None,
        )
        mock_completion.side_effect = [
            Timeout("timeout", llm_provider="openai", model="gpt"),
            Timeout("timeout", llm_provider="openai", model="gpt"),
            success,
        ]
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        result = llm(prompt)
        assert result.message.content == "final answer"
        assert mock_completion.call_count == 3


class TestCacheControlHelpers:
    """Tests for prompt-cache helper functions."""

    def test_is_anthropic_model_true_cases(self) -> None:
        for name in [
            "claude-3-5-sonnet-20241022",
            "anthropic/claude-3-5-haiku",
            "bedrock/anthropic.claude-sonnet-4-5",
            "vertex_ai/claude-opus-4-1",
        ]:
            assert _is_anthropic_model(name), name

    def test_is_anthropic_model_false_cases(self) -> None:
        # Includes a deliberate string-only-trap: substring "claude" appears under
        # an openai routing prefix; the canonical provider resolver classifies it
        # as openai so cache_control payloads don't leak.
        for name in [
            "gpt-4o",
            "gpt-5-mini",
            "azure/gpt-4o",
            "gemini/gemini-2.0-flash",
            "openai/something-claude-ish",
        ]:
            assert not _is_anthropic_model(name), name

    def test_injection_points_empty_messages(self) -> None:
        assert _build_cache_injection_points([]) == []

    def test_injection_points_system_only(self) -> None:
        # Single message → len < 2, no breakpoints emitted.
        assert _build_cache_injection_points([{"role": "system", "content": "x"}]) == []

    def test_injection_points_system_user_assistant(self) -> None:
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        points = _build_cache_injection_points(msgs)
        assert points == [
            {"location": "message", "index": 1, "control": {"type": "ephemeral"}},
            {"location": "message", "index": 2, "control": {"type": "ephemeral"}},
        ]

    def test_injection_points_uses_last_assistant(self) -> None:
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        points = _build_cache_injection_points(msgs)
        indices = sorted(p["index"] for p in points)
        assert indices == [1, 4]

    def test_injection_points_no_system(self) -> None:
        msgs = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u2"},
        ]
        points = _build_cache_injection_points(msgs)
        # No system message → only the rolling assistant breakpoint.
        assert points == [{"location": "message", "index": 1, "control": {"type": "ephemeral"}}]

    def test_injection_points_message_objects(self) -> None:
        msgs = [Message(role="system", content="s"), Message(role="assistant", content="a")]
        points = _build_cache_injection_points(msgs)
        indices = sorted(p["index"] for p in points)
        # Both breakpoints land at index 1: second message = assistant → deduplicated to one entry.
        assert indices == [1]

    def test_mark_last_tool_for_cache_empty(self) -> None:
        assert _mark_last_tool_for_cache([]) == []

    def test_mark_last_tool_for_cache_marks_last(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        marked = _mark_last_tool_for_cache(tools)
        assert marked[0] == tools[0]  # first untouched
        assert marked[1]["cache_control"] == {"type": "ephemeral"}
        # Original list not mutated.
        assert "cache_control" not in tools[1]


class TestLLMCacheControl:
    """Tests for LLM.__call__ cache_control wiring."""

    @patch("cube_harness.llm.litellm.completion")
    def test_no_cache_control_when_unset(self, mock_completion, sample_prompt) -> None:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None
        )
        llm = LLM(config=LLMConfig(model_name="claude-sonnet-4-5"))  # set_cache_control=None
        llm(sample_prompt)
        kwargs = mock_completion.call_args.kwargs
        assert "cache_control_injection_points" not in kwargs
        for tool in kwargs.get("tools", []):
            assert "cache_control" not in tool

    @patch("cube_harness.llm.litellm.completion")
    def test_no_cache_control_for_non_anthropic(self, mock_completion) -> None:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None
        )
        llm = LLM(config=LLMConfig(model_name="gpt-4o", set_cache_control="auto"))
        prompt = Prompt(
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
        llm(prompt)
        kwargs = mock_completion.call_args.kwargs
        # Non-Anthropic: don't inject cache_control even when set_cache_control="auto".
        assert "cache_control_injection_points" not in kwargs
        for tool in kwargs.get("tools", []):
            assert "cache_control" not in tool

    @patch("cube_harness.llm.litellm.completion")
    def test_auto_caching_for_anthropic(self, mock_completion) -> None:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None
        )
        llm = LLM(config=LLMConfig(model_name="claude-sonnet-4-5", set_cache_control="auto"))
        prompt = Prompt(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u2"},
            ],
            tools=[
                {"type": "function", "function": {"name": "a"}},
                {"type": "function", "function": {"name": "b"}},
            ],
        )
        llm(prompt)
        kwargs = mock_completion.call_args.kwargs
        points = kwargs["cache_control_injection_points"]
        indices = sorted(p["index"] for p in points)
        assert indices == [1, 2]
        # Last tool marked, first untouched. Original prompt.tools must not be mutated.
        assert kwargs["tools"][1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in kwargs["tools"][0]
        assert "cache_control" not in prompt.tools[1]

    @patch("cube_harness.llm.litellm.completion")
    def test_auto_caching_no_assistant_yet(self, mock_completion) -> None:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None
        )
        llm = LLM(config=LLMConfig(model_name="claude-sonnet-4-5", set_cache_control="auto"))
        prompt = Prompt(
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tools=[],
        )
        llm(prompt)
        kwargs = mock_completion.call_args.kwargs
        # Only the second-message anchor when no assistant has spoken yet.
        assert kwargs["cache_control_injection_points"] == [
            {"location": "message", "index": 1, "control": {"type": "ephemeral"}}
        ]


class TestPromptThinkingRoundTrip:
    """Round-trip invariant: thinking_blocks (+ signature) and reasoning_content survive Prompt coercion.

    Anthropic extended thinking with tool use requires the prior assistant turn's
    thinking_blocks (including the opaque signature) to be echoed back in the next
    call. The Prompt validator dumps litellm.Message via model_dump(exclude_none=True);
    this test locks in that the provider fields make it through.
    """

    def test_thinking_blocks_survive_prompt_coercion(self) -> None:
        msg = Message(
            role="assistant",
            content="here is my answer",
            thinking_blocks=[{"type": "thinking", "thinking": "let me think...", "signature": "sig_abc123"}],
            reasoning_content="let me think...",
        )
        prompt = Prompt(messages=[msg], tools=[])
        coerced = prompt.messages[0]

        assert isinstance(coerced, dict)
        assert coerced["thinking_blocks"][0]["signature"] == "sig_abc123"
        assert coerced["thinking_blocks"][0]["thinking"] == "let me think..."
        assert coerced["thinking_blocks"][0]["type"] == "thinking"
        assert coerced["reasoning_content"] == "let me think..."

    def test_reasoning_content_alone_survives(self) -> None:
        msg = Message(role="assistant", content="answer", reasoning_content="step by step...")
        prompt = Prompt(messages=[msg], tools=[])
        assert prompt.messages[0]["reasoning_content"] == "step by step..."


class TestLLMConfigAnthropicThinkingValidator:
    """LLMConfig must reject Anthropic + reasoning_effort + temperature != 1.0 at config time."""

    def test_anthropic_thinking_rejects_nonunity_temperature(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="temperature=1.0"):
            LLMConfig(model_name="claude-sonnet-4-5", temperature=0.7, reasoning_effort="medium")

    def test_anthropic_thinking_accepts_temperature_one(self) -> None:
        cfg = LLMConfig(model_name="claude-sonnet-4-5", temperature=1.0, reasoning_effort="medium")
        assert cfg.reasoning_effort == "medium"

    def test_anthropic_no_reasoning_allows_any_temperature(self) -> None:
        # Without reasoning_effort, Claude can use any temperature.
        cfg = LLMConfig(model_name="claude-sonnet-4-5", temperature=0.3)
        assert cfg.temperature == 0.3

    def test_openai_reasoning_allows_nonunity_temperature(self) -> None:
        # The constraint is Anthropic-specific; OpenAI o-series/gpt-5 are unaffected.
        cfg = LLMConfig(model_name="gpt-5-mini", temperature=0.5, reasoning_effort="high")
        assert cfg.temperature == 0.5

    def test_bedrock_claude_also_validated(self) -> None:
        import pytest

        # _is_anthropic_model covers bedrock/anthropic.* routing too.
        with pytest.raises(ValueError, match="temperature=1.0"):
            LLMConfig(model_name="bedrock/anthropic.claude-sonnet-4-5", temperature=0.5, reasoning_effort="low")


class TestGetReasoning:
    """get_reasoning(msg) is the canonical provider-agnostic reasoning extractor."""

    def test_returns_reasoning_content_when_set(self) -> None:
        msg = Message(role="assistant", content="answer", reasoning_content="my reasoning here")
        assert get_reasoning(msg) == "my reasoning here"

    def test_returns_thinking_blocks_concatenated_when_no_reasoning_content(self) -> None:
        msg = Message(
            role="assistant",
            content="answer",
            thinking_blocks=[
                {"type": "thinking", "thinking": "first thought"},
                {"type": "thinking", "thinking": "second thought"},
            ],
        )
        assert get_reasoning(msg) == "first thought second thought"

    def test_reasoning_content_takes_precedence_over_thinking_blocks(self) -> None:
        msg = Message(
            role="assistant",
            content="answer",
            reasoning_content="from reasoning_content",
            thinking_blocks=[{"type": "thinking", "thinking": "from thinking_blocks"}],
        )
        # Precedence: reasoning_content first, since both refer to the same underlying reasoning.
        assert get_reasoning(msg) == "from reasoning_content"

    def test_returns_empty_when_no_reasoning_fields(self) -> None:
        # Deliberately does NOT fall back to msg.content — the final response
        # text is already accessible via .message.content; reasoning_text is
        # strictly about reasoning, so callers can rely on truthiness as a
        # "did the model think?" signal.
        msg = Message(role="assistant", content="plain answer")
        assert get_reasoning(msg) == ""

    def test_empty_string_when_nothing_present(self) -> None:
        msg = Message(role="assistant", content=None)
        assert get_reasoning(msg) == ""


class TestLLMResponseReasoningText:
    """LLMResponse.reasoning_text exposes get_reasoning(message) as a property."""

    def test_reasoning_text_from_reasoning_content(self) -> None:
        msg = Message(role="assistant", content="answer", reasoning_content="rc here")
        resp = LLMResponse(message=msg, usage=Usage())
        assert resp.reasoning_text == "rc here"

    def test_reasoning_text_from_thinking_blocks(self) -> None:
        msg = Message(
            role="assistant",
            content="answer",
            thinking_blocks=[{"type": "thinking", "thinking": "tb here", "signature": "s"}],
        )
        resp = LLMResponse(message=msg, usage=Usage())
        assert resp.reasoning_text == "tb here"

    def test_reasoning_text_empty_when_none(self) -> None:
        # When there's no reasoning, the property falls back to content (empty string here).
        msg = Message(role="assistant", content="")
        resp = LLMResponse(message=msg, usage=Usage())
        assert resp.reasoning_text == ""


class TestUsageReasoningTokens:
    """LLM._extract_usage populates reasoning_tokens from completion_tokens_details."""

    @patch("cube_harness.llm.litellm.completion")
    def test_reasoning_tokens_extracted_from_openai_response(
        self, mock_completion, sample_llm_config, sample_prompt
    ) -> None:
        # OpenAI o-series / gpt-5 surface reasoning_tokens under completion_tokens_details.
        usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            prompt_tokens_details=None,
            completion_tokens_details=MagicMock(reasoning_tokens=42),
        )
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=usage
        )
        llm = LLM(config=sample_llm_config)
        resp = llm(sample_prompt)
        assert resp.usage.reasoning_tokens == 42
        assert resp.usage.completion_tokens == 200

    @patch("cube_harness.llm.litellm.completion")
    def test_reasoning_tokens_zero_when_no_details(self, mock_completion, sample_llm_config, sample_prompt) -> None:
        # Anthropic does not surface reasoning_tokens separately — folded into completion_tokens.
        usage = MagicMock(
            prompt_tokens=50,
            completion_tokens=80,
            total_tokens=130,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=usage
        )
        llm = LLM(config=sample_llm_config)
        resp = llm(sample_prompt)
        assert resp.usage.reasoning_tokens == 0


class TestInterleavedThinkingBeta:
    """auto-fix(412): the interleaved-thinking beta is gated on the flag.

    Three modes:
      off    = reasoning_effort=None                                   -> no beta
      once   = reasoning_effort=<level>, interleaved_thinking=False    -> no beta (default)
      always = reasoning_effort=<level>, interleaved_thinking=True     -> beta present
    """

    @staticmethod
    def _ok() -> MagicMock:
        return MagicMock(choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None)

    @patch("cube_harness.llm.litellm.completion")
    def test_always_mode_sets_beta(self, mock_completion, sample_prompt) -> None:
        mock_completion.return_value = self._ok()
        cfg = LLMConfig(
            model_name="claude-haiku-4-5",
            temperature=1.0,
            reasoning_effort="low",
            interleaved_thinking=True,
        )
        LLM(config=cfg)(sample_prompt)
        hdrs = mock_completion.call_args.kwargs.get("extra_headers") or {}
        assert "interleaved-thinking-2025-05-14" in hdrs.get("anthropic-beta", "")

    @patch("cube_harness.llm.litellm.completion")
    def test_once_mode_no_beta(self, mock_completion, sample_prompt) -> None:
        """Default cadence: anthropic + reasoning_effort but interleaved_thinking=False."""
        mock_completion.return_value = self._ok()
        cfg = LLMConfig(
            model_name="claude-haiku-4-5",
            temperature=1.0,
            reasoning_effort="low",  # interleaved_thinking defaults to False
        )
        LLM(config=cfg)(sample_prompt)
        hdrs = mock_completion.call_args.kwargs.get("extra_headers") or {}
        assert "interleaved-thinking" not in hdrs.get("anthropic-beta", "")

    @patch("cube_harness.llm.litellm.completion")
    def test_non_anthropic_flag_is_noop(self, mock_completion, sample_prompt) -> None:
        """The flag has no effect on non-Anthropic models (gpt-5 reasoning is server-managed)."""
        mock_completion.return_value = self._ok()
        cfg = LLMConfig(model_name="gpt-5-nano", reasoning_effort="low", interleaved_thinking=True)
        LLM(config=cfg)(sample_prompt)
        hdrs = mock_completion.call_args.kwargs.get("extra_headers") or {}
        assert "interleaved-thinking" not in hdrs.get("anthropic-beta", "")

    @patch("cube_harness.llm.litellm.completion")
    def test_off_mode_no_beta(self, mock_completion, sample_prompt) -> None:
        """No reasoning_effort -> no beta even if interleaved_thinking=True (degenerate)."""
        mock_completion.return_value = self._ok()
        cfg = LLMConfig(model_name="claude-haiku-4-5", temperature=1.0, interleaved_thinking=True)
        LLM(config=cfg)(sample_prompt)
        hdrs = mock_completion.call_args.kwargs.get("extra_headers") or {}
        assert "interleaved-thinking" not in hdrs.get("anthropic-beta", "")
