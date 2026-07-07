from __future__ import annotations

from typing import Annotated, Any, Callable, Literal

from pydantic import Field, SecretStr

from cube_harness.llm import _RETRY_TYPES, LLM, LLMConfig

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class RolloutTokenCounter:
    def __init__(self, tokenizer_name: str):
        if AutoTokenizer is None:
            raise ImportError("AutoTokenizer is required for RolloutTokenCounter. Please install transformers library.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

    def count_prompt_tokens(self, messages, tools=None) -> int:
        normalized = [
            m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
            for m in messages
        ]
        token_ids = self.tokenizer.apply_chat_template(
            normalized,
            tools=tools,
            add_special_tokens=True,
            add_generation_prompt=True,
            tokenize=True,
        )
        return len(token_ids)


class RolloutLLMConfig(LLMConfig):
    """Trainer-facing rollout LLM config.

    This config targets OpenAI/vLLM-compatible rollout endpoints. It intentionally
    disables benchmark-agent features such as tool choice, provider reasoning
    modes, Anthropic cache controls, and parallel tool calls.
    """

    api_base: str
    api_key: SecretStr = Field(exclude=True)
    tokenizer_name: Annotated[str, Field(min_length=1)]

    # RL rollout defaults.
    num_retries: int = 1
    retry_strategy: _RETRY_TYPES = "constant_retry"
    capture_training_metadata: Literal[True] = True

    # Generation/logprob controls exposed to rollout clients.
    top_p: float | None = None
    top_k: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)

    # Lock benchmark/agent-facing features.
    reasoning_effort: None = None
    interleaved_thinking: Literal[False] = False
    # tool_choice: Literal["auto"] = "auto"
    parallel_tool_calls: Literal[False] = False
    set_cache_control: None = None

    def make(self) -> "LLM":
        return LLM(config=self)

    def make_counter(self) -> Callable[..., int]:
        return RolloutTokenCounter(
            tokenizer_name=self.tokenizer_name,
        ).count_prompt_tokens
