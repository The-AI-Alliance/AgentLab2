"""LLM interaction abstractions, LiteLLM based."""

import pprint
from functools import partial
from typing import Callable, List, Literal

from litellm import Message, completion_with_retries
from litellm.utils import token_counter
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """Represents the input prompt to chat completion api of LLM."""

    messages: List[dict | Message]
    tools: List[dict] = Field(default_factory=list)

    def __str__(self) -> str:
        """Debug view of the prompt."""
        messages = "\n".join([f"[{i}]{m}" for i, m in enumerate(self.messages)])
        tools = pprint.pformat(self.tools, width=120)
        return f"Tools:\n{tools}\nMessages[{len(self.messages)}]:\n{messages}"


class LLM(BaseModel):
    """Thin LLM wrapper around LiteLLM completion API."""

    model_name: str
    temperature: float = 1.0
    max_tokens: int = 128000
    max_completion_tokens: int = 8192
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low"
    tool_choice: Literal["auto", "none", "required"] = "auto"
    parallel_tool_calls: bool = False
    num_retries: int = 5
    retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = "exponential_backoff_retry"

    def __call__(self, prompt: Prompt) -> Message:
        response = completion_with_retries(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens,
            reasoning_effort=self.reasoning_effort,
            num_retries=self.num_retries,
            retry_strategy=self.retry_strategy,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            tools=prompt.tools,
            messages=prompt.messages,
        )
        return response.choices[0].message  # type: ignore

    @property
    def counter(self) -> Callable[..., int]:
        """Get a token counter function for the LLM model."""
        return partial(token_counter, model=self.model_name)
