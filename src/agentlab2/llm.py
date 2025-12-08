"""LLM interaction abstractions, LiteLLM based."""

import json
import pprint
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Self

from litellm import Message, completion
from litellm.utils import token_counter
from pydantic import BaseModel, Field

from agentlab2.core import Action, ActionSchema


class LLMMessage(BaseModel):
    """Represents a message for the LLM input."""

    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | List[Dict[str, Any]] | None = None  # may be null for assistant messages with function calls
    name: Optional[str] = None  # required if the role is "function"
    action: Optional[Action] = None  # may be present if the assistant is calling a function
    tool_call_id: Optional[str] = None  # may be present if the message is from a tool result


class Prompt(BaseModel):
    """Represents the input prompt to chat completion api of LLM."""

    messages: List[LLMMessage]
    tools: List[ActionSchema] = Field(default_factory=list)

    def message_dicts(self) -> List[Dict[str, Any]]:
        """Convert messages to a list of dictionaries."""
        return [message.model_dump() for message in self.messages]

    def tool_dicts(self) -> List[Dict[str, Any]]:
        """Convert tools to a list of dictionaries."""
        return [tool.schema for tool in self.tools]

    def __str__(self) -> str:
        """Debug view of the prompt."""
        messages = pprint.pformat([str(m)[:500] for m in self.messages], width=120)
        tools = pprint.pformat([t.name for t in self.tools], width=120)
        return f"Tools:\n{tools}\n\nMessages:\n{messages}"


class LLMOutput(BaseModel):
    """LLMOutput represents the output from LLM."""

    text: str
    actions: List[Action] = Field(default_factory=list)
    thoughts: Optional[str] = None
    tokens: List[int] = Field(default_factory=list)
    logprobs: List[float] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_litellm_response(cls, response: Any) -> Self:
        """Create LLMOutput from a LiteLLM response object."""
        # Extract the message from the response
        message = response.choices[0].message

        # Extract contents
        text = message.content or ""
        actions = cls.actions_from_message(message)
        thoughts = cls.thoughts_from_message(message)

        # Extract metadata from usage if available
        metadata = {}
        if hasattr(response, "usage") and response.usage:
            metadata["usage"] = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }

        return cls(text=text, actions=actions, thoughts=thoughts, metadata=metadata)

    @classmethod
    def actions_from_message(cls, message: Message):
        actions = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            actions = []
            for tc in message.tool_calls:
                arguments = tc.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON arguments in tool call: {arguments}")
                if tc.function.name is None:
                    raise ValueError("Tool call must have a function name.")
                actions.append(Action(id=tc.id, name=tc.function.name, arguments=arguments))
        return actions

    @classmethod
    def thoughts_from_message(cls, message: Message) -> str:
        """Extract the agent's thoughts from the LLM message."""
        thoughts = []
        if reasoning := message.get("reasoning_content"):
            thoughts.append(reasoning)
        if blocks := message.get("thinking_blocks"):
            for block in blocks:
                if thinking := getattr(block, "content", None) or getattr(block, "thinking", None):
                    thoughts.append(thinking)
        if message.content:
            thoughts.append(message.content)
        return "\n\n".join(thoughts)


class LLM(BaseModel):
    """Thin LLM wrapper around LiteLLM completion API."""

    model_name: str
    temperature: float = 1.0
    max_total_tokens: int = 128000
    max_new_tokens: int = 8192
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low"
    num_retries: int = 3

    def __call__(self, prompt: Prompt) -> LLMOutput:
        response = completion(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_total_tokens,
            max_completion_tokens=self.max_new_tokens,
            reasoning_effort=self.reasoning_effort,
            num_retries=self.num_retries,
            tool_choice="auto",
            parallel_tool_calls=False,
            tools=prompt.tool_dicts(),
            messages=prompt.message_dicts(),
        )
        return LLMOutput.from_litellm_response(response)

    @property
    def counter(self) -> Callable[..., int]:
        """Get a token counter function for the LLM model."""
        return partial(token_counter, model=self.model_name)
