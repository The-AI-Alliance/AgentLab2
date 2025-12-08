"""LLM interaction abstractions, LiteLLM based."""

import logging
import pprint
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional

from litellm import Message, completion
from litellm.utils import token_counter
from pydantic import BaseModel, Field

from agentlab2.core import Action, ActionSchema, Observation
from agentlab2.utils import image_to_png_base64_url

logger = logging.getLogger(__name__)


class LLMMessage(BaseModel):
    """Represents a message for the LLM input."""

    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | List[Dict[str, Any]] | None = None  # may be null for assistant messages with function calls
    name: Optional[str] = None  # required if the role is "function"
    action: Optional[Action] = None  # may be present if the assistant is calling a function
    tool_call_id: Optional[str] = None  # may be present if the message is from a tool result


class Prompt(BaseModel):
    """Represents the input prompt to chat completion api of LLM."""

    messages: List[LLMMessage | Message]
    tools: List[ActionSchema] = Field(default_factory=list)

    def message_dicts(self) -> List[Dict[str, Any] | Message]:
        """Convert messages to a list of dictionaries."""
        return [
            message.model_dump(exclude_none=True) if isinstance(message, LLMMessage) else message
            for message in self.messages
        ]

    def tool_dicts(self) -> List[Dict[str, Any]]:
        """Convert tools to a list of dictionaries."""
        return [self.schema(tool) for tool in self.tools]

    def schema(self, tool: ActionSchema) -> Dict[str, Any]:
        """Produce dict that could be passed as tool schema into LLM api."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def __str__(self) -> str:
        """Debug view of the prompt."""
        messages = pprint.pformat([str(m)[:300] for m in self.messages], width=120)
        tools = pprint.pformat(self.tools, width=120)
        return f"Tools:\n{tools}\nMessages[{len(self.messages)}]:\n{messages}"


class LLM(BaseModel):
    """Thin LLM wrapper around LiteLLM completion API."""

    model_name: str
    temperature: float = 1.0
    max_total_tokens: int = 128000
    max_new_tokens: int = 8192
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low"
    tool_choice: Literal["auto", "none", "all"] = "auto"
    parallel_tool_calls: bool = False
    max_retries: int = 3

    def __call__(self, prompt: Prompt) -> Message:
        response = completion(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_total_tokens,
            max_completion_tokens=self.max_new_tokens,
            reasoning_effort=self.reasoning_effort,
            max_retries=self.max_retries,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            tools=prompt.tool_dicts(),
            messages=prompt.message_dicts(),
        )
        return response.choices[0].message  # type: ignore

    @property
    def counter(self) -> Callable[..., int]:
        """Get a token counter function for the LLM model."""
        return partial(token_counter, model=self.model_name)


def obs_to_messages(obs: Observation) -> List[LLMMessage]:
    """Convert observation to a list of messages suitable for sending to LLM."""

    messages = []
    images = {k: v for k, v in obs.contents.items() if v.type.startswith("image/")}
    non_images = {k: v for k, v in obs.contents.items() if k not in images}
    for name, content in non_images.items():
        message = LLMMessage(
            role="tool" if obs.tool_call_id else "user",
            content=f"##{name}\n{content.data}",
            tool_call_id=obs.tool_call_id,
        )
        messages.append(message)
    for name, content in images.items():
        message = LLMMessage(
            role="user",
            content=[
                {"type": "text", "text": name},
                {"type": "image_url", "image_url": {"url": image_to_png_base64_url(content.data)}},
            ],
            tool_call_id=obs.tool_call_id,
        )
        messages.append(message)
    return messages
