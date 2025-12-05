"""LLM interaction abstractions, LiteLLM based."""

from typing import Any, Dict, List, Literal, Optional, Self

from pydantic import BaseModel, Field

from agentlab2.core import Action


class LLMMessage(BaseModel):
    """Represents a message for the LLM input."""

    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | List[Dict[str, Any]] | None = (
        None  # may be null for assistant messages with function calls
    )
    name: Optional[str] = None  # required if the role is "function"
    action: Optional[Action] = (
        None  # may be present if the assistant is calling a function
    )
    tool_call_id: Optional[str] = (
        None  # may be present if the message is from a tool result
    )


class Prompt(BaseModel):
    """Represents the input prompt to chat completion api of LLM."""

    messages: List[LLMMessage]
    tools: List[dict]

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        """Convert messages to a list of dictionaries."""
        return [message.model_dump() for message in self.messages]


class LLMOutput(BaseModel):
    """LLMOutput represents the output from LLM."""

    text: str
    actions: List[Action] | None = None
    thoughts: Optional[str] = None
    tokens: List[int] | None = None
    logprobs: List[float] | None = None
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_litellm_response(cls, response: Any) -> Self:
        """Create LLMOutput from a LiteLLM response object."""
        # Extract the message from the response
        message = response.choices[0].message

        # Extract text content
        text = message.content or ""

        # Extract tool calls if present
        actions = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            actions = []
            for tc in message.tool_calls:
                import json

                arguments = tc.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}

                actions.append(
                    Action(id=tc.id, name=tc.function.name, arguments=arguments)
                )

        # Extract metadata from usage if available
        metadata = {}
        if hasattr(response, "usage") and response.usage:
            metadata["usage"] = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }

        return cls(text=text, actions=actions, metadata=metadata)
