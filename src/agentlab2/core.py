from typing import Any, Callable, Dict, List, Literal, Self

from pydantic import BaseModel, Field

from agentlab2 import LLMOutput
from agentlab2.llm import LLMMessage


class ActionSchema(BaseModel):
    """
    Represents a function specification with a type, name, description and arguments.
    Compatible with OAI, Anthropic and VLLM definitions.

    Attributes:
        type (Literal["function"]): The type of the tool, which is always "function".
        name (str): The name of the function.
        description (str): A brief description of the function.
        parameters (dict): A dictionary containing the parameters of the function.
    """

    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: dict

    @classmethod
    def from_function(cls, func: Callable) -> Self:
        """Create tool object from python function."""
        schema = {}
        return cls(**schema)

    @property
    def schema(self) -> Dict[str, Any]:
        """Produce dict that could be passed as tool schema into LLM api."""
        return self.model_dump()


class Action(BaseModel):
    """
    A class representing a function call.

    Attributes:
        id (str): The identifier for the tool call.
        name (str): The name of the function being called.
        arguments (Any): The arguments to be passed to the function.
    """

    id: str
    name: str
    arguments: Dict[str, Any]


class Content(BaseModel):
    """Represents a piece of content in an observation."""

    type: str  # e.g., "text/plain", "image/png"
    data: Any  # The actual content data


class Observation(BaseModel):
    """Represents an observation from the environment."""

    tool_call_id: str | None = (
        None  # first observation may not be linked to any tool call
    )
    contents: List[Content]
    metadata: dict = Field(default_factory=dict)
    reward_info: dict = Field(default_factory=dict)

    def to_messages(self) -> List["LLMMessage"]:
        """Convert observation to a list of messages suitable for sending to LLM."""

        messages = []
        for content in self.contents:
            if content.type == "text/plain":
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=str(content.data),
                        tool_call_id=self.tool_call_id,
                    )
                )
            elif content.type in ["image/png", "image/jpeg"]:
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=[
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": content.type,
                                    "data": content.data,
                                },
                            }
                        ],
                        tool_call_id=self.tool_call_id,
                    )
                )
            else:
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=str(content.data),
                        tool_call_id=self.tool_call_id,
                    )
                )
        return messages


class TraceStep(BaseModel):
    """A single step in the trace, consisting of an action or an observation."""

    observation: Observation | None = None
    llm_output: LLMOutput | None = None


class Trace(BaseModel):
    """
    Stores history of the previous interaction.

    Metadata contains info about agent, env and task.
    reward_info represents episode level reward data.
    """

    steps: List[TraceStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    reward_info: dict = Field(default_factory=dict)
