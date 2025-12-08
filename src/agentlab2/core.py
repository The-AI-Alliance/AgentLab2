import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Self

import litellm.utils
from pydantic import BaseModel, Field


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
        schema = litellm.utils.function_to_dict(func)
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

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Content(BaseModel):
    """Represents a piece of content in an observation."""

    type: str = "text/plain"  # e.g., "text/plain", "image/png"
    data: Any  # The actual content data


class Observation(BaseModel):
    """Represents an observation from the environment."""

    tool_call_id: str | None = None  # first observation may not be linked to any tool call
    contents: dict[str, Content]
    metadata: dict = Field(default_factory=dict)
    reward_info: dict = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """AgentOutput represents the output from LLM."""

    text: str
    actions: List[Action] = Field(default_factory=list)
    thoughts: List[str] = Field(default_factory=list)
    tokens: List[int] = Field(default_factory=list)
    logprobs: List[float] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class TraceStep(BaseModel):
    """A single step in the trace, consisting of an action or an observation."""

    observation: Optional[Observation] = None
    llm_output: Optional[AgentOutput] = None


class Trace(BaseModel):
    """
    Stores history of the previous interaction.

    Metadata contains info about agent, env and task.
    reward_info represents episode level reward data.
    """

    steps: List[TraceStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    reward_info: dict = Field(default_factory=dict)
