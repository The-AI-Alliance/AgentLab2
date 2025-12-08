import base64
import io
from typing import Any, Callable, Dict, List, Optional, Self

import litellm.utils
from litellm import Message
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


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

    name: str
    description: str
    parameters: dict

    @classmethod
    def from_function(cls, func: Callable) -> Self:
        """Create tool object from python function."""
        schema = litellm.utils.function_to_dict(func)
        return cls(**schema)

    def schema(self) -> dict[str, Any]:
        """Produce dict that could be passed as tool schema into LLM api."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Action(BaseModel):
    """
    A class representing a function call.

    Attributes:
        id (str): The identifier for the tool call.
        name (str): The name of the function being called.
        arguments (Any): The arguments to be passed to the function.
    """

    id: str | None = None
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


image_prefix = "data:image/png;base64,"


class Content(BaseModel):
    """Represents a piece of content in an observation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: str | Image.Image  # The actual content data

    @field_serializer("data")
    def serialize_image(self, data: str | Image.Image) -> str:
        if isinstance(data, str):
            return data
        byte_arr = io.BytesIO()
        data.save(byte_arr, format="PNG")
        encoded_image = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
        return f"{image_prefix}{encoded_image}"

    @field_validator("data", mode="before")
    @classmethod
    def deserialize_image(cls, v: str):
        if isinstance(v, str) and v.startswith(image_prefix):
            v = v[len(image_prefix) :]
            # Decode base64 string to bytes
            decoded_image = base64.b64decode(v)
            # Open bytes as PIL Image
            return Image.open(io.BytesIO(decoded_image))
        return v  # Return original value if not a string (e.g., already an Image object)


class Observation(BaseModel):
    """Represents an observation from the environment."""

    tool_call_id: str | None = None  # first observation may not be linked to any tool call
    contents: dict[str, Content]


AgentOutput = Message


class TraceStep(BaseModel):
    """A single step in the trace, consisting of an action or an observation."""

    observation: Optional[Observation] = None
    agent_output: Optional[AgentOutput] = None
    reward_info: dict = Field(default_factory=dict)


class Trace(BaseModel):
    """
    Stores history of the previous interaction.

    Metadata contains info about agent, env and task.
    reward_info represents episode level reward data.
    """

    steps: List[TraceStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    reward_info: dict = Field(default_factory=dict)
