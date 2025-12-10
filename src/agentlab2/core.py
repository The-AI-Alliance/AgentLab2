import base64
import io
from typing import Any, Callable, Dict, List, Self

import litellm.utils
from litellm import Message
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class ToolSchema(BaseModel):
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
    parameters: dict = Field(default_factory=dict)
    # examples: str | None = None # Maybe to-add

    @classmethod
    def from_function(cls, func: Callable) -> Self:
        """Create tool object from python function."""
        schema = litellm.utils.function_to_dict(func)
        return cls(**schema)

    def as_dict(self) -> dict[str, Any]:
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


class AgentOutput(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    llm_output: Message | None = None


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


class EnvironmentOutput(BaseModel):
    """Represents the result of an environment step."""

    obs: Observation
    reward: float = 0.0
    done: bool = False
    info: dict = Field(default_factory=dict)


class Trajectory(BaseModel):
    """
    Stores history of the previous interaction.

    Metadata contains info about agent, env and task.
    reward_info represents episode level reward data.
    """

    steps: List[EnvironmentOutput | AgentOutput] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def append(self, item: EnvironmentOutput | AgentOutput) -> None:
        self.steps.append(item)

    def last_env_step(self) -> EnvironmentOutput:
        for step in reversed(self.steps):
            if isinstance(step, EnvironmentOutput):
                return step
        raise ValueError("No EnvironmentOutput found in the trajectory.")

    def final_reward(self) -> float | None:
        return self.last_env_step().reward
