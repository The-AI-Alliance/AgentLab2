"""Agent abstraction."""

from typing import List

from pydantic import BaseModel, Field

from agentlab2.core import ActionSchema, Trace
from agentlab2.llm import LLMOutput


class Agent(BaseModel):
    """
    ACP based definition.

    See: https://agentcommunicationprotocol.dev/core-concepts/agent-manifest
    """

    name: str
    description: str | None = None
    input_content_types: List[
        str
    ]  # values ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: List[str]
    actions: List[ActionSchema]
    metadata: dict = Field(default_factory=dict)

    def reset(self) -> None:
        """Reset the agent state."""
        pass

    def step(self, trace: Trace) -> LLMOutput:
        """
        Agent given not the last observation, but the full trace of all previous steps.

        Returns:
            LLMOutput containing the agent's response and any tool calls.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def finished(self) -> bool:
        """Check if the agent has finished its task."""
        return False
