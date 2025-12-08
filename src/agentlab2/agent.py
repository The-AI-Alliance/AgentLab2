"""Agent abstraction."""

from pydantic import BaseModel

from agentlab2.core import Observation
from agentlab2.llm import LLMOutput


class AgentConfig(BaseModel):
    """Configuration for creating an Agent."""

    def make(self, **kwargs) -> "Agent":
        return Agent(self, **kwargs)


class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config

    def reset(self) -> None:
        """Reset the agent state."""
        pass

    def step(self, observation: Observation) -> LLMOutput:
        """
        Take a step given an observation.

        Returns:
            Actions to perform.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def finished(self) -> bool:
        """Check if the agent has finished its task."""
        return False
