"""Agent abstraction."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from agentlab2.core import AgentOutput, Observation


class AgentConfig(BaseModel, ABC):
    """Configuration for creating an Agent."""

    @abstractmethod
    def make(self, **kwargs) -> "Agent":
        pass


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    def step(self, obs: Observation) -> AgentOutput:
        """
        Take a step given an observation.

        Returns:
            Actions to perform.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def finished(self) -> bool:
        """Optional check if the agent has finished its task."""
        return False

    ## TODO: Add a good __repr__
