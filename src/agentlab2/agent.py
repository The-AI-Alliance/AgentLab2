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
        Perform a step given an observation and return the agent's output with actions.
        """
        pass

    def __repr__(self) -> str:
        return self.config.model_dump_json(indent=2, serialize_as_any=True)
