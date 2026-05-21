"""Agent abstraction."""

from abc import ABC, abstractmethod

from cube.core import ActionSchema, Observation, ValidatedConfig

from cube_harness.core import AgentOutput


class AgentConfig(ValidatedConfig, ABC):
    """Configuration for creating an Agent."""

    @property
    def agent_name(self) -> str:
        """Human-readable name for this agent configuration, used in xray and logging."""
        return type(self).__name__

    @abstractmethod
    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "Agent":
        pass


class Agent(ABC):
    name: str
    description: str
    input_content_types: list[str]
    output_content_types: list[str]

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
