"""Environment, Benchmark and Task abstractions."""

from typing import Any, List

from pydantic import BaseModel

from agentlab2.core import Action, ActionSchema, Observation


class Tool:
    """Base class for objects that can react on some actions"""

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        pass

    @property
    def actions(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        return []

    def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        raise NotImplementedError

    def close(self) -> None:
        """Clean up environment resources."""
        pass


class EnvironmentConfig(BaseModel):
    """Configuration for Environment."""

    def make(self) -> "Environment":
        return Environment()


class Environment:
    """Base class for environments that agents interact with."""

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        pass

    @property
    def actions(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        return []

    def step(self, action: Action) -> Observation:
        """Execute a single action and return the observation."""
        raise NotImplementedError("Subclasses must implement step()")

    def close(self) -> None:
        """Clean up environment resources."""
        pass
