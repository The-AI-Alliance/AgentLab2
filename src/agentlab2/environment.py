"""Environment, Benchmark and Task abstractions."""

from abc import ABC, abstractmethod
from typing import Any, List

from pydantic import BaseModel

from agentlab2.core import Action, EnvironmentOutput, Observation, ToolSchema

STOP_ACTION = ToolSchema(name="final_step", description="Stop the task execution.")


class Tool:
    """Base class for objects that can react on some actions"""

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        pass

    @property
    def actions(self) -> List[ToolSchema]:
        """Returns list of actions supported by that environment."""
        return []

    def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        raise NotImplementedError

    def close(self) -> None:
        """Clean up environment resources."""
        pass


class EnvironmentConfig(BaseModel, ABC):
    """Configuration for Environment."""

    @abstractmethod
    def make(self, task: "Task") -> "Environment":
        pass


class Environment(ABC):
    """Base class for environments that agents interact with."""

    def __init__(self, task: "Task", *args, **kwargs) -> None:
        super().__init__()
        self.task: Task = task

    @abstractmethod
    def setup(self) -> EnvironmentOutput:
        """Set up the environment before starting a task."""
        pass

    def actions(self) -> List[ToolSchema]:
        """Returns list of actions supported by that environment."""
        return []

    @abstractmethod
    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        """Execute a single or multiple actions and return the observation."""
        pass

    def close(self) -> None:
        """Optional clean up environment resources."""
        pass


class Task[E: Environment](BaseModel, ABC):
    """Represents a task that an agent must complete in an environment."""

    id: str
    validate_per_step: bool = False

    @abstractmethod
    def setup(self, env: E) -> tuple[str, dict]:
        """
        Set up the task in the given environment.

        Returns:
            Tuple of (list of initial observations, dict with additional task info)
        """
        pass

    def teardown(self, env: E) -> None:
        """Optional clean up after task completion."""
        pass

    @abstractmethod
    def validate_task(self, env: E, obs: Observation, action: Action) -> tuple[float, dict]:
        """Validate the whole trajectory and state of the env at the end of the run."""
        pass

    @abstractmethod
    def filter_actions(self, actions: list[ToolSchema]) -> list[ToolSchema]:
        """Allows the task to whitelist subset of all the actions provided by the environment."""
        pass

    def cheat(self):
        """
        Solve the task using a pre-defined solution (optional).
        """
        raise NotImplementedError

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Optional post-processing of observation before returning it to the agent."""
        return obs

    def finished(self, env: E) -> bool:
        """Check if the task is finished."""
        return False
