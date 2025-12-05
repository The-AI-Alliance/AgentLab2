"""Environment, Benchmark and Task abstractions."""

from typing import List, Tuple

from pydantic import BaseModel

from agentlab2.core import Action, ActionSchema, Observation, Trace


class Tool(BaseModel):
    """Base class for objects that can react on some actions"""

    metadata: dict = {}

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


class Environment(BaseModel):
    """Base class for environments that agents interact with."""

    metadata: dict = {}

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        pass

    @property
    def actions(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        return []

    def step(self, actions: list[Action]) -> Observation:
        """Execute a single action and return the observation."""
        raise NotImplementedError("Subclasses must implement step()")

    def close(self) -> None:
        """Clean up environment resources."""
        pass


class Task(BaseModel):
    """Represents a task that an agent must complete in an environment."""

    id: str
    evaluate_per_step: bool = False

    def setup(self, environment: Environment) -> Tuple[List[Observation], dict]:
        """
        Set up the task in the given environment.

        Returns:
            Tuple of (list of initial observations, dict with additional task info)
        """
        return [], {}

    def teardown(self, environment: Environment) -> None:
        """Clean up after task completion."""
        pass

    def validate(self, environment: Environment, trace: "Trace") -> dict:
        """Validate the whole trace and state of the env at the end of the run."""
        return {}

    def validate_step(
        self,
        environment: Environment,
        actions: List[Action],
        observation: Observation,
    ) -> None:
        """
        If evaluate_per_step=True this will be called to produce reward for each step.
        Updates observation.reward_info in-place.
        """
        pass

    def filter_actions(self, actions: List[ActionSchema]) -> List[ActionSchema]:
        """Allows the task to whitelist subset of all the actions provided by the environment."""
        return actions

    def finished(self) -> bool:
        """Check if the task is finished."""
        return False


class Benchmark(BaseModel):
    """Represents a benchmark consisting of multiple tasks and an environment."""

    name: str
    tasks: List[Task]
    environment: Environment

    def initialize(self) -> None:
        """
        Loads data for tasks from the storage, prepares the environment,
        creates list of AgentRun objects to run.
        """
        pass
