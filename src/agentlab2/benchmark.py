from pydantic import BaseModel, Field

from agentlab2.core import Action, ActionSchema, Observation, Trace
from agentlab2.environment import Environment


class Task(BaseModel):
    """Represents a task that an agent must complete in an environment."""

    id: str
    evaluate_per_step: bool = False

    def setup(self, environment: Environment) -> tuple[Observation, dict]:
        """
        Set up the task in the given environment.

        Returns:
            Tuple of (list of initial observations, dict with additional task info)
        """
        raise NotImplementedError

    def teardown(self, environment: Environment) -> None:
        """Optional clean up after task completion."""
        pass

    def validate(self, environment: Environment, trace: Trace) -> dict:
        """Validate the whole trace and state of the env at the end of the run."""
        raise NotImplementedError

    def validate_step(
        self,
        environment: Environment,
        actions: list[Action],
        observation: Observation,
    ) -> None:
        """
        If evaluate_per_step=True this will be called to produce reward for each step.
        Updates observation.reward_info in-place.
        """
        pass

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Allows the task to whitelist subset of all the actions provided by the environment."""
        return actions

    def cheat(self):
        """
        Solve the task using a pre-defined solution (optional).
        """
        raise NotImplementedError

    def obs_postprocess(self, obs: dict) -> dict:
        return obs

    def finished(self) -> bool:
        """Check if the task is finished."""
        return False


class Benchmark(BaseModel):
    """Represents a benchmark consisting of multiple tasks and an environment."""

    metadata: dict = Field(default_factory=dict)
    tasks: list[Task] = Field(default_factory=list)
