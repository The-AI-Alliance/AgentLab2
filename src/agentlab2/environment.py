"""Environment, Benchmark and Task abstractions."""

from abc import ABC, abstractmethod
from typing import Any, List

from pydantic import BaseModel

from agentlab2.core import Action, Content, EnvironmentOutput, Observation, ToolSchema

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
            Tuple of (goal string, dict with additional task info)
        """
        pass

    def teardown(self, env: E) -> None:
        """Optional clean up after task completion."""
        pass

    @abstractmethod
    def validate_task(self, env: E, obs: Observation) -> tuple[float, dict]:
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


class ToolboxEnv(Environment):
    """Environment that uses a collection of tools for interaction."""

    def __init__(self, task: Task, tools: list[Tool]):
        self.task = task
        self.tools = tools
        self._action_name_to_tool = {action.name: tool for tool in tools for action in tool.actions}

    def actions(self) -> list[ToolSchema]:
        actions_union = [action for tool in self.tools for action in tool.actions]
        return self.task.filter_actions(actions_union)

    def setup(self) -> EnvironmentOutput:
        for tool in self.tools:
            tool.reset()
        goal, info = self.task.setup(self)
        return EnvironmentOutput(obs=Observation.from_text(goal), info=info)

    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        actions = [action] if isinstance(action, Action) else action
        done = False
        reward = 0.0
        info = {}
        tool_results = []
        for action in actions:
            if self.is_stop_action(action):
                obs = Observation.from_text("Task finished by agent.")
                done = True
                break
            if action.name not in self._action_name_to_tool:
                raise ValueError(f"Action '{action.name}' is not supported by any tool in this environment.")
            tool = self._action_name_to_tool[action.name]
            tool_result = tool.execute_action(action)
            tool_results.append(tool_result)
        obs = self.merge_tool_results(tool_results)
        done = done or self.task.finished(self)
        if self.task.validate_per_step or done:
            reward, info = self.task.validate_task(self, obs)

        return EnvironmentOutput(obs=obs, reward=reward, info=info, done=done)

    def merge_tool_results(self, tool_results: list[Any]) -> Observation:
        merged_contents = []
        for result in tool_results:
            if isinstance(result, Observation):
                merged_contents += result.contents
            else:
                merged_contents.append(Content(data=result))
        return Observation(contents=merged_contents)

    def is_stop_action(self, action: Action) -> bool:
        return action.name == STOP_ACTION.name

    def close(self):
        self.task.teardown(self)
        for tool in self.tools:
            tool.close()
