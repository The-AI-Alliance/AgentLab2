"""Single run abstraction."""

from typing import Self
from pydantic import BaseModel

from agentlab2.agent import Agent
from agentlab2.core import Trace, TraceStep
from agentlab2.environment import Environment, Task


class Config(BaseModel):
    """Configuration for creating an AgentRun."""

    environment: dict
    agent: dict


class AgentRun(BaseModel):
    """Manages the execution of an agent on a specific task in an environment."""

    agent: Agent
    task: Task
    environment: Environment

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_config(cls, task: Task, config: Config) -> Self:
        """Create an AgentRun from a task and configuration."""
        environment = Environment(**config.environment)
        actions = task.filter_actions(environment.actions)
        agent = Agent(**config.agent, actions=actions)
        return cls(agent=agent, task=task, environment=environment)

    def run(self) -> Trace:
        """
        Main loop to run the agent on a single specific task.

        Returns:
            Trace containing the full history of the run.
        """
        self.agent.reset()
        self.environment.reset()
        observations, task_info = self.task.setup(self.environment)

        trace = Trace(
            steps=[TraceStep(observation=o) for o in observations],
            metadata={
                "agent_info": self.agent.metadata,
                "env_info": self.environment.metadata,
                "task_info": task_info,
            },
        )

        while not self.task.finished() and not self.agent.finished():
            llm_output = self.agent.step(trace)
            trace.steps.append(TraceStep(llm_output=llm_output))

            if llm_output.actions:
                observation = self.environment.step(llm_output.actions)

                if self.task.evaluate_per_step:
                    # validator will update obs.reward_info in-place
                    self.task.validate_step(
                        self.environment, llm_output.actions, observation
                    )
                trace.steps.append(TraceStep(observation=observation))

        trace.reward_info = self.task.validate(self.environment, trace)
        self.task.teardown(self.environment)
        return trace
