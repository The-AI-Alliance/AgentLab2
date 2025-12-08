import json
import logging

from pydantic import BaseModel
from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.benchmark import Task
from agentlab2.core import Action, AgentOutput, Trace, TraceStep
from agentlab2.environment import EnvironmentConfig

logger = logging.getLogger(__name__)


class AgentRun(BaseModel):
    """Manages the execution of an agent on a specific task in an environment."""

    agent_config: AgentConfig
    env_config: EnvironmentConfig
    task: Task

    def run(self) -> Trace:
        """
        Main loop to run the agent on a single specific task.

        Returns:
            Trace containing the full history of the run.
        """
        env = self.env_config.make()
        try:
            env.reset()
            trace = self._run_with_env(env)
        except Exception as e:
            logger.exception(f"Error during agent run: {e}")
            raise e
        finally:
            env.close()
        return trace

    def _run_with_env(self, env):
        agent = self.agent_config.make(actions=env.actions)
        agent.reset()
        obs, task_info = self.task.setup(env)
        logger.info(colored(f"Initial observation: {obs.model_dump_json(indent=2)}", "blue"))
        logger.info(f"Task info: {task_info}")
        trace = Trace(
            steps=[TraceStep(observation=obs)],
            metadata={"task_id": self.task.id, "task_info": task_info},
        )
        steps = 0
        while not self.task.finished(steps) and not agent.finished() and not env.finished():
            agent_output = agent.step(obs)
            steps += 1
            logger.info(colored(f"Step {steps} Agent output: {agent_output.model_dump_json(indent=2)}", "magenta"))
            trace.steps.append(TraceStep(agent_output=agent_output))
            actions = self._actions_from_output(agent_output)
            # TODO: support parallel actions if the environment allows it
            for action in actions:
                obs = env.step(action)
                obs = self.task.obs_postprocess(obs)
                logger.info(colored(f"Step {steps} Observation: {obs.model_dump_json(indent=2)}", "blue"))
                if self.task.validate_per_step:
                    reward_info = self.task.validate_step(action, obs)
                else:
                    reward_info = {}
                trace.steps.append(TraceStep(observation=obs, reward_info=reward_info))
        trace.reward_info = self.task.validate(trace)
        self.task.teardown()
        return trace

    def _actions_from_output(self, agent_output: AgentOutput) -> list[Action]:
        actions = []
        if hasattr(agent_output, "tool_calls") and agent_output.tool_calls:
            for tc in agent_output.tool_calls:
                arguments = tc.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON arguments in tool call: {arguments}")
                if tc.function.name is None:
                    raise ValueError("Tool call must have a function name.")
                actions.append(Action(id=tc.id, name=tc.function.name, arguments=arguments))
        return actions
