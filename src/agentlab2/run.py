import logging

from pydantic import BaseModel
from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.core import Trajectory
from agentlab2.environment import EnvironmentConfig, Task

logger = logging.getLogger(__name__)


class AgentRun(BaseModel):
    """Manages the execution of an agent on a specific task in an environment."""

    agent_config: AgentConfig
    env_config: EnvironmentConfig
    task: Task
    max_steps: int = 1000  # system-wide upper limit on steps

    def run(self) -> Trajectory:
        """
        Main loop to run the agent on a single specific task.

        Returns:
            Trace containing the full history of the run.
        """
        env = self.env_config.make(self.task)
        agent = self.agent_config.make(actions=env.actions)
        try:
            env_output = env.setup()
            logger.info(colored(f"Initial env output: {env_output.model_dump_json(indent=2)}", "blue"))
            trajectory = Trajectory(steps=[env_output], metadata={"task_id": self.task.id})
            steps = 0
            while not agent.finished() and not env_output.done and steps < self.max_steps:
                agent_output = agent.step(env_output.observation)
                steps += 1
                logger.info(colored(f"Step {steps} Agent output: {agent_output.model_dump_json(indent=2)}", "magenta"))
                trajectory.append(agent_output)
                for action in agent_output.actions:
                    env_output = env.step(action)
                    logger.info(colored(f"Step {steps} Env output: {env_output.model_dump_json(indent=2)}", "blue"))
                    trajectory.append(env_output)
            trajectory.reward_info = self.task.validate(trajectory)
            self.task.teardown()
        except Exception as e:
            logger.exception(f"Error during agent run: {e}")
            raise e
        finally:
            env.close()
        return trajectory
