import json
import logging
import os

from pydantic import BaseModel
from termcolor import colored

from agentlab2.agent import AgentConfig
from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory
from agentlab2.environment import EnvironmentConfig, Task
from agentlab2.metrics.tracer import get_tracer


logger = logging.getLogger(__name__)


class AgentRun(BaseModel):
    """Manages the execution of an agent on a specific task in an environment."""

    id: int
    exp_name: str
    output_dir: str
    agent_config: AgentConfig
    env_config: EnvironmentConfig
    task: Task
    max_steps: int = 1000  # system-wide upper limit on steps
    _output_name: str = ""


    def run(self) -> Trajectory:
        """
        Main loop to run the agent on a single specific task.

        Returns:
            Trajectory containing the full history of the run.
        """
        tracer = get_tracer(self.exp_name)

        env = self.env_config.make(self.task)
        agent = self.agent_config.make(actions=env.actions())
        try:
            with tracer.episode(self.task.id, experiment=self.exp_name):
                env_output = env.setup()
                logger.info(colored(f"Initial env output: {env_output}", "blue"))
                trajectory = Trajectory(steps=[env_output], metadata={"task_id": self.task.id})
                self.save_trajectory(trajectory)
                steps = 0
                while not agent.finished() and not env_output.done and steps < self.max_steps:
                    with tracer.step(f"step_{steps}") as span:
                        agent_output = agent.step(env_output.obs)
                        steps += 1
                        self.save_step(agent_output)
                        logger.info(colored(f"Step {steps} Agent output: {agent_output}", "magenta"))
                        trajectory.append(agent_output)
                        for action in agent_output.actions:
                            env_output = env.step(action)
                            self.save_step(env_output)
                            logger.info(colored(f"Step {steps} Env output: {env_output}", "blue"))
                            trajectory.append(env_output)
                        span.set_attribute("agent_output", agent_output.model_dump_json())
        except Exception as e:
            logger.exception(f"Error during agent run: {e}")
            raise e
        finally:
            env.close()
            tracer.shutdown()
        return trajectory


    def save_trajectory(self, trajectory: Trajectory) -> None:
        """Save the trajectory to the output directory."""
        # TODO: Replace with tracing implementation
        traj_dir = f"{self.output_dir}/trajectories"
        os.makedirs(traj_dir, exist_ok=True)
        self._output_name = f"{traj_dir}/run{self.id}_task_{self.task.id}"
        with open(f"{self._output_name}.metadata.json", "w") as f:
            f.write(json.dumps(trajectory.metadata, indent=2))
        with open(f"{self._output_name}.jsonl", "a") as f:
            for step in trajectory.steps:
                f.write(step.model_dump_json() + "\n")
        logger.info(f"Saved trajectory for task {self.task.id} to {self._output_name}")


    def save_step(self, step: AgentOutput | EnvironmentOutput) -> None:
        """Append a single step to the trajectory JSONL file."""
        # TODO: Replace with tracing implementation
        if not self._output_name:
            raise ValueError("Trajectory path not set. Call save_trajectory first.")
        try:
            with open(f"{self._output_name}.jsonl", "a") as f:
                line = step.model_dump_json(serialize_as_any=True)
                f.write(f"{line}\n")
        except Exception as e:
            logger.exception(f"Error saving step to trajectory {self._output_name}: {e}")
            raise e
