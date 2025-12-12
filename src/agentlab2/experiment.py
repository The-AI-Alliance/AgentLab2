import logging
import os

from pydantic import BaseModel, Field

from agentlab2.agent import AgentConfig
from agentlab2.benchmark import Benchmark
from agentlab2.core import Trajectory
from agentlab2.run import AgentRun

logger = logging.getLogger(__name__)


class ExpResult(BaseModel):
    tasks_num: int
    config: dict = Field(default_factory=dict)
    trajectories: dict[str, Trajectory] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)


class Experiment(BaseModel):
    name: str
    output_dir: str
    agent_config: AgentConfig
    benchmark: Benchmark

    @property
    def config(self) -> dict:
        return self.model_dump(serialize_as_any=True)

    def create_runs(self):
        runs = [
            AgentRun(
                id=i,
                exp_name=self.name,
                output_dir=self.output_dir,
                agent_config=self.agent_config,
                task=task,
                env_config=self.benchmark.env_config,
            )
            for i, task in enumerate(self.benchmark.tasks())
        ]
        logger.info(f"Prepared {len(runs)} runs for experiment '{self.name}'")
        return runs

    def save_config(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        config_json = self.model_dump_json(indent=2, serialize_as_any=True)
        with open(config_path, "w") as f:
            f.write(config_json)
        logger.info(f"Saved experiment config to {config_path}")

    def print_stats(self, results: ExpResult) -> None:
        if not results.trajectories:
            logger.info("No trajectories to compute stats")
            return

        total_steps = sum(len(trajectory.steps) for trajectory in results.trajectories.values())
        avg_steps = total_steps / len(results.trajectories)

        rewards = []
        for traj in results.trajectories.values():
            rewards.append(traj.final_reward())

        accuracy = sum(rewards) / len(rewards) if rewards else 0.0

        logger.info(f"Experiment '{self.name}' stats:")
        logger.info(f"  Total trajectories: {len(results.trajectories)}")
        logger.info(f"  Avg steps per trajectory: {avg_steps:.2f}")
        logger.info(f"  Accuracy (avg. final reward): {accuracy:.4f}")
        logger.info(f"  Failed tasks: {len(results.failures)}")
        logger.info(f"Saved to: {self.output_dir}")
