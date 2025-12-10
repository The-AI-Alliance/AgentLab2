import logging
import os

import ray
from pydantic import BaseModel

from agentlab2.agent import AgentConfig
from agentlab2.benchmark import Benchmark
from agentlab2.core import Trajectory
from agentlab2.run import AgentRun

logger = logging.getLogger(__name__)


class Experiment(BaseModel):
    name: str
    output_dir: str
    agent_config: AgentConfig
    benchmark: Benchmark

    def create_runs(self):
        runs = [
            AgentRun(agent_config=self.agent_config, task=task, env_config=self.benchmark.env_config)
            for task in self.benchmark.tasks()
        ]
        logger.info(f"Prepared {len(runs)} runs for experiment '{self.name}'")
        return runs

    def save_config(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2, serialize_as_any=True))
        logger.info(f"Saved experiment config to {config_path}")

    def run_ray(self, n_cpus: int = 4, save_results: bool = True) -> list[Trajectory]:
        self.save_config()

        @ray.remote
        def run_single(agent_run: AgentRun) -> Trajectory:
            return agent_run.run()

        if not ray.is_initialized():
            ray_context = ray.init(
                num_cpus=n_cpus,
                dashboard_host="0.0.0.0",
                include_dashboard=True,
                log_to_driver=True,
                runtime_env={"working_dir": None},
            )
            logger.info(f"Ray initialized, dashboard at {ray_context.dashboard_url}")

        self.benchmark.setup()
        try:
            runs = self.create_runs()[:10]
            futures = [run_single.remote(run) for run in runs]
            trajectories = ray.get(futures)
            if save_results:
                self.save_trajectories(trajectories)
            self.print_stats(trajectories)
            return trajectories
        finally:
            ray.shutdown()
            self.benchmark.close()

    def run_sequential(self, save_results: bool = True, debug_limit: int | None = None) -> list[Trajectory]:
        self.save_config()
        self.benchmark.setup()
        try:
            runs = self.create_runs()
            if debug_limit is not None:
                logger.info(f"Running only first {debug_limit} runs")
                runs = runs[:debug_limit]
            trajectories = [run.run() for run in runs]
            if save_results:
                self.save_trajectories(trajectories)
            self.print_stats(trajectories)
            return trajectories
        finally:
            self.benchmark.close()

    def print_stats(self, trajectories: list[Trajectory]) -> None:
        if not trajectories:
            logger.info("No trajectories to compute stats")
            return

        total_steps = sum(len(trajectory.steps) for trajectory in trajectories)
        avg_steps = total_steps / len(trajectories)

        rewards = []
        for traj in trajectories:
            rewards.append(traj.final_reward())

        accuracy = sum(rewards) / len(rewards) if rewards else 0.0

        logger.info(f"Experiment '{self.name}' stats:")
        logger.info(f"  Total trajectories: {len(trajectories)}")
        logger.info(f"  Avg steps per trajectory: {avg_steps:.2f}")
        logger.info(f"  Accuracy (avg. final reward): {accuracy:.4f}")

    def save_trajectories(self, trajectories: list[Trajectory]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        save_dir = os.path.join(self.output_dir, "trajectories")
        os.makedirs(save_dir, exist_ok=True)
        for i, traj in enumerate(trajectories):
            n = traj.metadata.get("task_id", i)
            traj_path = os.path.join(save_dir, f"traj_{n}.json")
            with open(traj_path, "w") as f:
                f.write(traj.model_dump_json(indent=2))
        logger.info(f"Saved {len(trajectories)} trajectories to {save_dir}")
