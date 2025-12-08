import logging
import os

import ray
from pydantic import BaseModel

from agentlab2.agent import AgentConfig
from agentlab2.benchmark import Benchmark
from agentlab2.core import Trace
from agentlab2.environment import EnvironmentConfig
from agentlab2.run import AgentRun

logger = logging.getLogger(__name__)


class Experiment(BaseModel):
    name: str
    output_dir: str
    agent_config: AgentConfig
    env_config: EnvironmentConfig
    benchmark: Benchmark

    def create_runs(self):
        runs = [
            AgentRun(
                agent_config=self.agent_config,
                env_config=self.env_config,
                task=task,
            )
            for task in self.benchmark.tasks
        ]
        logger.info(f"Prepared {len(runs)} runs for experiment '{self.name}'")
        return runs

    def run_ray(self, n_cpus: int = 4, save_results: bool = True) -> list[Trace]:
        @ray.remote
        def run_single(agent_run: AgentRun) -> Trace:
            return agent_run.run()

        if not ray.is_initialized():
            ray.init(num_cpus=n_cpus)
        runs = self.create_runs()
        futures = [run_single.remote(run) for run in runs]
        traces = ray.get(futures)
        if save_results:
            self.save_traces(traces)
        return traces

    def run_sequential(self, save_results: bool = True) -> list[Trace]:
        self.benchmark.prepare()  # initialize tasks
        runs = self.create_runs()
        traces = [run.run() for run in runs]
        if save_results:
            self.save_traces(traces)
        return traces

    def save_traces(self, traces: list[Trace]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        traces_dir = os.path.join(self.output_dir, "traces")
        os.makedirs(traces_dir, exist_ok=True)
        for i, trace in enumerate(traces):
            n = trace.metadata.get("task_id", i)
            trace_path = os.path.join(traces_dir, f"trace_{n}.json")
            with open(trace_path, "w") as f:
                f.write(trace.model_dump_json(indent=2))
        logger.info(f"Saved {len(traces)} traces to {traces_dir}")
