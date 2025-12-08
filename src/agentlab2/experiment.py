"""Single run abstraction."""

import logging
import os

import ray
from pydantic import BaseModel

from agentlab2.agent import AgentConfig
from agentlab2.benchmark import Benchmark, Task
from agentlab2.core import Trace, TraceStep
from agentlab2.environment import EnvironmentConfig
from agentlab2.llm import LLM

logger = logging.getLogger(__name__)


class AgentRun(BaseModel):
    """Manages the execution of an agent on a specific task in an environment."""

    agent_config: AgentConfig
    env_config: EnvironmentConfig
    task: Task
    llm: LLM

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
        agent = self.agent_config.make(llm=self.llm, actions=env.actions)
        agent.reset()
        initial_observation, task_info = self.task.setup(env)

        trace = Trace(
            steps=[TraceStep(observation=initial_observation)],
            metadata={"task_id": self.task.id, "task_info": task_info},
        )

        while not self.task.finished() and not agent.finished():
            llm_output = agent.step(initial_observation)
            trace.steps.append(TraceStep(llm_output=llm_output))

            if llm_output.actions:
                initial_observation = env.step(llm_output.actions)

                if self.task.evaluate_per_step:
                    # validator will update obs.reward_info in-place
                    self.task.validate_step(env, llm_output.actions, initial_observation)
                trace.steps.append(TraceStep(observation=initial_observation))

        trace.reward_info = self.task.validate(env, trace)
        self.task.teardown(env)
        return trace


class Experiment(BaseModel):
    name: str
    output_dir: str
    agent_config: AgentConfig
    env_config: EnvironmentConfig
    benchmark: Benchmark
    llm: LLM

    def get_runs(self):
        runs = [
            AgentRun(
                agent_config=self.agent_config,
                env_config=self.env_config,
                llm=self.llm,
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
        runs = self.get_runs()
        futures = [run_single.remote(run) for run in runs]
        traces = ray.get(futures)
        if save_results:
            self.save_traces(traces)
        return traces

    def run_sequential(self, save_results: bool = True) -> list[Trace]:
        runs = self.get_runs()
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
