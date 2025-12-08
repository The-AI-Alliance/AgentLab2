"""Single run abstraction."""

import logging

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
        env.reset()
        agent = self.agent_config.make(llm=self.llm, actions=env.actions)
        agent.reset()
        observation, task_info = self.task.setup(env)

        trace = Trace(
            steps=[TraceStep(observation=observation)],
            metadata={"task_info": task_info},
        )

        while not self.task.finished() and not agent.finished():
            llm_output = agent.step(observation)
            trace.steps.append(TraceStep(llm_output=llm_output))

            if llm_output.actions:
                observation = env.step(llm_output.actions)

                if self.task.evaluate_per_step:
                    # validator will update obs.reward_info in-place
                    self.task.validate_step(env, llm_output.actions, observation)
                trace.steps.append(TraceStep(observation=observation))

        trace.reward_info = self.task.validate(env, trace)
        self.task.teardown(env)
        return trace


class Experiment(BaseModel):
    name: str
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

    def run_ray(self, n_cpus: int = 4) -> list[Trace]:
        @ray.remote
        def run_single(agent_run: AgentRun) -> Trace:
            return agent_run.run()

        ray.init(num_cpus=n_cpus)
        runs = self.get_runs()
        futures = [run_single.remote(run) for run in runs]
        results = ray.get(futures)
        return results

    def run_sequential(self) -> list[Trace]:
        runs = self.get_runs()
        results = [run.run() for run in runs]
        return results
