"""Task and TaskConfig for tau2_cube.

Task owns a Tool instance and implements the episode loop:
  reset()    → (Observation, info dict)   called once before each episode
  evaluate() → (reward: float, info dict) called after every step
  finished() → bool                       optional early-termination check

For per-task data, prefer typed Pydantic fields over stringly-typed dicts:
  - Lightweight, eager-loaded values (always available, ship in
    task_metadata.json) → declare them on a `TaskMetadata` subclass.
  - Heavy, lazy values populated by `BenchmarkConfig.install()` (problem
    statements, patches, archives, …) → declare them on a
    `TaskExecutionInfo` subclass and surface them via `Task.execution_info`.

TaskConfig is the serialisable boundary that crosses process/network lines.
It carries its own metadata (stamped by BenchmarkConfig.get_task_configs()
on the driver) so workers never need to import the owning BenchmarkConfig.
Implement make() using self.metadata directly; populate execution_info from
self.load_task_execution_info() when your cube ships heavy data.
"""

from typing import Any, cast

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskExecutionInfo  # noqa: F401  (TaskExecutionInfo used in commented CubeExecutionInfo example below)
from tau2_cube.tool import CubeTool, CubeToolConfig


# Optional: declare a TaskExecutionInfo subclass when your cube ships heavy
# per-task data via BenchmarkConfig.install(). Cubes with no heavy data
# can leave this commented out and keep `Task.execution_info = None`.
#
# class CubeExecutionInfo(TaskExecutionInfo):
#     """Heavy per-task data populated on the worker by CubeTaskConfig.make()."""
#
#     instruction: str
#     # patch: str = ""
#     # ...


from cube.task import TaskMetadata
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task as Tau2Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation


class CubeTaskMetadata(TaskMetadata):
    domain: str = "mock"
    tau2_task: Tau2Task  # the full tau2 Task (scenario, initial_state, criteria)


class CubeTask(Task[CubeTaskMetadata]):
    """One episode: interact with CubeTool to satisfy the goal.

    Read static per-task config from `self.metadata.<field>` (typed) and
    heavy lazy data — when used — from `self.execution_info.<field>` (typed).
    """

    @property
    def tool(self) -> CubeTool:
        """Narrow the inherited AbstractTool to CubeTool (typing only)."""
        return cast(CubeTool, super().tool)

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Initialise the env + user simulator and return the opening observation.

        The agent is NOT shown the customer's private instructions — it must learn
        the request by conversing via send_message_to_user. The opening observation
        is the domain policy (what a real CS agent has), not the task brief.
        """
        self.tool.reset()
        init = self.metadata.tau2_task.initial_state
        if init is not None:  # apply task's starting state
            self.tool._env.set_state(
                initialization_data=init.initialization_data,
                initialization_actions=init.initialization_actions,
                message_history=init.message_history or [],
            )
        # Wire the user simulator for this episode (LLM-free until first message).
        self.tool.start_user_session(self.metadata.tau2_task.user_scenario)
        obs = Observation.from_text(
            f"You are a customer service agent for the {self.metadata.domain} domain. A customer "
            f"has contacted you. Greet them and use send_message_to_user to find out what they need, "
            f"then use the available tools to help. Follow this policy:\n\n{self.tool._env.get_policy()}"
        )
        return obs, {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        """Score the episode with tau2's evaluator over the recorded conversation.

        Delegates to ``evaluate_simulation``, which computes the reward per the
        task's ``reward_basis`` (DB / COMMUNICATE / NL_ASSERTION / ENV_ASSERTION /
        ACTION) from ``self.tool._messages``. termination_reason=AGENT_STOP since
        evaluate() runs only at episode end; if the agent never stopped, tau2 would
        score it 0 (premature) — we treat reaching here as a completed attempt.
        """
        task = self.metadata.tau2_task
        sim = SimulationRun(
            id=task.id,
            task_id=task.id,
            timestamp="",
            start_time="",
            end_time="",
            duration=0.0,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=self.tool._messages,
        )
        reward_info = evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=self.metadata.domain,
        )
        return reward_info.reward, reward_info.model_dump(mode="json")

    def finished(self, obs: Observation | None = None) -> bool:
        """Episodes end when the agent emits STOP or hits max_steps — never on a
        transient reward. tau2 scores the full conversation once, at the end."""
        return False


class CubeTaskConfig(TaskConfig):
    """Serialisable factory that produces a CubeTask.

    Self-contained: ``self.metadata`` carries the (possibly subclassed)
    TaskMetadata, stamped onto the config by
    ``CubeBenchmarkConfig.get_task_configs()`` on the driver.
    """

    def verify_installed(self) -> None:
        """Optional fail-fast check before workers attempt to load heavy data.

        Default base implementation is a no-op. Override when your cube ships
        heavy data via ``BenchmarkConfig.install()`` so misconfigured workers
        error early with an actionable message instead of timing out::

            cache_dir = type(self).task_execution_cache_dir()
            if not cache_dir.exists() or not any(cache_dir.iterdir()):
                raise RuntimeError(
                    f"Run `cube install tau2-cube` first."
                )
        """

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> CubeTask:
        # By convention, fail fast if this worker is misconfigured.
        self.verify_installed()
        # If your cube uses heavy per-task data, hydrate it here:
        # exec_info = CubeExecutionInfo.model_validate(
        #     self.load_task_execution_info()
        # )
        return CubeTask(
            metadata=self.metadata,
            execution_info=None,  # set to ``exec_info`` if you populate it above
            tool_config=self.tool_config or CubeToolConfig(domain=self.metadata.domain),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
