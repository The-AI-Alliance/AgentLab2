"""Agent abstraction."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cube.core import ActionSchema, Observation, StepError, ValidatedConfig
from pydantic import Field

from cube_harness.core import AgentOutput

if TYPE_CHECKING:
    from cube.tool import AbstractAsyncTool

    from cube_harness.streamer import EventStreamer

logger = logging.getLogger(__name__)


def apply_description_overrides(encoded_tools: list[dict], overrides: dict[str, str]) -> None:
    """Replace tool-schema descriptions in place from ``{action_name: description}``.

    ``encoded_tools`` are dicts in LLM tool-schema form (``ActionSchema.as_dict()``).
    Raises ``ValueError`` on a key that matches no tool in ``encoded_tools`` — this
    catches typos and keys left stale after an action was renamed. No-op when empty.
    """
    if not overrides:
        return
    by_name = {t["function"]["name"]: t for t in encoded_tools}
    unknown = set(overrides) - set(by_name)
    if unknown:
        raise ValueError(
            f"description_overrides target unknown actions {sorted(unknown)}; the action space has {sorted(by_name)}"
        )
    for name, description in overrides.items():
        by_name[name]["function"]["description"] = description


class AgentConfig(ValidatedConfig, ABC):
    """Configuration for creating an Agent."""

    description_overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Experiment-time overrides for action descriptions, keyed by action name "
            "(the unique name the LLM sees). Replaces the docstring-derived description "
            "before the tool schema is built — a toggleable knob for testing better "
            "wording without editing the tool. A proven override graduates into the "
            "tool's docstring at the source via a PR."
        ),
    )

    @property
    def agent_name(self) -> str:
        """Human-readable name for this agent configuration, used in xray and logging."""
        return type(self).__name__

    @abstractmethod
    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "Agent":
        """Instantiate the live Agent from this config + the task's action_set.

        Called by Episode after task.reset; subclasses wire in the
        action schemas, model handle, and any per-task overrides."""


class Agent(ABC):
    name: str
    description: str
    input_content_types: list[str]
    output_content_types: list[str]

    def __init__(self, config: AgentConfig):
        self.config = config
        # Set by Episode via `attach_recorder(recorder)` before `run`.
        # Subclasses that hold LLM(s) override `attach_recorder` to
        # propagate it down (Genny does this for its `self.llm`).
        # Agent-side reads of `self._recorder.budget` for graceful
        # self-stop and prompt injection rely on this being set.
        self._recorder: "EventStreamer | None" = None

    def attach_recorder(self, recorder: "EventStreamer") -> None:
        """Wire the recorder into this agent and its event-producing
        children (LLMs, etc). Called by `Episode` once, before `run()`.

        **Two reasons agent authors override this:**

        1. **Propagate to held LLMs** so `LLM.call(...)` auto-emits
           `LLMCallEvent`. Multi-LLM agents call `.attach_recorder()` on
           each LLM they want recorded; LLMs whose calls should NOT
           appear in the trajectory are simply left unattached::

               class MyAgent(Agent):
                   def attach_recorder(self, recorder):
                       super().attach_recorder(recorder)
                       self.llm.attach_recorder(recorder)
                       # self.scratch_llm intentionally NOT attached

        2. **Reach the live `Budget`** for graceful self-stop /
           prompt injection. The default impl stashes the recorder on
           `self._recorder`, exposing the live budget at
           `self._recorder.budget`::

               def step(self, obs):
                   budget = self._recorder.budget
                   if budget is not None and budget.exhausted:
                       # Soft self-stop — friendly path that returns a
                       # STOP_ACTION rather than letting MonitoredTool
                       # raise BudgetExceeded mid-call.
                       return AgentOutput(actions=[STOP_ACTION])
                   # Inject budget summary into the prompt every K turns
                   # so the LLM can plan against remaining budget:
                   budget_msg = str(budget) if budget.turns % 10 == 0 else None

           Read-only fields available on Budget: `turns`, `tool_calls`,
           `cost_usd`, `prompt_tokens`, `completion_tokens`, plus the
           `exhausted` property and `__str__` (concise human-readable
           summary of all configured caps and current usage).
        """
        self._recorder = recorder

    @abstractmethod
    def step(self, obs: Observation) -> AgentOutput:
        """Perform one agent turn against the observation.

        Returns the minimal `AgentOutput(actions, error)` the framework
        needs to dispatch through `env_tool` next. LLM calls inside
        `step()` auto-emit `LLMCallEvent`s through the attached
        recorder (see `LLM.attach_recorder`) — `step()` does NOT
        bundle LLM calls into its return value.
        """

    async def run(
        self,
        initial_obs: Observation,
        env_tool: "AbstractAsyncTool",
    ) -> None:
        """Default gym-style loop on top of `self.step` — Episode's canonical entry point.

        Sync `step()` agents get this for free. Override `run` directly
        for parallel tool dispatch, async LLM, or streaming (see
        `GennyParallel`).

        `env_tool` is uniformly `AbstractAsyncTool`: Episode adapts sync
        tools at the boundary. The recorder is attached out-of-band via
        `attach_recorder()` before `run` is called; LLM/tool events
        auto-emit. `self._recorder.budget` is available for self-stop.

        Termination:
          * Graceful: `step` returns empty actions with no error.
          * `TaskDone` from a MonitoredTool (task `finished()` or
            STOP_ACTION) — propagates; do NOT catch BaseException.
          * `BudgetExceeded` from a MonitoredTool — propagates.
        """
        obs = initial_obs
        while True:
            # Sync body under async signature — debugable on the main
            # thread. Override `run` for true async/concurrent work.
            agent_output = self.step(obs)
            # Bump turns AFTER step() (so its LLM calls emit) and BEFORE
            # dispatch (so a turn that crosses max_turns can't dispatch).
            if self._recorder is not None:
                self._recorder.on_step()
            if agent_output.error is not None:
                raise RuntimeError(f"Agent step returned error: {agent_output.error.exception_str}")
            if not agent_output.actions:
                return
            # Sequential dispatch. Multi-action agents needing fan-out
            # / result merging override `run` (see GennyParallel).
            last_obs: Observation | None = None
            for action in agent_output.actions:
                result = await env_tool.execute_action(action)
                if isinstance(result, StepError):
                    raise RuntimeError(f"Tool dispatch returned StepError: {result.exception_str}")
                last_obs = result
            if last_obs is not None:
                obs = last_obs

    def __repr__(self) -> str:
        return self.config.model_dump_json(indent=2, serialize_as_any=True)
