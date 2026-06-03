"""Agent abstraction."""

import asyncio
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
        """Default gym-style loop on top of `self.step` — the canonical
        entry point invoked by `Episode` (RFC `agent-owns-loop`).

        Sync `step()` agents get this for free — they never write a
        line of async code. Agents that want parallel tool calls,
        async LLM dispatch, or streaming observability override this
        method instead.

        The `env_tool` parameter is always `AbstractAsyncTool` — Episode
        wraps sync tools in a thin `asyncio.to_thread`-based adapter
        at the boundary so the agent's view is uniformly async. One
        `await env_tool.execute_action(action) -> Observation | StepError`
        call site regardless of the underlying tool's sync/async nature.

        The recorder is NOT a parameter — Episode attaches it via
        `agent.attach_recorder(recorder)` BEFORE calling `run()`. LLM
        calls inside `step()` auto-emit; tool calls auto-emit; the
        agent code never touches the recorder directly. For
        introspection (e.g. budget self-stop), `self._recorder.budget`
        is available.

        Termination:

          * Graceful: `self.step` returns empty actions with no error.
          * `TaskDone` raised by a `MonitoredTool` when the task's
            `finished()` check returned True OR the agent emitted the
            STOP_ACTION sentinel — propagates to Episode and is captured
            in its outer `except`. Agents must NOT catch BaseException.
          * `BudgetExceeded` raised by a monitored tool — same
            propagation pattern.
        """
        obs = initial_obs
        while True:
            agent_output = await asyncio.to_thread(self.step, obs)
            # Bump `budget.turns` (one agent step) and enforce caps —
            # AFTER step() so LLM calls inside step() emit first, but
            # BEFORE dispatching any actions so a turn that crosses
            # `max_turns` doesn't get to dispatch.
            if self._recorder is not None:
                self._recorder.on_step()
            # Graceful done: empty actions AND no error.
            # If error is set, raise via StepError so Episode tags the
            # episode FAILED. A bare return would silently look like
            # success.
            if agent_output.error is not None:
                raise RuntimeError(f"Agent step returned error: {agent_output.error.exception_str}")
            if not agent_output.actions:
                return
            # Dispatch each action sequentially. The default loop is
            # ONE action per step in practice (Genny.step returns 1
            # action even with parallel_tool_calls=False); for multi-
            # action turns, agents override `run` (see GennyParallel)
            # so they can fan out and merge observations correctly.
            # We dispatch all N here but accumulate observations into
            # the next prompt by way of MonitoredTool side-effects;
            # an env_tool.execute_action returning StepError aborts
            # the run — Episode finalizes with the failure recorded
            # via the MonitoredTool's emit + Episode's outer except.
            last_obs: Observation | None = None
            for action in agent_output.actions:
                result = await env_tool.execute_action(action)
                if isinstance(result, StepError):
                    # Surface as failure so Episode records it (record_failure
                    # via the outer except wraps it in AgentErrorEvent).
                    raise RuntimeError(f"Tool dispatch returned StepError: {result.exception_str}")
                last_obs = result
            # Feed the LAST observation back. Multi-action agents that
            # need result merging should override `run` (GennyParallel
            # does this via `_merge_results`).
            if last_obs is not None:
                obs = last_obs

    def __repr__(self) -> str:
        return self.config.model_dump_json(indent=2, serialize_as_any=True)
