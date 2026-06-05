"""Agent abstraction."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cube.core import ActionSchema, Observation, StepError, ValidatedConfig
from pydantic import Field

from cube_harness.core import AgentOutput

if TYPE_CHECKING:
    from cube.tool import AbstractTool

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

    parallel_actions: bool = Field(
        default=False,
        description=(
            "If True, `Agent.run` dispatches to `_arun` (async body, fans out the "
            "N actions per step via `asyncio.gather` over `MonitoredTool.async_execute_action`). "
            "If False (default), dispatches to `_run` (sync body, sequential dispatch via "
            "`MonitoredTool.execute_action` — no `await`, single-stack pdb). Pair with "
            "`LLMConfig.parallel_tool_calls=True` so the model emits multiple tool calls "
            "per turn; otherwise parallel dispatch fans out over a one-element list and "
            "wins nothing."
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
                   budget_msg = str(budget) if budget.agent_steps % 10 == 0 else None

           Read-only fields available on Budget: `agent_steps`, `tool_calls`,
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

    def run(
        self,
        initial_obs: Observation,
        env_tool: "AbstractTool",
    ) -> None:
        """Episode's canonical entry point — dispatches to one of two loop bodies.

        Selection is by `self.config.parallel_actions`:

          * `False` (default) → `_run` (sync body, called directly). No event loop
            on the calling thread — sync tools (Playwright, shell) work natively and
            pdb lands in a single stack with no thread hops.

          * `True` → `_arun` (async body). Spins its own `asyncio.run` scoped to the
            parallel gather — the event loop lives only here, not in Episode.

        `env_tool` is any `AbstractTool` — `Tool.execute_action` and
        `Tool.async_execute_action` both handle sync and async `@tool_action`
        methods, bridging when caller and method differ. The agent picks the
        dispatch shape; the tool doesn't need to know.

        The recorder is attached out-of-band via `attach_recorder()` before `run` is
        called; LLM/tool events auto-emit. `self._recorder.budget` is available for
        self-stop.

        Override `_run` / `_arun` to customize loop behavior; override `run` only
        when you need a fundamentally different dispatch (streaming LLM, custom
        backoff, …).

        Termination (shared by both bodies):
          * Graceful: `step` returns empty actions with no error.
          * `TaskDone` from a MonitoredTool (task `finished()` or STOP_ACTION).
          * `BudgetExceeded` from a MonitoredTool.
        """
        if self.config.parallel_actions:
            # Event loop scoped to just the parallel gather — not the whole episode.
            asyncio.run(self._arun(initial_obs, env_tool))
        else:
            self._run(initial_obs, env_tool)

    def _run(
        self,
        initial_obs: Observation,
        env_tool: "AbstractTool",
    ) -> None:
        """Sync gym-style loop — sequential, single-threaded, no `await`.

        Use for sync `step()` agents driving sync tools (the 99% case).
        `env_tool.execute_action(action)` runs the inner sync tool
        directly on the calling thread — pdb lands in the tool body
        with a single stack, no thread-pool worker, no async hop.

        Override to customize sync loop behavior. For parallel tool
        dispatch, see `_arun` + `AgentConfig.parallel_actions=True`.
        """
        obs = initial_obs
        while True:
            agent_output = self.step(obs)
            # Bump the step counter AFTER step() (so its LLM calls emit)
            # and BEFORE dispatch (so a step that crosses the cap can't
            # dispatch).
            if self._recorder is not None:
                self._recorder.on_step()
            if agent_output.error is not None:
                raise RuntimeError(f"Agent step returned error: {agent_output.error.exception_str}")
            if not agent_output.actions:
                return
            last_obs: Observation | None = None
            for action in agent_output.actions:
                # Sync call — no `await`, no thread hop. Debuggable.
                result = env_tool.execute_action(action)
                if isinstance(result, StepError):
                    raise RuntimeError(f"Tool dispatch returned StepError: {result.exception_str}")
                last_obs = result
            if last_obs is not None:
                obs = last_obs

    async def _arun(
        self,
        initial_obs: Observation,
        env_tool: "AbstractTool",
    ) -> None:
        """Async gym-style loop — N actions per step fan out via
        `asyncio.gather`. Tool calls run in real parallel (sync
        `@tool_action` methods hop to `asyncio.to_thread` inside
        `Tool.async_execute_action`).

        The default observation-merge concatenates results in original
        action order (`Observation.__iadd__`). Subclasses with bespoke
        merge logic (Genny's `_merge_results`) override this method.

        Override to customize async loop behavior. For sync default,
        see `_run` + `AgentConfig.parallel_actions=False`.
        """
        obs = initial_obs
        while True:
            # Step is sync. To exploit a truly async LLM (rare today —
            # LiteLLM's sync `completion()` is the canonical path),
            # override and use `asyncio.to_thread` or an async step.
            agent_output = self.step(obs)
            if self._recorder is not None:
                self._recorder.on_step()
            if agent_output.error is not None:
                raise RuntimeError(f"Agent step returned error: {agent_output.error.exception_str}")
            if not agent_output.actions:
                return
            # Parallel fan-out via the unified async call-site.
            results = await asyncio.gather(*(env_tool.async_execute_action(a) for a in agent_output.actions))
            merged = self._merge_results(results)
            if merged is None:
                return
            obs = merged

    @staticmethod
    def _merge_results(results: list["Observation | StepError"]) -> "Observation | None":
        """Merge N parallel tool results into one observation. StepError
        results surface as text inside the merged obs so the LLM sees the
        failure. Returns None when every result is an error and no
        useful observation was produced — the agent treats this as a
        graceful stop. Subclasses can override for bespoke merging."""
        merged: Observation | None = None
        any_observation = False
        for r in results:
            if isinstance(r, Observation):
                any_observation = True
                merged = r if merged is None else merged + r
            else:
                msg = f"[tool error: {r.error_type}: {r.exception_str}]"
                fragment = Observation.from_text(msg)
                merged = fragment if merged is None else merged + fragment
        if not any_observation and merged is None:
            return None
        return merged

    def __repr__(self) -> str:
        return self.config.model_dump_json(indent=2, serialize_as_any=True)
