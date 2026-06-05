"""MonitoredTool — wraps any cube-standard `Tool` and records `ToolCallEvent`s.

`MonitoredTool` is a thin pass-through that adds budget enforcement,
ToolCallEvent emission, and the cube-standard `Task.step` semantics
(STOP_ACTION, obs_postprocess, validate_per_step, finished). It exposes
the same dual call surface as its inner `Tool`:

  * `tool.execute_action(action)` — sync.
  * `await tool.async_execute_action(action)` — async.

The inner `Tool` handles per-method sync/async routing itself, so
`MonitoredTool` doesn't need to know whether the `@tool_action` method
is sync or async — it just delegates and records.
"""

import copy
import logging
import time
from typing import Any, Callable

from cube.core import Action, Observation, StepError
from cube.task import STOP_ACTION
from cube.tool import AbstractTool, ActionSchema, Toolbox

from cube_harness.budget import Budget, BudgetExceeded
from cube_harness.core import EvaluationEvent, ToolCallEvent, TrajectoryEvent

logger = logging.getLogger(__name__)

# Re-export Budget + BudgetExceeded so existing call sites
# `from cube_harness.tool import Budget` keep working. Canonical home
# is now `cube_harness.budget` — see that module's docstring.
__all__ = ["Budget", "BudgetExceeded", "TaskDone"]


class TaskDone(BaseException):
    """Raised by `MonitoredTool` when the task indicates the episode is over.

    Fires in two cases:

    - The agent emitted `STOP_ACTION` (cube-standard's sentinel) and
      `task.accept_agent_stop=True`. Recorded as agent-initiated stop.
    - `task.finished(obs)` returned `True` after a tool call.

    Like `BudgetExceeded`, subclasses `BaseException` so agent code's
    `try / except Exception` doesn't swallow it. Episode catches it in
    its outer `except` block and finalizes normally (no failure tag —
    a TaskDone is a clean end).
    """

    def __init__(self, action: Action | None = None) -> None:
        super().__init__("task finished")
        self.action = action


# ---------------------------------------------------------------------------
# Shared recording helpers
# ---------------------------------------------------------------------------


def _record_tool_call(
    emit: "Callable[[TrajectoryEvent], str]",
    budget: Budget,
    parent_event_id: str,
    action: Action,
    result: Observation | StepError,
    start: float,
    end: float,
) -> str:
    """Emit one `ToolCallEvent` through the streamer, bump
    `budget.tool_calls`, return the event's id (so a follow-up
    step-wise `EvaluationEvent` can reference it).

    Shared between `MonitoredTool` (single class, sync + async dispatch)."""
    is_error = isinstance(result, StepError)
    event = ToolCallEvent(
        parent_event_id=parent_event_id,
        action_id=action.id,
        action=action,
        obs=Observation() if is_error else result,
        error=result if is_error else None,
    )
    emit(TrajectoryEvent(output=event, start_time=start, end_time=end))
    budget.bump_tool_calls()
    return event.id


def _record_step_evaluation(
    emit: "Callable[[TrajectoryEvent], str]",
    parent_event_id: str,
    reward: float,
    info: dict,
    start: float,
    end: float,
) -> None:
    """Emit one step-wise `EvaluationEvent` (is_terminal=False) through
    the streamer.

    Called by `MonitoredTool` after each tool call when
    `task.validate_per_step=True`. The reward / info land on disk but
    are NOT returned to the agent — the agent's view of execute_action
    remains `Observation | StepError`."""
    event = EvaluationEvent(
        reward=float(reward),
        info=dict(info),
        is_terminal=False,
        parent_event_id=parent_event_id,
    )
    emit(TrajectoryEvent(output=event, start_time=start, end_time=end))


class _MonitorState:
    """State shared between a MonitoredTool wrapper and its build helper.

    Carries the streamer's `emit` callable, budget, parent-event-id
    getter (late-bound to the streamer's current parent), and an
    optional `task` reference used by the wrapper to absorb
    cube-standard `Task.step` semantics (STOP_ACTION, obs_postprocess,
    finished, validate_per_step).
    """

    __slots__ = (
        "emit",
        "budget",
        "parent_event_id_getter",
        "task",
    )

    def __init__(
        self,
        emit: "Callable[[TrajectoryEvent], str]",
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None,
        task: Any | None = None,
    ) -> None:
        self.emit = emit
        self.budget = budget
        self.parent_event_id_getter = parent_event_id_getter
        self.task = task

    def parent_event_id(self) -> str:
        """Resolve the parent_event_id to attribute to the next recorded
        ToolCallEvent — late-bound to the EventStreamer's current turn,
        with a `RESET` sentinel fallback when no turn is active."""
        if self.parent_event_id_getter is not None:
            value = self.parent_event_id_getter()
            if value is not None:
                return value
        return "no-parent"


def _maybe_stop_action(state: "_MonitorState", action: Action) -> bool:
    """Return True iff this action is the cube-standard STOP sentinel
    and the task accepts it. Caller should raise TaskDone."""
    task = state.task
    if task is None:
        return False
    return getattr(task, "accept_agent_stop", False) and action.name == STOP_ACTION.name


def _post_execute_wrapping(
    state: "_MonitorState",
    action: Action,
    result: Observation | StepError,
    tool_call_event_id: str,
) -> Observation | StepError:
    """Run the post-execute wrapping cube-standard's `Task.step` would
    have done: obs_postprocess, optional step-wise evaluate, and the
    finished-check that triggers `TaskDone`.

    Returns the (possibly post-processed) result the agent will see.
    """
    task = state.task
    if task is None:
        return result

    # 1. obs_postprocess (only when we got an Observation, not a StepError).
    if isinstance(result, Observation):
        postprocess = getattr(task, "obs_postprocess", None)
        if postprocess is not None:
            result = postprocess(result)

    # 2. Step-wise evaluate (when task.validate_per_step is True).
    if getattr(task, "validate_per_step", False):
        try:
            eval_start = time.time()
            reward, info = task.evaluate(result if isinstance(result, Observation) else None)
            _record_step_evaluation(
                state.emit,
                parent_event_id=tool_call_event_id,
                reward=reward,
                info=info,
                start=eval_start,
                end=time.time(),
            )
        except Exception:  # noqa: BLE001
            # Step-eval failures don't stop the run; log but continue.
            # The terminal evaluate in Episode.finally re-attempts.
            pass

    # 3. finished() check — triggers TaskDone on True.
    finished = getattr(task, "finished", None)
    if finished is not None and finished(result if isinstance(result, Observation) else None):
        raise TaskDone(action=action)

    return result


# ---------------------------------------------------------------------------
# MonitoredTool — single wrapper for any cube tool (sync OR async)
# ---------------------------------------------------------------------------


class MonitoredTool(AbstractTool):
    """Wraps a `cube.tool.AbstractTool` to add budget enforcement,
    `ToolCallEvent` emission, and the cube-standard `Task.step`
    semantics (STOP_ACTION, obs_postprocess, validate_per_step,
    finished).

    Both `execute_action` (sync) and `async_execute_action` (async) are
    available — each delegates to the inner tool's same-named dispatch
    method. The inner `Tool` handles per-method sync/async routing,
    bridging to a thread+loop or `asyncio.to_thread` when caller and
    `@tool_action` method differ.

    Construction is per-episode. Re-using across episodes is a bug —
    the budget counter and trajectory are episode-scoped.
    """

    def __init__(
        self,
        inner: AbstractTool,
        emit: "Callable[[TrajectoryEvent], str]",
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        task: Any | None = None,
    ) -> None:
        if not isinstance(inner, AbstractTool):
            raise TypeError(f"MonitoredTool wraps a cube.tool.AbstractTool; got {type(inner).__name__}.")
        self.inner = inner
        self._state = _MonitorState(emit, budget, parent_event_id_getter, task)
        # TEMP (F4 / design-debt): when set, this leaf does NOT surface
        # STOP_ACTION. `_dedup_stop_actions` flips it on every monitored leaf
        # but one so a multi-leaf toolbox surfaces `stop` exactly once. See
        # `_dedup_stop_actions` for why and the proper fix.
        self._suppress_stop = False

    # --- delegation ---

    @property
    def action_set(self) -> list[ActionSchema]:
        """The wrapped tool's action set, plus `STOP_ACTION` when the
        attached task accepts it.

        `STOP_ACTION` is the cube-standard sentinel an agent emits to
        signal graceful end-of-episode. `Task.action_set` includes it
        (when `task.accept_agent_stop` is True), but `Toolbox` only
        sees member-tool action sets. Surfacing it here lets a
        containing toolbox dispatch `STOP_ACTION` to this wrapper,
        where `_maybe_stop_action` translates it into `TaskDone`.
        """
        actions = list(self.inner.action_set)
        task = self._state.task
        if task is not None and getattr(task, "accept_agent_stop", False) and not self._suppress_stop:
            if not any(a.name == STOP_ACTION.name for a in actions):
                actions.append(STOP_ACTION)
        return actions

    def reset(self) -> None:
        self.inner.reset()

    def close(self) -> None:
        self.inner.close()

    # --- monitored execution ---

    def execute_action(self, action: Action) -> Observation | StepError:
        """Sync monitored dispatch. Layers full cube-standard `Task.step`
        semantics over the wrapped tool's `execute_action`:

          1. STOP_ACTION short-circuit (raises TaskDone if accepted).
          2. Budget check (raises BudgetExceeded if exhausted).
          3. Inner tool dispatch (records a ToolCallEvent).
          4. obs_postprocess (when task is attached and result is Observation).
          5. Step-wise evaluate (when task.validate_per_step is True) —
             emits an EvaluationEvent referencing the ToolCallEvent.id,
             does NOT bleed reward/info back to the agent.
          6. task.finished() check — raises TaskDone on True.

        Works for any inner `Tool`: a sync `@tool_action` runs directly
        on the calling thread; an async `@tool_action` is bridged via a
        one-shot worker thread inside `Tool.execute_action` (~2-5 ms).
        """
        if _maybe_stop_action(self._state, action):
            raise TaskDone(action=action)
        if self._state.budget.exhausted:
            raise BudgetExceeded(action=action)
        start = time.time()
        result = self.inner.execute_action(action)
        end = time.time()
        tool_call_event_id = _record_tool_call(
            self._state.emit,
            self._state.budget,
            self._state.parent_event_id(),
            action,
            result,
            start,
            end,
        )
        return _post_execute_wrapping(self._state, action, result, tool_call_event_id)

    async def async_execute_action(self, action: Action) -> Observation | StepError:
        """Async monitored dispatch — the parallel-safe call-site.

        Delegates to the inner tool's `async_execute_action`, which
        routes per-method: an async `@tool_action` is awaited directly;
        a sync `@tool_action` hops through `asyncio.to_thread` so
        `asyncio.gather` over N calls runs them in real OS-thread
        parallel.

        Same wrapping steps as `execute_action` (STOP, budget, record,
        obs_postprocess, step-eval, finished).
        """
        if _maybe_stop_action(self._state, action):
            raise TaskDone(action=action)
        if self._state.budget.exhausted:
            raise BudgetExceeded(action=action)
        start = time.time()
        result = await self.inner.async_execute_action(action)
        end = time.time()
        tool_call_event_id = _record_tool_call(
            self._state.emit,
            self._state.budget,
            self._state.parent_event_id(),
            action,
            result,
            start,
            end,
        )
        return _post_execute_wrapping(self._state, action, result, tool_call_event_id)

    def __getattr__(self, name: str) -> object:
        # Forward direct attribute / method access to the wrapped tool.
        # cube-standard's `@tool_action`-decorated methods (`tool.bash`,
        # `tool.read`, …) are called DIRECTLY by tasks for setup,
        # verification, and oracle paths — those paths are not agent
        # tool calls and should NOT be recorded as `ToolCallEvent`s.
        # `__getattr__` only fires for attrs Python didn't find on
        # MonitoredTool itself, so `execute_action` / `async_execute_action`
        # / `action_set` / `reset` / `close` keep going through the
        # monitored path.
        if name.startswith("_") or name == "inner":
            # Don't proxy dunders or our own state — produces infinite
            # recursion on partially-constructed instances.
            raise AttributeError(name)
        return getattr(self.inner, name)


# ---------------------------------------------------------------------------
# build_monitored_env_tool helper
# ---------------------------------------------------------------------------


def wrap_tool(
    inner: AbstractTool,
    emit: "Callable[[TrajectoryEvent], str]",
    budget: Budget,
    parent_event_id_getter: Callable[[], str] | None = None,
    task: Any | None = None,
) -> MonitoredTool:
    """Wrap a tool in `MonitoredTool`. Already-wrapped tools pass through
    (idempotent). When `task` is provided, the wrapper also absorbs
    cube-standard `Task.step` semantics (STOP_ACTION, obs_postprocess,
    validate_per_step, finished).
    """
    if isinstance(inner, MonitoredTool):
        return inner
    return MonitoredTool(inner, emit, budget, parent_event_id_getter, task)


def build_monitored_env_tool(task: Any, streamer: Any) -> AbstractTool | None:
    """Build the monitored ``env_tool`` the agent drives — WITHOUT mutating
    the task's own tool.

    The agent sees a `MonitoredTool` (single tool) or a shallow copy of the
    task's toolbox whose leaves are `MonitoredTool`-wrapped. Crucially the
    wrappers share the SAME inner tool instances as the task, so browser /
    container state is shared between the agent's calls and the task's own.

    The task keeps its concrete `tool` / `toolbox`, so `task.setup` / `reset`
    / `evaluate` / `finished` still reach concrete-tool methods (`bash`,
    `evaluate_js`), private attrs (`_container`, `_config`), and type checks
    (`isinstance` / `Toolbox.find_tool`). Previously these were wrapped in
    place, which broke every cube whose lifecycle code touched its own tool.
    The RFC's contract is "the agent receives only env_tool; the task
    reference never leaks" — the wrapper still absorbs `Task.step` semantics
    (STOP_ACTION, obs_postprocess, validate_per_step, finished) via the
    baked-in task ref, but the task's tool is no longer clobbered.

    `streamer` is an `EventStreamer`; we read `.emit`, `.budget`, and
    `.current_parent_event_id`. Looks up the env tool via `task.toolbox`
    first, then `task.tool`. Returns None when the task exposes neither.
    """
    emit = streamer.emit
    budget = streamer.budget
    parent_event_id_getter = streamer.current_parent_event_id
    container = getattr(task, "toolbox", None) or getattr(task, "tool", None)
    if container is None:
        return None
    env_tool = _monitored_view(container, emit, budget, parent_event_id_getter, task)
    _dedup_stop_actions(env_tool)
    return env_tool


def _dedup_stop_actions(env_tool: AbstractTool) -> None:
    """TEMP (F4 / design-debt): surface STOP_ACTION on exactly one monitored leaf.

    `MonitoredTool.action_set` appends `STOP_ACTION` per leaf, so a multi-leaf
    monitored toolbox advertises `stop` N times — which both trips
    `Toolbox.__init__`'s duplicate-name guard (crashing `parallel_actions`
    on multi-tool cubes) and sends duplicate `stop` tool schemas to the LLM.
    Keep STOP on the first monitored leaf and suppress it on the rest.

    Proper fix (deferred): STOP is a task/container-level action, not a per-leaf
    one — surface it once at the toolbox boundary (or have the toolbox treat
    STOP_ACTION as a shared sentinel) and drop both this pass and the per-leaf
    append. Tracked as design-debt F4.
    """
    seen_stop = False
    stack: list = [env_tool]
    while stack:
        node = stack.pop()
        if isinstance(node, Toolbox):
            stack.extend(node.tools)
        elif isinstance(node, MonitoredTool):
            if seen_stop:
                node._suppress_stop = True
            else:
                seen_stop = True


def _monitored_view(
    container: AbstractTool,
    emit: "Callable[[TrajectoryEvent], str]",
    budget: Budget,
    parent_event_id_getter: Callable[[], str] | None,
    task: Any | None = None,
) -> AbstractTool:
    """Return a monitored view of `container` sharing its inner instances,
    without mutating it. A `Toolbox` is shallow-copied with `MonitoredTool`
    leaves (nested toolboxes copied recursively); a single tool is wrapped
    directly. Idempotent via `wrap_tool`."""
    if isinstance(container, Toolbox):
        view = copy.copy(container)  # same type + attrs; we replace tools below
        view.tools = [
            _monitored_view(leaf, emit, budget, parent_event_id_getter, task)
            if isinstance(leaf, Toolbox)
            else wrap_tool(leaf, emit, budget, parent_event_id_getter, task)
            for leaf in container.tools
        ]
        # Rebuild the dispatch index so action-name lookups resolve to the
        # wrappers. Overwrite-style (last wins) — tolerant of leaves that each
        # surface STOP_ACTION, matching the prior in-place behavior.
        view._action_name_to_tool = {action.name: tool for tool in view.tools for action in tool.action_set}
        return view
    return wrap_tool(container, emit, budget, parent_event_id_getter, task)
