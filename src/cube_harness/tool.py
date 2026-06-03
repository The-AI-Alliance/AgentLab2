"""MonitoredTool — single wrapper that records `ToolCallEvent`s.

Part of RFC `agent-owns-loop`. One `MonitoredTool` class wraps ANY
cube-standard tool (sync `AbstractTool` or async `AbstractAsyncTool`)
and exposes a dual call surface:

  * `tool.execute_action(action)` — sync. Sync-inner only; raises
    TypeError for async inner. Runs the inner on the calling thread
    with NO `to_thread` hop. Pick this in sequential loops where
    single-stack debuggability matters (default `Agent._run`).

  * `await tool.async_execute_action(action)` — async. Works for BOTH
    inner kinds. Sync inner is dispatched via `asyncio.to_thread` so
    `asyncio.gather` over N calls runs them in real parallel. Async
    inner is awaited directly. Pick this for parallel dispatch
    (`Agent._arun` / `Genny[parallel_actions=True]`) or when the inner
    may be async.

The dual API lets agent authors pick the right call shape for their
loop semantics. cube-harness internals route through the right one
based on `Agent._run` vs `_arun`.

The collapse to one class was enabled by cube-standard adding
`async_execute_action` as a default method on `AbstractTool` /
`AbstractAsyncTool` and relaxing `AsyncToolbox` to accept mixed sync +
async leaves (see cube-standard PR #152).
"""

import asyncio
import logging
import time
from typing import Any, Callable

from cube.core import Action, Observation, StepError
from cube.task import STOP_ACTION
from cube.tool import (
    AbstractAsyncTool,
    AbstractTool,
    ActionSchema,
    AsyncToolbox,
    Toolbox,
)

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
        turn_id=parent_event_id,
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
    """State shared between a MonitoredTool wrapper and install_monitoring.

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
    """Single wrapper around a `cube.tool.AbstractTool` or `AbstractAsyncTool`.

    Subclasses `AbstractTool` so it's structurally compatible with sync
    `Toolbox` containment, but also exposes `async_execute_action` (the
    one declared on `AbstractTool` by cube-standard) which works for
    both inner kinds.

    Dual call surface:

      * `tool.execute_action(action)` — sync. Sync inner only; raises
        TypeError if the inner is async.

      * `await tool.async_execute_action(action)` — async. Works for
        BOTH inner kinds. Sync inner is called synchronously on the
        current task (no `to_thread` hop — debuggable). Async inner
        is awaited.

    Construction is per-episode. Re-using across episodes is a bug —
    the budget counter and trajectory are episode-scoped.
    """

    def __init__(
        self,
        inner: AbstractTool | AbstractAsyncTool,
        emit: "Callable[[TrajectoryEvent], str]",
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        task: Any | None = None,
    ) -> None:
        if not isinstance(inner, (AbstractTool, AbstractAsyncTool)):
            raise TypeError(
                f"MonitoredTool wraps a cube.tool.AbstractTool or AbstractAsyncTool; got {type(inner).__name__}."
            )
        self.inner = inner
        self._inner_is_async = isinstance(inner, AbstractAsyncTool)
        self._state = _MonitorState(emit, budget, parent_event_id_getter, task)

    # --- delegation ---

    @property
    def action_set(self) -> list[ActionSchema]:
        """The wrapped tool's action set, plus `STOP_ACTION` when the
        attached task accepts it.

        `STOP_ACTION` is the cube-standard sentinel an agent emits to
        signal graceful end-of-episode. `Task.action_set` includes it
        (when `task.accept_agent_stop` is True), but `Toolbox` /
        `AsyncToolbox` only see member-tool action sets. Surfacing it
        here lets a containing toolbox dispatch `STOP_ACTION` to this
        wrapper, where `_maybe_stop_action` translates it into
        `TaskDone`.
        """
        actions = list(self.inner.action_set)
        task = self._state.task
        if task is not None and getattr(task, "accept_agent_stop", False):
            if not any(a.name == STOP_ACTION.name for a in actions):
                actions.append(STOP_ACTION)
        return actions

    def reset(self) -> None:
        """Sync reset. Async inner returns a coroutine; we close it and
        log a debug — callers wanting proper async cleanup should go
        through `AsyncToolbox.reset` (which awaits) or the live agent
        loop's shutdown path."""
        r = self.inner.reset()
        if asyncio.iscoroutine(r):
            r.close()
            logger.debug(
                "MonitoredTool.reset: async inner returned a coroutine; "
                "closed without awaiting. Use AsyncToolbox.reset for proper cleanup."
            )

    def close(self) -> None:
        """Sync close. Same async-coroutine handling as `reset`."""
        c = self.inner.close()
        if asyncio.iscoroutine(c):
            c.close()
            logger.debug(
                "MonitoredTool.close: async inner returned a coroutine; "
                "closed without awaiting. Use AsyncToolbox.close for proper cleanup."
            )

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

        Sync inner only. For async inner, use `async_execute_action`."""
        if self._inner_is_async:
            raise TypeError(
                "MonitoredTool.execute_action (sync) does not support async inner tools. "
                "Use `await tool.async_execute_action(action)` for async-shaped dispatch "
                "(works for both sync and async inners), or wrap with asyncio.to_thread "
                "inside asyncio.gather for parallel dispatch of sync inners."
            )
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
        """Async monitored dispatch — the parallel-safe call-site that
        works for BOTH sync and async inners.

        Sync inner: dispatched via `asyncio.to_thread`. Lets
        `asyncio.gather` over N sync-inner calls run them in real
        parallel (each on its own OS thread). The lock on `Budget` +
        `EventStreamer._lock` keeps stats coherent under that
        concurrency.

        Async inner: `await self.inner.execute_action(action)`.

        For agents that want sync-on-main-thread dispatch (default
        `Agent._run` — single-stack pdb, no thread hop), call the sync
        `execute_action` directly instead.

        Same wrapping steps as `execute_action` (STOP, budget, record,
        obs_postprocess, step-eval, finished).
        """
        if _maybe_stop_action(self._state, action):
            raise TaskDone(action=action)
        if self._state.budget.exhausted:
            raise BudgetExceeded(action=action)
        start = time.time()
        if self._inner_is_async:
            result = await self.inner.execute_action(action)
        else:
            result = await asyncio.to_thread(self.inner.execute_action, action)
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
# install_monitoring helper
# ---------------------------------------------------------------------------


def wrap_tool(
    inner: AbstractTool | AbstractAsyncTool,
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


def install_monitoring(task: Any, streamer: Any) -> None:
    """Wrap every leaf tool of `task`'s toolbox in place + bake the
    task reference into each wrapper for cube-standard Task.step semantics.

    `streamer` is an `EventStreamer` instance — we read three pieces
    from it: `streamer.emit` (the single fan-out callable),
    `streamer.budget` (the per-episode Budget), and
    `streamer.current_parent_event_id` (the late-bound getter for
    ToolCallEvent.parent_event_id).

    After this call, any path through the toolbox emits monitoring +
    triggers the task.step wrapping inline:

      - STOP_ACTION short-circuit → raises TaskDone.
      - obs_postprocess on every Observation result.
      - Step-wise evaluate when task.validate_per_step=True, recorded
        as an EvaluationEvent (no bleed to the agent).
      - task.finished() check after every tool call → raises TaskDone.

    The function mutates `task.toolbox.tools` / `task.tool` so call
    sites that hold the original reference see the wrappers.

    Nested `Toolbox` / `AsyncToolbox` instances are recursively
    flattened. Already-wrapped tools pass through (idempotent).

    Looks up the toolbox via `task.toolbox` first, then `task.tool` —
    cube-standard `Task` exposes the latter (a single Toolbox usually).
    """
    emit = streamer.emit
    budget = streamer.budget
    parent_event_id_getter = streamer.current_parent_event_id
    container = getattr(task, "toolbox", None) or getattr(task, "tool", None)
    if container is None:
        return

    if isinstance(container, (Toolbox, AsyncToolbox)):
        _wrap_toolbox_in_place(container, emit, budget, parent_event_id_getter, task)
        return

    # Single tool. Wrap and stash it back on the attribute it came from.
    wrapped = wrap_tool(container, emit, budget, parent_event_id_getter, task)
    if hasattr(task, "toolbox") and getattr(task, "toolbox", None) is container:
        task.toolbox = wrapped
    elif hasattr(task, "tool") and getattr(task, "tool", None) is container:
        # cube.task.Task stores the live tool on _tool (Pydantic
        # PrivateAttr) and exposes it via the `tool` property. We
        # write _tool when present so the property returns the wrapper.
        if hasattr(task, "_tool"):
            task._tool = wrapped
        else:
            task.tool = wrapped


def _wrap_toolbox_in_place(
    toolbox: Toolbox | AsyncToolbox,
    emit: "Callable[[TrajectoryEvent], str]",
    budget: Budget,
    parent_event_id_getter: Callable[[], str] | None,
    task: Any | None = None,
) -> None:
    """Recursively wrap each leaf tool of a Toolbox / AsyncToolbox with
    `MonitoredTool`, rebuilding the dispatch index so action-name
    lookups resolve to the wrappers."""
    new_tools: list = []
    for tool in toolbox.tools:
        if isinstance(tool, (Toolbox, AsyncToolbox)):
            _wrap_toolbox_in_place(tool, emit, budget, parent_event_id_getter, task)
            new_tools.append(tool)
        else:
            new_tools.append(wrap_tool(tool, emit, budget, parent_event_id_getter, task))
    toolbox.tools = new_tools
    # Rebuild action-name → tool index so dispatch resolves to the wrappers.
    toolbox._action_name_to_tool = {action.name: tool for tool in toolbox.tools for action in tool.action_set}


# ---------------------------------------------------------------------------
# as_async — expose any toolbox as `AbstractAsyncTool` for the agent
# ---------------------------------------------------------------------------


def as_async(tool: AbstractTool | AbstractAsyncTool) -> AbstractAsyncTool:
    """Return `tool` as an `AbstractAsyncTool` for `Agent.run`.

    `AbstractAsyncTool` instance passes through unchanged. A sync
    `Toolbox` (post-`install_monitoring`, with `MonitoredTool` leaves)
    is rebuilt as an `AsyncToolbox` containing the same leaves —
    `AsyncToolbox` now accepts mixed sync + async leaves (cube-standard
    PR #152) and dispatches each through `async_execute_action`,
    running sync leaves synchronously on the current task without a
    `to_thread` hop. A single sync tool (not wrapped in a Toolbox) is
    wrapped in a one-element `AsyncToolbox`.
    """
    if isinstance(tool, AbstractAsyncTool):
        return tool
    if isinstance(tool, Toolbox):
        return AsyncToolbox(tools=tool.tools)
    if isinstance(tool, AbstractTool):
        return AsyncToolbox(tools=[tool])
    raise TypeError(f"as_async expects AbstractTool / AbstractAsyncTool; got {type(tool).__name__}")
