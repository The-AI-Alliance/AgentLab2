"""MonitoredTool — Tool-compatible decorator that records `ToolCallEvent`s.

Part of RFC `agent-owns-loop`. `MonitoredTool` (sync) and
`AsyncMonitoredTool` (async) each subclass the corresponding
cube-standard ABC and expose the same `execute_action` signature as the
wrapped tool — agents call them identically, monitoring fires
transparently on every call.

Drop-in replacement: a `Toolbox` may contain a mix of `MonitoredTool`
wrappers and unmonitored tools side-by-side. Dispatch by action name
routes each call to the right member; the agent doesn't know which is
which.

The previous `ToolWithTelemetry` / `AsyncToolWithTelemetry` (an OTel
shim) was removed in commit e760f9e5. This module re-creates the
harness-side tool layer around the new event model. Per-tool-call OTel
spans are NOT re-introduced (no consumer); the trajectory event stream
is the harness's structured per-call observability.

Design choice (changed from RFC draft): the RFC originally specified a
single async `MonitoredTool`. In practice every in-tree cube uses sync
`Tool` subclasses (ArithmeticTool, BgymTool, ComputerBase, WebSearchTool,
…), and `Toolbox.execute_action` asserts `isinstance(tool, AbstractTool)`.
Forcing everything through `asyncio.to_thread` would add overhead AND
fail the sync-Toolbox isinstance check. Solution: provide both wrappers,
with shared recording logic, and let `install_monitoring` pick.
"""

import time
from typing import TYPE_CHECKING, Any, Callable

from cube.core import Action, Observation, StepError, TypedBaseModel
from cube.tool import (
    AbstractAsyncTool,
    AbstractTool,
    ActionSchema,
    AsyncToolbox,
    Toolbox,
)
from pydantic import Field

from cube_harness.core import EvaluationEvent, ToolCallEvent, TrajectoryEvent
from cube_harness.recorder import EventCounter

if TYPE_CHECKING:
    from cube_harness.summary import SummaryProcessor


# ---------------------------------------------------------------------------
# Budget + termination signal
# ---------------------------------------------------------------------------


class Budget(TypedBaseModel):
    """Per-episode resource budget enforced by `MonitoredTool` wrappers.

    Phase 1 enforces only `max_turns` (and optional `max_tool_calls` if
    set). The cost / wallclock limits are declared so the field names are
    stable; their enforcement lands when there's end-to-end cost
    accounting (Phase 2).
    """

    max_turns: int = 1_000
    max_tool_calls: int | None = None
    max_cost_usd: float | None = None
    max_wallclock_s: float | None = None

    # Counters mutated in place during the run.
    turns: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    started_at: float = Field(default_factory=time.time)

    @property
    def exhausted(self) -> bool:
        """True iff any configured limit (turns / tool_calls / cost /
        wallclock) is at-or-past its cap. Checked by MonitoredTool on
        entry to every execute_action and by TurnRecorder after every
        AgentEvent flush."""
        if self.turns >= self.max_turns:
            return True
        if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
            return True
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            return True
        if self.max_wallclock_s is not None and (time.time() - self.started_at) >= self.max_wallclock_s:
            return True
        return False


class BudgetExceeded(BaseException):
    """Raised by `MonitoredTool` wrappers when the per-episode budget is exhausted.

    Subclasses `BaseException` (not `Exception`) so an agent's
    `try / except Exception` cannot swallow it. The Episode `try/except`
    captures it explicitly and finalizes the trajectory.
    """

    def __init__(self, action: Action | None = None) -> None:
        super().__init__("budget exhausted")
        self.action = action


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
    trajectory_id: str,
    budget: Budget,
    parent_event_id: str,
    action: Action,
    result: Observation | StepError,
    start: float,
    end: float,
    storage: object | None,
    summary: "SummaryProcessor | None",
    event_counter: EventCounter,
) -> str:
    """Stream one `ToolCallEvent` to storage + summary, bump
    `budget.tool_calls`, return the event's id (so a follow-up
    step-wise `EvaluationEvent` can reference it).

    Shared between `MonitoredTool` (sync) and `AsyncMonitoredTool` (async).
    Events stream to disk via `storage.save_event(event, trajectory_id, n)`
    where `n` comes from the shared `EventCounter`. There is no in-memory
    accumulation."""
    if isinstance(result, StepError):
        event = ToolCallEvent(
            parent_event_id=parent_event_id,
            action_id=action.id,
            obs=Observation(),
            error=result,
            turn_id=parent_event_id,
        )
    else:
        event = ToolCallEvent(
            parent_event_id=parent_event_id,
            action_id=action.id,
            obs=result,
            turn_id=parent_event_id,
        )
    trajectory_event = TrajectoryEvent(output=event, start_time=start, end_time=end)

    if storage is not None:
        save_event = getattr(storage, "save_event", None)
        if save_event is not None:
            save_event(trajectory_event, trajectory_id, event_counter.next())

    if summary is not None:
        on_event = getattr(summary, "on_event", None)
        if on_event is not None:
            on_event(trajectory_event)

    budget.tool_calls += 1
    return event.id


def _record_step_evaluation(
    trajectory_id: str,
    parent_event_id: str,
    reward: float,
    info: dict,
    start: float,
    end: float,
    storage: object | None,
    summary: "SummaryProcessor | None",
    event_counter: EventCounter,
) -> None:
    """Stream one step-wise `EvaluationEvent` (is_terminal=False).

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
    trajectory_event = TrajectoryEvent(output=event, start_time=start, end_time=end)
    if storage is not None:
        save_event = getattr(storage, "save_event", None)
        if save_event is not None:
            save_event(trajectory_event, trajectory_id, event_counter.next())
    if summary is not None:
        on_event = getattr(summary, "on_event", None)
        if on_event is not None:
            on_event(trajectory_event)


class _MonitorState:
    """State shared between a MonitoredTool / AsyncMonitoredTool wrapper
    and the install_monitoring helper.

    Carries the per-episode trajectory id, budget, storage / summary
    handles, parent-event-id getter (late-bound to the recorder's
    current turn), a shared `EventCounter` so monitored tool writes
    and recorder writes use a single global event number sequence on
    disk, and an optional `task` reference used by the wrapper to
    absorb cube-standard `Task.step` semantics (STOP_ACTION,
    obs_postprocess, finished, validate_per_step).
    """

    __slots__ = (
        "trajectory_id",
        "budget",
        "parent_event_id_getter",
        "storage",
        "summary",
        "event_counter",
        "task",
    )

    def __init__(
        self,
        trajectory_id: str,
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None,
        storage: object | None,
        summary: "SummaryProcessor | None",
        event_counter: EventCounter,
        task: Any | None = None,
    ) -> None:
        self.trajectory_id = trajectory_id
        self.budget = budget
        self.parent_event_id_getter = parent_event_id_getter
        self.storage = storage
        self.summary = summary
        self.event_counter = event_counter
        self.task = task

    def parent_event_id(self) -> str:
        """Resolve the parent_event_id to attribute to the next recorded
        ToolCallEvent — late-bound to the TurnRecorder's current turn,
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
    try:
        from cube.task import STOP_ACTION
    except ImportError:
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
                state.trajectory_id,
                parent_event_id=tool_call_event_id,
                reward=reward,
                info=info,
                start=eval_start,
                end=time.time(),
                storage=state.storage,
                summary=state.summary,
                event_counter=state.event_counter,
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
# Sync MonitoredTool — wraps AbstractTool, exposes sync execute_action
# ---------------------------------------------------------------------------


class MonitoredTool(AbstractTool):
    """Sync wrapper over a `cube.tool.AbstractTool`.

    Mixable in a sync `Toolbox` alongside unmonitored sync tools. The
    wrapper has the same `execute_action(action) -> Observation |
    StepError` signature as any other `AbstractTool`.

    Construction is per-episode. Re-using across episodes is a bug —
    the budget counter and trajectory are episode-scoped.
    """

    def __init__(
        self,
        inner: AbstractTool,
        trajectory_id: str,
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        storage: object | None = None,
        summary: "SummaryProcessor | None" = None,
        event_counter: EventCounter | None = None,
        task: Any | None = None,
    ) -> None:
        if not isinstance(inner, AbstractTool):
            raise TypeError(
                f"MonitoredTool wraps sync AbstractTool; got {type(inner).__name__}. "
                f"Use AsyncMonitoredTool for AbstractAsyncTool."
            )
        self.inner = inner
        self._state = _MonitorState(
            trajectory_id,
            budget,
            parent_event_id_getter,
            storage,
            summary,
            event_counter if event_counter is not None else EventCounter(),
            task,
        )

    # --- delegation ---

    @property
    def action_set(self) -> list[ActionSchema]:
        """Delegate to the wrapped tool — transparent action discovery."""
        return self.inner.action_set

    def reset(self) -> None:
        """Delegate reset to the wrapped tool."""
        self.inner.reset()

    def close(self) -> None:
        """Delegate close to the wrapped tool."""
        self.inner.close()

    # --- monitored execution ---

    def execute_action(self, action: Action) -> Observation | StepError:
        """Run the wrapped tool's execute_action with full cube-standard
        `Task.step` semantics layered on top:

          1. STOP_ACTION short-circuit (raises TaskDone if accepted).
          2. Budget check (raises BudgetExceeded if exhausted).
          3. Inner tool dispatch (records a ToolCallEvent).
          4. obs_postprocess (when task is attached and result is Observation).
          5. Step-wise evaluate (when task.validate_per_step is True) —
             emits an EvaluationEvent referencing the ToolCallEvent.id,
             does NOT bleed reward/info back to the agent.
          6. task.finished() check — raises TaskDone on True.

        The agent's view of the return value remains `Observation | StepError`."""
        # 1. STOP sentinel: agent-initiated stop.
        if _maybe_stop_action(self._state, action):
            raise TaskDone(action=action)
        # 2. Budget.
        if self._state.budget.exhausted:
            raise BudgetExceeded(action=action)
        # 3. Inner dispatch + record.
        start = time.time()
        result = self.inner.execute_action(action)
        end = time.time()
        tool_call_event_id = _record_tool_call(
            self._state.trajectory_id,
            self._state.budget,
            self._state.parent_event_id(),
            action,
            result,
            start,
            end,
            self._state.storage,
            self._state.summary,
            self._state.event_counter,
        )
        # 4-6. Post-execute wrapping (obs_postprocess, step-eval, finished).
        return _post_execute_wrapping(self._state, action, result, tool_call_event_id)

    def __getattr__(self, name: str) -> object:
        # Forward direct attribute / method access to the wrapped tool.
        # cube-standard's `@tool_action`-decorated methods (`tool.bash`,
        # `tool.read`, …) are called DIRECTLY by tasks for setup,
        # verification, and oracle paths — those paths are not agent
        # tool calls and should NOT be recorded as `ToolCallEvent`s.
        # Without this delegate, terminalbench2's `task.tool.bash(...)`
        # would AttributeError as soon as Episode installs monitoring.
        # `__getattr__` only fires for attrs Python didn't find on
        # MonitoredTool itself, so `execute_action` / `action_set` /
        # `reset` / `close` keep going through the monitored path.
        if name.startswith("_") or name == "inner":
            # Don't proxy dunders or our own state — that produces
            # infinite recursion on partially-constructed instances.
            raise AttributeError(name)
        return getattr(self.inner, name)


# ---------------------------------------------------------------------------
# Async MonitoredTool — wraps AbstractAsyncTool, exposes async execute_action
# ---------------------------------------------------------------------------


class AsyncMonitoredTool(AbstractAsyncTool):
    """Async wrapper over a `cube.tool.AbstractAsyncTool`.

    Mixable in an `AsyncToolbox` alongside unmonitored async tools.
    Same `execute_action` signature as any other `AbstractAsyncTool`.
    """

    def __init__(
        self,
        inner: AbstractAsyncTool,
        trajectory_id: str,
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        storage: object | None = None,
        summary: "SummaryProcessor | None" = None,
        event_counter: EventCounter | None = None,
        task: Any | None = None,
    ) -> None:
        if not isinstance(inner, AbstractAsyncTool):
            raise TypeError(
                f"AsyncMonitoredTool wraps AbstractAsyncTool; got {type(inner).__name__}. "
                f"Use MonitoredTool for AbstractTool."
            )
        self.inner = inner
        self._state = _MonitorState(
            trajectory_id,
            budget,
            parent_event_id_getter,
            storage,
            summary,
            event_counter if event_counter is not None else EventCounter(),
            task,
        )

    @property
    def action_set(self) -> list[ActionSchema]:
        """Delegate to the wrapped tool — transparent action discovery."""
        return self.inner.action_set

    async def reset(self) -> None:
        """Delegate reset to the wrapped async tool."""
        await self.inner.reset()

    async def close(self) -> None:
        """Delegate close to the wrapped async tool."""
        await self.inner.close()

    async def execute_action(self, action: Action) -> Observation | StepError:
        """Async variant of `MonitoredTool.execute_action` — same wrapping
        steps (STOP sentinel, budget, record, obs_postprocess, step-eval,
        finished) layered around an awaited inner call."""
        if _maybe_stop_action(self._state, action):
            raise TaskDone(action=action)
        if self._state.budget.exhausted:
            raise BudgetExceeded(action=action)
        start = time.time()
        result = await self.inner.execute_action(action)
        end = time.time()
        tool_call_event_id = _record_tool_call(
            self._state.trajectory_id,
            self._state.budget,
            self._state.parent_event_id(),
            action,
            result,
            start,
            end,
            self._state.storage,
            self._state.summary,
            self._state.event_counter,
        )
        return _post_execute_wrapping(self._state, action, result, tool_call_event_id)

    def __getattr__(self, name: str) -> object:
        # Same rationale as MonitoredTool.__getattr__: cube-standard
        # tasks call @tool_action methods directly (`tool.bash(...)`)
        # for setup / verification / oracle paths, separate from the
        # agent's `execute_action` calls.
        if name.startswith("_") or name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


# ---------------------------------------------------------------------------
# install_monitoring helper
# ---------------------------------------------------------------------------


def wrap_tool(
    inner: AbstractTool | AbstractAsyncTool,
    trajectory_id: str,
    budget: Budget,
    parent_event_id_getter: Callable[[], str] | None = None,
    storage: object | None = None,
    summary: "SummaryProcessor | None" = None,
    event_counter: EventCounter | None = None,
    task: Any | None = None,
) -> AbstractTool | AbstractAsyncTool:
    """Wrap a tool in the right MonitoredTool variant for its sync/async nature.

    Already-wrapped tools pass through (idempotent). When `task` is
    provided, the wrapper also absorbs cube-standard `Task.step`
    semantics (STOP_ACTION, obs_postprocess, validate_per_step, finished).
    """
    if isinstance(inner, (MonitoredTool, AsyncMonitoredTool)):
        return inner
    if isinstance(inner, AbstractAsyncTool):
        return AsyncMonitoredTool(
            inner, trajectory_id, budget, parent_event_id_getter, storage, summary, event_counter, task
        )
    if isinstance(inner, AbstractTool):
        return MonitoredTool(
            inner, trajectory_id, budget, parent_event_id_getter, storage, summary, event_counter, task
        )
    raise TypeError(f"Cannot wrap {type(inner).__name__}: not a cube.tool.Tool / AsyncTool")


def install_monitoring(
    task: Any,
    trajectory_id: str,
    budget: Budget,
    parent_event_id_getter: Callable[[], str] | None = None,
    storage: object | None = None,
    summary: "SummaryProcessor | None" = None,
    event_counter: EventCounter | None = None,
) -> None:
    """Wrap every leaf tool of `task`'s toolbox in place + bake the
    task reference into each wrapper for cube-standard Task.step semantics.

    After this call, any path through the toolbox (whether
    `task.step` → `tool.execute_action`, or direct
    `task.toolbox.execute_action(action)`, or
    `task.tool.execute_action(action)` for single-tool tasks) emits
    monitoring AND triggers the task.step wrapping inline:

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

    `event_counter` must be the same instance shared with the episode's
    `TurnRecorder` so that monitored tool writes and recorder writes
    use a single global event-numbering sequence on disk. Defaults to
    a fresh counter for tests / standalone use.
    """
    container = getattr(task, "toolbox", None) or getattr(task, "tool", None)
    if container is None:
        return

    if event_counter is None:
        event_counter = EventCounter()

    if isinstance(container, (Toolbox, AsyncToolbox)):
        _wrap_toolbox_in_place(
            container, trajectory_id, budget, parent_event_id_getter, storage, summary, event_counter, task
        )
        return

    # Single tool. Wrap and stash it back on the attribute it came from.
    wrapped = wrap_tool(container, trajectory_id, budget, parent_event_id_getter, storage, summary, event_counter, task)
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
    trajectory_id: str,
    budget: Budget,
    parent_event_id_getter: Callable[[], str] | None,
    storage: object | None,
    summary: "SummaryProcessor | None",
    event_counter: EventCounter,
    task: Any | None = None,
) -> None:
    """Recursively wrap each leaf tool of a Toolbox / AsyncToolbox with
    its sync/async MonitoredTool variant, rebuilding the dispatch index
    so action-name lookups resolve to the wrappers."""
    new_tools: list = []
    for tool in toolbox.tools:
        if isinstance(tool, (Toolbox, AsyncToolbox)):
            _wrap_toolbox_in_place(
                tool, trajectory_id, budget, parent_event_id_getter, storage, summary, event_counter, task
            )
            new_tools.append(tool)
        else:
            new_tools.append(
                wrap_tool(tool, trajectory_id, budget, parent_event_id_getter, storage, summary, event_counter, task)
            )
    toolbox.tools = new_tools
    # Rebuild action-name → tool index so dispatch resolves to the wrappers.
    toolbox._action_name_to_tool = {action.name: tool for tool in toolbox.tools for action in tool.action_set}
