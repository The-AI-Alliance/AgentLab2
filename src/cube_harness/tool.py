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

import asyncio
import threading
import time
from typing import Any, Callable

from cube.core import Action, Observation, StepError, TypedBaseModel
from cube.task import STOP_ACTION
from cube.tool import (
    AbstractAsyncTool,
    AbstractTool,
    ActionSchema,
    AsyncToolbox,
    Toolbox,
)
from pydantic import Field, PrivateAttr

from cube_harness.core import EvaluationEvent, ToolCallEvent, TrajectoryEvent

# ---------------------------------------------------------------------------
# Budget + termination signal
# ---------------------------------------------------------------------------


class Budget(TypedBaseModel):
    """Per-episode resource budget enforced by `MonitoredTool` + `EventStreamer`.

    Caps:
      - `max_turns`: agent step() calls.
      - `max_tool_calls`: monitored tool dispatches.
      - `max_cost_usd`: cumulative LLM call cost (from `LLMCall.usage.cost`).
      - `max_prompt_tokens` / `max_completion_tokens`: per-direction
        cumulative token usage.
      - `max_wallclock_s`: elapsed seconds since the Budget was created.

    Counters bumped during the run:
      - `turns` / `cost_usd` / `prompt_tokens` / `completion_tokens` —
        by `EventStreamer._flush_agent_event` from `AgentEvent.llm_calls`.
      - `tool_calls` — by `MonitoredTool._record_tool_call`.
      - `started_at` — set once at construction; elapsed time derived from it.

    `Budget.exhausted` returns True iff any configured cap is at-or-past
    its limit. `MonitoredTool` raises `BudgetExceeded(BaseException)`
    when it is. Agents can also introspect the live budget via
    `recorder.budget` for graceful self-stop and prompt-injection
    ("you have X% budget left") — see `Budget.__str__`.
    """

    max_turns: int = 1_000
    max_tool_calls: int | None = None
    max_cost_usd: float | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_wallclock_s: float | None = None

    # Counters mutated in place during the run.
    turns: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started_at: float = Field(default_factory=time.time)

    # Guards the bump methods + the `exhausted` read. GennyParallel
    # dispatches N tool calls via `asyncio.gather` over an
    # `_SyncToolAsAsync` adapter, whose `execute_action` hops into a
    # real OS thread via `asyncio.to_thread`. Without this lock the
    # `tool_calls += 1` in `_record_tool_call` would race across N
    # workers and `max_tool_calls` could overrun. Mirrors the
    # `SummaryProcessor` lock added for the same parallel path.
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def bump_tool_calls(self) -> None:
        """Atomic +1 on `tool_calls`. Called by `_record_tool_call`
        from MonitoredTool workers — may run on multiple threads in
        parallel."""
        with self._lock:
            self.tool_calls += 1

    def bump_turn(self) -> None:
        """Atomic +1 on `turns` — bumped once per Agent.run loop
        iteration (one agent step). Distinct from LLM-call count: an
        agent step may make 0..N LLM calls.

        Called by the agent loop (`Agent.run` default + GennyParallel.run
        override) after each `self.step(obs)` returns, so `max_turns`
        caps agent steps the way callers expect.
        """
        with self._lock:
            self.turns += 1

    def bump_llm_usage(self, cost: float, prompt: int, completion: int) -> None:
        """Atomic bump of LLM-call cost + token counters. Called by
        `EventStreamer.on_llm_call` per LLM API call (so a multi-LLM-call
        step accumulates correctly). Does NOT bump `turns` — that's
        `bump_turn`'s job."""
        with self._lock:
            self.cost_usd += cost
            self.prompt_tokens += prompt
            self.completion_tokens += completion

    @property
    def exhausted(self) -> bool:
        """True iff any configured cap is at-or-past its limit. Checked
        by MonitoredTool on entry to every execute_action and by
        EventStreamer after every AgentEvent flush.

        Lock-protected so the multi-field read is coherent against
        concurrent bumps from parallel tool-call workers."""
        with self._lock:
            if self.turns >= self.max_turns:
                return True
            if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
                return True
            if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
                return True
            if self.max_prompt_tokens is not None and self.prompt_tokens >= self.max_prompt_tokens:
                return True
            if self.max_completion_tokens is not None and self.completion_tokens >= self.max_completion_tokens:
                return True
            if self.max_wallclock_s is not None and (time.time() - self.started_at) >= self.max_wallclock_s:
                return True
            return False

    def __getstate__(self) -> dict:
        # Strip the unpicklable Lock so Budget can ride along inside
        # any future serialized config without blowing up on Ray.
        state = super().__getstate__()
        state = dict(state)
        private = dict(state.get("__pydantic_private__") or {})
        private.pop("_lock", None)
        state["__pydantic_private__"] = private
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        # Recreate the lock — the deserialized instance is a fresh one,
        # no contention to inherit.
        self._lock = threading.Lock()

    def __str__(self) -> str:
        """Concise human-readable summary of budget usage — suitable for
        injection into an LLM prompt so the agent can plan against
        what's left. Only configured caps are listed; empty string when
        no cap is set (no signal to convey).

        Format: `"budget used: turns 34/150 (23%), cost $1.20/$5.00 (24%), tokens 1200/5000 prompt (24%), tokens 100/1000 completion (10%), 60s/300s wallclock (20%)"`
        """
        parts: list[str] = []
        # max_turns has a non-None default (1000), so we always show it.
        # Guard the zero case so a max_turns=0 (born-exhausted, used in
        # tests) doesn't div-by-zero in the percent calc.
        if self.max_turns > 0:
            parts.append(f"turns {self.turns}/{self.max_turns} ({self.turns / self.max_turns * 100:.0f}%)")
        else:
            parts.append(f"turns {self.turns}/{self.max_turns}")
        if self.max_tool_calls is not None:
            pct = self.tool_calls / self.max_tool_calls * 100 if self.max_tool_calls > 0 else 0.0
            parts.append(f"tool_calls {self.tool_calls}/{self.max_tool_calls} ({pct:.0f}%)")
        if self.max_cost_usd is not None:
            pct = self.cost_usd / self.max_cost_usd * 100 if self.max_cost_usd > 0 else 0.0
            parts.append(f"cost ${self.cost_usd:.2f}/${self.max_cost_usd:.2f} ({pct:.0f}%)")
        if self.max_prompt_tokens is not None:
            pct = self.prompt_tokens / self.max_prompt_tokens * 100 if self.max_prompt_tokens > 0 else 0.0
            parts.append(f"prompt_tokens {self.prompt_tokens}/{self.max_prompt_tokens} ({pct:.0f}%)")
        if self.max_completion_tokens is not None:
            pct = self.completion_tokens / self.max_completion_tokens * 100 if self.max_completion_tokens > 0 else 0.0
            parts.append(f"completion_tokens {self.completion_tokens}/{self.max_completion_tokens} ({pct:.0f}%)")
        if self.max_wallclock_s is not None:
            elapsed = time.time() - self.started_at
            pct = elapsed / self.max_wallclock_s * 100 if self.max_wallclock_s > 0 else 0.0
            parts.append(f"{elapsed:.0f}s/{self.max_wallclock_s:.0f}s wallclock ({pct:.0f}%)")
        return "budget used: " + ", ".join(parts)


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

    Shared between `MonitoredTool` (sync) and `AsyncMonitoredTool` (async)."""
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
    """State shared between a MonitoredTool / AsyncMonitoredTool wrapper
    and the install_monitoring helper.

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
        emit: "Callable[[TrajectoryEvent], str]",
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        task: Any | None = None,
    ) -> None:
        if not isinstance(inner, AbstractTool):
            raise TypeError(
                f"MonitoredTool wraps sync AbstractTool; got {type(inner).__name__}. "
                f"Use AsyncMonitoredTool for AbstractAsyncTool."
            )
        self.inner = inner
        self._state = _MonitorState(emit, budget, parent_event_id_getter, task)

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
            self._state.emit,
            self._state.budget,
            self._state.parent_event_id(),
            action,
            result,
            start,
            end,
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
        emit: "Callable[[TrajectoryEvent], str]",
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        task: Any | None = None,
    ) -> None:
        if not isinstance(inner, AbstractAsyncTool):
            raise TypeError(
                f"AsyncMonitoredTool wraps AbstractAsyncTool; got {type(inner).__name__}. "
                f"Use MonitoredTool for AbstractTool."
            )
        self.inner = inner
        self._state = _MonitorState(emit, budget, parent_event_id_getter, task)

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
    emit: "Callable[[TrajectoryEvent], str]",
    budget: Budget,
    parent_event_id_getter: Callable[[], str] | None = None,
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
        return AsyncMonitoredTool(inner, emit, budget, parent_event_id_getter, task)
    if isinstance(inner, AbstractTool):
        return MonitoredTool(inner, emit, budget, parent_event_id_getter, task)
    raise TypeError(f"Cannot wrap {type(inner).__name__}: not a cube.tool.Tool / AsyncTool")


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
    its sync/async MonitoredTool variant, rebuilding the dispatch index
    so action-name lookups resolve to the wrappers."""
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
# Sync → async adapter (Episode boundary)
# ---------------------------------------------------------------------------


class _SyncToolAsAsync(AbstractAsyncTool):
    """Wraps a sync `AbstractTool` so it exposes the `AbstractAsyncTool`
    surface — `await execute_action(action)` works regardless of the
    underlying tool's sync/async nature.

    Episode applies this at the boundary right before handing the
    toolbox to `agent.run`. Agent authors who override `run()` write
    fully-async code (`await toolbox.execute_action(a)`, parallel
    `asyncio.gather`) without branching on tool type. Sync inner
    `execute_action` is dispatched via `asyncio.to_thread` so the
    event loop stays responsive.

    Sync-tool authors don't see this — they override `step()` and
    inherit `Agent.run`, never touching async.
    """

    def __init__(self, inner: AbstractTool) -> None:
        self._inner = inner

    @property
    def action_set(self) -> list[ActionSchema]:
        return self._inner.action_set

    async def execute_action(self, action: Action) -> Observation | StepError:
        return await asyncio.to_thread(self._inner.execute_action, action)

    async def reset(self) -> None:
        reset = getattr(self._inner, "reset", None)
        if reset is None:
            return
        if asyncio.iscoroutinefunction(reset):
            await reset()
        else:
            await asyncio.to_thread(reset)

    async def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is None:
            return
        if asyncio.iscoroutinefunction(close):
            await close()
        else:
            await asyncio.to_thread(close)

    def __getattr__(self, name: str) -> object:
        # Forward direct attribute / method access to the wrapped tool
        # (same pattern as MonitoredTool — cube-standard tasks call
        # @tool_action methods directly for setup / verification paths).
        if name.startswith("_") or name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def as_async(tool: AbstractTool | AbstractAsyncTool) -> AbstractAsyncTool:
    """Return `tool` as an `AbstractAsyncTool`. No-op when already async;
    wraps in `_SyncToolAsAsync` when sync."""
    if isinstance(tool, AbstractAsyncTool):
        return tool
    if isinstance(tool, AbstractTool):
        return _SyncToolAsAsync(tool)
    raise TypeError(f"as_async expects AbstractTool / AbstractAsyncTool; got {type(tool).__name__}")
