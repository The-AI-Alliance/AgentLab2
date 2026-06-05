"""Tests for MonitoredTool — drop-in compatibility,
mixed toolbox dispatch, budget enforcement, and build_monitored_env_tool.

Post-tool-consolidation: one `MonitoredTool` wraps any `cube.tool.Tool`,
whether its `@tool_action` methods are sync, async, or a mix. The inner
`Tool` handles per-method dispatch (bridging via thread+loop or
`asyncio.to_thread` when needed); `MonitoredTool` just adds budget
enforcement, `ToolCallEvent` emission, and the `Task.step` wrapping.
"""

import asyncio
from typing import Callable

import pytest
from cube.core import Action, ActionSchema, Observation, StepError
from cube.task import STOP_ACTION
from cube.tool import AbstractTool, Tool, Toolbox, tool_action

from cube_harness.core import ToolCallEvent, TrajectoryEvent
from cube_harness.tool import (
    Budget,
    BudgetExceeded,
    MonitoredTool,
    build_monitored_env_tool,
    wrap_tool,
)


class _SyncEchoTool(AbstractTool):
    @property
    def action_set(self) -> list[ActionSchema]:
        return [ActionSchema(name="sync_echo", description="echo", parameters={"type": "object", "properties": {}})]

    def execute_action(self, action: Action) -> Observation | StepError:
        return Observation.from_text(f"sync:{action.arguments.get('msg', '')}")


class _SyncOtherTool(AbstractTool):
    """Second sync tool with a distinct action name — used to validate that
    a Toolbox dispatches monitored + unmonitored siblings correctly."""

    @property
    def action_set(self) -> list[ActionSchema]:
        return [ActionSchema(name="sync_other", description="echo", parameters={"type": "object", "properties": {}})]

    def execute_action(self, action: Action) -> Observation | StepError:
        return Observation.from_text("other")


class _AsyncEchoTool(Tool):
    """A `Tool` with an async `@tool_action` — the post-consolidation way
    to express what used to be `AbstractAsyncTool`. Dispatch routes per
    method's kind: `execute_action` bridges via thread+loop,
    `async_execute_action` awaits directly."""

    @tool_action
    async def async_echo(self, msg: str = "") -> str:
        """Echo the given message."""
        await asyncio.sleep(0)
        return f"async:{msg}"


def _action(name: str, **args: object) -> Action:
    return Action(id=f"id-{name}", name=name, arguments=args)


class _FakeStorage:
    """Captures every save_event call so tests can inspect what
    MonitoredTool streamed without needing a real FileStorage.

    Storage assigns + returns the event_num (matches the new
    `Storage.save_event(event, trajectory_id) -> int` contract)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int, TrajectoryEvent]] = []
        self._next_num = 0

    def save_event(self, te: TrajectoryEvent, trajectory_id: str) -> int:
        n = self._next_num
        self._next_num += 1
        self.events.append((trajectory_id, n, te))
        return n

    def tool_call_events(self) -> list[ToolCallEvent]:
        """Convenience: only the ToolCallEvent outputs in order."""
        return [te.output for _, _, te in self.events if isinstance(te.output, ToolCallEvent)]


def _make_monitored(
    inner: AbstractTool,
    budget: Budget,
    *,
    storage: _FakeStorage | None = None,
    parent_event_id_getter: Callable[[], str] | None = None,
) -> MonitoredTool:
    """Construct a MonitoredTool. One class wraps any inner kind."""
    storage = storage if storage is not None else _FakeStorage()

    def emit(te: TrajectoryEvent) -> str:
        storage.save_event(te, "t")
        return te.output.id

    return MonitoredTool(
        inner,
        emit=emit,
        budget=budget,
        parent_event_id_getter=parent_event_id_getter,
    )


# ---------------------------------------------------------------------------
# Drop-in compatibility — same API as wrapped tool
# ---------------------------------------------------------------------------


def test_sync_monitored_tool_exposes_inner_action_set() -> None:
    budget = Budget(max_agent_steps=5)
    tool = _make_monitored(_SyncEchoTool(), budget)
    assert [s.name for s in tool.action_set] == ["sync_echo"]


def test_sync_monitored_tool_returns_observation_unchanged() -> None:
    budget = Budget(max_agent_steps=5)
    tool = _make_monitored(_SyncEchoTool(), budget)
    result = tool.execute_action(_action("sync_echo", msg="hi"))
    assert isinstance(result, Observation)


def test_async_action_via_async_execute_action() -> None:
    """A `Tool` with an async `@tool_action` is dispatched directly when
    the caller goes through `async_execute_action` — no thread hop."""
    budget = Budget(max_agent_steps=5)
    tool = _make_monitored(_AsyncEchoTool(), budget)
    result = asyncio.run(tool.async_execute_action(_action("async_echo", msg="hi")))
    assert isinstance(result, Observation)


def test_async_action_via_sync_execute_action_bridges() -> None:
    """Calling sync `execute_action` on a tool with an async `@tool_action`
    bridges through a one-shot worker thread + event loop. The caller
    gets a normal `Observation` synchronously — no TypeError."""
    budget = Budget(max_agent_steps=5)
    tool = _make_monitored(_AsyncEchoTool(), budget)
    result = tool.execute_action(_action("async_echo", msg="hi"))
    assert isinstance(result, Observation)


def test_sync_action_via_async_execute_action() -> None:
    """Sync action through async dispatch hops through `asyncio.to_thread`
    inside `Tool.async_execute_action`. Lets one code path serve both
    method kinds; `asyncio.gather` over N sync actions runs them in real
    OS-thread parallel."""
    budget = Budget(max_agent_steps=5)
    tool = _make_monitored(_SyncEchoTool(), budget)
    result = asyncio.run(tool.async_execute_action(_action("sync_echo", msg="hi")))
    assert isinstance(result, Observation)


def test_monitored_tool_records_event_per_call() -> None:
    budget = Budget(max_agent_steps=5)
    storage = _FakeStorage()
    tool = _make_monitored(_SyncEchoTool(), budget, storage=storage)
    tool.execute_action(_action("sync_echo", msg="hi"))
    tool.execute_action(_action("sync_echo", msg="bye"))
    events = storage.tool_call_events()
    assert len(events) == 2
    assert events[0].action_id == "id-sync_echo"


def test_wrong_inner_type_raises() -> None:
    """MonitoredTool accepts any `AbstractTool` — not arbitrary objects."""

    class _NotATool:
        pass

    budget = Budget(max_agent_steps=5)
    with pytest.raises(TypeError, match="AbstractTool"):
        MonitoredTool(
            _NotATool(),  # type: ignore[arg-type]
            emit=lambda te: te.output.id,
            budget=budget,
        )


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_monitored_tool_budget_exhaustion_raises() -> None:
    budget = Budget(max_agent_steps=0)
    tool = _make_monitored(_SyncEchoTool(), budget)
    with pytest.raises(BudgetExceeded):
        tool.execute_action(_action("sync_echo"))


def test_budget_exhaustion_via_max_tool_calls() -> None:
    budget = Budget(max_agent_steps=100, max_tool_calls=2)
    tool = _make_monitored(_SyncEchoTool(), budget)
    tool.execute_action(_action("sync_echo"))
    tool.execute_action(_action("sync_echo"))
    with pytest.raises(BudgetExceeded):
        tool.execute_action(_action("sync_echo"))


def test_budget_exceeded_is_base_exception() -> None:
    """BudgetExceeded must NOT be caught by `except Exception:`."""
    try:
        raise BudgetExceeded()
    except Exception:  # noqa: BLE001
        raise AssertionError("BudgetExceeded must subclass BaseException, not Exception") from None
    except BudgetExceeded:
        pass


# ---------------------------------------------------------------------------
# Toolbox mixing — the key drop-in property the RFC asks for
# ---------------------------------------------------------------------------


def test_sync_toolbox_with_mixed_monitored_and_unmonitored() -> None:
    """A `Toolbox` can contain `MonitoredTool` wrappers AND bare tools
    side-by-side. Dispatch by action name routes each call correctly.

    Validates the design goal: `MonitoredTool` is API-identical to any
    `AbstractTool` from the caller's perspective. The agent doesn't
    know or care which tools are monitored."""
    budget = Budget(max_agent_steps=10)
    storage = _FakeStorage()
    monitored = _make_monitored(_SyncEchoTool(), budget, storage=storage)
    bare = _SyncOtherTool()
    box = Toolbox([monitored, bare])

    # Monitored call → event recorded
    box.execute_action(_action("sync_echo", msg="m"))
    # Unmonitored call → no event recorded
    box.execute_action(_action("sync_other"))
    events = storage.tool_call_events()
    assert len(events) == 1
    assert events[0].action_id == "id-sync_echo"


def test_toolbox_async_dispatch_over_mixed_leaves() -> None:
    """A single `Toolbox` may hold a mix of sync-action and async-action
    leaves; `async_execute_action` routes per the leaf's action method's
    kind. Only the monitored leaf records a ToolCallEvent."""

    class _AsyncOther(Tool):
        @tool_action
        async def async_other(self) -> str:
            """Bare async tool."""
            return "other"

    budget = Budget(max_agent_steps=10)
    storage = _FakeStorage()
    monitored = _make_monitored(_AsyncEchoTool(), budget, storage=storage)
    bare = _AsyncOther()
    box = Toolbox([monitored, bare])

    asyncio.run(box.async_execute_action(_action("async_echo", msg="m")))
    asyncio.run(box.async_execute_action(_action("async_other")))
    assert len(storage.tool_call_events()) == 1


# ---------------------------------------------------------------------------
# wrap_tool factory
# ---------------------------------------------------------------------------


def test_monitored_tool_forwards_direct_method_calls_to_inner() -> None:
    """Regression: cube-standard tasks call @tool_action methods directly
    (e.g. terminalbench2's `task.tool.bash(...)`) for setup / verification
    / oracle paths. Those direct calls must reach the inner tool — they
    are NOT agent tool calls and must not be recorded as ToolCallEvents.
    Without __getattr__ delegation, `monitored_tool.bash(...)` raises
    AttributeError as soon as Episode installs monitoring."""

    class _BashLikeTool(AbstractTool):
        @property
        def action_set(self) -> list[ActionSchema]:
            return [ActionSchema(name="run", description="x", parameters={"type": "object", "properties": {}})]

        def execute_action(self, action: Action) -> Observation:
            return Observation.from_text("noop")

        def bash(self, cmd: str, timeout: int = 0) -> str:
            return f"bash:{cmd}:t{timeout}"

    budget = Budget(max_agent_steps=5)
    storage = _FakeStorage()
    wrapped = _make_monitored(_BashLikeTool(), budget, storage=storage)
    # The direct method call must reach the inner tool unchanged.
    assert wrapped.bash("echo ok", timeout=15) == "bash:echo ok:t15"
    # And it must NOT have recorded a ToolCallEvent — those direct
    # method calls aren't agent tool calls.
    assert len(storage.tool_call_events()) == 0


def _noop_emit(te: TrajectoryEvent) -> str:
    return te.output.id


def test_wrap_tool_returns_monitored_tool_for_both_action_kinds() -> None:
    """One `MonitoredTool` class handles both sync-action and async-action inners."""
    budget = Budget(max_agent_steps=5)
    sync_wrapped = wrap_tool(_SyncEchoTool(), emit=_noop_emit, budget=budget)
    async_wrapped = wrap_tool(_AsyncEchoTool(), emit=_noop_emit, budget=budget)
    assert isinstance(sync_wrapped, MonitoredTool)
    assert isinstance(async_wrapped, MonitoredTool)


def test_wrap_tool_is_idempotent() -> None:
    budget = Budget(max_agent_steps=5)
    once = wrap_tool(_SyncEchoTool(), emit=_noop_emit, budget=budget)
    twice = wrap_tool(once, emit=_noop_emit, budget=budget)
    assert twice is once


# ---------------------------------------------------------------------------
# build_monitored_env_tool
# ---------------------------------------------------------------------------


class _FakeTask:
    def __init__(self, tools: list[AbstractTool], *, attr: str = "toolbox") -> None:
        setattr(self, attr, Toolbox(tools))


def _make_streamer(
    budget: Budget,
    storage: _FakeStorage | None = None,
    parent_event_id_getter: Callable[[], str] | None = None,
) -> object:
    """Lightweight streamer stand-in matching the duck-typed surface
    `build_monitored_env_tool` reads: `.emit`, `.budget`,
    `.current_parent_event_id`. Avoids importing EventStreamer to
    keep this test module agnostic of streamer wiring."""

    class _S:
        def __init__(self) -> None:
            self.budget = budget
            self._store = storage if storage is not None else _FakeStorage()

        def emit(self, te: TrajectoryEvent) -> str:
            self._store.save_event(te, "t")
            return te.output.id

        def current_parent_event_id(self) -> str:
            return parent_event_id_getter() if parent_event_id_getter is not None else "reset"

    s = _S()
    s.storage = s._store  # noqa: SLF001 — convenience accessor used by tests below
    return s


def test_build_monitored_env_tool_wraps_each_member() -> None:
    task = _FakeTask([_SyncEchoTool()])
    env_tool = build_monitored_env_tool(task, _make_streamer(Budget(max_agent_steps=5)))
    assert all(isinstance(t, MonitoredTool) for t in env_tool.tools)
    assert "sync_echo" in env_tool._action_name_to_tool


def test_build_monitored_env_tool_does_not_mutate_task_tool() -> None:
    """The task keeps its concrete tool — only the returned env_tool is
    monitored. This is the contract that lets task.evaluate/setup reach
    concrete-tool methods, private attrs, and isinstance/find_tool."""
    task = _FakeTask([_SyncEchoTool()])
    original_leaves = list(task.toolbox.tools)
    env_tool = build_monitored_env_tool(task, _make_streamer(Budget(max_agent_steps=5)))
    # task's toolbox is untouched: same object, same concrete leaves.
    assert task.toolbox.tools == original_leaves
    assert all(not isinstance(t, MonitoredTool) for t in task.toolbox.tools)
    # env_tool is a distinct toolbox whose wrappers share the SAME inner.
    assert env_tool is not task.toolbox
    assert env_tool.tools[0].inner is task.toolbox.tools[0]
    # find_tool by concrete type still resolves on the task's toolbox.
    assert task.toolbox.find_tool(_SyncEchoTool) is task.toolbox.tools[0]


def test_build_monitored_env_tool_is_idempotent() -> None:
    task = _FakeTask([_SyncEchoTool()])
    budget = Budget(max_agent_steps=5)
    env_tool = build_monitored_env_tool(task, _make_streamer(budget))
    assert len(env_tool.tools) == 1
    assert isinstance(env_tool.tools[0], MonitoredTool)
    assert not isinstance(env_tool.tools[0].inner, MonitoredTool)


def test_build_monitored_env_tool_dispatch_records_event() -> None:
    """Calling execute_action on the returned env_tool writes a
    ToolCallEvent; the task's own concrete toolbox does NOT record."""
    task = _FakeTask([_SyncEchoTool()])
    storage = _FakeStorage()
    env_tool = build_monitored_env_tool(task, _make_streamer(Budget(max_agent_steps=5), storage=storage))
    env_tool.execute_action(_action("sync_echo", msg="hi"))
    assert len(storage.tool_call_events()) == 1
    # The task's own concrete tool calls are NOT monitored.
    task.toolbox.execute_action(_action("sync_echo", msg="hi"))
    assert len(storage.tool_call_events()) == 1


def test_build_monitored_env_tool_recurses_into_nested_toolboxes() -> None:
    inner_box = Toolbox([_SyncEchoTool()])
    outer_box = Toolbox([inner_box, _SyncOtherTool()])

    class _Task:
        def __init__(self) -> None:
            self.toolbox = outer_box

    task = _Task()
    env_tool = build_monitored_env_tool(task, _make_streamer(Budget(max_agent_steps=5)))
    # Returned env_tool: every leaf wrapped; toolboxes stay as toolboxes.
    assert isinstance(env_tool.tools[0], Toolbox)
    assert isinstance(env_tool.tools[0].tools[0], MonitoredTool)
    assert isinstance(env_tool.tools[1], MonitoredTool)
    # Original toolbox tree is untouched (concrete leaves).
    assert isinstance(outer_box.tools[0].tools[0], _SyncEchoTool)
    assert isinstance(outer_box.tools[1], _SyncOtherTool)


def test_build_monitored_env_tool_surfaces_stop_once_for_multi_leaf() -> None:
    """F4: a multi-leaf monitored toolbox must advertise STOP_ACTION exactly
    once. Otherwise `Toolbox.__init__`'s duplicate-name guard trips, and
    the LLM gets duplicate stop tool schemas."""

    class _StopTask:
        accept_agent_stop = True

        def __init__(self) -> None:
            self.toolbox = Toolbox([_SyncEchoTool(), _SyncOtherTool()])

    task = _StopTask()
    env_tool = build_monitored_env_tool(task, _make_streamer(Budget(max_agent_steps=5)))
    stop_count = sum(a.name == STOP_ACTION.name for a in env_tool.action_set)
    assert stop_count == 1, f"expected exactly one STOP_ACTION, got {stop_count}"
    # Rebuilding a fresh Toolbox over the same leaves must not trip
    # its duplicate action-name guard.
    Toolbox(tools=env_tool.tools)


def test_build_monitored_env_tool_with_parent_event_id_getter() -> None:
    """parent_event_id_getter is late-bound to the streamer's current turn.
    Recorded events carry the value the getter returns at call time."""
    task = _FakeTask([_SyncEchoTool()])
    storage = _FakeStorage()
    current_turn = {"v": "agent-001"}
    env_tool = build_monitored_env_tool(
        task,
        _make_streamer(Budget(max_agent_steps=5), storage=storage, parent_event_id_getter=lambda: current_turn["v"]),
    )
    env_tool.execute_action(_action("sync_echo"))
    current_turn["v"] = "agent-002"
    env_tool.execute_action(_action("sync_echo"))
    events = storage.tool_call_events()
    assert events[0].parent_event_id == "agent-001"
    assert events[1].parent_event_id == "agent-002"
