"""Tests for MonitoredTool / AsyncMonitoredTool — drop-in compatibility,
mixed toolbox dispatch, budget enforcement, and install_monitoring.

Design note: the RFC originally specified a single async wrapper, but in
practice every in-tree cube uses sync `Tool` subclasses (and
`Toolbox.execute_action` asserts `isinstance(tool, AbstractTool)`), so
we ship both sync and async wrappers with shared recording logic. See
src/cube_harness/tool.py docstring.

Post-Trajectory-removal: events stream to a Storage hook; tests inspect
what was sent there rather than walking an in-memory list.
"""

import asyncio
from typing import Callable

import pytest
from cube.core import Action, ActionSchema, Observation, StepError
from cube.tool import AbstractAsyncTool, AbstractTool, AsyncToolbox, Toolbox

from cube_harness.core import ToolCallEvent, TrajectoryEvent
from cube_harness.tool import (
    AsyncMonitoredTool,
    Budget,
    BudgetExceeded,
    MonitoredTool,
    install_monitoring,
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


class _AsyncEchoTool(AbstractAsyncTool):
    @property
    def action_set(self) -> list[ActionSchema]:
        return [ActionSchema(name="async_echo", description="echo", parameters={"type": "object", "properties": {}})]

    async def execute_action(self, action: Action) -> Observation | StepError:
        await asyncio.sleep(0)
        return Observation.from_text(f"async:{action.arguments.get('msg', '')}")


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
    inner: AbstractTool | AbstractAsyncTool,
    budget: Budget,
    *,
    storage: _FakeStorage | None = None,
    parent_event_id_getter: Callable[[], str] | None = None,
) -> MonitoredTool | AsyncMonitoredTool:
    storage = storage if storage is not None else _FakeStorage()
    # Inline a minimal emit callable that forwards to the test FakeStorage.
    # Mirrors what `EventStreamer.emit` does — no stats fold needed since
    # these tests inspect storage directly, not stats counters.

    def emit(te: TrajectoryEvent) -> str:
        storage.save_event(te, "t")
        return te.output.id

    if isinstance(inner, AbstractAsyncTool):
        return AsyncMonitoredTool(
            inner,
            emit=emit,
            budget=budget,
            parent_event_id_getter=parent_event_id_getter,
        )
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
    budget = Budget(max_turns=5)
    tool = _make_monitored(_SyncEchoTool(), budget)
    assert [s.name for s in tool.action_set] == ["sync_echo"]


def test_sync_monitored_tool_returns_observation_unchanged() -> None:
    budget = Budget(max_turns=5)
    tool = _make_monitored(_SyncEchoTool(), budget)
    result = tool.execute_action(_action("sync_echo", msg="hi"))
    assert isinstance(result, Observation)


def test_async_monitored_tool_returns_observation_unchanged() -> None:
    budget = Budget(max_turns=5)
    tool = _make_monitored(_AsyncEchoTool(), budget)
    result = asyncio.run(tool.execute_action(_action("async_echo", msg="hi")))
    assert isinstance(result, Observation)


def test_monitored_tool_records_event_per_call() -> None:
    budget = Budget(max_turns=5)
    storage = _FakeStorage()
    tool = _make_monitored(_SyncEchoTool(), budget, storage=storage)
    tool.execute_action(_action("sync_echo", msg="hi"))
    tool.execute_action(_action("sync_echo", msg="bye"))
    events = storage.tool_call_events()
    assert len(events) == 2
    assert events[0].action_id == "id-sync_echo"


def test_wrong_inner_type_raises() -> None:
    budget = Budget(max_turns=5)
    with pytest.raises(TypeError):
        MonitoredTool(_AsyncEchoTool(), trajectory_id="t", budget=budget)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AsyncMonitoredTool(_SyncEchoTool(), trajectory_id="t", budget=budget)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_monitored_tool_budget_exhaustion_raises() -> None:
    budget = Budget(max_turns=0)
    tool = _make_monitored(_SyncEchoTool(), budget)
    with pytest.raises(BudgetExceeded):
        tool.execute_action(_action("sync_echo"))


def test_budget_exhaustion_via_max_tool_calls() -> None:
    budget = Budget(max_turns=100, max_tool_calls=2)
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
    """A sync Toolbox can contain MonitoredTool wrappers AND bare tools
    side-by-side. Dispatch by action name routes each call correctly.

    Validates the RFC design goal: MonitoredTool is API-identical to any
    AbstractTool from the caller's perspective. The agent doesn't know
    or care which tools are monitored."""
    budget = Budget(max_turns=10)
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


def test_async_toolbox_with_mixed_monitored_and_unmonitored() -> None:
    """Symmetric check for AsyncToolbox + AsyncMonitoredTool."""

    class _AsyncOther(AbstractAsyncTool):
        @property
        def action_set(self) -> list[ActionSchema]:
            return [ActionSchema(name="async_other", description="x", parameters={"type": "object", "properties": {}})]

        async def execute_action(self, action: Action) -> Observation | StepError:
            return Observation.from_text("other")

    budget = Budget(max_turns=10)
    storage = _FakeStorage()
    monitored = _make_monitored(_AsyncEchoTool(), budget, storage=storage)
    bare = _AsyncOther()
    box = AsyncToolbox([monitored, bare])

    asyncio.run(box.execute_action(_action("async_echo", msg="m")))
    asyncio.run(box.execute_action(_action("async_other")))
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

    budget = Budget(max_turns=5)
    storage = _FakeStorage()
    wrapped = _make_monitored(_BashLikeTool(), budget, storage=storage)
    # The direct method call must reach the inner tool unchanged.
    assert wrapped.bash("echo ok", timeout=15) == "bash:echo ok:t15"
    # And it must NOT have recorded a ToolCallEvent — those direct
    # method calls aren't agent tool calls.
    assert len(storage.tool_call_events()) == 0


def _noop_emit(te: TrajectoryEvent) -> str:
    return te.output.id


def test_wrap_tool_picks_sync_or_async_by_inner_type() -> None:
    budget = Budget(max_turns=5)
    sync_wrapped = wrap_tool(_SyncEchoTool(), emit=_noop_emit, budget=budget)
    async_wrapped = wrap_tool(_AsyncEchoTool(), emit=_noop_emit, budget=budget)
    assert isinstance(sync_wrapped, MonitoredTool)
    assert isinstance(async_wrapped, AsyncMonitoredTool)


def test_wrap_tool_is_idempotent() -> None:
    budget = Budget(max_turns=5)
    once = wrap_tool(_SyncEchoTool(), emit=_noop_emit, budget=budget)
    twice = wrap_tool(once, emit=_noop_emit, budget=budget)
    assert twice is once


# ---------------------------------------------------------------------------
# install_monitoring
# ---------------------------------------------------------------------------


class _FakeTask:
    def __init__(self, tools: list[AbstractTool | AbstractAsyncTool], *, attr: str = "toolbox") -> None:
        setattr(self, attr, Toolbox(tools))


def _make_streamer(
    budget: Budget,
    storage: _FakeStorage | None = None,
    parent_event_id_getter: Callable[[], str] | None = None,
) -> object:
    """Lightweight streamer stand-in matching the duck-typed surface
    `install_monitoring` reads: `.emit`, `.budget`,
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


def test_install_monitoring_wraps_each_member_in_place() -> None:
    task = _FakeTask([_SyncEchoTool()])
    install_monitoring(task, _make_streamer(Budget(max_turns=5)))
    assert all(isinstance(t, MonitoredTool) for t in task.toolbox.tools)
    assert "sync_echo" in task.toolbox._action_name_to_tool


def test_install_monitoring_is_idempotent() -> None:
    task = _FakeTask([_SyncEchoTool()])
    budget = Budget(max_turns=5)
    install_monitoring(task, _make_streamer(budget))
    install_monitoring(task, _make_streamer(budget))
    assert len(task.toolbox.tools) == 1
    assert isinstance(task.toolbox.tools[0], MonitoredTool)
    assert not isinstance(task.toolbox.tools[0].inner, MonitoredTool)


def test_install_monitoring_dispatch_records_event() -> None:
    """After install_monitoring, calling task.toolbox.execute_action
    transitively writes a ToolCallEvent."""
    task = _FakeTask([_SyncEchoTool()])
    storage = _FakeStorage()
    install_monitoring(task, _make_streamer(Budget(max_turns=5), storage=storage))
    task.toolbox.execute_action(_action("sync_echo", msg="hi"))
    assert len(storage.tool_call_events()) == 1


def test_install_monitoring_recurses_into_nested_toolboxes() -> None:
    inner_box = Toolbox([_SyncEchoTool()])
    outer_box = Toolbox([inner_box, _SyncOtherTool()])

    class _Task:
        def __init__(self) -> None:
            self.toolbox = outer_box

    task = _Task()
    install_monitoring(task, _make_streamer(Budget(max_turns=5)))
    # Every leaf is wrapped; toolboxes stay as toolboxes.
    assert isinstance(outer_box.tools[0], Toolbox)
    assert isinstance(outer_box.tools[0].tools[0], MonitoredTool)
    assert isinstance(outer_box.tools[1], MonitoredTool)


def test_install_monitoring_with_parent_event_id_getter() -> None:
    """parent_event_id_getter is late-bound to the streamer's current turn.
    Recorded events carry the value the getter returns at call time."""
    task = _FakeTask([_SyncEchoTool()])
    storage = _FakeStorage()
    current_turn = {"v": "agent-001"}
    install_monitoring(
        task,
        _make_streamer(Budget(max_turns=5), storage=storage, parent_event_id_getter=lambda: current_turn["v"]),
    )
    task.toolbox.execute_action(_action("sync_echo"))
    current_turn["v"] = "agent-002"
    task.toolbox.execute_action(_action("sync_echo"))
    events = storage.tool_call_events()
    assert events[0].parent_event_id == "agent-001"
    assert events[1].parent_event_id == "agent-002"
