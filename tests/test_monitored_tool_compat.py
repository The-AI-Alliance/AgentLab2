"""Tests for MonitoredTool / AsyncMonitoredTool — drop-in compatibility,
mixed toolbox dispatch, budget enforcement, and install_monitoring.

Design note: the RFC originally specified a single async wrapper, but in
practice every in-tree cube uses sync `Tool` subclasses (and
`Toolbox.execute_action` asserts `isinstance(tool, AbstractTool)`), so
we ship both sync and async wrappers with shared recording logic. See
src/cube_harness/tool.py docstring.
"""

import asyncio

import pytest
from cube.core import Action, ActionSchema, Observation, StepError
from cube.tool import AbstractAsyncTool, AbstractTool, AsyncToolbox, Toolbox

from cube_harness.core import ToolCallEvent, Trajectory
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


# ---------------------------------------------------------------------------
# Drop-in compatibility — same API as wrapped tool
# ---------------------------------------------------------------------------


def test_sync_monitored_tool_exposes_inner_action_set() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    tool = MonitoredTool(_SyncEchoTool(), traj, budget)
    assert [s.name for s in tool.action_set] == ["sync_echo"]


def test_sync_monitored_tool_returns_observation_unchanged() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    tool = MonitoredTool(_SyncEchoTool(), traj, budget)
    result = tool.execute_action(_action("sync_echo", msg="hi"))
    assert isinstance(result, Observation)


def test_async_monitored_tool_returns_observation_unchanged() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    tool = AsyncMonitoredTool(_AsyncEchoTool(), traj, budget)
    result = asyncio.run(tool.execute_action(_action("async_echo", msg="hi")))
    assert isinstance(result, Observation)


def test_monitored_tool_records_event_per_call() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    tool = MonitoredTool(_SyncEchoTool(), traj, budget)
    tool.execute_action(_action("sync_echo", msg="hi"))
    tool.execute_action(_action("sync_echo", msg="bye"))
    assert traj.n_tool_calls == 2
    assert isinstance(traj.events[0].output, ToolCallEvent)
    assert traj.events[0].output.action_id == "id-sync_echo"


def test_wrong_inner_type_raises() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    with pytest.raises(TypeError):
        MonitoredTool(_AsyncEchoTool(), traj, budget)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AsyncMonitoredTool(_SyncEchoTool(), traj, budget)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_monitored_tool_budget_exhaustion_raises() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=0)
    tool = MonitoredTool(_SyncEchoTool(), traj, budget)
    with pytest.raises(BudgetExceeded):
        tool.execute_action(_action("sync_echo"))


def test_budget_exhaustion_via_max_tool_calls() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=100, max_tool_calls=2)
    tool = MonitoredTool(_SyncEchoTool(), traj, budget)
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
    traj = Trajectory(id="t")
    budget = Budget(max_turns=10)
    monitored = MonitoredTool(_SyncEchoTool(), traj, budget)
    bare = _SyncOtherTool()
    box = Toolbox([monitored, bare])

    # Monitored call → event recorded
    box.execute_action(_action("sync_echo", msg="m"))
    # Unmonitored call → no event recorded
    box.execute_action(_action("sync_other"))
    assert traj.n_tool_calls == 1
    # The recorded event is for the monitored tool's action name.
    assert isinstance(traj.events[0].output, ToolCallEvent)
    assert traj.events[0].output.action_id == "id-sync_echo"


def test_async_toolbox_with_mixed_monitored_and_unmonitored() -> None:
    """Symmetric check for AsyncToolbox + AsyncMonitoredTool."""

    class _AsyncOther(AbstractAsyncTool):
        @property
        def action_set(self) -> list[ActionSchema]:
            return [ActionSchema(name="async_other", description="x", parameters={"type": "object", "properties": {}})]

        async def execute_action(self, action: Action) -> Observation | StepError:
            return Observation.from_text("other")

    traj = Trajectory(id="t")
    budget = Budget(max_turns=10)
    monitored = AsyncMonitoredTool(_AsyncEchoTool(), traj, budget)
    bare = _AsyncOther()
    box = AsyncToolbox([monitored, bare])

    asyncio.run(box.execute_action(_action("async_echo", msg="m")))
    asyncio.run(box.execute_action(_action("async_other")))
    assert traj.n_tool_calls == 1


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

    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    wrapped = MonitoredTool(_BashLikeTool(), traj, budget)
    # The direct method call must reach the inner tool unchanged.
    assert wrapped.bash("echo ok", timeout=15) == "bash:echo ok:t15"
    # And it must NOT have recorded a ToolCallEvent — those direct
    # method calls aren't agent tool calls.
    assert traj.n_tool_calls == 0


def test_wrap_tool_picks_sync_or_async_by_inner_type() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    sync_wrapped = wrap_tool(_SyncEchoTool(), traj, budget)
    async_wrapped = wrap_tool(_AsyncEchoTool(), traj, budget)
    assert isinstance(sync_wrapped, MonitoredTool)
    assert isinstance(async_wrapped, AsyncMonitoredTool)


def test_wrap_tool_is_idempotent() -> None:
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    once = wrap_tool(_SyncEchoTool(), traj, budget)
    twice = wrap_tool(once, traj, budget)
    assert twice is once


# ---------------------------------------------------------------------------
# install_monitoring
# ---------------------------------------------------------------------------


class _FakeTask:
    def __init__(self, tools: list[AbstractTool | AbstractAsyncTool], *, attr: str = "toolbox") -> None:
        setattr(self, attr, Toolbox(tools))


def test_install_monitoring_wraps_each_member_in_place() -> None:
    task = _FakeTask([_SyncEchoTool()])
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    install_monitoring(task, traj, budget)
    assert all(isinstance(t, MonitoredTool) for t in task.toolbox.tools)
    assert "sync_echo" in task.toolbox._action_name_to_tool


def test_install_monitoring_is_idempotent() -> None:
    task = _FakeTask([_SyncEchoTool()])
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    install_monitoring(task, traj, budget)
    install_monitoring(task, traj, budget)
    assert len(task.toolbox.tools) == 1
    assert isinstance(task.toolbox.tools[0], MonitoredTool)
    assert not isinstance(task.toolbox.tools[0].inner, MonitoredTool)


def test_install_monitoring_dispatch_records_event() -> None:
    """After install_monitoring, calling task.toolbox.execute_action
    transitively writes a ToolCallEvent."""
    task = _FakeTask([_SyncEchoTool()])
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    install_monitoring(task, traj, budget)
    task.toolbox.execute_action(_action("sync_echo", msg="hi"))
    assert traj.n_tool_calls == 1


def test_install_monitoring_recurses_into_nested_toolboxes() -> None:
    inner_box = Toolbox([_SyncEchoTool()])
    outer_box = Toolbox([inner_box, _SyncOtherTool()])

    class _Task:
        def __init__(self) -> None:
            self.toolbox = outer_box

    task = _Task()
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    install_monitoring(task, traj, budget)
    # Every leaf is wrapped; toolboxes stay as toolboxes.
    assert isinstance(outer_box.tools[0], Toolbox)
    assert isinstance(outer_box.tools[0].tools[0], MonitoredTool)
    assert isinstance(outer_box.tools[1], MonitoredTool)


def test_install_monitoring_with_parent_event_id_getter() -> None:
    """parent_event_id_getter is late-bound to the recorder's current turn.
    Recorded events carry the value the getter returns at call time."""
    task = _FakeTask([_SyncEchoTool()])
    traj = Trajectory(id="t")
    budget = Budget(max_turns=5)
    current_turn = {"v": "agent-001"}
    install_monitoring(task, traj, budget, parent_event_id_getter=lambda: current_turn["v"])
    task.toolbox.execute_action(_action("sync_echo"))
    current_turn["v"] = "agent-002"
    task.toolbox.execute_action(_action("sync_echo"))
    assert traj.events[0].output.parent_event_id == "agent-001"
    assert traj.events[1].output.parent_event_id == "agent-002"
