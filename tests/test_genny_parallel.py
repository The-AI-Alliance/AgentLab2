"""GennyParallel tests — verifies parallel tool-call dispatch records
sibling ToolCallEvents under one AgentEvent (same turn_id), with
budget enforcement still firing across the parallel calls.

Uses a hand-rolled mock task + a thin agent override to avoid hitting
a real LLM; the structural assertion (sibling ToolCallEvents in one
turn) is what we care about. Real-LLM behavioral check happens in
Phase K smokes."""

import asyncio
import time

from cube.core import Action, ActionSchema, Observation
from cube.tool import AbstractTool

from cube_harness.agents.genny_parallel import GennyParallel
from cube_harness.core import AgentEvent, AgentOutput, ToolCallEvent, Trajectory
from cube_harness.recorder import TurnRecorder
from cube_harness.tool import Budget, install_monitoring


class _SleepyTool(AbstractTool):
    """Tool whose action_set declares two actions; each `sleep(ms)` waits
    `ms` milliseconds before returning. Lets us check that parallel
    dispatch wins on wall-clock time."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def action_set(self) -> list[ActionSchema]:
        return [
            ActionSchema(
                name="sleep",
                description="wait ms milliseconds",
                parameters={"type": "object", "properties": {"ms": {"type": "integer"}}},
            )
        ]

    def execute_action(self, action: Action) -> Observation:
        ms = int(action.arguments.get("ms", 0))
        time.sleep(ms / 1000.0)
        self.calls += 1
        return Observation.from_text(f"slept-{ms}ms-{self.calls}")


class _FakeTask:
    """Minimal Task look-alike for GennyParallel: exposes `.tool`
    (matching cube-standard's Task.tool attribute)."""

    def __init__(self) -> None:
        self.tool = _SleepyTool()


class _ScriptedParallel(GennyParallel):
    """GennyParallel subclass that bypasses the real LLM by returning a
    pre-set AgentOutput from `step()`. Used to drive the parallel
    `run()` path deterministically.

    The first call to step() emits N parallel `sleep` actions; the
    second call emits empty actions to graceful-stop the loop.
    """

    def __init__(self, n_actions: int, sleep_ms: int) -> None:  # noqa: D401
        # Skip Genny's __init__ — we don't need the LLM machinery.
        self._n_actions = n_actions
        self._sleep_ms = sleep_ms
        self._called = 0

    def step(self, obs: Observation) -> AgentOutput:
        self._called += 1
        if self._called > 1:
            return AgentOutput(actions=[])  # done
        return AgentOutput(
            actions=[
                Action(id=f"a-{i}", name="sleep", arguments={"ms": self._sleep_ms}) for i in range(self._n_actions)
            ],
            thoughts=f"firing {self._n_actions} parallel calls",
        )


def test_parallel_dispatch_records_sibling_tool_calls() -> None:
    """N actions returned from one assistant turn should produce N
    sibling ToolCallEvents sharing the parent AgentEvent's id as
    turn_id — the back-reference invariant the RFC asks for."""
    task = _FakeTask()
    traj = Trajectory(id="t")
    budget = Budget(max_turns=10)
    recorder = TurnRecorder(traj, budget=budget)
    install_monitoring(task, traj, budget, parent_event_id_getter=recorder.current_turn_id)

    agent = _ScriptedParallel(n_actions=4, sleep_ms=20)
    asyncio.run(agent.run(initial_obs=Observation(), task=task, recorder=recorder))

    # One AgentEvent + 4 sibling ToolCallEvents + a graceful-stop
    # AgentEvent (the second step returned empty actions).
    agent_events = [e for e in traj.events if isinstance(e.output, AgentEvent)]
    tool_calls = [e for e in traj.events if isinstance(e.output, ToolCallEvent)]
    assert len(agent_events) == 2
    assert len(tool_calls) == 4
    # All four tool calls share the same turn_id (the parent
    # AgentEvent's id) — XRay uses this to render them as siblings.
    parent_id = agent_events[0].output.id
    assert all(t.output.turn_id == parent_id for t in tool_calls)
    assert all(t.output.parent_event_id == parent_id for t in tool_calls)


def test_parallel_dispatch_is_faster_than_serial() -> None:
    """Wall-clock check: 4 × 50ms sleeps in parallel must run in well
    under 200ms (their serial sum). 130ms gives a safe margin for
    thread-pool startup variance on slower CI runners."""
    task = _FakeTask()
    traj = Trajectory(id="t")
    budget = Budget(max_turns=10)
    recorder = TurnRecorder(traj, budget=budget)
    install_monitoring(task, traj, budget, parent_event_id_getter=recorder.current_turn_id)

    agent = _ScriptedParallel(n_actions=4, sleep_ms=50)
    start = time.time()
    asyncio.run(agent.run(initial_obs=Observation(), task=task, recorder=recorder))
    elapsed = time.time() - start
    # Serial: 4 × 50ms = 200ms.  Parallel: ~50ms + overhead.
    assert elapsed < 0.13, f"parallel dispatch took {elapsed:.3f}s — slower than expected"


def test_budget_still_fires_across_parallel_calls() -> None:
    """Budget.max_tool_calls counts across parallel dispatch too —
    each MonitoredTool.execute_action bumps the counter."""
    from cube_harness.tool import BudgetExceeded

    task = _FakeTask()
    traj = Trajectory(id="t")
    # max_tool_calls=2 — the agent fires 4 in one turn; the 3rd should
    # raise BudgetExceeded.
    budget = Budget(max_turns=10, max_tool_calls=2)
    recorder = TurnRecorder(traj, budget=budget)
    install_monitoring(task, traj, budget, parent_event_id_getter=recorder.current_turn_id)

    agent = _ScriptedParallel(n_actions=4, sleep_ms=10)
    raised: list[BaseException] = []
    try:
        asyncio.run(agent.run(initial_obs=Observation(), task=task, recorder=recorder))
    except BaseException as e:  # noqa: BLE001
        raised.append(e)
    assert any(isinstance(e, BudgetExceeded) for e in raised)
