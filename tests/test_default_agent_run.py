"""Default Agent.run tests — verify the base class implementation
reproduces today's gym-style loop and integrates with TurnRecorder +
MonitoredTool.

Uses a hand-rolled mock task (no LLM, no cube), so the test runs fast
and deterministically. The structural-parity check for real cubes lives
in Phase G (`cube test <name>` for every in-tree benchmark).

Post-Trajectory-removal: events stream to a Storage hook; tests inspect
the captured event stream rather than walking an in-memory list.
"""

import asyncio

from cube.core import Action, ActionSchema, Observation
from cube.tool import AbstractTool

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentEvent, AgentOutput, ToolCallEvent, TrajectoryEvent
from cube_harness.recorder import EventCounter, TurnRecorder
from cube_harness.tool import Budget, install_monitoring

# ---------------------------------------------------------------------------
# Mock pieces: a tiny "task" with a sync step that increments a counter
# ---------------------------------------------------------------------------


class _CounterTool(AbstractTool):
    """Single-action tool. The action just bumps the task's counter."""

    def __init__(self, task: "_MockTask") -> None:
        self._task = task

    @property
    def action_set(self) -> list[ActionSchema]:
        return [ActionSchema(name="inc", description="bump counter", parameters={"type": "object", "properties": {}})]

    def execute_action(self, action: Action) -> Observation:
        self._task.counter += 1
        return Observation.from_text(f"counter={self._task.counter}")


class _MockTask:
    """Minimal Task look-alike: exposes `toolbox` (a single Tool, the
    counter) and a `finished(obs)` hook that returns True after N counter
    increments. MonitoredTool polls `finished()` after each tool call
    and raises TaskDone — that's how the loop terminates under the
    agent-owns-loop design."""

    def __init__(self, done_after_n: int = 3) -> None:
        self.counter = 0
        self.done_after_n = done_after_n
        self.toolbox = _CounterTool(self)
        self.accept_agent_stop = True
        self.validate_per_step = False

    def finished(self, obs=None) -> bool:
        _ = obs
        return self.counter >= self.done_after_n

    def obs_postprocess(self, obs: Observation) -> Observation:
        return obs


class _CounterAgentConfig(AgentConfig):
    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "Agent":
        return _CounterAgent(self)


class _CounterAgent(Agent):
    """Scripted "agent" — every step emits one inc action. Mirrors what
    a deterministic debug agent does in a cube's debug.py."""

    name = "counter-agent"
    description = "increments a counter every step"
    input_content_types = ["text/plain"]
    output_content_types = ["application/json"]

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self.steps_taken = 0

    def step(self, obs: Observation) -> AgentOutput:
        self.steps_taken += 1
        return AgentOutput(
            actions=[Action(id=f"a-{self.steps_taken}", name="inc", arguments={})],
            thoughts=f"bumping at step {self.steps_taken}",
        )


def _action(name: str = "inc") -> Action:
    return Action(id=f"id-{name}", name=name, arguments={})


class _FakeStorage:
    """Captures every save_event call so the default-run tests can
    inspect what the recorder + MonitoredTool streamed."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int, TrajectoryEvent]] = []

    def save_event(self, te: TrajectoryEvent, trajectory_id: str, n: int) -> None:
        self.events.append((trajectory_id, n, te))

    def outputs(self) -> list:
        return [te.output for _, _, te in self.events]


def _setup(task, budget: Budget) -> tuple[TurnRecorder, _FakeStorage]:
    """Build TurnRecorder + storage + install monitoring with a shared
    EventCounter — the way Episode does it."""
    storage = _FakeStorage()
    counter = EventCounter()
    recorder = TurnRecorder(trajectory_id="t", storage=storage, budget=budget, event_counter=counter)
    install_monitoring(
        task,
        trajectory_id="t",
        budget=budget,
        parent_event_id_getter=recorder.current_turn_id,
        storage=storage,
        event_counter=counter,
    )
    return recorder, storage


# ---------------------------------------------------------------------------
# Default Agent.run shape
# ---------------------------------------------------------------------------


def test_default_run_completes_when_task_signals_done() -> None:
    """task.finished() returning True triggers TaskDone from MonitoredTool;
    Episode catches it normally. The unit test catches here since there's
    no Episode to drive."""
    from cube_harness.tool import TaskDone

    task = _MockTask(done_after_n=3)
    budget = Budget(max_turns=100)
    recorder, storage = _setup(task, budget)

    agent = _CounterAgent(_CounterAgentConfig())
    try:
        asyncio.run(agent.run(initial_obs=Observation(), toolbox=task.toolbox, recorder=recorder))
    except TaskDone:
        pass  # expected: task.finished() returned True after 3 counter increments

    outputs = storage.outputs()
    n_agent = sum(1 for e in outputs if isinstance(e, AgentEvent))
    n_tool = sum(1 for e in outputs if isinstance(e, ToolCallEvent))
    # Three rounds: each emits one AgentEvent + one ToolCallEvent. The
    # 3rd tool call's MonitoredTool raises TaskDone AFTER recording.
    assert n_agent == 3
    assert n_tool == 3
    assert task.counter == 3


def test_default_run_terminates_on_empty_actions() -> None:
    """An agent that returns empty actions and no error signals 'done'
    — the loop must return without an env step."""

    class _NoopAgent(Agent):
        name = "noop"
        description = ""
        input_content_types = []
        output_content_types = []

        def step(self, obs: Observation) -> AgentOutput:
            return AgentOutput(actions=[])

    task = _MockTask(done_after_n=100)
    budget = Budget(max_turns=10)
    recorder, storage = _setup(task, budget)
    agent = _NoopAgent(_CounterAgentConfig())
    asyncio.run(agent.run(initial_obs=Observation(), toolbox=task.toolbox, recorder=recorder))
    outputs = storage.outputs()
    assert sum(1 for e in outputs if isinstance(e, AgentEvent)) == 1
    assert sum(1 for e in outputs if isinstance(e, ToolCallEvent)) == 0
    assert task.counter == 0


def test_default_run_records_parent_event_id_on_tool_calls() -> None:
    """Each ToolCallEvent in the stream must reference the AgentEvent
    that spawned it (Phase A/B/C back-reference invariant)."""
    from cube_harness.tool import TaskDone

    task = _MockTask(done_after_n=2)
    budget = Budget(max_turns=10)
    recorder, storage = _setup(task, budget)
    try:
        asyncio.run(_CounterAgent(_CounterAgentConfig()).run(Observation(), task.toolbox, recorder))
    except TaskDone:
        pass

    agent_event_ids: list[str] = []
    for ev in storage.outputs():
        if isinstance(ev, AgentEvent):
            agent_event_ids.append(ev.id)
        elif isinstance(ev, ToolCallEvent):
            assert ev.parent_event_id in agent_event_ids, (
                "ToolCallEvent.parent_event_id must reference a preceding AgentEvent.id"
            )


def test_default_run_propagates_budget_exceeded() -> None:
    """BudgetExceeded from a monitored tool must surface through
    agent.run for Episode to capture."""
    from cube_harness.tool import BudgetExceeded

    task = _MockTask(done_after_n=100)
    budget = Budget(max_turns=100, max_tool_calls=1)
    recorder, storage = _setup(task, budget)

    agent = _CounterAgent(_CounterAgentConfig())
    # The second tool call (turn 2) raises.
    raised: list[BaseException] = []
    try:
        asyncio.run(agent.run(initial_obs=Observation(), toolbox=task.toolbox, recorder=recorder))
    except BaseException as e:  # noqa: BLE001
        raised.append(e)
    assert any(isinstance(e, BudgetExceeded) for e in raised)
    # At least one full round completed before the budget kicked in.
    assert sum(1 for e in storage.outputs() if isinstance(e, ToolCallEvent)) >= 1
