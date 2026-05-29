"""Default Agent.run tests — verify the base class implementation
reproduces today's gym-style loop and integrates with TurnRecorder +
MonitoredTool.

Uses a hand-rolled mock task (no LLM, no cube), so the test runs fast
and deterministically. The structural-parity check for real cubes lives
in Phase G (`cube test <name>` for every in-tree benchmark)."""

import asyncio

from cube.core import Action, ActionSchema, EnvironmentOutput, Observation
from cube.tool import AbstractTool

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentEvent, AgentOutput, ToolCallEvent, Trajectory
from cube_harness.recorder import TurnRecorder
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
    counter), `step(actions)` returning an EnvironmentOutput, and a
    `done_after_n` knob so we can deterministically end the loop."""

    def __init__(self, done_after_n: int = 3) -> None:
        self.counter = 0
        self.done_after_n = done_after_n
        self.toolbox = _CounterTool(self)

    def step(self, actions: list[Action]) -> EnvironmentOutput:
        # Forward each action to the toolbox (which is a single tool).
        last_obs = Observation()
        for action in actions:
            result = self.toolbox.execute_action(action)
            assert isinstance(result, Observation)
            last_obs = result
        done = self.counter >= self.done_after_n
        return EnvironmentOutput(obs=last_obs, reward=1.0 if done else 0.0, done=done, info={})


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


# ---------------------------------------------------------------------------
# Default Agent.run shape
# ---------------------------------------------------------------------------


def test_default_run_completes_when_task_signals_done() -> None:
    task = _MockTask(done_after_n=3)
    traj = Trajectory(id="t")
    budget = Budget(max_turns=100)
    recorder = TurnRecorder(traj)
    install_monitoring(task, traj, budget, parent_event_id_getter=recorder.current_turn_id)

    agent = _CounterAgent(_CounterAgentConfig())
    asyncio.run(agent.run(initial_obs=Observation(), task=task, recorder=recorder))

    # Three rounds: each emits one AgentEvent + one ToolCallEvent.
    assert traj.n_agent_events == 3
    assert traj.n_tool_calls == 3
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
    traj = Trajectory(id="t")
    budget = Budget(max_turns=10)
    recorder = TurnRecorder(traj)
    install_monitoring(task, traj, budget, parent_event_id_getter=recorder.current_turn_id)
    agent = _NoopAgent(_CounterAgentConfig())
    asyncio.run(agent.run(initial_obs=Observation(), task=task, recorder=recorder))
    assert traj.n_agent_events == 1
    assert traj.n_tool_calls == 0
    assert task.counter == 0


def test_default_run_records_parent_event_id_on_tool_calls() -> None:
    """Each ToolCallEvent in the stream must reference the AgentEvent
    that spawned it (Phase A/B/C back-reference invariant)."""
    task = _MockTask(done_after_n=2)
    traj = Trajectory(id="t")
    budget = Budget(max_turns=10)
    recorder = TurnRecorder(traj)
    install_monitoring(task, traj, budget, parent_event_id_getter=recorder.current_turn_id)
    asyncio.run(_CounterAgent(_CounterAgentConfig()).run(Observation(), task, recorder))

    agent_event_ids: list[str] = []
    for ev in traj.events:
        if isinstance(ev.output, AgentEvent):
            agent_event_ids.append(ev.output.id)
        elif isinstance(ev.output, ToolCallEvent):
            assert ev.output.parent_event_id in agent_event_ids, (
                "ToolCallEvent.parent_event_id must reference a preceding AgentEvent.id"
            )


def test_default_run_propagates_budget_exceeded() -> None:
    """BudgetExceeded from a monitored tool must surface through
    agent.run for Episode to capture."""
    from cube_harness.tool import BudgetExceeded

    task = _MockTask(done_after_n=100)
    traj = Trajectory(id="t")
    budget = Budget(max_turns=100, max_tool_calls=1)
    recorder = TurnRecorder(traj)
    install_monitoring(task, traj, budget, parent_event_id_getter=recorder.current_turn_id)

    agent = _CounterAgent(_CounterAgentConfig())
    # The second tool call (turn 2) raises.
    raised: list[BaseException] = []
    try:
        asyncio.run(agent.run(initial_obs=Observation(), task=task, recorder=recorder))
    except BaseException as e:  # noqa: BLE001
        raised.append(e)
    assert any(isinstance(e, BudgetExceeded) for e in raised)
    # At least one full round completed before the budget kicked in.
    assert traj.n_tool_calls >= 1
