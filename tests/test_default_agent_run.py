"""Default Agent.run tests — verify the base class implementation
reproduces today's gym-style loop and integrates with EventStreamer +
MonitoredTool.

Uses a hand-rolled mock task (no LLM, no cube), so the test runs fast
and deterministically. The structural-parity check for real cubes lives
in Phase G (`cube test <name>` for every in-tree benchmark).

Post-Trajectory-removal: events stream to a Storage hook; tests inspect
the captured event stream rather than walking an in-memory list.
"""

from cube.core import Action, ActionSchema, Observation
from cube.tool import AbstractTool

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCallEvent, ToolCallEvent, TrajectoryEvent
from cube_harness.streamer import EventStreamer
from cube_harness.tool import Budget, BudgetExceeded, TaskDone, build_monitored_env_tool

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

    def finished(self, obs: Observation | None = None) -> bool:
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
        )


def _action(name: str = "inc") -> Action:
    return Action(id=f"id-{name}", name=name, arguments={})


class _FakeStorage:
    """Captures every save_event call + assigns event_nums itself
    (matches the new Storage.save_event(event, id) -> int contract)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int, TrajectoryEvent]] = []
        self._next_num = 0

    def save_event(self, te: TrajectoryEvent, trajectory_id: str) -> int:
        n = self._next_num
        self._next_num += 1
        self.events.append((trajectory_id, n, te))
        return n

    def outputs(self) -> list:
        return [te.output for _, _, te in self.events]


def _setup(task, budget: Budget) -> tuple[EventStreamer, _FakeStorage, object]:
    """Build EventStreamer + storage + the monitored env_tool — the way
    Episode does it. Storage owns event numbering; nothing to thread. The
    returned env_tool is what the agent drives (task's own tool is left
    concrete)."""
    storage = _FakeStorage()
    streamer = EventStreamer(trajectory_id="t", storage=storage, budget=budget)
    env_tool = build_monitored_env_tool(task, streamer)
    return streamer, storage, env_tool


# ---------------------------------------------------------------------------
# Default Agent.run shape
# ---------------------------------------------------------------------------


def test_default_run_completes_when_task_signals_done() -> None:
    """task.finished() returning True triggers TaskDone from MonitoredTool;
    Episode catches it normally. The unit test catches here since there's
    no Episode to drive."""

    task = _MockTask(done_after_n=3)
    budget = Budget(max_agent_steps=100)
    recorder, storage, env_tool = _setup(task, budget)

    agent = _CounterAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)
    try:
        agent.run(initial_obs=Observation(), env_tool=env_tool)
    except TaskDone:
        pass  # expected: task.finished() returned True after 3 counter increments

    outputs = storage.outputs()
    n_tool = sum(1 for e in outputs if isinstance(e, ToolCallEvent))
    # MockAgent has no LLM → no LLMCallEvent. Three rounds emit three
    # ToolCallEvents; the 3rd's MonitoredTool raises TaskDone AFTER
    # recording.
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
    budget = Budget(max_agent_steps=10)
    recorder, storage, env_tool = _setup(task, budget)
    agent = _NoopAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)
    agent.run(initial_obs=Observation(), env_tool=env_tool)
    outputs = storage.outputs()
    # No LLM call + empty actions => nothing was emitted by the agent
    # loop (LLM auto-emit doesn't fire; ToolCallEvent dispatch doesn't fire).
    assert sum(1 for e in outputs if isinstance(e, LLMCallEvent)) == 0
    assert sum(1 for e in outputs if isinstance(e, ToolCallEvent)) == 0
    assert task.counter == 0


def test_default_run_records_parent_event_id_on_tool_calls() -> None:
    """Each ToolCallEvent must reference either the RESET sentinel
    (LLM-less paths) or a preceding LLMCallEvent."""

    task = _MockTask(done_after_n=2)
    budget = Budget(max_agent_steps=10)
    recorder, storage, env_tool = _setup(task, budget)
    agent = _CounterAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)
    try:
        agent.run(Observation(), env_tool)
    except TaskDone:
        pass

    valid_parents: set[str] = {"reset"}
    for ev in storage.outputs():
        if isinstance(ev, LLMCallEvent):
            valid_parents.add(ev.id)
        elif isinstance(ev, ToolCallEvent):
            assert ev.parent_event_id in valid_parents, (
                "ToolCallEvent.parent_event_id must reference RESET or a preceding LLMCallEvent.id"
            )


def test_default_run_propagates_budget_exceeded() -> None:
    """BudgetExceeded from a monitored tool must surface through
    agent.run for Episode to capture."""

    task = _MockTask(done_after_n=100)
    budget = Budget(max_agent_steps=100, max_tool_calls=1)
    recorder, storage, env_tool = _setup(task, budget)

    agent = _CounterAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)
    # The second tool call (turn 2) raises.
    raised: list[BaseException] = []
    try:
        agent.run(initial_obs=Observation(), env_tool=env_tool)
    except BaseException as e:  # noqa: BLE001
        raised.append(e)
    assert any(isinstance(e, BudgetExceeded) for e in raised)
    # At least one full round completed before the budget kicked in.
    assert sum(1 for e in storage.outputs() if isinstance(e, ToolCallEvent)) >= 1
