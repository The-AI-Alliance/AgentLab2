"""TurnRecorder tests — coarse vs granular APIs produce equivalent
AgentEvents, Episode-only helpers (reset/failure/evaluation) work as
expected, current_turn_id surfaces correctly for MonitoredTool's
parent_event_id_getter, and record_external_run carries metadata for
opaque framework connectors.

Post-Trajectory-removal: events stream to a Storage hook; tests inspect
what was sent there rather than walking an in-memory list.
"""

from cube.core import Action, ActionSchema, EnvironmentOutput, Observation
from cube.tool import AbstractTool

# Import litellm.Message lazily through the existing module's exports
from litellm import Message

from cube_harness.core import AgentEvent, AgentOutput, EvaluationEvent, ToolCallEvent, TrajectoryEvent
from cube_harness.llm import LLMCall, LLMConfig, Prompt, Usage
from cube_harness.recorder import (
    RESET_PARENT_EVENT_ID,
    EventCounter,
    TurnRecorder,
    equivalent_agent_events,
)
from cube_harness.tool import Budget, MonitoredTool

# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _action(name: str = "noop") -> Action:
    return Action(id=f"id-{name}", name=name, arguments={})


def _agent_output(thoughts: str = "t", response_text: str | None = None) -> AgentOutput:
    return AgentOutput(actions=[_action()], thoughts=thoughts)


def _llm_call() -> LLMCall:
    return LLMCall(
        tag="act",
        llm_config=LLMConfig(model_name="openai/gpt-4o-mini"),
        prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
        output=Message(content="ok", role="assistant"),
        usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12, cost=0.01),
    )


class _FakeStorage:
    """Captures every save_event call so tests can inspect what the
    recorder streamed without needing a real FileStorage backend."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int, TrajectoryEvent]] = []

    def save_event(self, te: TrajectoryEvent, trajectory_id: str, n: int) -> None:
        self.events.append((trajectory_id, n, te))

    def event_outputs(self) -> list:
        """Convenience: list[TrajectoryEvent.output] in save order."""
        return [te.output for _, _, te in self.events]


class _FakeSummary:
    def __init__(self) -> None:
        self.seen = 0

    def on_event(self, te: TrajectoryEvent) -> None:
        self.seen += 1


def _make_recorder(**kwargs) -> tuple[TurnRecorder, _FakeStorage, dict]:
    storage = kwargs.pop("storage", None) or _FakeStorage()
    metadata_updates: dict = kwargs.pop("metadata_updates", {})
    r = TurnRecorder(
        trajectory_id=kwargs.pop("trajectory_id", "t"),
        storage=storage,
        metadata_updates=metadata_updates,
        **kwargs,
    )
    return r, storage, metadata_updates


# ---------------------------------------------------------------------------
# Coarse path
# ---------------------------------------------------------------------------


def test_record_emits_one_agent_event() -> None:
    r, st, _ = _make_recorder()
    r.record(_agent_output(thoughts="x", response_text="resp"))
    outputs = st.event_outputs()
    assert len(outputs) == 1
    assert isinstance(outputs[0], AgentEvent)
    assert outputs[0].thoughts == "x"


def test_record_returns_agent_event_id() -> None:
    r, st, _ = _make_recorder()
    eid = r.record(_agent_output())
    assert isinstance(st.event_outputs()[0], AgentEvent)
    assert st.event_outputs()[0].id == eid


# ---------------------------------------------------------------------------
# Granular path
# ---------------------------------------------------------------------------


def test_begin_turn_flushes_on_exit() -> None:
    r, st, _ = _make_recorder()
    with r.begin_turn() as turn:
        turn.add_thought("thinking...")
        turn.add_response_text("hello")
        turn.add_llm_call(_llm_call())
        turn.add_action(_action("foo"))
        turn.add_profile("llm", 1.0, 1.5)
    outputs = st.event_outputs()
    assert len(outputs) == 1
    ev = outputs[0]
    assert isinstance(ev, AgentEvent)
    assert ev.thoughts == "thinking..."
    assert ev.response_text == "hello"
    assert len(ev.llm_calls) == 1
    assert [a.name for a in ev.actions] == ["foo"]


def test_begin_turn_accumulates_thought_chunks() -> None:
    """Streaming agents emit reasoning in chunks; the recorder must
    concatenate so we don't lose history."""
    r, st, _ = _make_recorder()
    with r.begin_turn() as turn:
        turn.add_thought("chunk-1 ")
        turn.add_thought("chunk-2")
    ev = st.event_outputs()[0]
    assert isinstance(ev, AgentEvent)
    assert ev.thoughts == "chunk-1 chunk-2"


def test_record_and_begin_turn_produce_equivalent_events() -> None:
    """Same payload through both paths should yield equivalent AgentEvents
    (ignoring id and timestamp/profiling)."""
    out = _agent_output(thoughts="t", response_text="rt")
    out.llm_calls.append(_llm_call())

    r1, st1, _ = _make_recorder(trajectory_id="t1")
    r1.record(out, response_text="rt")
    coarse = st1.event_outputs()[0]
    assert isinstance(coarse, AgentEvent)

    r2, st2, _ = _make_recorder(trajectory_id="t2")
    with r2.begin_turn() as turn:
        for a in out.actions:
            turn.add_action(a)
        for c in out.llm_calls:
            turn.add_llm_call(c)
        if out.thoughts:
            turn.add_thought(out.thoughts)
        turn.add_response_text("rt")
    granular = st2.event_outputs()[0]
    assert isinstance(granular, AgentEvent)
    assert equivalent_agent_events(coarse, granular)


# ---------------------------------------------------------------------------
# Episode-only helpers
# ---------------------------------------------------------------------------


def test_record_reset_emits_synthetic_tool_call_event() -> None:
    r, st, _ = _make_recorder()
    initial = EnvironmentOutput(obs=Observation(), reward=0.0, done=False, info={})
    r.record_reset(initial)
    outputs = st.event_outputs()
    assert len(outputs) == 1
    ev = outputs[0]
    assert isinstance(ev, ToolCallEvent)
    assert ev.parent_event_id == RESET_PARENT_EVENT_ID
    assert ev.turn_id == RESET_PARENT_EVENT_ID


def test_record_failure_records_agent_event_with_error() -> None:
    r, st, _ = _make_recorder()
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        r.record_failure(e)
    outputs = st.event_outputs()
    assert len(outputs) == 1
    ev = outputs[0]
    assert isinstance(ev, AgentEvent)
    assert ev.error is not None
    assert ev.error.error_type == "RuntimeError"
    assert "boom" in ev.error.exception_str


def test_record_evaluation_emits_evaluation_event() -> None:
    r, st, _ = _make_recorder()
    r.record_evaluation(reward=0.75, info={"score": 0.75})
    outputs = st.event_outputs()
    assert len(outputs) == 1
    ev = outputs[0]
    assert isinstance(ev, EvaluationEvent)
    assert ev.reward == 0.75


# ---------------------------------------------------------------------------
# current_turn_id wiring (drives MonitoredTool.parent_event_id_getter)
# ---------------------------------------------------------------------------


class _SyncEchoTool(AbstractTool):
    @property
    def action_set(self) -> list[ActionSchema]:
        return [ActionSchema(name="echo", description="x", parameters={"type": "object", "properties": {}})]

    def execute_action(self, action: Action) -> Observation:
        return Observation.from_text("ok")


def test_current_turn_id_propagates_to_monitored_tool_via_getter() -> None:
    """Tool calls fired inside a turn record that turn's id as parent."""
    counter = EventCounter()
    storage = _FakeStorage()
    r = TurnRecorder(trajectory_id="t", storage=storage, event_counter=counter)
    budget = Budget(max_turns=5)
    tool = MonitoredTool(
        _SyncEchoTool(),
        trajectory_id="t",
        budget=budget,
        parent_event_id_getter=r.current_turn_id,
        storage=storage,
        event_counter=counter,
    )

    eid = r.record(_agent_output())
    tool.execute_action(_action("echo"))

    # Sequence: AgentEvent → ToolCallEvent with parent_event_id == eid.
    outputs = storage.event_outputs()
    assert isinstance(outputs[0], AgentEvent)
    assert isinstance(outputs[1], ToolCallEvent)
    assert outputs[1].parent_event_id == eid


def test_current_turn_id_before_any_record_is_reset_sentinel() -> None:
    r, _, _ = _make_recorder()
    assert r.current_turn_id() == RESET_PARENT_EVENT_ID


# ---------------------------------------------------------------------------
# Lossy capture for Phase-2 connectors
# ---------------------------------------------------------------------------


def test_record_external_run_emits_one_event_with_final_text() -> None:
    r, st, _ = _make_recorder()
    r.record_external_run(final_text="the answer is 42")
    ev = st.event_outputs()[0]
    assert isinstance(ev, AgentEvent)
    assert ev.response_text == "the answer is 42"


def test_record_external_run_stashes_usage_on_metadata_updates() -> None:
    r, _, meta_updates = _make_recorder()
    r.record_external_run(
        final_text=None,
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.05),
    )
    assert "external_run_usage" in meta_updates
    assert meta_updates["external_run_usage"][0]["prompt_tokens"] == 100


def test_record_external_run_stashes_raw_events_on_metadata_updates() -> None:
    r, _, meta_updates = _make_recorder()
    r.record_external_run(
        final_text="ok",
        raw_events=[{"event": "message_start"}, {"event": "tool_call"}],
    )
    assert meta_updates["external_run_raw_events"] == [
        {"event": "message_start"},
        {"event": "tool_call"},
    ]


# ---------------------------------------------------------------------------
# Storage / summary hook plumbing
# ---------------------------------------------------------------------------


def test_recorder_invokes_storage_and_summary_hooks() -> None:
    storage = _FakeStorage()
    summary = _FakeSummary()
    r = TurnRecorder(trajectory_id="t", storage=storage, summary=summary)
    r.record(_agent_output())
    r.record_evaluation(reward=1.0)
    assert [(tid, n) for tid, n, _ in storage.events] == [("t", 0), ("t", 1)]
    assert summary.seen == 2
