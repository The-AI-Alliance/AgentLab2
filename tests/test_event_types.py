"""Unit tests for the new trajectory event model (RFC: agent-owns-loop).

Covers AgentEvent / ToolCallEvent / EvaluationEvent / TrajectoryEvent
construction, serialization round-trip, back-references, and the
helpers on `Trajectory` that walk the event stream.
"""

from cube.core import Action, EnvironmentOutput, Observation

from cube_harness.core import (
    AgentEvent,
    AgentOutput,
    EvaluationEvent,
    ToolCallEvent,
    Trajectory,
    TrajectoryEvent,
    TrajectoryStep,
)


def _make_action(name: str = "noop", action_id: str | None = "a-1") -> Action:
    return Action(id=action_id, name=name, arguments={})


def _make_env_output(reward: float = 0.0, done: bool = False) -> EnvironmentOutput:
    return EnvironmentOutput(obs=Observation(), reward=reward, done=done, info={})


def test_agent_event_default_id_unique() -> None:
    e1 = AgentEvent()
    e2 = AgentEvent()
    assert e1.id != e2.id
    assert len(e1.id) > 0


def test_agent_event_round_trip() -> None:
    src = AgentEvent(
        actions=[_make_action("foo"), _make_action("bar", action_id="a-2")],
        thoughts="reasoning",
        response_text="here are two calls",
        profiling={"llm": (1.0, 2.5)},
    )
    blob = src.model_dump_json()
    dst = AgentEvent.model_validate_json(blob)
    assert dst.id == src.id
    assert [a.name for a in dst.actions] == ["foo", "bar"]
    assert dst.thoughts == "reasoning"
    assert dst.response_text == "here are two calls"
    assert dst.profiling == {"llm": (1.0, 2.5)}


def test_agent_event_from_agent_output() -> None:
    out = AgentOutput(actions=[_make_action()], thoughts="t")
    ev = AgentEvent.from_agent_output(out, response_text="resp")
    assert ev.actions == out.actions
    assert ev.thoughts == "t"
    assert ev.response_text == "resp"


def test_tool_call_event_back_references() -> None:
    parent = AgentEvent(actions=[_make_action("foo", action_id="a-1")])
    tc = ToolCallEvent(
        parent_event_id=parent.id,
        action_id="a-1",
        output=_make_env_output(reward=0.5),
        turn_id=parent.id,
    )
    assert tc.parent_event_id == parent.id
    assert tc.turn_id == parent.id
    assert tc.output.reward == 0.5


def test_tool_call_event_round_trip() -> None:
    tc = ToolCallEvent(
        parent_event_id="p-1",
        action_id="a-1",
        output=_make_env_output(reward=1.0, done=True),
        turn_id="t-1",
    )
    blob = tc.model_dump_json()
    dst = ToolCallEvent.model_validate_json(blob)
    assert dst.parent_event_id == "p-1"
    assert dst.action_id == "a-1"
    assert dst.turn_id == "t-1"
    assert dst.output.reward == 1.0
    assert dst.output.done is True


def test_evaluation_event_round_trip() -> None:
    ev = EvaluationEvent(reward=0.75, info={"k": "v"})
    dst = EvaluationEvent.model_validate_json(ev.model_dump_json())
    assert dst.reward == 0.75
    assert dst.info == {"k": "v"}


def test_trajectory_event_union_serialization() -> None:
    """TrajectoryEvent.output must round-trip through the polymorphic union."""
    cases: list[TrajectoryEvent] = [
        TrajectoryEvent(output=AgentEvent(thoughts="x")),
        TrajectoryEvent(
            output=ToolCallEvent(parent_event_id="p", action_id=None, output=_make_env_output(), turn_id="t")
        ),
        TrajectoryEvent(output=EvaluationEvent(reward=1.0)),
    ]
    for src in cases:
        dst = TrajectoryEvent.model_validate_json(src.model_dump_json())
        assert type(dst.output) is type(src.output)


def test_trajectory_event_helpers_and_counters() -> None:
    parent = AgentEvent(
        actions=[
            _make_action("foo", action_id="a-1"),
            _make_action("bar", action_id="a-2"),
        ]
    )
    tc1 = ToolCallEvent(parent_event_id=parent.id, action_id="a-1", output=_make_env_output(0.0), turn_id=parent.id)
    tc2 = ToolCallEvent(parent_event_id=parent.id, action_id="a-2", output=_make_env_output(0.0), turn_id=parent.id)
    final = EvaluationEvent(reward=1.0, info={"x": 1})
    traj = Trajectory(id="t-1")
    traj.events.append(TrajectoryEvent(output=parent))
    traj.events.append(TrajectoryEvent(output=tc1))
    traj.events.append(TrajectoryEvent(output=tc2))
    traj.events.append(TrajectoryEvent(output=final))

    assert traj.n_agent_events == 1
    assert traj.n_tool_calls == 2
    assert traj.n_evaluations == 1
    siblings = traj.events_of_turn(parent.id)
    assert len(siblings) == 2
    assert all(isinstance(e.output, ToolCallEvent) for e in siblings)
    assert traj.last_env_output() is not None


def test_trajectory_legacy_steps_still_counted() -> None:
    """The legacy `steps` field stays a writable target during migration.
    Counters fold both streams together so XRay and Summary keep working."""
    traj = Trajectory(id="t-1")
    traj.steps.append(TrajectoryStep(output=_make_env_output()))
    traj.steps.append(TrajectoryStep(output=AgentOutput()))
    assert traj.n_env_steps == 1
    assert traj.n_agent_steps == 1


def test_trajectory_last_env_prefers_events() -> None:
    """When the event stream is populated, last_env_step pulls from there
    rather than the legacy steps field."""
    traj = Trajectory(id="t-1")
    traj.steps.append(TrajectoryStep(output=_make_env_output(reward=0.1)))
    parent = AgentEvent()
    traj.events.append(TrajectoryEvent(output=parent))
    traj.events.append(
        TrajectoryEvent(
            output=ToolCallEvent(
                parent_event_id=parent.id,
                action_id=None,
                output=_make_env_output(reward=0.9),
                turn_id=parent.id,
            )
        )
    )
    last = traj.last_env_step()
    assert last.reward == 0.9


def test_trajectory_last_env_output_returns_none_when_empty() -> None:
    traj = Trajectory(id="t-1")
    assert traj.last_env_output() is None
