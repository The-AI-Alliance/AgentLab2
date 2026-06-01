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
    obs = _make_env_output(reward=0.5).obs
    tc = ToolCallEvent(
        parent_event_id=parent.id,
        action_id="a-1",
        obs=obs,
        turn_id=parent.id,
    )
    assert tc.parent_event_id == parent.id
    assert tc.turn_id == parent.id
    assert tc.obs == obs


def test_tool_call_event_round_trip() -> None:
    obs = _make_env_output(reward=1.0, done=True).obs
    tc = ToolCallEvent(
        parent_event_id="p-1",
        action_id="a-1",
        obs=obs,
        turn_id="t-1",
    )
    blob = tc.model_dump_json()
    dst = ToolCallEvent.model_validate_json(blob)
    assert dst.parent_event_id == "p-1"
    assert dst.action_id == "a-1"
    assert dst.turn_id == "t-1"
    assert dst.obs == obs


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
            output=ToolCallEvent(parent_event_id="p", action_id=None, obs=_make_env_output().obs, turn_id="t")
        ),
        TrajectoryEvent(output=EvaluationEvent(reward=1.0)),
    ]
    for src in cases:
        dst = TrajectoryEvent.model_validate_json(src.model_dump_json())
        assert type(dst.output) is type(src.output)


def test_trajectory_legacy_steps_still_counted() -> None:
    """The legacy `steps` field stays a writable target on the slim
    Trajectory shape (XRay consumer). Counters work as the old code
    expected for the XRay legacy view."""
    traj = Trajectory(id="t-1")
    traj.steps.append(TrajectoryStep(output=_make_env_output()))
    traj.steps.append(TrajectoryStep(output=AgentOutput()))
    assert traj.n_env_steps == 1
    assert traj.n_agent_steps == 1


def test_trajectory_last_env_step_walks_steps() -> None:
    """Trajectory.last_env_step walks the legacy steps list. Event-stream
    consumers should use TrajectoryView.last_env_output instead — see
    test_episode_view.py."""
    traj = Trajectory(id="t-1")
    traj.steps.append(TrajectoryStep(output=_make_env_output(reward=0.1)))
    traj.steps.append(TrajectoryStep(output=_make_env_output(reward=0.9)))
    last = traj.last_env_step()
    assert last.reward == 0.9


def test_trajectory_last_env_output_returns_none_when_empty() -> None:
    traj = Trajectory(id="t-1")
    assert traj.last_env_output() is None
