"""Unit tests for the flat event-stream view model (`analyze.xray_events`).

These assert the parent-link pairing XRay relies on: an observation pairs with
the LLM call that produced it, an LLM call pairs with the observation(s) it
produced, and parallel siblings (several tool calls sharing one parent) all
pair back to that single LLM call. No Gradio, no disk.
"""

from cube.core import Action, Observation, StepError

from cube_harness.analyze import xray_events as xe
from cube_harness.core import (
    AgentErrorEvent,
    EvaluationEvent,
    LLMCallEvent,
    ToolCallEvent,
    TrajectoryEvent,
)


def _llm(event_id: str) -> TrajectoryEvent:
    return TrajectoryEvent(output=LLMCallEvent(id=event_id, call=None))


def _tool(event_id: str, parent: str, action_name: str = "click") -> TrajectoryEvent:
    return TrajectoryEvent(
        output=ToolCallEvent(
            id=event_id,
            parent_event_id=parent,
            turn_id=parent,
            action=Action(name=action_name, arguments={"element_id": "btn1"}),
            obs=Observation.from_text(f"after {action_name}"),
        )
    )


def _eval(reward: float, *, terminal: bool) -> TrajectoryEvent:
    return TrajectoryEvent(output=EvaluationEvent(reward=reward, is_terminal=terminal))


def _err(error_type: str = "Boom") -> StepError:
    return StepError(error_type=error_type, exception_str=f"{error_type}!", stack_trace="…")


def _gym_stream() -> xe.EpisodeEvents:
    """A gym-style episode: reset obs, then (llm -> obs) pairs, then eval."""
    return xe.EpisodeEvents(
        [
            _tool("obs0", xe.RESET_PARENT),  # initial observation, no parent
            _llm("llm1"),
            _tool("obs1", "llm1"),
            _llm("llm2"),
            _tool("obs2", "llm2"),
            _eval(1.0, terminal=True),
        ]
    )


def test_observation_pairs_with_parent_llm() -> None:
    ep = _gym_stream()
    # obs1 is at index 2, produced by llm1 at index 1.
    assert ep.parent_index(2) == 1
    assert ep.accompanying_indices(2) == [1]


def test_llm_pairs_with_child_observation() -> None:
    ep = _gym_stream()
    # llm1 at index 1 produced obs1 at index 2.
    assert ep.child_indices(1) == [2]
    assert ep.accompanying_indices(1) == [2]


def test_reset_observation_has_no_parent() -> None:
    ep = _gym_stream()
    assert ep.parent_index(0) is None
    assert ep.accompanying_indices(0) == []


def test_resolve_pair_normalizes_either_side() -> None:
    ep = _gym_stream()
    # Selecting the LLM call and selecting its observation resolve to the same pair.
    assert ep.resolve_pair(1) == (1, [2])
    assert ep.resolve_pair(2) == (1, [2])


def test_parallel_siblings_share_one_parent() -> None:
    ep = xe.EpisodeEvents(
        [
            _llm("llm1"),
            _tool("a", "llm1", "read"),
            _tool("b", "llm1", "write"),
            _tool("c", "llm1", "list"),
        ]
    )
    # The LLM call lights up all three parallel observations...
    assert ep.child_indices(0) == [1, 2, 3]
    # ...and each observation pairs back to the single LLM call.
    for obs_idx in (1, 2, 3):
        assert ep.parent_index(obs_idx) == 0
        assert ep.resolve_pair(obs_idx) == (0, [obs_idx])
    assert ep.resolve_pair(0) == (0, [1, 2, 3])


def test_eval_and_error_render_alone() -> None:
    ep = xe.EpisodeEvents([_eval(0.0, terminal=True), TrajectoryEvent(output=AgentErrorEvent(error=_err()))])
    assert ep.resolve_pair(0) == (None, [])
    assert ep.resolve_pair(1) == (None, [])


def test_typed_extractors_are_none_safe() -> None:
    ep = _gym_stream()
    assert ep.observation(2) is not None
    assert ep.action(2) is not None and ep.action(2).name == "click"
    assert ep.llm_call(1) is None  # call=None on this synthetic event
    assert ep.observation(1) is None  # index 1 is an LLM call, not a tool call
    assert ep.action(None) is None


def test_cards_cover_every_event_with_kinds() -> None:
    ep = _gym_stream()
    cards = ep.cards()
    assert len(cards) == len(ep)
    kinds = [c.kind for c in cards]
    assert kinds == [
        xe.KIND_OBSERVATION,  # reset obs
        xe.KIND_LLM,
        xe.KIND_OBSERVATION,
        xe.KIND_LLM,
        xe.KIND_OBSERVATION,
        xe.KIND_EVALUATION,
    ]
    # Every card carries a colour and the reset card is labelled distinctly.
    assert all(c.color for c in cards)
    assert cards[0].title == "Initial observation"
    # The observation card's accompanying link points at its parent LLM call.
    assert cards[2].accompanying == [1]


def test_error_events_flagged_and_coloured() -> None:
    ep = xe.EpisodeEvents(
        [
            TrajectoryEvent(output=LLMCallEvent(id="llm1", call=None, error=_err("Boom"))),
            _tool("obs1", "llm1"),
        ]
    )
    card = ep.cards()[0]
    assert card.is_error
    assert card.kind == xe.KIND_ERROR
    assert card.color == xe.KIND_COLORS[xe.KIND_ERROR]
