"""Unit tests for the flat event-stream view model (`analyze.xray_events`).

These assert the dependency-graph grouping XRay relies on: an LLM call, the
observation(s) it produced, their step-wise rewards, and any error in the
chain all resolve to one logical group, so selecting any card surfaces the
whole "why did the agent do this, what did it observe, what reward, any error"
story. Parallel siblings share one group; terminal evals with no parent link
attach to the most recent group. No Gradio, no disk.
"""

from cube.core import Action, Observation, StepError

from cube_harness.analyze import xray_events as xe
from cube_harness.core import (
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


def test_group_normalizes_either_side() -> None:
    ep = _gym_stream()
    # Selecting the LLM call and selecting its observation yield the same group.
    g_llm = ep.group_for(1)
    g_obs = ep.group_for(2)
    assert g_llm.members == g_obs.members == [1, 2]
    assert g_llm.llm_index == 1  # index 1 is the LLM call
    assert g_obs.observation_indices == [2]


def test_stepwise_eval_groups_with_its_toolcall() -> None:
    # llm1 -> obs1 -> step-wise reward(obs1): all one logical group.
    ev_obs = _tool("obs1", "llm1")
    step_eval = TrajectoryEvent(output=EvaluationEvent(reward=0.5, is_terminal=False, parent_event_id="obs1"))
    ep = xe.EpisodeEvents([_llm("llm1"), ev_obs, step_eval])
    g = ep.group_for(2)  # select the reward
    assert g.members == [0, 1, 2]
    assert g.llm_index == 0
    assert g.observation_indices == [1]
    assert g.evaluation_indices == [2]


def test_parallel_siblings_share_one_group() -> None:
    ep = xe.EpisodeEvents(
        [
            _llm("llm1"),
            _tool("a", "llm1", "read"),
            _tool("b", "llm1", "write"),
            _tool("c", "llm1", "list"),
        ]
    )
    # The LLM call and all three parallel observations form one group.
    assert ep.child_indices(0) == [1, 2, 3]
    g = ep.group_for(2)
    assert g.members == [0, 1, 2, 3]
    assert g.observation_indices == [1, 2, 3]
    # Selecting any observation highlights the LLM call + the two siblings.
    assert ep.accompanying_indices(2) == [0, 1, 3]


def test_grouping_is_order_independent() -> None:
    # A child decoded BEFORE its parent must still resolve to the parent's group
    # (grouping follows parent_event_id links, not stream position).
    ep = xe.EpisodeEvents([_tool("obs1", "llm1"), _llm("llm1")])
    assert ep.group_for(0).members == [0, 1]
    assert ep.group_for(1).members == [0, 1]
    assert ep.accompanying_indices(0) == [1]


def test_self_referential_parent_forms_own_group() -> None:
    # A malformed self-parent link must not blow the recursion stack.
    ep = xe.EpisodeEvents([_tool("x", "x")])
    assert ep.group_for(0).members == [0]


def test_group_navigation_moves_one_step_per_press() -> None:
    # gym stream groups: {0:reset}, {1:llm,2:obs}, {3:llm,4:obs,5:terminal-eval}.
    ep = _gym_stream()
    assert ep.group_roots() == [0, 1, 3]
    # Next from anywhere in a group jumps to the NEXT group's root (one press).
    assert ep.next_group_root(0) == 1
    assert ep.next_group_root(1) == 3
    assert ep.next_group_root(2) == 3  # from the observation inside group {1,2}
    assert ep.next_group_root(3) == 3  # last group, clamped
    # Prev mirrors it.
    assert ep.prev_group_root(4) == 1  # from inside the last group
    assert ep.prev_group_root(1) == 0
    assert ep.prev_group_root(0) == 0  # first group, clamped


def test_first_and_last_group_root() -> None:
    # groups: {0:reset}, {1:llm,2:obs}, {3:llm,4:obs,5:terminal-eval}.
    ep = _gym_stream()
    assert ep.group_roots() == [0, 1, 3]
    # "First step" skips the initial-observation group (root 0) → first real step.
    assert ep.first_group_root() == 1
    # "End" → the last group's root.
    assert ep.last_group_root() == 3


def test_first_group_root_falls_back_when_only_reset() -> None:
    ep = xe.EpisodeEvents([_tool("obs0", xe.RESET_PARENT)])
    assert ep.group_roots() == [0]
    assert ep.first_group_root() == 0  # nothing after the reset → clamp to it
    assert ep.last_group_root() == 0


def test_terminal_eval_attaches_to_last_group() -> None:
    # No parent link on the terminal eval -> it joins the most recent group.
    ep = xe.EpisodeEvents([_llm("llm1"), _tool("obs1", "llm1"), _eval(1.0, terminal=True)])
    g = ep.group_for(2)
    assert g.members == [0, 1, 2]
    assert g.llm_index == 0
    assert g.evaluation_indices == [2]


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
