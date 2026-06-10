"""Tests for the LaMer-paper Sokoban reflection provider (``sokoban_cube.reflection``)."""

from sokoban_cube.reflection import (
    SokobanReflectionProvider,
    SokobanReflectionProviderConfig,
    summarize_sokoban_history,
)

# A failed single-episode history: initial board, one move turn, the resulting (still unsolved) board.
_HISTORY = [
    {
        "role": "user",
        "content": "Push the box onto the goal.\nLegend: # wall, @ you.\n######\n#@ $.#\n######\nBoxes on goals: 0/1.",
    },
    {
        "role": "assistant",
        "content": "I'll walk right toward the box.",
        "tool_calls": [{"function": {"name": "move", "arguments": '{"directions": "right, right"}'}}],
    },
    {"role": "tool", "content": "Moves: right=ok right=blocked\n######\n# @$.#\n######\nBoxes on goals: 0/1."},
]


def test_summarize_extracts_boards_and_actions() -> None:
    initial, actions, final = summarize_sokoban_history(_HISTORY)
    assert initial == "######\n#@ $.#\n######"  # only the glyph grid, not instructions/legend/status
    assert actions == "right, right"
    assert final == "######\n# @$.#\n######"


def test_build_prompt_replays_trajectory_and_asks_for_remark() -> None:
    prompt = SokobanReflectionProvider().build_reflection_prompt(
        _HISTORY, reward=0.0, max_actions=10, requirements=None
    )
    system, user = prompt.messages[0]["content"], prompt.messages[1]["content"]
    assert "reflect" in system.lower()
    # the paper structure: initial board, the actions, final board, the failure statement, the tag
    assert "#@ $.#" in user and "right, right" in user and "# @$.#" in user
    assert "NOT successfully completed" in user
    assert "<remark>" in user and "step-by-step" in user.lower()


def test_extract_lesson_pulls_remark() -> None:
    provider = SokobanReflectionProvider()
    raw = "Reasoning: the box got stuck.\n<remark>Approach the box from the left first.</remark>"
    assert provider.extract_lesson(raw) == "Approach the box from the left first."
    # last remark wins; missing tag falls back to the whole text
    assert provider.extract_lesson("<remark>a</remark> then <remark>b</remark>") == "b"
    assert provider.extract_lesson("no tag here") == "no tag here"


def test_config_make_roundtrip() -> None:
    provider = SokobanReflectionProviderConfig().make()
    assert isinstance(provider, SokobanReflectionProvider)
    assert provider.extract_lesson("<remark>x</remark>") == "x"
