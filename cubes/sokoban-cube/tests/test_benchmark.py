"""Tests for sokoban_cube.

Run with:  pytest tests/
"""

import sokoban_cube.debug as _debug_mod
from cube.testing import assert_debug_tasks_reward_one

from sokoban_cube.benchmark import SokobanBenchmarkConfig
from sokoban_cube.grid import SokobanState, simulate
from sokoban_cube.task import SokobanTaskConfig, SokobanTaskMetadata
from sokoban_cube.tool import SokobanToolConfig


def test_benchmark_metadata() -> None:
    """Benchmark metadata is valid, non-empty, and has no unfilled placeholders."""
    meta = SokobanBenchmarkConfig.benchmark_metadata
    assert meta.name, "benchmark name must not be empty"
    assert meta.version, "benchmark version must not be empty"
    assert meta.description, "benchmark description must not be empty"
    assert "TODO" not in meta.description, "benchmark description still contains a TODO placeholder"
    assert meta.num_tasks == len(SokobanBenchmarkConfig.task_metadata), "num_tasks must match task count"


def test_task_metadata_keys_match() -> None:
    """Every task_metadata key matches its TaskMetadata.id and deserialises to the subclass."""
    for key, meta in SokobanBenchmarkConfig.task_metadata.items():
        assert key == meta.id, f"Key {key!r} does not match TaskMetadata.id {meta.id!r}"
        assert isinstance(meta, SokobanTaskMetadata), f"{key!r} did not load as SokobanTaskMetadata"


def test_all_reference_solutions_solve() -> None:
    """Every shipped level's reference solution actually solves the level."""
    for key, meta in SokobanBenchmarkConfig.task_metadata.items():
        assert isinstance(meta, SokobanTaskMetadata)
        assert simulate(meta.board, meta.solution).is_solved(), f"reference solution for {key!r} does not solve"


def test_push_and_block_semantics() -> None:
    """A box is pushed into free space but not into a wall."""
    # Player @ pushes box $ right onto goal . -> solved.
    state = SokobanState.parse(["#####", "#@$.#", "#####"])
    assert state.step("right") is True
    assert state.is_solved()
    # Box against the wall cannot be pushed further (player on a goal so counts match).
    blocked = SokobanState.parse(["####", "#+$#", "####"])
    assert blocked.step("right") is False
    assert blocked.player == (1, 1)  # player did not move


def test_blocked_move_keeps_state() -> None:
    """Walking into a wall is a no-op that does not move the player."""
    state = SokobanState.parse(["###", "#@#", "###"])
    assert state.step("up") is False
    assert state.player == (1, 1)


def test_tool_multi_move() -> None:
    """move() applies a comma/space-separated sequence in order and stops on solve."""
    tool = SokobanToolConfig().make()
    # # @ _ $ . #  -> push the box right twice onto the goal.
    tool.set_board(["######", "#@ $.#", "######"])
    out = tool.move("right, right")
    assert tool.is_solved, out
    assert out.count("=ok") == 2, out
    # A single direction still works, and an invalid token is reported, not raised.
    tool.set_board(["######", "#@ $.#", "######"])
    out2 = tool.move("nope right")
    assert "nope=invalid" in out2 and "right=ok" in out2


def test_sparse_vs_proportional_reward() -> None:
    """sparse_reward toggles evaluate() between proportional and the paper's sparse success reward."""
    # Two boxes / two goals: "#@$.$.#" — push the first box right onto its goal (1 of 2), not solved.
    board = ["#######", "#@$.$.#", "#######"]
    for sparse, partial in ((False, 0.5), (True, 0.0)):
        meta = SokobanTaskMetadata(id="t", board=board, solution=["right"], num_boxes=2, seed=0)
        task = SokobanTaskConfig(metadata=meta, tool_config=SokobanToolConfig(sparse_reward=sparse)).make()
        task.reset()
        task.tool.move("right")  # one box on its goal, level not yet solved
        assert not task.tool.is_solved
        assert task.evaluate()[0] == partial  # proportional 0.5 vs sparse 0.0


def test_debug_tasks() -> None:
    """Every debug task completes with reward == 1.0 (no LLM required)."""
    assert_debug_tasks_reward_one(_debug_mod)
