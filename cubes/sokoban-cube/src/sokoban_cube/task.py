"""Task and TaskConfig for sokoban_cube.

Per-task data (the start board, its reference solution) lives on a typed
:class:`SokobanTaskMetadata` — lightweight and shipped inline in
``task_metadata.json``, so workers are self-contained and never touch a heavy
execution cache. :class:`SolveSokobanTask` is parameterised on the concrete
tool, so ``self.tool`` is typed as :class:`~sokoban_cube.tool.SokobanTool`
without ``isinstance`` narrowing.
"""

from typing import Any

from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata

from sokoban_cube.grid import LEGEND
from sokoban_cube.tool import SokobanTool, SokobanToolConfig

_INSTRUCTIONS = (
    "You are playing Sokoban. Push every box ($) onto a goal (.) by walking into it; "
    "a box slides one cell in the direction you move if the cell behind it is free. "
    "Boxes cannot be pushed into walls (#) or other boxes, and there is no pull move — "
    "a box shoved into a corner can become permanently stuck, so plan ahead. "
    "Move with move(directions): pass one or more of up, down, left, right, separated by "
    'commas (e.g. move("up, up, left")); they are applied in order and you then see the '
    "resulting board. You win when every box sits on a goal (shown as *)."
)


class SokobanTaskMetadata(TaskMetadata):
    """Typed per-task fields for one Sokoban level."""

    board: list[str]  # start position as ASCII rows
    solution: list[str]  # a move sequence known to solve it (reference / debug)
    num_boxes: int
    seed: int  # generation provenance
    dataset: str = "sokoban_6x6_2b"  # registry label; PipelineRL filters by extra_info["dataset"]

    @property
    def extra_info(self) -> dict:
        # cube_rl's registry filters tasks via ``task_metadata.extra_info["dataset"]`` and uses it
        # for the per-cube metric label (mirrors math-tool-use). Keeps that working without adding a
        # generic extra_info field to cube-standard's TaskMetadata.
        return {"dataset": self.dataset}


class SolveSokobanTask(Task[SokobanTaskMetadata, SokobanTool]):
    """One episode: push all boxes onto goals on the level from ``self.metadata``."""

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.set_board(self.metadata.board)
        self.tool.reset()
        prompt = f"{_INSTRUCTIONS}\n\n{LEGEND}\n\n{self.tool.render()}"
        info = {"num_boxes": self.metadata.num_boxes, "board": self.metadata.board}
        return Observation.from_text(prompt), info

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        num_boxes = self.tool.num_boxes
        on_goal = self.tool.boxes_on_goal
        solved = self.tool.is_solved
        if self.tool.sparse_reward:
            reward = 1.0 if solved else 0.0  # LaMer-paper sparse success reward
        else:
            reward = on_goal / num_boxes if num_boxes else 0.0  # proportional cold-start aid
        return reward, {"solved": solved, "boxes_on_goal": on_goal, "num_boxes": num_boxes}

    def finished(self, obs: Observation | None = None) -> bool:
        return self.tool.is_solved


class SokobanTaskConfig(TaskConfig[SokobanTaskMetadata]):
    """Serialisable factory that produces a :class:`SolveSokobanTask`."""

    def make(self, runtime_context: RuntimeContext | None = None) -> SolveSokobanTask:
        return SolveSokobanTask(
            metadata=self.metadata,
            tool_config=self.tool_config or SokobanToolConfig(),
            runtime_context=runtime_context,
        )
