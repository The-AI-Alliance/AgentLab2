"""Tool layer for sokoban_cube.

A single action, :meth:`SokobanTool.move`, walks the player through a *sequence*
of steps (pushing a box whenever one is directly ahead). Multi-step turns mirror
the LaMer paper's ``<action>a,b,c</action>`` format and keep the agent loop cheap
(one LLM call per batch of moves rather than per step). The board is per-task
data, so the task loads it into the tool at ``reset()`` time via
:meth:`SokobanTool.set_board`; the tool itself just holds the live
:class:`~sokoban_cube.grid.SokobanState` and exposes read-only views for
``Task.evaluate`` / ``Task.finished``.
"""

import re
from typing import Any

from cube.container import Container
from cube.tool import Tool, ToolConfig, tool_action

from sokoban_cube.grid import DIRECTIONS, SokobanState


class SokobanEnv:
    """Holds the mutable game state and the board to reset back to."""

    def __init__(self) -> None:
        self._initial_board: list[str] | None = None
        self.state: SokobanState | None = None

    def set_board(self, board: list[str]) -> None:
        """Install the level the player will solve and parse the live state."""
        self._initial_board = list(board)
        self.state = SokobanState.parse(self._initial_board)

    def reset(self) -> None:
        """Restore the live state to the installed board's starting position."""
        if self._initial_board is not None:
            self.state = SokobanState.parse(self._initial_board)


class SokobanToolConfig(ToolConfig):
    """Serialisable config for the Sokoban tool.

    ``sparse_reward`` selects the reward shape read by :meth:`SokobanTask.evaluate`: ``False``
    (default) gives proportional ``boxes_on_goal / num_boxes``; ``True`` gives the LaMer paper's sparse
    success reward (1.0 iff fully solved, else 0.0). Sparse is the paper-faithful signal and also makes
    the multi-episode early-stop (reward > 0 ⇔ solved) exact; proportional is a denser cold-start aid.
    """

    sparse_reward: bool = False

    def make(self, container: Container | None = None) -> "SokobanTool":
        return SokobanTool(self)


class SokobanTool(Tool):
    """Agent-facing tool: move around the grid and push boxes onto goals."""

    def __init__(self, config: SokobanToolConfig) -> None:
        self._env = SokobanEnv()
        self._config = config

    def reset(self) -> None:
        self._env.reset()

    def set_board(self, board: list[str]) -> None:
        self._env.set_board(board)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @tool_action
    def move(self, directions: str) -> str:
        """Move one or more steps in sequence, pushing a box when one is directly ahead.

        Parameters
        ----------
        directions : str
            One or more directions from {"up", "down", "left", "right"}, separated by
            commas and/or whitespace (e.g. "up" or "up, up, left"). Moves are applied
            in order and stop early once the level is solved; a blocked move is a no-op.
            The board you see afterwards reflects the whole sequence.
        """
        state = self._env.state
        if state is None:
            return "No level loaded."
        tokens = [t for t in re.split(r"[\s,]+", directions.strip().lower()) if t]
        if not tokens:
            return "No direction given. Provide one or more of: up, down, left, right."
        trace: list[str] = []
        for key in tokens:
            if key not in DIRECTIONS:
                trace.append(f"{key}=invalid")
                continue
            trace.append(f"{key}={'ok' if state.step(key) else 'blocked'}")
            if state.is_solved():
                break
        return f"Moves: {' '.join(trace)}\n{self.render()}"

    @tool_action
    def _unknown_tool(self, name: str, arguments: dict[str, Any]) -> str:
        # Agents sometimes hallucinate tool names; return guidance instead of raising
        # so RL rollouts stay alive. Mirrors arithmetic-cube / math-tool-use.
        return f"Unknown tool {name!r} with arguments {arguments}. Use move(directions: str)."

    # ------------------------------------------------------------------
    # Read-only views for Task.evaluate / Task.finished
    # ------------------------------------------------------------------

    @property
    def num_boxes(self) -> int:
        return 0 if self._env.state is None else self._env.state.num_boxes

    @property
    def boxes_on_goal(self) -> int:
        return 0 if self._env.state is None else self._env.state.boxes_on_goal()

    @property
    def is_solved(self) -> bool:
        return self._env.state is not None and self._env.state.is_solved()

    @property
    def sparse_reward(self) -> bool:
        """Reward-shape knob read by ``Task.evaluate`` (see :class:`SokobanToolConfig`)."""
        return self._config.sparse_reward

    def render(self) -> str:
        """Board + status line, shown to the agent after each move."""
        state = self._env.state
        if state is None:
            return "No level loaded."
        status = f"Boxes on goals: {state.boxes_on_goal()}/{state.num_boxes}."
        if state.is_solved():
            status += " Solved!"
        return f"{state.render()}\n{status}"
