"""Pure Sokoban simulator — board parsing, push semantics, rendering, scoring.

No cube-standard imports: this module is the deterministic game core, reused by
the tool (runtime), the generator (level creation), and the tests. Coordinates
are ``(row, col)`` with row 0 at the top.

Board glyphs (standard Sokoban)::

    #  wall            .  goal (empty)
    @  player          +  player on goal
    $  box             *  box on goal
       floor (space)

A level is *solved* when every box sits on a goal. Since a level always has
exactly ``len(goals)`` boxes, that is equivalent to every goal being covered.
"""

from __future__ import annotations

from typing import Final

Cell = tuple[int, int]

# Movement directions, keyed by the names the agent uses.
DIRECTIONS: Final[dict[str, Cell]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE: Final[dict[str, str]] = {"up": "down", "down": "up", "left": "right", "right": "left"}

LEGEND: Final[str] = "Legend: # wall, @ you, $ box, . goal, * box-on-goal, + you-on-goal, (space) floor."


class SokobanState:
    """Mutable Sokoban position over a fixed wall/goal layout.

    Static layout (``walls``, ``goals``, ``height``, ``width``) never changes;
    only ``player`` and ``boxes`` move. Build one with :meth:`parse` and mutate
    it with :meth:`step`.
    """

    def __init__(
        self,
        height: int,
        width: int,
        walls: set[Cell],
        goals: set[Cell],
        boxes: set[Cell],
        player: Cell,
    ) -> None:
        self.height = height
        self.width = width
        self.walls = walls
        self.goals = goals
        self.boxes = boxes
        self.player = player

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, board: list[str]) -> "SokobanState":
        """Build a state from ASCII rows. Rows may be ragged; missing cells are floor."""
        height = len(board)
        width = max((len(row) for row in board), default=0)
        walls: set[Cell] = set()
        goals: set[Cell] = set()
        boxes: set[Cell] = set()
        player: Cell | None = None
        for r, row in enumerate(board):
            for c, ch in enumerate(row):
                pos = (r, c)
                if ch == "#":
                    walls.add(pos)
                elif ch == "@":
                    player = pos
                elif ch == "+":
                    player = pos
                    goals.add(pos)
                elif ch == "$":
                    boxes.add(pos)
                elif ch == "*":
                    boxes.add(pos)
                    goals.add(pos)
                elif ch == ".":
                    goals.add(pos)
                # ' ' (floor) and any other char: nothing to record
        if player is None:
            raise ValueError("Board has no player ('@' or '+').")
        if len(boxes) != len(goals):
            raise ValueError(f"Board has {len(boxes)} boxes but {len(goals)} goals; counts must match.")
        return cls(height, width, walls, goals, boxes, player)

    def copy(self) -> "SokobanState":
        return SokobanState(
            self.height,
            self.width,
            self.walls,  # static — safe to share
            self.goals,  # static — safe to share
            set(self.boxes),
            self.player,
        )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _blocked(self, pos: Cell) -> bool:
        """A cell the player or a box can never occupy: a wall or off-board."""
        r, c = pos
        if r < 0 or c < 0 or r >= self.height or c >= self.width:
            return True
        return pos in self.walls

    def step(self, direction: str) -> bool:
        """Move the player one cell in ``direction``, pushing a box if present.

        Returns ``True`` if the player (and possibly a box) moved, ``False`` if
        the move was blocked by a wall, the board edge, or an immovable box.
        Raises ``KeyError`` only for an unknown direction name.
        """
        dr, dc = DIRECTIONS[direction]
        pr, pc = self.player
        target = (pr + dr, pc + dc)
        if self._blocked(target):
            return False
        if target in self.boxes:
            beyond = (target[0] + dr, target[1] + dc)
            if self._blocked(beyond) or beyond in self.boxes:
                return False  # box can't be pushed into a wall or another box
            self.boxes.remove(target)
            self.boxes.add(beyond)
        self.player = target
        return True

    # ------------------------------------------------------------------
    # Scoring / rendering
    # ------------------------------------------------------------------

    def boxes_on_goal(self) -> int:
        return len(self.boxes & self.goals)

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)

    def is_solved(self) -> bool:
        return self.boxes <= self.goals

    def render(self) -> str:
        """Render the current position back to ASCII (inverse of :meth:`parse`)."""
        rows: list[str] = []
        for r in range(self.height):
            chars: list[str] = []
            for c in range(self.width):
                pos = (r, c)
                if pos in self.walls:
                    ch = "#"
                elif pos == self.player:
                    ch = "+" if pos in self.goals else "@"
                elif pos in self.boxes:
                    ch = "*" if pos in self.goals else "$"
                elif pos in self.goals:
                    ch = "."
                else:
                    ch = " "
                chars.append(ch)
            rows.append("".join(chars))
        return "\n".join(rows)


def simulate(board: list[str], moves: list[str]) -> SokobanState:
    """Replay ``moves`` from ``board`` and return the resulting state.

    Used by the generator and tests to confirm a recorded solution actually
    solves its level.
    """
    state = SokobanState.parse(board)
    for move in moves:
        state.step(move)
    return state
