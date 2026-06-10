"""Seeded reverse-play generator for solvable Sokoban levels.

The hard guarantee: every level ships with a solution that provably solves it.
We get this for free from *reverse play*. Starting from the solved position
(all boxes on goals), we apply a random sequence of "reverse moves" — the
player walks, and may *pull* a box that sits directly behind it. The exact
forward inverse of a reverse move is an ordinary Sokoban move:

    reverse move: player P -> P+d, box at P-d pulled to P
    forward undo: player P+d -> P, pushing the box at P back to P-d

So the forward solution is just ``[opposite(d) for d in reversed(reverse_moves)]``
applied from the scrambled (start) position — it walks the boxes back onto the
goals. :func:`generate_level` returns that solution, and the caller
(:mod:`sokoban_cube` tests / ``scripts/create_task_metadata.py``) re-validates
it by forward simulation before shipping, so a buggy generator can never emit a
level whose solution does not solve.

Levels are ``dim`` x ``dim`` with a solid wall border (so the open interior is
``(dim-2)`` x ``(dim-2)``) and ``n_boxes`` boxes. The training default is the
canonical 6x6 board with 2 boxes.
"""

from __future__ import annotations

import random

from sokoban_cube.grid import DIRECTIONS, OPPOSITE, Cell, SokobanState, simulate


def _interior(dim: int) -> list[Cell]:
    """Floor cells inside the wall border of a ``dim`` x ``dim`` board."""
    return [(r, c) for r in range(1, dim - 1) for c in range(1, dim - 1)]


def _render(dim: int, goals: set[Cell], boxes: set[Cell], player: Cell) -> list[str]:
    walls = {(r, c) for r in range(dim) for c in range(dim) if r in (0, dim - 1) or c in (0, dim - 1)}
    state = SokobanState(dim, dim, walls, goals, set(boxes), player)
    return state.render().split("\n")


def generate_level(
    seed: int,
    dim: int = 6,
    n_boxes: int = 2,
    scramble_steps: int = 12,
    min_solution: int = 2,
    pull_bias: float = 0.85,
) -> tuple[list[str], list[str]] | None:
    """Generate one level via reverse play.

    Returns ``(board, solution)`` where ``board`` is the start position as ASCII
    rows and ``solution`` is a list of move names that solves it — or ``None``
    if this seed produced a degenerate level (no scramble, or boxes still on
    their goals). Callers iterate over seeds and keep the non-``None`` results.
    """
    rng = random.Random(seed)
    interior = _interior(dim)
    if len(interior) < n_boxes + 1:
        raise ValueError(f"{dim}x{dim} board has no room for {n_boxes} boxes plus the player.")

    goals: set[Cell] = set(rng.sample(interior, n_boxes))
    boxes: set[Cell] = set(goals)  # solved position: every box on its goal
    player: Cell = rng.choice([c for c in interior if c not in boxes])

    reverse_moves: list[str] = []
    for _ in range(scramble_steps):
        pulling: list[str] = []
        walking: list[str] = []
        for name, (dr, dc) in DIRECTIONS.items():
            ahead = (player[0] + dr, player[1] + dc)
            if ahead in boxes or ahead not in interior:
                continue  # player can only step onto open interior floor
            behind = (player[0] - dr, player[1] - dc)
            (pulling if behind in boxes else walking).append(name)
        if not pulling and not walking:
            break
        if pulling and (not walking or rng.random() < pull_bias):
            name = rng.choice(pulling)
        else:
            name = rng.choice(walking)
        dr, dc = DIRECTIONS[name]
        behind = (player[0] - dr, player[1] - dc)
        if behind in boxes:  # pull the box into the cell the player vacates
            boxes.discard(behind)
            boxes.add(player)
        player = (player[0] + dr, player[1] + dc)
        reverse_moves.append(name)

    solution = [OPPOSITE[name] for name in reversed(reverse_moves)]
    if len(solution) < min_solution or boxes <= goals:
        return None  # degenerate: nothing to solve

    board = _render(dim, goals, boxes, player)
    return board, solution


def generate_levels(
    count: int,
    *,
    start_seed: int = 0,
    dim: int = 6,
    n_boxes: int = 2,
    scramble_steps: int = 12,
    max_seeds: int | None = None,
) -> list[dict]:
    """Generate ``count`` distinct, validated levels.

    Each returned dict has ``seed``, ``board``, ``solution``, ``num_boxes`` and
    ``solution_length``. Boards are deduplicated, and every solution is
    re-checked by forward simulation — a level only ships if its solution
    actually reaches the solved state.
    """
    if max_seeds is None:
        max_seeds = start_seed + count * 200
    levels: list[dict] = []
    seen: set[str] = set()
    seed = start_seed
    while len(levels) < count and seed < max_seeds:
        result = generate_level(seed, dim=dim, n_boxes=n_boxes, scramble_steps=scramble_steps)
        if result is not None:
            board, solution = result
            key = "\n".join(board)
            if key not in seen and simulate(board, solution).is_solved():
                seen.add(key)
                levels.append(
                    {
                        "seed": seed,
                        "board": board,
                        "solution": solution,
                        "num_boxes": n_boxes,
                        "solution_length": len(solution),
                    }
                )
        seed += 1
    if len(levels) < count:
        raise RuntimeError(
            f"Only generated {len(levels)}/{count} levels within {max_seeds - start_seed} seeds; "
            "raise max_seeds or lower count."
        )
    return levels
