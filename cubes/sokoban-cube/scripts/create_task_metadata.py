#!/usr/bin/env python3
"""Regenerate ``src/sokoban_cube/task_metadata.json``.

Deterministically generates a pool of solvable 6x6 / 2-box Sokoban levels (see
:mod:`sokoban_cube.generator`), stamps each with the ``_type`` of
:class:`sokoban_cube.task.SokobanTaskMetadata` so the framework deserialises it
into the typed subclass, tags each with a ``dataset`` label (so PipelineRL's
cube_rl registry can filter via ``extra_info["dataset"]``), assigns train/val/test
splits, and writes the JSON the benchmark auto-loads (Option B). Committed but not
shipped in the wheel.

The default (5000 levels) reproduces the shipped dataset for RL training; the
held-out RL eval split is carved by index via ``train_subset``/``test_subset`` in
the training config (the per-task ``split`` field is informational).

Run from the package root::

    PYTHONPATH=src python scripts/create_task_metadata.py --force          # 5000 levels (shipped)
    PYTHONPATH=src python scripts/create_task_metadata.py --num 40 --force  # tiny pool for quick tests
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokoban_cube.generator import generate_levels
from sokoban_cube.grid import simulate

# recommended_max_steps: generous budget. Optimal solutions on a 6x6/2-box board
# are short; reference solutions are <= the scramble length. 40 leaves real
# agents room to explore without making the episode run forever.
RECOMMENDED_MAX_STEPS = 40

_TYPE = "sokoban_cube.task.SokobanTaskMetadata"
_OUT = Path(__file__).resolve().parent.parent / "src" / "sokoban_cube" / "task_metadata.json"


def _split_for(index: int, total: int) -> str:
    """80/10/10 train/val/test split, deterministic in index order."""
    val_start = int(total * 0.8)
    test_start = int(total * 0.9)
    if index < val_start:
        return "train"
    return "val" if index < test_start else "test"


def build(num: int, *, dim: int, n_boxes: int, start_seed: int) -> list[dict]:
    levels = generate_levels(num, start_seed=start_seed, dim=dim, n_boxes=n_boxes)
    dataset = f"sokoban_{dim}x{dim}_{n_boxes}b"  # registry label (extra_info["dataset"]); see task.py
    records: list[dict] = []
    for i, lv in enumerate(levels):
        # Defensive re-validation: never ship a level whose solution doesn't solve.
        assert simulate(lv["board"], lv["solution"]).is_solved(), f"level {i} solution does not solve"
        task_id = f"sokoban-{dim}x{dim}-{n_boxes}b-{i:05d}"
        records.append(
            {
                "_type": _TYPE,
                "id": task_id,
                "split": _split_for(i, len(levels)),
                "dataset": dataset,
                "abstract_description": f"Push the {n_boxes} boxes onto the goals (level {i:05d}).",
                "recommended_max_steps": RECOMMENDED_MAX_STEPS,
                "container_config": None,
                "board": lv["board"],
                "solution": lv["solution"],
                "num_boxes": n_boxes,
                "seed": lv["seed"],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sokoban task_metadata.json")
    parser.add_argument("--num", type=int, default=5000, help="number of levels to generate")
    parser.add_argument("--dim", type=int, default=6, help="board side length (incl. wall border)")
    parser.add_argument("--boxes", type=int, default=2, help="number of boxes")
    parser.add_argument("--start-seed", type=int, default=1, help="first generation seed")
    parser.add_argument("--out", type=Path, default=_OUT, help="output JSON path")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite.")

    records = build(args.num, dim=args.dim, n_boxes=args.boxes, start_seed=args.start_seed)
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    splits = {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}
    print(f"Wrote {len(records)} levels to {args.out}  (splits: {splits})")
    print(f"Remember to set benchmark_metadata.json num_tasks = {len(records)}")


if __name__ == "__main__":
    main()
