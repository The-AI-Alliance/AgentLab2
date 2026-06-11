"""Build a hint bank for hint-conditioned RL from a JefHinter run's mined hints.

Reads a JefHinter ``metrics.json`` (its ``hint_db`` = {task_id: [hints]}) and emits
the three per-task hint maps the RL cubes consume:

  good       : {task_id: <best mined hint>}     # inject the real hint
  none       : {}                               # empty (agent gets no hint)
  distractor : {task_id: <a hint mined for a DIFFERENT task>}

Usage:
    .venv/bin/python scripts/build_hint_bank.py <metrics.json> <out_bank.json> [--seed 0]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def build_bank(metrics_path: Path, seed: int = 0) -> dict:
    hint_db: dict[str, list[str]] = json.loads(metrics_path.read_text()).get("hint_db", {})
    good = {tid: hints[0] for tid, hints in hint_db.items() if hints}  # first/best hint per task
    task_ids = sorted(good)
    rng = random.Random(seed)
    distractor: dict[str, str] = {}
    for tid in task_ids:
        others = [t for t in task_ids if t != tid]
        if others:
            distractor[tid] = good[rng.choice(others)]  # a real hint, but for the wrong task
    return {"good": good, "none": {}, "distractor": distractor, "task_ids": task_ids}


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    bank = build_bank(src, seed)
    out.write_text(json.dumps(bank, indent=2))
    print(f"bank -> {out}  ({len(bank['good'])} good, {len(bank['distractor'])} distractor)")
