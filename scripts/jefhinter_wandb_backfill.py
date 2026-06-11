"""Backfill a finished JefHinter run's metrics.json into Weights & Biases.

For runs launched before W&B logging was wired into recipes/jefhinter.py. Reads
<run_dir>/metrics.json and logs the per-iteration success curve, per-task rates,
the curve image, and the mined-hints table into the given project.

    .venv/bin/python scripts/jefhinter_wandb_backfill.py <run_dir> <run_name> [project]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb


def backfill(
    run_dir: Path,
    run_name: str,
    project: str = "jeffhinter",
    group: str | None = None,
    tags: list[str] | None = None,
    config: dict | None = None,
) -> str:
    data = json.loads((run_dir / "metrics.json").read_text())
    run = wandb.init(
        project=project,
        name=run_name,
        group=group,
        tags=tags or ["eval"],
        reinit=True,
        config={"backfilled": True, "run_dir": str(run_dir), **(config or {})},
    )
    for record in data["iterations"]:
        row = {"success_rate": record["overall"], "n_hints": record["hints_in_db"]}
        for task_id, rate in record["per_task"].items():
            row[f"task_success/{task_id.split('.')[-1]}"] = rate
        run.log(row, step=record["iteration"])
    curve = run_dir / "curve.png"
    if curve.exists():
        run.log({"curve": wandb.Image(str(curve))})
    table = wandb.Table(columns=["task_id", "hint"])
    for task_id, hints in data.get("hint_db", {}).items():
        for hint in hints:
            table.add_data(task_id, hint)
    run.log({"hints": table})
    url = run.url
    run.finish()
    return url


if __name__ == "__main__":
    rd, name = Path(sys.argv[1]), sys.argv[2]
    project = sys.argv[3] if len(sys.argv) > 3 else "jeffhinter"
    group = sys.argv[4] if len(sys.argv) > 4 else None
    print("W&B run:", backfill(rd, name, project, group=group))
