"""Solvable-subset analysis for gold-patch runs.

After a gold-patch run (see ``recipe.py``), these read the trajectory rewards
to report which tasks the environment can actually resolve. Pure post-hoc
helpers — no CLI, no side effects.

Requires cube-harness (not a swebench-verified-cube runtime dependency).
"""

from __future__ import annotations

from pathlib import Path

from cube_harness.storage import FileStorage


def extract_solvable(run_dir: Path) -> list[str]:
    """Task IDs that resolved (reward == 1.0) in a completed gold run."""
    trajs = FileStorage(run_dir).load_all_trajectory_metadata()
    return sorted(t.id.rsplit("_ep", 1)[0] for t in trajs if t.reward_info.get("reward") == 1.0)


def intersect_solvable(run_dirs: list[Path]) -> tuple[list[str], list[str]]:
    """Across N runs return (stable, flaky).

    stable — solved in ALL runs (safe eval subset).
    flaky  — solved in SOME runs (environment instability).
    """
    sets = [set(extract_solvable(d)) for d in run_dirs]
    stable = sorted(set.intersection(*sets))
    flaky = sorted(set.union(*sets) - set(stable))
    return stable, flaky
