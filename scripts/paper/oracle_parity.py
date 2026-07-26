#!/usr/bin/env python3
"""oracle_parity.py — report layer-3 parity: does the wrapper reward the reference solution?

A benchmark ships its own answer key. If a wrapper's evaluator does not award
full credit to that answer key, the wrapper is broken regardless of what any
agent scores on it. This script summarises completed oracle runs (the scripted,
LLM-free gold-patch / ``solve.sh`` agents each CUBE ships) into the parity table
the paper reports, and writes the resolved-id list that pins the gold-solvable
subset for later agent evaluation.

Running the oracles (each writes an ordinary experiment directory):

    python -m swebench_verified_cube.gold_patch.recipe --experiment toolkit --ray 50
    python -m swebench_live_cube.gold_patch.recipe     --experiment daytona --ray 20

Then summarise one or more runs:

    scripts/paper/oracle_parity.py ~/cube_harness_results/gold-patch-verified-* \\
        --expected 500 --label "SWE-bench Verified"

Passing several directories for the same benchmark reports the *stable* subset
(resolved in every run) alongside the flaky one, because a task that resolves
only sometimes is an infrastructure signal, not an evaluator signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.analyze.investigator.episode_discovery import discover_episodes


def resolved_ids(run_dir: Path) -> tuple[set[str], set[str]]:
    """Return (resolved, attempted) task ids for one oracle run."""
    resolved: set[str] = set()
    attempted: set[str] = set()
    for ref in discover_episodes(run_dir):
        record = ref.record
        if record is None:
            continue
        attempted.add(record.sample_id)
        if record.score >= 1.0:
            resolved.add(record.sample_id)
    return resolved, attempted


def main(
    run_dirs: Annotated[list[Path], typer.Argument(help="One or more completed oracle run directories.")],
    label: Annotated[str, typer.Option(help="Benchmark name for the report.")] = "benchmark",
    expected: Annotated[int, typer.Option(help="Upstream task count the oracle should resolve.")] = 0,
    dump_solvable: Annotated[Path | None, typer.Option(help="Write the stable resolved-id list here.")] = None,
) -> None:
    """Summarise oracle runs into a parity result."""
    per_run = [resolved_ids(d) for d in run_dirs]
    per_run = [(r, a) for r, a in per_run if a]
    if not per_run:
        raise typer.BadParameter("no episodes with records found in the given directories")

    attempted = set.union(*(a for _, a in per_run))
    stable = set.intersection(*(r for r, _ in per_run))
    ever = set.union(*(r for r, _ in per_run))
    flaky = ever - stable
    denominator = expected or len(attempted)

    typer.echo(f"{label}: {len(run_dirs)} oracle run(s)")
    typer.echo(f"  attempted            : {len(attempted)}")
    typer.echo(f"  resolved in all runs : {len(stable)}  ({len(stable) / denominator * 100:.1f}% of {denominator})")
    if len(per_run) > 1:
        typer.echo(f"  resolved in some runs: {len(flaky)}  (infrastructure flake, not evaluator semantics)")
    unresolved = sorted(attempted - ever)
    if unresolved:
        typer.echo(f"  never resolved       : {len(unresolved)}  e.g. {', '.join(unresolved[:6])}")

    typer.echo(f"\nLaTeX row:\n{label} & {denominator} & {len(stable)} & {len(stable) / denominator * 100:.1f}\\% \\\\")

    if dump_solvable:
        dump_solvable.write_text(
            json.dumps(
                {
                    "benchmark": label,
                    "n_tasks": len(stable),
                    "description": "Task ids resolved by the reference solution under the CUBE evaluator in every listed run.",
                    "runs": [str(d) for d in run_dirs],
                    "task_ids": sorted(stable),
                },
                indent=2,
            )
        )
        typer.echo(f"\nwrote {dump_solvable}")


if __name__ == "__main__":
    typer.run(main)
