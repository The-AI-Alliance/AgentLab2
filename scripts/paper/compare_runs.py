#!/usr/bin/env python3
"""compare_runs.py — task-level comparison of two experiment runs.

Built for the scaffold-sensitivity question: when two agent scaffolds are run
over the identical task set with the identical tool stack, a single aggregate
pass rate hides the thing that matters. Two scaffolds can land within a point of
each other while disagreeing on a third of the tasks, which would mean the
benchmark ranks scaffolds unreliably; or they can agree task for task, which
means the aggregate difference is real and small.

Reports, for the tasks both runs attempted:

  pass rate per run, and the paired difference with a McNemar exact test
  per-task agreement, and the discordant tasks in both directions
  Spearman rank correlation over per-task scores, when tasks carry partial credit

    scripts/paper/compare_runs.py ~/cube_harness_results/<run-a> ~/cube_harness_results/<run-b> \\
        --label-a genny --label-b react
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.analyze.investigator.episode_discovery import discover_episodes


def _scores(run_dir: Path) -> dict[str, float]:
    """Map task id → final score for one run, averaging repeated episodes."""
    totals: dict[str, list[float]] = {}
    for ref in discover_episodes(run_dir):
        record = ref.record
        if record is None:
            continue
        totals.setdefault(record.sample_id, []).append(record.score)
    return {task: sum(vals) / len(vals) for task, vals in totals.items()}


def _binomial_tail(k: int, n: int) -> float:
    """Two-sided exact binomial p-value at p=0.5, for McNemar on small samples."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main(
    run_a: Annotated[Path, typer.Argument(help="First experiment directory.")],
    run_b: Annotated[Path, typer.Argument(help="Second experiment directory.")],
    label_a: Annotated[str, typer.Option(help="Name for the first run.")] = "A",
    label_b: Annotated[str, typer.Option(help="Name for the second run.")] = "B",
    out: Annotated[Path | None, typer.Option(help="Write the comparison as JSON.")] = None,
) -> None:
    """Compare two runs task by task."""
    a, b = _scores(run_a), _scores(run_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise typer.BadParameter("the two runs share no task ids")

    pass_a = sum(a[t] >= 1.0 for t in shared)
    pass_b = sum(b[t] >= 1.0 for t in shared)
    only_a = [t for t in shared if a[t] >= 1.0 > b[t]]
    only_b = [t for t in shared if b[t] >= 1.0 > a[t]]
    agree = len(shared) - len(only_a) - len(only_b)
    p_value = _binomial_tail(len(only_a), len(only_a) + len(only_b))
    rho = _spearman([a[t] for t in shared], [b[t] for t in shared])

    n = len(shared)
    typer.echo(f"{n} shared tasks ({len(set(a) ^ set(b))} attempted by only one run)\n")
    typer.echo(f"  {label_a:10s} pass rate: {pass_a / n * 100:5.1f}%  ({pass_a}/{n})")
    typer.echo(f"  {label_b:10s} pass rate: {pass_b / n * 100:5.1f}%  ({pass_b}/{n})")
    typer.echo(f"  difference          : {(pass_b - pass_a) / n * 100:+5.1f} pp\n")
    typer.echo(f"  same outcome        : {agree}/{n}  ({agree / n * 100:.1f}%)")
    typer.echo(f"  solved by {label_a} only : {len(only_a):3d}  {', '.join(only_a[:6])}")
    typer.echo(f"  solved by {label_b} only : {len(only_b):3d}  {', '.join(only_b[:6])}")
    typer.echo(f"  McNemar exact p     : {p_value:.3f}")
    typer.echo(f"  Spearman rho        : {rho:.3f}")

    if out:
        out.write_text(
            json.dumps(
                {
                    "label_a": label_a,
                    "label_b": label_b,
                    "n_shared": n,
                    "pass_a": pass_a,
                    "pass_b": pass_b,
                    "agreement": agree / n,
                    "only_a": only_a,
                    "only_b": only_b,
                    "mcnemar_p": p_value,
                    "spearman_rho": rho,
                },
                indent=2,
            )
        )
        typer.echo(f"\nwrote {out}")


if __name__ == "__main__":
    typer.run(main)
