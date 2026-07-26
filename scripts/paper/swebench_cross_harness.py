#!/usr/bin/env python3
"""swebench_cross_harness.py — re-grade CUBE episodes with the upstream SWE-bench harness.

The sharpest parity test available for a wrapped benchmark: hold the *artefact*
constant and vary the *grader*. Every CUBE SWE-bench episode records the agent's
model patch; this script exports those patches in the upstream predictions
format, hands them to the official SWE-bench harness, and compares the two
verdicts task by task.

Any disagreement is a wrapper-vs-original difference in evaluation, measured
directly rather than inferred from a judge. Our evaluator makes three
deliberate, documented relaxations against upstream (pre-existing
``PASS_TO_PASS`` failures are not charged to the agent; known
network-dependent tests are skipped; ``pytest`` "no tests collected" is not a
failure), so a small, one-directional disagreement is the expected result. The
point is to publish its size and direction instead of assuming it is zero.

    # 1. export patches from a completed CUBE run
    scripts/paper/swebench_cross_harness.py export ~/cube_harness_results/<run> --out preds.jsonl

    # 2. grade them upstream (either path)
    python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \\
        --predictions_path preds.jsonl --run_id cube_parity --max_workers 8
    sb-cli submit swe-bench_verified test --predictions_path preds.jsonl --run_id cube_parity

    # 3. compare verdicts
    scripts/paper/swebench_cross_harness.py compare ~/cube_harness_results/<run> \\
        --upstream cube_parity.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from cube_harness.analyze.investigator.episode_discovery import discover_episodes

app = typer.Typer(add_completion=False, help=__doc__)

MODEL_FIELD = "model_name_or_path"


def _episodes(run_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Yield (instance_id, reward_info) for every completed episode in a run."""
    out: list[tuple[str, dict[str, Any]]] = []
    for ref in discover_episodes(run_dir):
        record = ref.record
        if record is None:
            continue
        info = dict(getattr(record, "reward_info", {}) or {})
        out.append((record.sample_id, info))
    return out


@app.command()
def export(
    run_dir: Annotated[Path, typer.Argument(help="A completed CUBE SWE-bench experiment directory.")],
    out: Annotated[Path, typer.Option(help="Where to write predictions.jsonl.")] = Path("predictions.jsonl"),
    model_name: Annotated[str, typer.Option(help="Value for the predictions' model field.")] = "cube-genny",
) -> None:
    """Export recorded model patches in the upstream predictions format."""
    rows, missing = [], []
    for instance_id, info in _episodes(run_dir):
        patch = info.get("model_patch")
        if not patch:
            missing.append(instance_id)
            continue
        rows.append({"instance_id": instance_id, MODEL_FIELD: model_name, "model_patch": patch})

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    typer.echo(f"wrote {len(rows)} predictions to {out}")
    if missing:
        typer.echo(
            f"{len(missing)} episodes carried no model_patch (runs predating patch capture, "
            f"or episodes that crashed before evaluate): {', '.join(missing[:5])}"
        )


def _upstream_resolved(path: Path) -> dict[str, bool]:
    """Parse resolved-instance ids out of an upstream harness report."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "resolved_ids" in data:  # run_evaluation report
        resolved = set(data.get("resolved_ids", []))
        attempted = set(data.get("resolved_ids", [])) | set(data.get("unresolved_ids", []))
        return {i: i in resolved for i in attempted}
    if isinstance(data, dict):  # sb-cli style {instance_id: {resolved: bool}}
        return {k: bool(v.get("resolved", False)) for k, v in data.items() if isinstance(v, dict)}
    raise typer.BadParameter(f"unrecognised upstream report shape in {path}")


@app.command()
def compare(
    run_dir: Annotated[Path, typer.Argument(help="The same CUBE run that produced the predictions.")],
    upstream: Annotated[Path, typer.Option(help="Upstream harness report JSON.")] = Path("report.json"),
    out: Annotated[Path, typer.Option(help="Where to write the LaTeX row.")] = Path("cross_harness.tex"),
) -> None:
    """Compare CUBE verdicts against upstream verdicts on the identical patches."""
    cube = {iid: bool(info.get("resolved", False)) for iid, info in _episodes(run_dir)}
    up = _upstream_resolved(upstream)
    shared = sorted(set(cube) & set(up))
    if not shared:
        raise typer.BadParameter("no overlapping instance ids between the run and the upstream report")

    both = sum(cube[i] and up[i] for i in shared)
    neither = sum(not cube[i] and not up[i] for i in shared)
    cube_only = [i for i in shared if cube[i] and not up[i]]
    up_only = [i for i in shared if not cube[i] and up[i]]
    agreement = (both + neither) / len(shared)

    typer.echo(f"n = {len(shared)} instances graded by both harnesses")
    typer.echo(f"  resolved by both      : {both}")
    typer.echo(f"  resolved by neither   : {neither}")
    typer.echo(f"  CUBE only (lenient)   : {len(cube_only)}  {', '.join(cube_only[:8])}")
    typer.echo(f"  upstream only (strict): {len(up_only)}  {', '.join(up_only[:8])}")
    typer.echo(f"  verdict agreement     : {agreement * 100:.1f}%")
    typer.echo(
        f"  pass rate  CUBE {sum(cube[i] for i in shared) / len(shared) * 100:.1f}%"
        f"  upstream {sum(up[i] for i in shared) / len(shared) * 100:.1f}%"
    )

    out.write_text(
        f"{len(shared)} & {both} & {neither} & {len(cube_only)} & {len(up_only)} & {agreement * 100:.1f}\\% \\\\\n"
    )
    (out.with_suffix(".json")).write_text(
        json.dumps(
            {
                "n": len(shared),
                "both": both,
                "neither": neither,
                "cube_only": cube_only,
                "upstream_only": up_only,
                "agreement": agreement,
            },
            indent=2,
        )
    )
    typer.echo(f"\nwrote {out}")


if __name__ == "__main__":
    app()
