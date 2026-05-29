#!/usr/bin/env python3
"""submit_to_eee.py — convert an experiment dir into an Every Eval Ever (EEE)
record and write it to the EEE-expected on-disk layout
(``data/{benchmark}/{developer}/{model}/{uuid}.json``).

Pushing to the upstream HuggingFace dataset is left to the operator — V1 just
materializes the JSON next to the experiment so it can be inspected and
uploaded via ``huggingface-cli upload`` separately.

Usage:
  scripts/submit_to_eee.py <experiment_dir>
  scripts/submit_to_eee.py <experiment_dir> --out-dir ~/eee-staging
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.reproducibility.eee import build_eee_record


def _eee_path(out_dir: Path, record: dict) -> Path:
    """Render the EEE-expected file path ``data/{benchmark}/{developer}/{model}/{uuid}.json``."""
    benchmark = record["evaluation_results"][0]["evaluation_name"].split("[", 1)[0]
    developer = record["model_info"].get("developer") or "unknown"
    model = record["model_info"]["id"].replace("/", "__") or "unknown"
    target_dir = out_dir / "data" / benchmark / developer / model
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{uuid.uuid4().hex}.json"


def main(
    experiment_dir: Annotated[Path, typer.Argument(help="Path to the experiment output dir.")],
    submitter_org: Annotated[
        str,
        typer.Option(
            "--submitter-org",
            help="Free-text identifier for the submitting organization "
            "(shown in EEE's UI as source_metadata.source_organization_name).",
        ),
    ] = "cube-harness submitter",
    submitter_url: Annotated[
        str | None,
        typer.Option("--submitter-url", help="Optional URL for the submitter org."),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Root for the EEE-expected on-disk layout (data/<benchmark>/<dev>/<model>/<uuid>.json).",
        ),
    ] = Path("./eee-out"),
) -> None:
    """Build an EEE record from a completed experiment and write it under <out-dir>/data/."""
    record = build_eee_record(
        experiment_dir,
        source_organization_name=submitter_org,
        source_organization_url=submitter_url,
    )
    target = _eee_path(out_dir, record)
    target.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"wrote: {target}")
    typer.echo(
        f"  {record['evaluation_results'][0]['evaluation_name']} · "
        f"{record['model_info']['id']} · "
        f"score={record['evaluation_results'][0]['score_details']['score']:.3f}"
    )
    typer.echo("")
    typer.echo("To submit upstream:")
    typer.echo(f"  huggingface-cli upload <hf-dataset-id> {target} {target.relative_to(out_dir)} --repo-type dataset")


if __name__ == "__main__":
    typer.run(main)
