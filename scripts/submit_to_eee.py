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
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.reproducibility import submissions
from cube_harness.reproducibility.eee import EEE_SCHEMA_VERSION, build_eee_record

# The community EEE datastore. `--upload` opens a PR here (which EEE's own
# validation + a maintainer review then process). Override for a private dataset.
DEFAULT_EEE_DATASET = "evaleval/EEE_datastore"


def _eee_path(out_dir: Path, record: dict) -> Path:
    """Render the EEE datastore path ``data/{benchmark}/{developer}/{model}/{uuid}.json``.

    Mirrors the live datastore's layout: lowercase developer slug, model
    directory without the provider prefix — e.g.
    ``data/GAIA/anthropic/claude-3-7-sonnet-20250219/<uuid>.json``.
    """
    benchmark = record["evaluation_results"][0]["evaluation_name"].split("[", 1)[0]
    developer = record["model_info"].get("developer") or "unknown"
    model = record["model_info"]["id"].split("/")[-1] or "unknown"
    target_dir = out_dir / "data" / benchmark / developer / model
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{uuid.uuid4().hex}.json"


def _validate_record(target: Path) -> None:
    """Validate *target* against EEE's schema via the ``every-eval-ever`` package.

    Raises ``typer.Exit(1)`` on a validation failure. Skips (with a warning) when
    the package isn't installed — it's an optional dev dependency; the upstream PR
    runs the same validation, so this is a fast local pre-check, not the gate."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "every_eval_ever", "validate", str(target)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        typer.echo("  (every-eval-ever not installed — skipping local schema check)", err=True)
        return
    if "No module named" in proc.stderr:
        typer.echo("  (every-eval-ever not installed — skipping local schema check)", err=True)
        return
    if proc.returncode != 0:
        typer.echo(proc.stdout + proc.stderr, err=True)
        typer.echo("EEE schema validation failed — not uploading.", err=True)
        raise typer.Exit(code=1)
    typer.echo("  ✓ validates against the EEE schema")


def _upload_to_eee(target: Path, out_dir: Path, dataset_id: str) -> str:
    """Open a PR adding *target* to the EEE HF dataset. Returns the PR url.

    Lazy-imports ``huggingface_hub`` (optional, upload-only dep). Uses
    ``create_pr=True`` so it lands as a reviewable PR — never a direct push to the
    community dataset's main branch. Auth via the standard HF token resolution
    (``HF_TOKEN`` / ``huggingface-cli login``)."""
    from huggingface_hub import HfApi  # noqa: PLC0415 — optional upload-only dependency

    path_in_repo = str(target.relative_to(out_dir))
    info = HfApi().upload_file(
        path_or_fileobj=str(target),
        path_in_repo=path_in_repo,
        repo_id=dataset_id,
        repo_type="dataset",
        create_pr=True,
        commit_message=f"results: add {target.stem} ({path_in_repo})",
    )
    return info.pr_url


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
    upload: Annotated[
        bool,
        typer.Option(
            "--upload/--no-upload",
            help="Open a PR adding the record to the EEE HuggingFace dataset (an outward "
            "publish). Without it, the record is only materialized locally.",
        ),
    ] = False,
    dataset_id: Annotated[
        str,
        typer.Option("--dataset-id", help="EEE HF dataset to open the PR against."),
    ] = DEFAULT_EEE_DATASET,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-emit even when submissions.json already records an 'eee' decision.",
        ),
    ] = False,
) -> None:
    """Build an EEE record from a completed experiment, validate it, and optionally
    open a PR to the EEE HuggingFace dataset (``--upload``)."""
    if not force and submissions.has_decision(experiment_dir, "eee"):
        prior = submissions.read(experiment_dir).get("eee", {})
        typer.echo(
            f"experiment_dir already has an 'eee' decision: {prior.get('status')} "
            f"({prior.get('reason') or prior.get('evaluation_id')}).",
            err=True,
        )
        typer.echo("Pass --force to override.", err=True)
        raise typer.Exit(code=2)

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

    _validate_record(target)

    if not upload:
        typer.echo("")
        typer.echo("Materialized locally (not submitted). To publish, re-run with --upload, or by hand:")
        typer.echo(f"  hf upload {dataset_id} {target} {target.relative_to(out_dir)} --repo-type dataset")
        return

    # Mark in-progress so a crash leaves a retryable 'pending', not a false 'submitted'.
    submissions.record_pending(experiment_dir, "eee")
    try:
        pr_url = _upload_to_eee(target, out_dir, dataset_id)
    except Exception as e:  # noqa: BLE001 — surface any HF/network failure as a recorded failure
        submissions.record_failed(experiment_dir, "eee", reason=f"{type(e).__name__}: {e}")
        typer.echo(f"EEE upload failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"PR opened: {pr_url}")
    submissions.record_submitted(
        experiment_dir,
        "eee",
        evaluation_id=record["evaluation_id"],
        schema_version=EEE_SCHEMA_VERSION,
        pr_url=pr_url,
        local_path=str(target),
    )
    typer.echo(f"recorded in {experiment_dir / submissions.SUBMISSIONS_FILENAME}")


if __name__ == "__main__":
    typer.run(main)
