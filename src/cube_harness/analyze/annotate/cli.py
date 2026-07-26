"""`ch-annotate` — serve and inspect the human judge-validation study.

    ch-annotate serve judge_study --share          # the portal, one shared link
    ch-annotate progress judge_study               # coverage so far
    ch-annotate export judge_study --out labels.json

`serve` seeds the pool from `<study_dir>/sample.json` on every start (idempotent)
and never reads `judge_key.json` — the judge's verdict has no path into the
portal.

Scoring stays in `scripts/paper/judge_validation.py score <study_dir>`, which
reads this study's SQLite file directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.analyze.annotate.app import build_app
from cube_harness.analyze.annotate.panel import EpisodeViewer
from cube_harness.analyze.annotate.store import (
    DEFAULT_LEASE_TTL,
    DEFAULT_TARGET_COVERAGE,
    AnnotationStore,
)

SAMPLE_FILENAME = "sample.json"

app = typer.Typer(add_completion=False, help=__doc__)


def _store(study_dir: Path, *, target_coverage: int, lease_ttl: float = DEFAULT_LEASE_TTL) -> AnnotationStore:
    return AnnotationStore(study_dir, target_coverage=target_coverage, lease_ttl=lease_ttl)


@app.command()
def serve(
    study_dir: Annotated[Path, typer.Argument(help="Study directory created by `judge_validation.py sample`.")],
    share: Annotated[bool, typer.Option(help="Create a public Gradio share link.")] = False,
    port: Annotated[int | None, typer.Option(help="Server port (Gradio picks one if unset).")] = None,
    target_coverage: Annotated[int, typer.Option(help="Independent labels wanted per episode.")] = (
        DEFAULT_TARGET_COVERAGE
    ),
    lease_minutes: Annotated[float, typer.Option(help="How long a claimed episode stays reserved.")] = (
        DEFAULT_LEASE_TTL / 60
    ),
    passphrase: Annotated[str | None, typer.Option(help="Gate the landing screen (share links are public).")] = None,
) -> None:
    """Launch the annotation portal against a study directory."""
    sample_path = study_dir / SAMPLE_FILENAME
    if not sample_path.exists():
        raise typer.BadParameter(f"{sample_path} not found — run `judge_validation.py sample` first")

    store = _store(study_dir, target_coverage=target_coverage, lease_ttl=lease_minutes * 60)
    n = store.seed_from_sample(sample_path)
    progress = store.progress()
    typer.echo(f"pool: {n} episodes, {progress.at_target}/{progress.total} at ≥{target_coverage} labels")
    if progress.per_annotator:
        typer.echo("labels so far: " + ", ".join(f"{a}={c}" for a, c in progress.per_annotator.items()))
    if share and not passphrase:
        typer.echo("note: the share link is unauthenticated — pass --passphrase to gate it")

    build_app(store, EpisodeViewer(), passphrase=passphrase).launch(
        share=share, server_port=port, show_api=False, quiet=False
    )


@app.command()
def progress(
    study_dir: Annotated[Path, typer.Argument(help="Study directory.")],
    target_coverage: Annotated[int, typer.Option(help="Labels wanted per episode.")] = DEFAULT_TARGET_COVERAGE,
) -> None:
    """Print per-annotator counts and the coverage histogram."""
    p = _store(study_dir, target_coverage=target_coverage).progress()
    typer.echo(f"{p.total} episodes; {p.at_target} at ≥{p.target_coverage} labels\n")
    typer.echo("labels per annotator:")
    for annotator, count in p.per_annotator.items():
        typer.echo(f"  {annotator:20s} {count}")
    if not p.per_annotator:
        typer.echo("  (none yet)")
    typer.echo("\ncoverage histogram (labels per episode → episodes):")
    for n_labels, count in p.coverage.items():
        typer.echo(f"  {n_labels} → {count}")


@app.command()
def export(
    study_dir: Annotated[Path, typer.Argument(help="Study directory.")],
    out: Annotated[Path, typer.Option(help="Where to write the JSON dump.")] = Path("labels.json"),
) -> None:
    """Dump every label as JSON — a portable backup of the study's output."""
    labels = _store(study_dir, target_coverage=DEFAULT_TARGET_COVERAGE).export_labels()
    out.write_text(json.dumps(labels, indent=2))
    typer.echo(f"wrote {out} ({sum(len(v) for v in labels.values())} labels from {len(labels)} annotators)")


@app.command()
def release(
    study_dir: Annotated[Path, typer.Argument(help="Study directory.")],
) -> None:
    """Reclaim every expired lease (the portal does this on each claim anyway)."""
    n = _store(study_dir, target_coverage=DEFAULT_TARGET_COVERAGE).expire_stale()
    typer.echo(f"reclaimed {n} expired lease(s)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
