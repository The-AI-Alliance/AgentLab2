#!/usr/bin/env python3
"""submit_to_journal.py — convert an experiment dir into a cube-registry
community-journal record and (optionally) open a PR against cube-registry.

Usage:
  scripts/submit_to_journal.py <experiment_dir>
  scripts/submit_to_journal.py <experiment_dir> --auto-pr

Without ``--auto-pr`` the script writes the JSON to
``./journal-out/results/<cube>/<file>.json`` and prints the ``gh`` command you
can run to open the PR by hand. With ``--auto-pr``, it clones cube-registry to
``/tmp``, creates a branch, copies the file in, commits, pushes, and opens the
PR via ``gh``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.reproducibility import (
    JOURNAL_SCHEMA_VERSION,
    build_journal_record,
    sanitize_filename,
)

CUBE_REGISTRY_REPO = "The-AI-Alliance/cube-registry"
CUBE_REGISTRY_URL = f"https://github.com/{CUBE_REGISTRY_REPO}.git"


def _git_user_handle() -> str:
    """Best-effort GitHub handle: ``GIT_AUTHOR_NAME`` env > ``git config user.name``."""
    env = os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("USER")
    if env:
        return env
    try:
        out = subprocess.check_output(["git", "config", "user.name"], text=True, stderr=subprocess.DEVNULL)
        return out.strip() or "submitter"
    except subprocess.CalledProcessError:
        return "submitter"


def _write_record(record: dict, out_root: Path) -> Path:
    cube_id = record["benchmark_name"]
    eid = record["evaluation_id"]
    target_dir = out_root / "results" / cube_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{sanitize_filename(eid)}.json"
    target_path.write_text(json.dumps(record, indent=2) + "\n")
    return target_path


def _open_pr(record_path: Path, record: dict, branch: str) -> str:
    """Clone cube-registry to a temp dir, copy *record_path* in, push, gh pr create."""
    with tempfile.TemporaryDirectory(prefix="cube-registry-submit-") as tmp:
        clone_dir = Path(tmp) / "cube-registry"
        subprocess.run(
            ["git", "clone", "--depth", "1", CUBE_REGISTRY_URL, str(clone_dir)],
            check=True,
        )
        subprocess.run(["git", "-C", str(clone_dir), "checkout", "-b", branch], check=True)
        # Copy the record into the clone at the expected path.
        cube_id = record["benchmark_name"]
        dst_dir = clone_dir / "results" / cube_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / record_path.name
        shutil.copyfile(record_path, dst_path)
        subprocess.run(
            ["git", "-C", str(clone_dir), "add", str(dst_path.relative_to(clone_dir))],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(clone_dir),
                "commit",
                "-s",
                "-m",
                f"results: add {record['evaluation_id']}",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(clone_dir), "push", "-u", "origin", branch], check=True)
        title = f"results: {record['benchmark_name']} — {record['agent']['llm_model']}"
        body = (
            f"Adds one community evaluation result for `{record['benchmark_name']}`.\n\n"
            f"- agent: `{record['agent']['config_type']}` on `{record['agent']['llm_model']}`\n"
            f"- score: **{record['results']['avg_score']:.3f}** "
            f"± {record['results']['std_err']:.3f}\n"
            f"- subset: `{record['benchmark_subset']['name']}` "
            f"({record['benchmark_subset']['n_tasks']} tasks)\n"
            f"- outcomes: {record['results']['outcomes']}\n\n"
            f"_Submitted via cube-harness `scripts/submit_to_journal.py`._"
        )
        pr = subprocess.check_output(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                CUBE_REGISTRY_REPO,
                "--title",
                title,
                "--body",
                body,
            ],
            text=True,
            cwd=clone_dir,
        ).strip()
        return pr


def main(
    experiment_dir: Annotated[Path, typer.Argument(help="Path to the experiment output dir.")],
    submitter: Annotated[
        str | None,
        typer.Option(
            "--submitter",
            help="GitHub handle for the evaluation_id namespace (defaults to git config user.name).",
        ),
    ] = None,
    cube_id: Annotated[
        str | None,
        typer.Option(
            "--cube-id",
            help="Override the cube id when it differs from the benchmark name "
            "(e.g. 'miniwob[level=all]' is benchmark_name; cube-id is 'miniwob').",
        ),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Local output directory. The file is written under <out-dir>/results/<cube>/.",
        ),
    ] = Path("./journal-out"),
    auto_pr: Annotated[
        bool,
        typer.Option(
            "--auto-pr",
            help="Clone cube-registry, commit the record on a fresh branch, push, and open the PR via gh.",
        ),
    ] = False,
) -> None:
    """Build a cube-registry community-journal record and (optionally) open a PR."""
    submitter = submitter or _git_user_handle()
    typer.echo(f"submitter: {submitter}")
    typer.echo(f"experiment_dir: {experiment_dir}")

    record = build_journal_record(experiment_dir, submitter=submitter, cube_id=cube_id)
    assert record["schema_version"] == JOURNAL_SCHEMA_VERSION
    target_path = _write_record(record, out_dir)
    typer.echo(f"wrote: {target_path}")
    typer.echo(
        f"  {record['benchmark_name']} v{record['benchmark_version']} · "
        f"{record['agent']['config_type']} / {record['agent']['llm_model']} · "
        f"score={record['results']['avg_score']:.3f}"
    )

    if not auto_pr:
        typer.echo("")
        typer.echo("To submit by hand:")
        typer.echo("  1. Fork https://github.com/The-AI-Alliance/cube-registry and clone it locally.")
        typer.echo(f"  2. Copy {target_path} into <clone>/results/{record['benchmark_name']}/")
        typer.echo("  3. Commit (with -s) and open a PR — CI will validate + auto-merge.")
        typer.echo("Or re-run with --auto-pr.")
        return

    branch = f"results/{record['benchmark_name']}/{sanitize_filename(record['evaluation_id'])}"
    pr_url = _open_pr(target_path, record, branch)
    typer.echo(f"PR opened: {pr_url}")


if __name__ == "__main__":
    typer.run(main)
