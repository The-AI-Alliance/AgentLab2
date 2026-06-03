#!/usr/bin/env python3
"""submit_to_journal.py — convert an experiment dir into a cube-registry
community-journal record and (optionally) open a PR against cube-registry.

Usage:
  scripts/submit_to_journal.py <experiment_dir>
  scripts/submit_to_journal.py <experiment_dir> --auto-pr

Without ``--auto-pr`` the script writes the JSON to
``./journal-out/results/<cube>/<file>.json`` and prints the ``gh`` command you
can run to open the PR by hand. With ``--auto-pr``, it forks cube-registry
(via ``gh repo fork``, idempotent — gh reuses an existing fork), clones the
fork to ``/tmp``, creates a branch, copies the file in, commits, pushes to
the fork, and opens the PR against the upstream repo. Works for any GitHub
user, not just members of The-AI-Alliance org.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.reproducibility import (
    JOURNAL_SCHEMA_VERSION,
    build_journal_record,
    sanitize_filename,
    submissions,
)

CUBE_REGISTRY_REPO = "The-AI-Alliance/cube-registry"
CUBE_REGISTRY_URL = f"https://github.com/{CUBE_REGISTRY_REPO}.git"

# The framing wall — every code path that produces a journal record sees this
# before any actual work happens. Mirrors the amber callout rendered on each
# cube's registry page. Edit both at the same time if you change the wording.
_LEADERBOARD_DISCLAIMER = """\
╔══════════════════════════════════════════════════════════════════════════╗
║  REPRODUCIBILITY JOURNAL  —  NOT A LEADERBOARD                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  You're about to submit an evaluation result to the cube-registry        ║
║  community journal.                                                      ║
║                                                                          ║
║  This is for publishing REFERENCE values — to let others detect drift    ║
║  across infrastructures, cube versions, and package versions.            ║
║                                                                          ║
║  This is NOT a place to publish a new agent or fine-tune to "win" the    ║
║  benchmark. There is no ranking. Submissions are self-reported and       ║
║  not independently verified.                                             ║
║                                                                          ║
║  Want to showcase a new agent or model? Use ATLAS / EEE / your own       ║
║  benchmark page instead.                                                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def _confirm_not_a_leaderboard(*, acknowledged: bool) -> None:
    """Show the framing wall and require explicit confirmation.

    The framing matters more than the data — the registry journal's value
    depends on submitters understanding it's not a competition. *acknowledged*
    is true when the caller passed ``--i-understand-this-is-not-a-leaderboard``
    (skips the prompt; the flag name is itself the friction). On a non-TTY
    stdin (CI / pipes / nohup), refuse without that flag rather than auto-
    answering yes — accidental scripted submissions would defeat the point.
    """
    typer.echo(_LEADERBOARD_DISCLAIMER)
    if acknowledged:
        typer.echo("Acknowledgement flag set — proceeding.")
        return
    if not sys.stdin.isatty():
        typer.echo(
            "Non-interactive stdin and no --i-understand-this-is-not-a-leaderboard flag — refusing.",
            err=True,
        )
        raise typer.Exit(code=2)
    answer = typer.prompt("Continue with submission? [y/N]", default="N", show_default=False)
    if answer.strip().lower() not in {"y", "yes"}:
        typer.echo("Aborted — no record written.")
        raise typer.Exit(code=1)


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
    """Fork cube-registry, copy *record_path* into the fork, push, gh pr create.

    Uses ``gh repo fork --clone`` so this works for any GitHub user, not just
    members of The-AI-Alliance org. The fork is idempotent — gh detects an
    existing user fork and reuses it. After the clone, ``origin`` points at
    the user's fork and ``upstream`` at the canonical repo; we push to
    ``origin`` and open the PR against ``upstream``.
    """
    with tempfile.TemporaryDirectory(prefix="cube-registry-submit-") as tmp:
        clone_dir = Path(tmp) / "cube-registry"
        # `gh repo fork --clone --remote` forks (if needed) and clones into
        # the cwd's subdirectory. We feed it the parent and let gh pick the
        # directory name. Quiet output keeps the user-visible echo clean.
        subprocess.run(
            [
                "gh",
                "repo",
                "fork",
                CUBE_REGISTRY_REPO,
                "--clone",
                "--remote",
                "--clone-flags=--depth=1",
            ],
            cwd=tmp,
            check=True,
        )
        # gh names the directory after the repo basename ("cube-registry").
        # The created remotes are `origin` (the fork) and `upstream` (canonical).
        if not clone_dir.exists():
            raise FileNotFoundError(f"expected `gh repo fork` to create {clone_dir}; got {list(Path(tmp).iterdir())}")

        subprocess.run(["git", "-C", str(clone_dir), "checkout", "-b", branch], check=True)
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
        # Push to the *fork* (origin). The canonical repo is `upstream`.
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
        # `gh pr create` from inside the fork clone defaults to opening the PR
        # against the parent (upstream) repo. We pass --repo explicitly to
        # make the target unambiguous.
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
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-submit even when submissions.json already records a 'journal' decision "
            "(default: refuse to avoid duplicate submissions).",
        ),
    ] = False,
    i_understand_this_is_not_a_leaderboard: Annotated[
        bool,
        typer.Option(
            "--i-understand-this-is-not-a-leaderboard",
            help="Acknowledge the framing wall and skip the interactive prompt. "
            "The flag name is deliberately verbose — reading it IS the friction.",
        ),
    ] = False,
) -> None:
    """Build a cube-registry community-journal record and (optionally) open a PR."""
    # Idempotency check first — refuse early before showing the framing wall
    # or building the record. Use --force to override (e.g. for re-submission
    # after a correction).
    if not force and submissions.has_decision(experiment_dir, "journal"):
        prior = submissions.read(experiment_dir).get("journal", {})
        typer.echo(
            f"experiment_dir already has a 'journal' decision: {prior.get('status')} "
            f"({prior.get('reason') or prior.get('evaluation_id')}).",
            err=True,
        )
        typer.echo("Pass --force to override.", err=True)
        raise typer.Exit(code=2)

    _confirm_not_a_leaderboard(acknowledged=i_understand_this_is_not_a_leaderboard)
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
    # Stamp idempotency so a repeat invocation (or the scan script) sees that
    # this experiment has already been submitted to the journal.
    submissions.record_submitted(
        experiment_dir,
        "journal",
        evaluation_id=record["evaluation_id"],
        schema_version=record["schema_version"],
        submitted_by=submitter,
        pr_url=pr_url,
        local_path=str(target_path),
    )
    typer.echo(f"recorded in {experiment_dir / submissions.SUBMISSIONS_FILENAME}")


if __name__ == "__main__":
    typer.run(main)
