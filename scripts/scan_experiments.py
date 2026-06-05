#!/usr/bin/env python3
"""scan_experiments.py — survey ``~/cube_harness_results/`` for journal-
eligible experiments.

For each direct child directory that contains an ``experiment_record.json``,
the script classifies it into one of:

  • already_submitted  — ``submissions.json`` already has a 'journal' decision
  • broken             — cannot produce a meaningful submission (silent
                          rejection persisted into ``submissions.json``)
  • unfinished         — still running, or tasks missing a status file; state
                          may change, re-scan later
  • subset_review      — passed integrity but not a "complete named subset";
                          requires ``--yes`` to submit
  • submittable        — clean run of a full benchmark or a complete named
                          subset; queue for submission

Output: a markdown table + a one-line summary. With ``--submit`` it
invokes ``submit_to_journal.py --auto-pr`` on each submittable run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.reproducibility import submissions
from cube_harness.reproducibility.scan import (
    DEFAULT_SYSTEM_ERROR_THRESHOLD,
    ScanCategory,
    ScanResult,
    archive_experiment_dir,
    walk,
)

DEFAULT_RESULTS_DIR = Path.home() / "cube_harness_results"

_CATEGORY_ICON = {
    ScanCategory.already_submitted: "✓",
    ScanCategory.submittable: "→",
    ScanCategory.subset_review: "?",
    ScanCategory.unfinished: "…",
    ScanCategory.broken: "✗",
}


def _format_table(results: list[ScanResult]) -> str:
    """Compact markdown summary, one row per experiment."""
    if not results:
        return "(no experiment directories found)"
    rows = [
        ["", "experiment", "category", "subset", "n_tasks", "errors", "diagnosis"],
        ["---"] * 7,
    ]
    for r in results:
        subset = r.benchmark_subset_name
        if r.benchmark_subset_filter:
            subset += f" / {r.benchmark_subset_filter}"
        if r.has_explicit_task_list:
            subset += " [list]"
        n_tasks_cell = ""
        if r.n_tasks is not None:
            n_tasks_cell = f"{r.n_terminal}/{r.n_tasks}"
        err_cell = f"{r.n_system_error}" if r.n_terminal else ""
        diagnosis = "; ".join(r.reasons) if r.reasons else "—"
        rows.append(
            [
                _CATEGORY_ICON[r.category],
                r.experiment_dir.name,
                r.category.value,
                subset or "—",
                n_tasks_cell,
                err_cell,
                diagnosis,
            ]
        )
    # Pad columns to align.
    widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
    return "\n".join("| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |" for row in rows)


def _persist_broken_decisions(results: list[ScanResult]) -> int:
    """Write submissions.json rejection entries for every broken experiment.

    Returns the number of dirs newly stamped (those that already had a
    decision are left alone)."""
    n = 0
    for r in results:
        if r.category is not ScanCategory.broken:
            continue
        if submissions.has_decision(r.experiment_dir, "journal"):
            continue
        reason = r.reasons[0] if r.reasons else "unclassified"
        submissions.record_rejected(r.experiment_dir, "journal", reason=f"broken: {reason}")
        n += 1
    return n


def _invoke_submitter(
    submit_to_journal: Path,
    experiment_dir: Path,
    *,
    auto_pr: bool,
) -> int:
    """Spawn ``submit_to_journal.py`` for one experiment dir. Returns its rc.

    The scan script's ``--yes`` controls eligibility upstream (whether
    ``subset_review`` rows make it into the ``eligible`` list); the per-row
    submission itself doesn't need to re-ask. submit_to_journal trusts the
    classification that produced its argv and skips its own framing wall
    via ``--i-understand-this-is-not-a-leaderboard``.
    """
    cmd = [
        sys.executable,
        str(submit_to_journal),
        str(experiment_dir),
        "--i-understand-this-is-not-a-leaderboard",
    ]
    if auto_pr:
        cmd.append("--auto-pr")
    typer.echo(f"  invoking: {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def main(
    root: Annotated[
        Path,
        typer.Argument(help="Experiments root (default: ~/cube_harness_results)."),
    ] = DEFAULT_RESULTS_DIR,
    system_error_threshold: Annotated[
        float,
        typer.Option(
            "--system-error-threshold",
            help="Fraction of episodes that must hit FAILED/STALE/INVALID_CONFIG "
            "before an experiment is marked broken. Default 0.10 (10%).",
            min=0.0,
            max=1.0,
        ),
    ] = DEFAULT_SYSTEM_ERROR_THRESHOLD,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Invoke submit_to_journal.py for each submittable experiment "
            "(default: dry-run, just print the table).",
        ),
    ] = False,
    auto_pr: Annotated[
        bool,
        typer.Option(
            "--auto-pr",
            help="When invoking submit_to_journal, pass --auto-pr through "
            "(forks cube-registry and opens the PR via gh).",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Also submit experiments classified as subset_review (debug "
            "runs / hand-picked task lists). Default: skip them.",
        ),
    ] = False,
    persist_broken: Annotated[
        bool,
        typer.Option(
            "--persist-broken/--no-persist-broken",
            help="Write rejection entries to submissions.json for broken "
            "experiments so future scans skip them. On by default.",
        ),
    ] = True,
    sweep_stale: Annotated[
        bool,
        typer.Option(
            "--sweep-stale/--no-sweep-stale",
            help="Before classifying, reuse cube_harness.experiment.sweep_stale_statuses "
            "to mark dead RUNNING/QUEUED episodes as STALE based on heartbeat age. "
            "Catches experiments where Ray crashed mid-run. On by default — pass "
            "--no-sweep-stale for a strictly read-only scan.",
        ),
    ] = True,
    archive_broken: Annotated[
        bool,
        typer.Option(
            "--archive-broken",
            help="After scanning, rename every broken experiment dir to "
            "<name>.archived_<ts> so future scans skip them and the results "
            "root stays uncluttered. Uses the same ARCHIVED_MARKER convention "
            "as FileStorage's per-episode archive.",
        ),
    ] = False,
    archive_debug: Annotated[
        bool,
        typer.Option(
            "--archive-debug",
            help="Also archive experiments classified as subset_review (debug "
            "runs / hand-picked task lists). Implies --archive-broken=True.",
        ),
    ] = False,
) -> None:
    """Survey *root* and either report or hand off to the submitter."""
    results = walk(
        root,
        system_error_threshold=system_error_threshold,
        sweep_stale=sweep_stale,
    )
    typer.echo(_format_table(results))

    # Summary counts
    counts: dict[ScanCategory, int] = {c: 0 for c in ScanCategory}
    for r in results:
        counts[r.category] += 1
    typer.echo("")
    typer.echo(" · ".join(f"{c.value}: {counts[c]}" for c in ScanCategory if counts[c] > 0) or "(nothing classifiable)")

    if persist_broken:
        n_stamped = _persist_broken_decisions(results)
        if n_stamped:
            typer.echo(f"persisted {n_stamped} broken decision(s) into submissions.json")

    def _archive_pass() -> None:
        """Rename broken (and optionally subset_review) dirs to .archived_<ts>."""
        targets = [r for r in results if r.category is ScanCategory.broken]
        if archive_debug:
            targets.extend(r for r in results if r.category is ScanCategory.subset_review)
        for r in targets:
            try:
                new_path = archive_experiment_dir(r.experiment_dir)
                typer.echo(f"archived: {r.experiment_dir.name} -> {new_path.name}")
            except Exception as e:
                typer.echo(f"  ✗ archive failed for {r.experiment_dir.name}: {e}", err=True)

    if not submit:
        # Archive only if we're not also submitting from this invocation —
        # otherwise we'd rename dirs that submit_to_journal still has to
        # touch. Re-run with --submit to actually push.
        if archive_broken or archive_debug:
            _archive_pass()
        if counts[ScanCategory.submittable] or (yes and counts[ScanCategory.subset_review]):
            typer.echo("")
            typer.echo("Re-run with --submit to hand off to submit_to_journal.py.")
        return

    submit_to_journal = Path(__file__).with_name("submit_to_journal.py")
    if not submit_to_journal.exists():
        typer.echo(f"submit_to_journal.py not found at {submit_to_journal}", err=True)
        raise typer.Exit(code=1)

    eligible = [r for r in results if r.category is ScanCategory.submittable]
    if yes:
        eligible.extend(r for r in results if r.category is ScanCategory.subset_review)

    if not eligible:
        typer.echo("Nothing eligible to submit.")
        return

    typer.echo("")
    typer.echo(f"Submitting {len(eligible)} experiment(s) …")
    n_failed = 0
    for r in eligible:
        rc = _invoke_submitter(submit_to_journal, r.experiment_dir, auto_pr=auto_pr)
        if rc != 0:
            n_failed += 1
            typer.echo(f"  ✗ {r.experiment_dir.name} (exit {rc})", err=True)
    # Now safe to archive — submit_to_journal already read everything it needed
    # from the broken/debug dirs (only broken get persisted-rejected; the
    # submittable ones were eligible and won't be archived here).
    if archive_broken or archive_debug:
        _archive_pass()
    if n_failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
