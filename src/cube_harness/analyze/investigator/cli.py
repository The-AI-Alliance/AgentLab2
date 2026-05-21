"""CLI entry point for the trajectory investigator (`ch-investigate`).

Two subcommands:
- `ch-investigate run <experiment_dir> [options]` — batch investigator episodes.
- `ch-investigate init-context <experiment_dir>` — (re)generate `investigation_context.md`
  via the benchmark-context sub-agent.

`ch-investigate <experiment_dir>` (no subcommand) is also accepted — it dispatches
to `run` so existing scripts keep working.

Defaults are aggressive: synthesis always runs, journaling always happens.
Programmatic callers (Auto-CUBE, recipe scripts) can disable either by
constructing `InvestigationConfig` directly with `synthesis_model=""` or
`journal_dir=Path(os.devnull)`. The CLI is for the common case.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.analyze.investigator.agent_driver import AgentDriver, ClaudeCodeSDKDriver, TerminalClaudeDriver
from cube_harness.analyze.investigator.benchmark_context_agent import generate_context_file
from cube_harness.analyze.investigator.core import InvestigationConfig, investigate_experiment
from cube_harness.analyze.investigator.recipe import InvestigatorRecipe
from cube_harness.analyze.investigator.use_cases import RECIPE_CATALOG
from cube_harness.eval_log import BaseFindings, InvestigationMetadata

logger = logging.getLogger(__name__)


def _print_summary_table(results: dict[str, tuple[BaseFindings, InvestigationMetadata]]) -> None:
    if not results:
        print("(no episodes investigated)")
        return
    print(f"\n{'trajectory_id':50s}  {'outcome':25s}  {'blame':24s}  {'conf':4s}  {'h_conf':6s}")
    print(f"{'-' * 50}  {'-' * 25}  {'-' * 24}  {'-' * 4}  {'-' * 6}")
    for tid, (o, _) in results.items():
        print(
            f"{tid[:50]:50s}  {o.outcome.value:25s}  {o.primary_blame.value:24s}  "
            f"{o.primary_blame_confidence:4d}  {o.hypothesis_confidence:6d}"
        )

    outcomes = Counter(o.outcome.value for o, _ in results.values())
    blames = Counter(o.primary_blame.value for o, _ in results.values())
    total_cost = sum(m.cost_usd for _, m in results.values())
    n = len(results)
    print(f"\nInvestigated {n} episodes  |  total cost: ${total_cost:.2f}  |  avg: ${total_cost / n:.3f}/ep")
    print(f"Outcomes:       {dict(outcomes)}")
    print(f"Primary blame:  {dict(blames)}")


def _resolve_recipe(name: str) -> InvestigatorRecipe:
    """Look up a recipe by name from the catalog."""
    if name not in RECIPE_CATALOG:
        known = ", ".join(sorted(RECIPE_CATALOG))
        raise typer.BadParameter(f"unknown recipe {name!r} — known: {known}")
    return RECIPE_CATALOG[name]


def _make_driver(name: str) -> AgentDriver:
    """Build a concrete driver from a CLI flag value."""
    if name == "claude-code-sdk":
        return ClaudeCodeSDKDriver()
    if name == "claude-terminal":
        return TerminalClaudeDriver()
    raise typer.BadParameter(f"unknown driver {name!r} — choose one of: claude-code-sdk, claude-terminal")


app = typer.Typer(
    add_completion=False,
    help="Post-hoc trajectory investigator for cube-harness experiments.",
    no_args_is_help=True,
)


@app.command("run")
def run_cmd(
    path: Annotated[Path, typer.Argument(help="Experiment directory or single episode directory.")],
    recipe: Annotated[str, typer.Option(help="Recipe name (see use_cases/ subdirectories).")] = "general_blame",
    driver: Annotated[
        str, typer.Option(help="Coding-agent driver: 'claude-code-sdk' (API key) or 'claude-terminal' (subscription).")
    ] = "claude-code-sdk",
    investigator_model: Annotated[
        str | None,
        typer.Option("--investigator-model", help="Override the recipe's per-episode investigator model."),
    ] = None,
    synthesis_model: Annotated[
        str, typer.Option("--synthesis-model", help="Model for the post-batch meta-analysis. Opus by default.")
    ] = "claude-opus-4-7",
    audit: Annotated[bool, typer.Option(help="Run the audit pass after each finding (writes audit.json).")] = False,
    n_seeds: Annotated[int, typer.Option("--n-seeds", help="Investigate each episode this many times.")] = 1,
    ids: Annotated[
        str | None,
        typer.Option(help="Comma-separated trajectory IDs (or task IDs). Default: all eligible episodes."),
    ] = None,
    failures_only: Annotated[
        bool, typer.Option("--failures-only", help="Restrict to episodes where is_correct=False.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-investigate episodes that already have findings.")
    ] = False,
    n_parallel: Annotated[int, typer.Option("--n-parallel", help="Concurrent investigator sub-processes.")] = 1,
    trace_mode: Annotated[
        str, typer.Option("--trace", help="Investigator action trace level: actions, full, off.")
    ] = "actions",
    journal_dir: Annotated[
        Path,
        typer.Option(
            "--journal-dir",
            help="Mirror meta_analysis.{json,md} into <journal-dir>/<experiment>/.",
        ),
    ] = Path("~/cube_auto_cube_journal").expanduser(),
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Stream tool calls + text to stderr.")] = False,
) -> None:
    """Batch-investigate episodes in an experiment directory.

    By default investigates every eligible (uninvestigated) episode, runs the post-batch
    meta-analysis, and mirrors the synthesis into ~/cube_auto_cube_journal/.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if trace_mode not in ("actions", "full", "off"):
        raise typer.BadParameter(f"--trace must be one of: actions, full, off (got {trace_mode!r})")

    chosen_recipe = _resolve_recipe(recipe)
    if investigator_model is not None and investigator_model != chosen_recipe.model:
        chosen_recipe = chosen_recipe.model_copy(update={"model": investigator_model})

    config = InvestigationConfig(
        recipe=chosen_recipe,
        driver=_make_driver(driver),
        audit=audit,
        n_seeds=n_seeds,
        ids=[s.strip() for s in ids.split(",")] if ids else None,
        failures_only=failures_only,
        overwrite=overwrite,
        n_parallel=n_parallel,
        verbose=verbose,
        trace_mode=trace_mode,  # type: ignore[arg-type]
        synthesis_model=synthesis_model,
        journal_dir=journal_dir,
    )
    results = investigate_experiment(path, config)
    _print_summary_table(results)


@app.command("init-context")
def init_context_cmd(
    experiment_dir: Annotated[Path, typer.Argument(help="Experiment directory.")],
    driver: Annotated[str, typer.Option(help="Driver to invoke the context agent through.")] = "claude-code-sdk",
    model: Annotated[str, typer.Option(help="Model name for the context agent.")] = "claude-opus-4-7",
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Invoke the benchmark-context sub-agent to (re)generate investigation_context.md."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    chosen_driver = _make_driver(driver)
    path = asyncio.run(
        generate_context_file(experiment_dir, driver=chosen_driver, model=model, verbose=verbose),
    )
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point.

    `ch-investigate <experiment_dir>` (no subcommand keyword) dispatches to `run`
    by prepending it — keeps the simplest invocation working.
    """
    if argv is None:
        argv = sys.argv[1:]
    subcommand_names = {"run", "init-context", "--help", "-h"}
    if argv and argv[0] not in subcommand_names:
        argv = ["run", *argv]
    try:
        app(args=argv, standalone_mode=False)
    except typer.Exit as e:
        return int(e.exit_code or 0)
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
