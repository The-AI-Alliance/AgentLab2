"""Public investigator API: investigate_episode, investigate_experiment, and supporting helpers."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from cube.core import TypedBaseModel
from pydantic import ConfigDict, Field, ValidationError

from cube_harness.analyze.cross_experiment.cross_investigation_agreement import (
    write_cross_investigation_agreement,
)
from cube_harness.analyze.investigator.agent_driver import (
    AgentDriver,
    ClaudeCodeSDKDriver,
    DriverResult,
    ToolAction,
    TraceMode,
)
from cube_harness.analyze.investigator.audit import AUDIT_FILENAME, run_audit_pass, write_audit
from cube_harness.analyze.investigator.benchmark_context_agent import generate_context_file
from cube_harness.analyze.investigator.context import (
    _load_experiment_view,
    find_default_context_file,
    validate_context_file,
)
from cube_harness.analyze.investigator.episode_discovery import (
    EpisodeRef,
    discover_episodes,
    load_episode_record,
    select_episodes,
)
from cube_harness.analyze.investigator.meta_analysis import (
    copy_to_journal,
    run_meta_analysis,
    write_meta_analysis,
)
from cube_harness.analyze.investigator.parse import extract_json_block
from cube_harness.analyze.investigator.recipe import InvestigatorRecipe, get_default_recipe
from cube_harness.analyze.investigator.selectors import Selector
from cube_harness.analyze.investigator.transcript import extract_transcript
from cube_harness.core import Trajectory
from cube_harness.eval_log import (
    FINDINGS_SCHEMA_VERSION,
    BaseFindings,
    InvestigationMetadata,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SAMPLE_FRACTION = 0.10
EXPERIMENT_INVESTIGATION_SUMMARY_FILENAME = "experiment_investigation_summary.json"
EXPERIMENT_INVESTIGATION_REPORT_FILENAME = "experiment_investigation_report.csv"
EXPERIMENT_INVESTIGATION_REPORT_JSON_FILENAME = "experiment_investigation_report.json"


class InvestigationConfig(TypedBaseModel):
    """Everything `investigate_experiment` needs except the experiment directory.

    Behaviour: `recipe`, `driver`, `selector`, `audit`, `n_seeds`.
    Selection: `ids`, `failures_only`, `overwrite`. The default is "all
    eligible (uninvestigated) episodes" — random sampling is intentionally NOT a
    field; if you need a subset, pass `ids` explicitly (Auto-CUBE
    builds the list in Python based on whatever logic it wants).
    Execution: `n_parallel`, `verbose`, `trace_mode`.
    Post-batch synthesis is always on; `synthesis_model` and `journal_dir`
    are tunable. To truly skip synthesis, programmatic callers can override
    `investigate_experiment` directly — the CLI does not expose a skip flag.

    `arbitrary_types_allowed=True` is required because `driver` and
    `selector` are Protocols, not Pydantic models. The recipe holds a
    Pydantic `output_model` type via the same mechanism.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Behaviour
    recipe: InvestigatorRecipe = Field(default_factory=get_default_recipe)
    driver: AgentDriver = Field(default_factory=ClaudeCodeSDKDriver)
    selector: Selector | None = None
    audit: bool = False
    n_seeds: int = 1
    # Hard wall-clock cap per episode (context-agent + primary + audit). A
    # hung driver subprocess raises nothing, so without this the serial
    # batch blocks forever (observed: 8h+). On timeout the episode is
    # logged + skipped and the batch continues. See auto-fix(409).
    episode_timeout_s: float = 1800.0

    # Selection — explicit only. No sampling / no seed; Auto-CUBE
    # builds `ids` lists programmatically when it needs a subset.
    ids: list[str] | None = None
    failures_only: bool = False
    overwrite: bool = False

    # Execution
    n_parallel: int = 1
    verbose: bool = False
    trace_mode: TraceMode = "actions"

    # Post-batch synthesis (always on; `synthesis_model=""` skips, but the
    # CLI does not expose that — it is a programmatic-only escape hatch).
    synthesis_model: str = "claude-opus-4-7"

    # Journal mirror — `meta_analysis.{json,md}` is copied into
    # `<journal_dir>/<experiment_basename>/`. Default is the conventional
    # machine-local journal dir; override or point at a tempdir to redirect.
    journal_dir: Path = Field(default_factory=lambda: Path("~/cube_auto_cube_journal").expanduser())


def _load_trajectory_meta(path: Path) -> Trajectory | None:
    """Load episode.metadata.json as a Trajectory. The `steps` field will be empty
    (steps live in `steps/*.msgpack.zst`); reward_info/metadata/summary_stats are populated."""
    if not path.exists():
        return None
    try:
        return Trajectory.model_validate_json(path.read_text())
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("Could not parse %s as Trajectory: %s", path, e)
        return None


def _validate_invariants(obj: BaseFindings) -> None:
    """Enforce the V1 invariants from the spec (post-parse, pre-write)."""
    if obj.primary_blame.value != "none" and not obj.evidence:
        raise ValueError("evidence must be non-empty when primary_blame != 'none'")
    if obj.primary_blame in obj.other_blames:
        raise ValueError("other_blames must not repeat primary_blame")
    if obj.outcome.value in ("success", "success_lucky") and obj.primary_blame.value != "none":
        # Soft-correct: tighten to spec rather than fail.
        logger.warning(
            "Investigator returned outcome=%s with primary_blame=%s; spec requires 'none'. Coercing.",
            obj.outcome.value,
            obj.primary_blame.value,
        )
        obj.primary_blame = obj.primary_blame.__class__("none")


def _resolve_recipe(recipe: InvestigatorRecipe | None, model_override: str | None) -> InvestigatorRecipe:
    """Pick the recipe for this run and apply a (deprecated) `model=` override.

    `model=` is preserved as a thin shim — old callers passed it directly; the
    new path puts the model on the recipe. We honour it here with a deprecation
    warning rather than silently breaking existing scripts.
    """
    base = recipe or get_default_recipe()
    if model_override is not None and model_override != base.model:
        warnings.warn(
            "Passing `model=` to investigate_episode/investigate_experiment is deprecated; "
            "set it on the recipe instead. Synthesising a recipe override.",
            DeprecationWarning,
            stacklevel=3,
        )
        return base.model_copy(update={"model": model_override})
    return base


def _resolve_driver(driver: AgentDriver | None) -> AgentDriver:
    """Default to `ClaudeCodeSDKDriver()` when no driver is given."""
    return driver or ClaudeCodeSDKDriver()


def _build_user_prompt(
    *,
    recipe: InvestigatorRecipe,
    trajectory_id: str,
    task_id: str,
    reward: float | None,
    total_steps: int | None,
    agent_name: str,
    benchmark_name: str,
    transcript_dir: Path,
    episode_metadata_path: Path,
    episode_config_path: Path,
    task_description: str,
    source_paths: dict[str, Path],
    related_paths: list[Path],
) -> str:
    """Render the recipe's user-prompt template with per-episode fields."""
    src_block = (
        "\n".join(f"  {name}: {p}" for name, p in source_paths.items())
        if source_paths
        else "  (none resolved — investigator from transcript only)"
    )
    if related_paths:
        src_block += "\n  related_episodes:\n" + "\n".join(f"    - {p}" for p in related_paths)

    return recipe.user_prompt_template.format(
        trajectory_id=trajectory_id,
        task_id=task_id,
        reward=reward if reward is not None else "unknown",
        total_steps=total_steps if total_steps is not None else "unknown",
        agent_name=agent_name,
        benchmark_name=benchmark_name,
        transcript_dir=transcript_dir,
        episode_metadata_path=episode_metadata_path,
        episode_config_path=episode_config_path,
        task_description=task_description or "(none)",
        source_paths_block=src_block,
    )


async def _ensure_context_file(experiment_dir: Path, driver: AgentDriver) -> Path:
    """Find or generate `investigation_context.md` and verify every listed path exists."""
    try:
        path = find_default_context_file(experiment_dir)
    except FileNotFoundError:
        logger.info("investigation_context.md missing under %s — invoking benchmark-context-agent", experiment_dir)
        path = await generate_context_file(experiment_dir, driver=driver)
    return path


async def _investigate_episode_impl(
    episode_dir: Path,
    experiment_dir: Path,
    *,
    recipe: InvestigatorRecipe,
    driver: AgentDriver,
    selector: Selector | None = None,
    audit: bool = False,
    verbose: bool = False,
    trace_mode: TraceMode = "actions",
    all_refs: list[EpisodeRef] | None = None,
) -> tuple[BaseFindings, InvestigationMetadata, list[ToolAction], DriverResult, float]:
    """Async core shared by investigate_episode (single) and investigate_experiment (parallel)."""
    transcript_dir = episode_dir / "_investigation_transcript"
    extract_transcript(episode_dir, transcript_dir)

    metadata_path = episode_dir / "episode.metadata.json"
    config_path = episode_dir / "episode_config.json"
    experiment_config_path = experiment_dir / "experiment_config.json"

    trajectory = _load_trajectory_meta(metadata_path)
    view = _load_experiment_view(experiment_config_path)

    if trajectory is not None:
        task_id = trajectory.metadata.get("task_id") or trajectory.id
        reward = trajectory.reward_info.get("reward")
        total_steps = (trajectory.summary_stats or {}).get("n_agent_steps")
        task_description = trajectory.metadata.get("task_description", "")
    else:
        task_id, reward, total_steps, task_description = "unknown", None, None, ""

    context_path = await _ensure_context_file(experiment_dir, driver)
    source_paths = validate_context_file(context_path)

    related_paths: list[Path] = []
    if selector is not None:
        if all_refs is None:
            all_refs = discover_episodes(experiment_dir)
        # Find the matching ref (by directory) so the selector has the record.
        main_ref = next((r for r in all_refs if r.episode_dir == episode_dir), None)
        if main_ref is not None:
            related_paths = selector.select(
                main_episode=main_ref,
                experiment_dir=experiment_dir,
                all_refs=all_refs,
            )

    user_prompt = _build_user_prompt(
        recipe=recipe,
        trajectory_id=episode_dir.name,
        task_id=task_id,
        reward=reward,
        total_steps=total_steps,
        agent_name=view.agent_dotted,
        benchmark_name=view.benchmark_dotted,
        transcript_dir=transcript_dir,
        episode_metadata_path=metadata_path,
        episode_config_path=config_path,
        task_description=task_description,
        source_paths=source_paths,
        related_paths=related_paths,
    )

    additional_dirs = list(source_paths.values()) + [transcript_dir] + related_paths
    logger.info(
        "Investigating %s (reward=%s, steps=%s) with recipe=%s driver=%s model=%s",
        episode_dir.name,
        reward,
        total_steps,
        recipe.name,
        driver.name,
        recipe.model,
    )

    result = await driver.run(
        system_prompt=recipe.system_prompt,
        user_prompt=user_prompt,
        cwd=episode_dir,
        additional_dirs=additional_dirs,
        model=recipe.model,
        allowed_tools=recipe.allowed_tools,
        permission_mode=recipe.permission_mode,
        verbose=verbose,
        trace_mode=trace_mode,
    )

    obj = extract_json_block(result.output_text)
    findings = recipe.output_model.model_validate(obj)
    _validate_invariants(findings)

    investigation_metadata = InvestigationMetadata(
        model=recipe.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        duration_s=result.duration_s,
        timestamp=time.time(),
        findings_schema_version=FINDINGS_SCHEMA_VERSION,
    )

    audit_cost = 0.0
    if audit or recipe.audit:
        try:
            audit_obj, audit_cost = await run_audit_pass(
                recipe=recipe,
                driver=driver,
                findings_json=findings.model_dump_json(),
                investigation_trace_path=episode_dir / "investigation_trace.json",
                session_id=result.session_id,
                cwd=episode_dir,
                additional_dirs=additional_dirs,
                verbose=verbose,
            )
            write_audit(episode_dir, audit_obj)
        except Exception as e:
            logger.warning("Audit pass failed for %s: %s", episode_dir.name, e)

    return findings, investigation_metadata, result.actions, result, audit_cost


def investigate_episode(
    episode_dir: Path,
    *,
    experiment_dir: Path | None = None,
    recipe: InvestigatorRecipe | None = None,
    driver: AgentDriver | None = None,
    selector: Selector | None = None,
    audit: bool = False,
    verbose: bool = False,
    trace_mode: TraceMode = "actions",
    model: str | None = None,
) -> tuple[BaseFindings, InvestigationMetadata]:
    """Run a post-hoc investigator on a single episode trajectory directory.

    `experiment_dir` is needed only to locate `experiment_config.json`; if omitted,
    we look one level up from `episode_dir`.

    `model=` is deprecated — set it on the recipe; it triggers a DeprecationWarning
    and synthesises a recipe override for one release window.
    """
    episode_dir = Path(episode_dir).resolve()
    if experiment_dir is None:
        experiment_dir = episode_dir.parent.parent
    chosen_recipe = _resolve_recipe(recipe, model)
    chosen_driver = _resolve_driver(driver)
    findings, investigation_metadata, _, _, _ = asyncio.run(
        _investigate_episode_impl(
            episode_dir,
            Path(experiment_dir).resolve(),
            recipe=chosen_recipe,
            driver=chosen_driver,
            selector=selector,
            audit=audit,
            verbose=verbose,
            trace_mode=trace_mode,
        )
    )
    return findings, investigation_metadata


def persist_findings(
    ref: EpisodeRef,
    findings: BaseFindings,
    investigation_metadata: InvestigationMetadata,
    actions: list[ToolAction] | None = None,
    trace_mode: TraceMode = "actions",
) -> None:
    """Write `findings` and `investigation_metadata` into the episode_record.json.

    If the record file does not exist yet (older runs without atlas-eval-log enabled),
    write a sidecar `findings.json` so the result is not lost.

    When `actions` is non-empty, writes a `investigation_trace.json` sidecar alongside the
    episode record with the investigator's tool-call sequence.
    """
    if ref.record is not None:
        updated = ref.record.model_copy(update={"findings": findings, "investigation_metadata": investigation_metadata})
        ref.record_path.write_text(updated.model_dump_json(indent=2))
    else:
        sidecar = ref.episode_dir / "findings.json"
        sidecar.write_text(
            json.dumps(
                {
                    "findings": findings.model_dump(mode="json"),
                    "investigation_metadata": investigation_metadata.model_dump(mode="json"),
                },
                indent=2,
            )
        )
    if actions:
        (ref.episode_dir / "investigation_trace.json").write_text(
            json.dumps(
                {"trace_mode": trace_mode, "actions": [a.model_dump(mode="json") for a in actions]},
                indent=2,
            )
        )


def investigate_experiment(
    experiment_dir: Path,
    config: InvestigationConfig | None = None,
) -> dict[str, tuple[BaseFindings, InvestigationMetadata]]:
    """Batch investigator selected episodes in an experiment output directory.

    `config` bundles every behavioural / selection / execution knob — see
    `InvestigationConfig`. `config=None` uses defaults (general_blame recipe,
    SDK driver, no selector, no audit, no seeding, all uninvestigated episodes,
    sequential, sonnet).

    Writes per-episode results into `episode_record.json` (or a sidecar if
    missing) and aggregate stats into `experiment_investigation_summary.json`,
    `experiment_investigation_report.csv`, and `experiment_investigation_report.json`.
    When `config.n_seeds > 1`, also writes `cross_investigation_agreement.csv`.
    """
    experiment_dir = Path(experiment_dir).resolve()
    cfg = config or InvestigationConfig()

    refs = discover_episodes(experiment_dir)
    selected = select_episodes(
        refs,
        ids=cfg.ids,
        failures_only=cfg.failures_only,
        overwrite=cfg.overwrite,
    )

    if not selected:
        logger.info("No episodes selected to investigator in %s", experiment_dir)
        return {}

    n_parallel = cfg.n_parallel
    if n_parallel > cfg.driver.max_parallelism:
        logger.info(
            "Clamping n_parallel from %d to driver max_parallelism=%d (driver=%s)",
            n_parallel,
            cfg.driver.max_parallelism,
            cfg.driver.name,
        )
        n_parallel = cfg.driver.max_parallelism

    logger.info(
        "Investigating %d / %d episodes in %s (recipe=%s driver=%s n_parallel=%d n_seeds=%d audit=%s)",
        len(selected),
        len(refs),
        experiment_dir.name,
        cfg.recipe.name,
        cfg.driver.name,
        n_parallel,
        cfg.n_seeds,
        bool(cfg.audit or cfg.recipe.audit),
    )

    # The `seeded` runs map for cross-investigator agreement when n_seeds > 1.
    # Keyed by (trajectory_id, recipe_name) → list of (output, metadata).
    seeded_runs: dict[tuple[str, str], list[tuple[BaseFindings, InvestigationMetadata]]] = {}
    # Per-episode audit cost (only populated when audit enabled).
    audit_costs: dict[str, float] = {}

    results: dict[str, tuple[BaseFindings, InvestigationMetadata]] = asyncio.run(
        _investigate_experiment_async(
            selected,
            experiment_dir,
            recipe=cfg.recipe,
            driver=cfg.driver,
            selector=cfg.selector,
            audit=cfg.audit,
            n_parallel=max(1, n_parallel),
            n_seeds=max(1, cfg.n_seeds),
            verbose=cfg.verbose,
            trace_mode=cfg.trace_mode,
            episode_timeout_s=cfg.episode_timeout_s,
            seeded_runs_out=seeded_runs,
            audit_costs_out=audit_costs,
            all_refs=refs,
        )
    )

    if cfg.n_seeds > 1 and seeded_runs:
        write_cross_investigation_agreement(experiment_dir, investigations=seeded_runs)

    write_summary(
        experiment_dir,
        selected,
        results,
        recipe=cfg.recipe,
        driver=cfg.driver,
        audit_costs=audit_costs,
    )

    # Synthesis always runs unless explicitly disabled by passing an empty
    # `synthesis_model` (programmatic-only escape hatch — the CLI does not
    # expose it). Failures are logged but do not block primary results: every
    # per-episode finding and the flat aggregates are already on disk.
    if results and cfg.synthesis_model:
        try:
            analysis = asyncio.run(
                run_meta_analysis(
                    experiment_dir=experiment_dir,
                    experiment_id=experiment_dir.name,
                    recipe_name=cfg.recipe.name,
                    driver=cfg.driver,
                    results=results,
                    model=cfg.synthesis_model,
                    verbose=cfg.verbose,
                )
            )
            json_path, md_path = write_meta_analysis(experiment_dir, analysis)
            logger.info("Wrote meta-analysis to %s and %s", json_path.name, md_path.name)
            copy_to_journal(cfg.journal_dir, experiment_dir.name, json_path, md_path)
        except Exception as e:
            logger.exception("Meta-analysis failed for %s: %s", experiment_dir.name, e)

    return results


async def _investigate_experiment_async(
    selected: list[EpisodeRef],
    experiment_dir: Path,
    *,
    recipe: InvestigatorRecipe,
    driver: AgentDriver,
    selector: Selector | None,
    audit: bool,
    n_parallel: int,
    n_seeds: int,
    verbose: bool,
    trace_mode: TraceMode,
    episode_timeout_s: float,
    seeded_runs_out: dict[tuple[str, str], list[tuple[BaseFindings, InvestigationMetadata]]],
    audit_costs_out: dict[str, float],
    all_refs: list[EpisodeRef],
) -> dict[str, tuple[BaseFindings, InvestigationMetadata]]:
    """Run the investigator across `selected` × `n_seeds`, bounded by `n_parallel`."""
    semaphore = asyncio.Semaphore(n_parallel)
    primary_results: dict[str, tuple[BaseFindings, InvestigationMetadata]] = {}

    async def _one(ref: EpisodeRef, seed_index: int) -> None:
        async with semaphore:
            try:
                # auto-fix(409)↓
                findings, investigation_metadata, actions, _, audit_cost = await asyncio.wait_for(
                    _investigate_episode_impl(
                        ref.episode_dir,
                        experiment_dir,
                        recipe=recipe,
                        driver=driver,
                        selector=selector,
                        audit=audit,
                        verbose=verbose,
                        trace_mode=trace_mode,
                        all_refs=all_refs,
                    ),
                    timeout=episode_timeout_s,
                )
                # /auto-fix(409)
            except Exception as e:
                logger.exception("Investigator failed on %s (seed %d): %s", ref.trajectory_id, seed_index, e)
                return
            # Persist the *first* finding per episode (seed 0). Additional
            # seeds feed the cross-investigator agreement table; we don't overwrite the
            # primary record with a later seed.
            if seed_index == 0:
                ref.record = load_episode_record(ref.record_path)
                persist_findings(ref, findings, investigation_metadata, actions, trace_mode)
                primary_results[ref.trajectory_id] = (findings, investigation_metadata)
            audit_costs_out[ref.trajectory_id] = audit_costs_out.get(ref.trajectory_id, 0.0) + audit_cost
            seeded_runs_out.setdefault((ref.trajectory_id, recipe.name), []).append((findings, investigation_metadata))

    tasks = [_one(ref, seed_idx) for ref in selected for seed_idx in range(n_seeds)]
    await asyncio.gather(*tasks)
    return primary_results


def write_summary(
    experiment_dir: Path,
    selected: list[EpisodeRef],
    results: dict[str, tuple[BaseFindings, InvestigationMetadata]],
    *,
    recipe: InvestigatorRecipe | None = None,
    driver: AgentDriver | None = None,
    audit_costs: dict[str, float] | None = None,
    model: str | None = None,  # deprecated — accepted for ch-investigation-report compat
) -> None:
    if not results:
        return
    if recipe is None:
        # Legacy path (investigation_report.py): synthesise a minimal recipe so we can
        # still emit the summary's recipe/model fields. Uses the model carried
        # by an existing investigation_metadata, or the deprecated `model=` kwarg.
        effective_model = model or next(iter(results.values()))[1].model
        recipe = get_default_recipe().model_copy(update={"model": effective_model})
    if driver is None:
        driver = ClaudeCodeSDKDriver()
    audit_costs = audit_costs or {}
    outcomes = Counter(o.outcome.value for o, _ in results.values())
    blames = Counter(o.primary_blame.value for o, _ in results.values())
    total_cost = sum(m.cost_usd for _, m in results.values())
    total_audit_cost = sum(audit_costs.values())
    total_prompt = sum(m.prompt_tokens for _, m in results.values())
    total_completion = sum(m.completion_tokens for _, m in results.values())
    n = len(results)

    investigated_episodes = [
        {
            "trajectory_id": ref.trajectory_id,
            "episode_record": str(ref.record_path.relative_to(experiment_dir)),
        }
        for ref in selected
        if ref.trajectory_id in results
    ]

    summary: dict[str, Any] = {
        "n_investigated": n,
        "model": recipe.model,
        "recipe": recipe.name,
        "driver": driver.name,
        "findings_schema_version": FINDINGS_SCHEMA_VERSION,
        "timestamp": time.time(),
        "total_investigation_cost_usd": round(total_cost, 4),
        "avg_investigation_cost_usd": round(total_cost / n, 4) if n else 0.0,
        "total_investigation_prompt_tokens": total_prompt,
        "total_investigation_completion_tokens": total_completion,
        "outcomes": dict(outcomes),
        "primary_blame": dict(blames),
        "report_csv": EXPERIMENT_INVESTIGATION_REPORT_FILENAME,
        "report_json": EXPERIMENT_INVESTIGATION_REPORT_JSON_FILENAME,
        "investigated_episodes": investigated_episodes,
    }
    if total_audit_cost > 0.0:
        summary["total_audit_cost_usd"] = round(total_audit_cost, 4)
        summary["audit_report"] = AUDIT_FILENAME
    (experiment_dir / EXPERIMENT_INVESTIGATION_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    write_csv_report(experiment_dir, selected, results)
    write_json_report(experiment_dir, selected, results, recipe=recipe, driver=driver)


def write_csv_report(
    experiment_dir: Path,
    selected: list[EpisodeRef],
    results: dict[str, tuple[BaseFindings, InvestigationMetadata]],
) -> None:
    """Write one row per investigated episode for spreadsheet-friendly inspection.

    Excludes `analysis` and `evidence` — they're too verbose for LLM consumption.
    Read them from the per-episode `episode_record.json` when needed.
    """
    fields = [
        "trajectory_id",
        "episode_record",
        "reward",
        "n_steps",
        "outcome",
        "primary_blame",
        "primary_blame_confidence",
        "other_blames",
        "hypothesis_confidence",
        "summary",
        "hypothesis",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "duration_s",
    ]
    path = experiment_dir / EXPERIMENT_INVESTIGATION_REPORT_FILENAME
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ref in selected:
            if ref.trajectory_id not in results:
                continue
            o, m = results[ref.trajectory_id]
            reward = ref.record.score if ref.record is not None else None
            n_steps = ref.record.n_agent_steps if ref.record is not None else None
            w.writerow(
                {
                    "trajectory_id": ref.trajectory_id,
                    "episode_record": str(ref.record_path.relative_to(experiment_dir)),
                    "reward": reward,
                    "n_steps": n_steps,
                    "outcome": o.outcome.value,
                    "primary_blame": o.primary_blame.value,
                    "primary_blame_confidence": o.primary_blame_confidence,
                    "other_blames": ";".join(b.value for b in o.other_blames),
                    "hypothesis_confidence": o.hypothesis_confidence,
                    "summary": o.summary,
                    "hypothesis": o.hypothesis,
                    "cost_usd": round(m.cost_usd, 4),
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "duration_s": round(m.duration_s, 2),
                }
            )


def write_json_report(
    experiment_dir: Path,
    selected: list[EpisodeRef],
    results: dict[str, tuple[BaseFindings, InvestigationMetadata]],
    *,
    recipe: InvestigatorRecipe,
    driver: AgentDriver,
) -> None:
    """Write `experiment_investigation_report.json` — preserves per-recipe `OutputModel` shape.

    The CSV flattens to the base schema (loses recipe-specific extension fields);
    this artefact keeps the typed shape for downstream consumers."""
    path = experiment_dir / EXPERIMENT_INVESTIGATION_REPORT_JSON_FILENAME
    rows: list[dict[str, Any]] = []
    for ref in selected:
        if ref.trajectory_id not in results:
            continue
        o, m = results[ref.trajectory_id]
        rows.append(
            {
                "trajectory_id": ref.trajectory_id,
                "episode_record": str(ref.record_path.relative_to(experiment_dir)),
                "findings": o.model_dump(mode="json"),
                "investigation_metadata": m.model_dump(mode="json"),
            }
        )
    payload = {
        "recipe": recipe.name,
        "driver": driver.name,
        "model": recipe.model,
        "findings_schema_version": FINDINGS_SCHEMA_VERSION,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2))


# === auto-fix notes ===  (spec: openspec/specs/auto-fix/spec.md)
# auto-fix-note(409) {class=L1 issue=409 hash=PENDING ctx=macos-arm64/claude-sonnet-4-6/cube-harness@5ca4e565}
#   symptoms:  Autonomous overnight Round 4 (ch-investigate run --audit,
#              detached): the bundled claude SDK subprocess hung 8h12m on
#              episode 1 (count-dataset-tokens_ep1) under bypassPermissions
#              — a genuinely unbounded wait, not a perms prompt. The serial
#              batch never advanced. Trigger non-deterministic (long model
#              loop or SDK/network stall); the defect is the unbounded wait.
#   invariant: one episode's investigation cannot stall the batch — a
#              non-terminating driver/audit/context call is time-bounded
#              and becomes a logged failure so later episodes still run.
#   why:       agent_driver.py awaits the subprocess with no timeout;
#              core.py:563 already logs+continues on per-episode Exception
#              (the #2 crash-symmetry) but a hang raises nothing. Wrapping
#              the per-episode await in asyncio.wait_for makes the hang
#              *reach* that existing guard — minimal, right layer (the
#              batch runner owns "one episode can't stall the batch").
#   tested:    tests/test_investigator.py — a hanging fake driver is
#              bounded; the batch records nothing for it and completes the
#              rest (asserts the invariant, not a reproduction).
#   hash=PENDING: stamped by scripts/auto_fix_lint.py (Tier-1) on first run.
