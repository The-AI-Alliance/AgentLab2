"""Post-batch synthesis: turn N per-episode investigations into one analysis.

Runs after `investigate_experiment` finishes and `write_summary` has produced the
flat aggregates. The synthesis call asks an LLM-driven sub-agent (Opus by
default) to read the per-episode investigator outputs and produce:

  - clustered failure patterns,
  - hypothesised root causes,
  - concrete intervention suggestions,
  - a human-readable markdown summary.

The structured payload is written as `<experiment_dir>/meta_analysis.json`;
the prose body is written as `<experiment_dir>/meta_analysis.md`.

When `journal_dir` is supplied, both files are also mirrored into
`<journal_dir>/<experiment_basename>/` so Auto-CUBE's outer loop can
collect a running history of what it observed across iterations.

Off-by-default for cheap repeat runs; the CLI's `--synthesize/--no-synthesize`
toggle controls it.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from cube.core import TypedBaseModel
from pydantic import Field, ValidationError

from cube_harness.analyze.investigator.audit import AUDIT_FILENAME
from cube_harness.analyze.investigator.parse import extract_json_block
from cube_harness.analyze.investigator.schema_prompt import model_to_json_example

if TYPE_CHECKING:
    from cube_harness.analyze.investigator.agent_driver import AgentDriver
    from cube_harness.eval_log import BaseFindings, InvestigationMetadata

logger = logging.getLogger(__name__)

META_ANALYSIS_FILENAME = "meta_analysis.json"
META_ANALYSIS_MARKDOWN_FILENAME = "meta_analysis.md"
META_ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_META_ANALYSIS_MODEL = "claude-opus-4-7"


class FailurePattern(TypedBaseModel):
    """A cluster of trajectories exhibiting the same failure mode.

    Field order is CoT-deliberate: describe what you see → cite the
    trajectories that show it → name the dominant blame → only then assign
    a short label. Models token-emit in this order, so naming-last forces
    the cluster to be understood before it is summarised.
    """

    description: str = Field(description="One or two sentences naming the pattern.")
    affected_trajectories: list[str] = Field(description="trajectory_ids the investigator attributed to this pattern.")
    dominant_blame: str = Field(description="BlameCategory value most often cited for these trajectories.")
    name: str = Field(description="Short kebab-case label (e.g. 'inspects-but-never-edits').")


class RootCauseHypothesis(TypedBaseModel):
    """A guess at *why* one or more patterns are happening.

    Field order is CoT-deliberate: reason first, link to patterns second,
    score confidence last (after the reasoning has played out).
    """

    description: str = Field(description="One to three sentences. Cite patterns and per-episode evidence.")
    pattern_names: list[str] = Field(description="`FailurePattern.name` values this hypothesis covers.")
    confidence: int = Field(ge=0, le=5, description="0=guess, 5=load-bearing evidence across multiple patterns.")


class Intervention(TypedBaseModel):
    """A concrete, tryable change.

    Field order is CoT-deliberate: identify the target → justify what
    change should be made and why → then state the concrete change →
    finally score how confident we are it will help.
    """

    target: str = Field(
        description="What to change. e.g. 'agent system prompt', 'scaffold loop', 'evaluation criterion'.",
    )
    rationale: str = Field(description="Why this addresses one or more root causes.")
    change: str = Field(description="Specific edit, prompt addition, or scaffold tweak.")
    confidence: int = Field(ge=0, le=5, description="0=long shot, 5=highly likely to help.")


class MetaAnalysis(TypedBaseModel):
    """Cross-episode synthesis of an experiment's investigations.

    `markdown_summary` carries the LLM's human-readable prose; the rest is
    structured for programmatic aggregation across experiments by
    Auto-CUBE's outer loop.
    """

    schema_version: int = META_ANALYSIS_SCHEMA_VERSION
    experiment_id: str
    recipe: str
    driver: str
    model: str
    timestamp: float
    n_episodes_investigated: int

    # Aggregates — duplicated from `experiment_investigation_summary.json` for
    # self-contained reading.
    outcome_distribution: dict[str, int] = Field(default_factory=dict)
    primary_blame_distribution: dict[str, int] = Field(default_factory=dict)
    success_rate: float = 0.0

    # Audit signal (present only when `--audit` was on): verdict counts plus
    # every non-`sound` episode with the alternative blames the auditor named.
    # Surfaces the self-critique at synthesis instead of leaving it write-only
    # in per-episode `audit.json` files. `None` when no episode was audited.
    audit_summary: dict | None = None

    # The LLM-authored synthesis.
    patterns: list[FailurePattern] = Field(default_factory=list)
    root_cause_hypotheses: list[RootCauseHypothesis] = Field(default_factory=list)
    suggested_interventions: list[Intervention] = Field(default_factory=list)
    markdown_summary: str = Field(description="Human-readable body; also written as meta_analysis.md.")

    notes: str | None = None
    cost_usd: float = 0.0
    duration_s: float = 0.0


# Fields the runtime supplies post-parse; the model must not emit these.
_RUNTIME_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "recipe",
        "driver",
        "model",
        "timestamp",
        "n_episodes_investigated",
        "outcome_distribution",
        "primary_blame_distribution",
        "success_rate",
        "audit_summary",
        "cost_usd",
        "duration_s",
    }
)


def _collect_audit_summary(experiment_dir: Path, trajectory_ids: list[str]) -> dict | None:
    """Aggregate per-episode `audit.json` into a synthesis-visible signal.

    Returns `None` when no episode has an audit (so `--no-audit` runs are
    unaffected). Otherwise: the verdict distribution plus, for every
    non-`sound` episode, the verdict and the alternative blames the auditor
    named — the part a synthesiser needs to discount a thin primary
    attribution instead of trusting it blindly.
    """
    verdicts: Counter[str] = Counter()
    flagged: list[dict] = []
    for tid in sorted(trajectory_ids):
        path = experiment_dir / "episodes" / tid / AUDIT_FILENAME
        if not path.exists():
            continue
        try:
            audit = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read audit %s: %s", path, e)
            continue
        verdict = audit.get("verdict", "unknown")
        verdicts[verdict] += 1
        if verdict != "sound":
            flagged.append(
                {
                    "trajectory_id": tid,
                    "verdict": verdict,
                    "alternative_blames": [b.get("blame") for b in audit.get("alternative_blames", [])],
                }
            )
    if not verdicts:
        return None
    return {"verdict_distribution": dict(verdicts), "flagged": flagged}


def _render_audit_block(audit_summary: dict | None) -> str:
    """Compact prompt text so the synthesiser weighs the auditor's dissent.

    Empty string when no audits ran — the prompt is unchanged for
    `--no-audit`, keeping that path byte-for-byte identical.
    """
    if not audit_summary:
        return ""
    lines = [
        "",
        "Auditor self-critique (independent re-read of each finding):",
        f"  verdict distribution: {audit_summary['verdict_distribution']}",
    ]
    flagged = audit_summary.get("flagged") or []
    if flagged:
        lines.append("  flagged (non-`sound`) episodes — treat their primary blame as a weak prior:")
        for f in flagged:
            alts = ", ".join(a for a in f["alternative_blames"] if a) or "(none named)"
            lines.append(f"    - {f['trajectory_id']}: {f['verdict']}; auditor's alternatives: {alts}")
    lines.append(
        "Weigh this: do not report a pattern's blame with high confidence when "
        "the auditor flagged the underlying episodes `questionable`/`wrong` "
        "unless you independently re-confirm it from the transcript."
    )
    return "\n".join(lines)


def _render_system_prompt() -> str:
    """Build the system prompt by deriving the JSON example from `MetaAnalysis`.

    The Pydantic class is the single spec; the prompt section is generated
    from `model_fields`, so renaming or reordering a field updates the prompt
    automatically. The runtime-provenance fields are stripped because the
    runtime fills them in after parsing.
    """
    example = model_to_json_example(MetaAnalysis, skip=_RUNTIME_PROVENANCE_FIELDS)
    return f"""You are a meta-analyst for a trajectory-investigator sweep.

The trajectory investigator has just produced one structured finding per episode
in a single experiment. Your job is to read those investigations, synthesise
patterns across them, hypothesise root causes, and propose concrete
interventions a human engineer can act on next.

Stay tightly grounded in what the per-episode investigations support. Do not
invent failure modes the investigations don't already describe. Every
trajectory_id in `affected_trajectories` must come from the list of investigated
trajectories in the user prompt.

You have read-only tools (Read / Glob / Grep / Bash). Use them to drill into
specific `findings.json` files when the summaries are too thin. Do not
modify any files. Your only output is the assistant message containing the
final JSON object.

The Pydantic-validated output must match this shape EXACTLY — do not rename,
omit, or add top-level fields. Each leaf is annotated `<type — description>`:

```json
{example}
```

Field order in your emitted JSON should follow the example above; the order
is deliberate (describe before naming, justify before changing, score after
reasoning).

The runtime fills in provenance and aggregate fields after parsing — you do
not need to emit `schema_version`, `experiment_id`, `recipe`, `driver`,
`model`, `timestamp`, `n_episodes_investigated`, `outcome_distribution`,
`primary_blame_distribution`, `success_rate`, `cost_usd`, or `duration_s`."""


META_ANALYSIS_SYSTEM_PROMPT = _render_system_prompt()


def _digest_investigations(
    results: dict[str, tuple["BaseFindings", "InvestigationMetadata"]],
) -> str:
    """Render the per-episode investigations as a compact text digest for the prompt.

    Each entry lists outcome, primary blame, confidence, summary, hypothesis,
    and the first two evidence quotes (enough signal without bloating the
    prompt for 100-episode sweeps).
    """
    lines: list[str] = []
    for tid, (findings, _meta) in sorted(results.items()):
        evidence_lines = []
        for ev in findings.evidence[:2]:
            evidence_lines.append(f"      step {ev.step}: {ev.quote[:140]!r}")
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "      (no evidence)"
        lines.append(
            "\n".join(
                [
                    f"- {tid}: outcome={findings.outcome.value} "
                    f"primary_blame={findings.primary_blame.value} "
                    f"conf={findings.primary_blame_confidence}",
                    f"    summary: {findings.summary}",
                    f"    hypothesis: {findings.hypothesis}",
                    f"    evidence:\n{evidence_block}",
                ]
            )
        )
    return "\n".join(lines)


def _build_user_prompt(
    *,
    experiment_dir: Path,
    experiment_id: str,
    n_investigated: int,
    outcome_distribution: dict[str, int],
    primary_blame_distribution: dict[str, int],
    success_rate: float,
    digest: str,
    audit_summary: dict | None = None,
) -> str:
    return f"""Experiment: {experiment_id}
Experiment directory (drill-in via Read/Glob if needed): {experiment_dir}

Episodes investigated: {n_investigated}
Success rate: {success_rate:.2%}

Outcome distribution: {dict(outcome_distribution)}
Primary blame distribution: {dict(primary_blame_distribution)}
{_render_audit_block(audit_summary)}

Per-episode digest:
{digest}

Files in the experiment directory you can drill into for richer signal
(read with Glob/Read; do not assume they all exist):
  - `episodes/<id>/findings.json` — the structured per-episode finding
  - `episodes/<id>/investigation_trace.json` — the investigator's tool-call log
  - `episodes/<id>/audit.json` — the investigator's self-critique (when audit was on)
  - `cross_investigation_agreement.csv` — modal blame + agreement fraction across
    seeds, when n_seeds > 1. Low agreement = the investigator disagreed with itself.
  - `investigation_context.md` — the source-paths the investigator had access to. If your
    patterns suggest the investigator missed evidence that lived in a missing path,
    flag the gap as a `tooling_gaps` entry on `audit.json`-derived findings.

Produce a single JSON object matching the `MetaAnalysis` schema. Wrap in
a ```json fence. Include a non-empty `markdown_summary`. Reference only
trajectory_ids that appear in the digest above."""


def _build_retry_prompt(prior_raw_output: str, validation_error: str) -> str:
    """Compose a follow-up prompt that asks the agent to fix invalid JSON.

    Used after a `ValidationError` on the first attempt. The retry is cheap
    relative to the initial pass: no transcript digest, no tool use — just
    the prior output and the validator's complaint.
    """
    return f"""Your prior reply failed Pydantic validation. Re-emit the JSON
object with the schema corrections noted below — do NOT redo the analysis,
just fix the structural mismatch. Reply with a single ```json fence and
nothing else.

Prior reply:
{prior_raw_output}

Validation error from Pydantic:
{validation_error}

Re-emit a valid JSON object matching the schema in the system prompt."""


async def run_meta_analysis(
    *,
    experiment_dir: Path,
    experiment_id: str,
    recipe_name: str,
    driver: "AgentDriver",
    results: dict[str, tuple["BaseFindings", "InvestigationMetadata"]],
    model: str = DEFAULT_META_ANALYSIS_MODEL,
    verbose: bool = False,
    max_retries: int = 1,
) -> MetaAnalysis:
    """Run the synthesis sub-agent and return a populated `MetaAnalysis`.

    On `ValidationError`, retries up to `max_retries` times with a cheap
    follow-up call that ships only the prior raw output + the validator's
    error message — no transcript digest, no tool use — and asks the model
    to fix the structural mismatch. Bounds retry cost while still self-
    correcting on the most common failure mode (field renames, missing
    fields, extra fields).

    Aggregates are computed here from `results` rather than re-reading the
    summary file — keeps the function pure(-ish) and lets tests pass a fake
    driver without on-disk dependencies.
    """
    outcomes = Counter(o.outcome.value for o, _ in results.values())
    blames = Counter(o.primary_blame.value for o, _ in results.values())
    n_investigated = len(results)
    n_success = outcomes.get("success", 0) + outcomes.get("success_lucky", 0)
    success_rate = n_success / n_investigated if n_investigated else 0.0
    audit_summary = _collect_audit_summary(experiment_dir, list(results))

    initial_prompt = _build_user_prompt(
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        n_investigated=n_investigated,
        outcome_distribution=dict(outcomes),
        primary_blame_distribution=dict(blames),
        success_rate=success_rate,
        digest=_digest_investigations(results),
        audit_summary=audit_summary,
    )

    total_cost = 0.0
    total_duration = 0.0
    prior_raw_output: str | None = None
    last_validation_error: str | None = None

    for attempt in range(max_retries + 1):
        user_prompt = (
            initial_prompt if attempt == 0 else _build_retry_prompt(prior_raw_output or "", last_validation_error or "")
        )
        result = await driver.run(
            system_prompt=META_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cwd=experiment_dir,
            additional_dirs=[],
            model=model,
            verbose=verbose,
        )
        total_cost += result.cost_usd
        total_duration += result.duration_s
        prior_raw_output = result.output_text

        try:
            raw = extract_json_block(result.output_text)
        except ValueError as e:
            last_validation_error = f"could not find a JSON block: {e}"
            logger.warning("Meta-analysis attempt %d failed to extract JSON: %s", attempt + 1, e)
            continue

        # Provenance is ours, not the model's. Override (don't setdefault) so
        # any model-emitted values are corrected.
        raw["experiment_id"] = experiment_id
        raw["recipe"] = recipe_name
        raw["driver"] = driver.name
        raw["model"] = model
        raw["timestamp"] = time.time()
        raw["n_episodes_investigated"] = n_investigated
        raw["outcome_distribution"] = dict(outcomes)
        raw["primary_blame_distribution"] = dict(blames)
        raw["success_rate"] = success_rate
        raw["audit_summary"] = audit_summary
        raw["cost_usd"] = total_cost
        raw["duration_s"] = total_duration
        raw["schema_version"] = META_ANALYSIS_SCHEMA_VERSION

        try:
            return MetaAnalysis.model_validate(raw)
        except ValidationError as e:
            last_validation_error = str(e)
            logger.warning("Meta-analysis attempt %d failed validation: %s", attempt + 1, e)

    raise RuntimeError(
        f"Meta-analysis output failed schema validation after {max_retries + 1} attempt(s).\n"
        f"Last validation error: {last_validation_error}"
    )


def _render_audit_md(audit_summary: dict | None) -> str:
    """Deterministic `## Audit signal` block prepended to the human-facing
    `.md` so the self-critique is visible even if the LLM prose omits it.
    Empty string when no audits ran."""
    if not audit_summary:
        return ""
    out = ["## Audit signal", "", f"Verdict distribution: `{audit_summary['verdict_distribution']}`", ""]
    flagged = audit_summary.get("flagged") or []
    if flagged:
        out.append("Episodes the auditor did **not** rate `sound` (treat their primary blame as a weak prior):")
        out.append("")
        for f in flagged:
            alts = ", ".join(a for a in f["alternative_blames"] if a) or "_(none named)_"
            out.append(f"- `{f['trajectory_id']}` — **{f['verdict']}**; auditor's alternatives: {alts}")
    else:
        out.append("All audited episodes rated `sound`.")
    out.append("")
    out.append("---")
    out.append("")
    return "\n".join(out)


def write_meta_analysis(experiment_dir: Path, analysis: MetaAnalysis) -> tuple[Path, Path]:
    """Write `meta_analysis.json` and `meta_analysis.md`. Returns the two paths."""
    json_path = experiment_dir / META_ANALYSIS_FILENAME
    md_path = experiment_dir / META_ANALYSIS_MARKDOWN_FILENAME
    json_path.write_text(json.dumps(analysis.model_dump(mode="json"), indent=2))
    md_path.write_text(_render_audit_md(analysis.audit_summary) + analysis.markdown_summary.rstrip() + "\n")
    return json_path, md_path


def copy_to_journal(
    journal_dir: Path,
    experiment_id: str,
    json_path: Path,
    md_path: Path,
) -> tuple[Path, Path]:
    """Mirror the synthesis into `<journal_dir>/<experiment_id>/`.

    Auto-CUBE later reads this directory to accumulate a running narrative
    across iterations. The Investigator only deposits files; Auto-CUBE owns
    the narrative log.
    """
    out_dir = Path(journal_dir).expanduser() / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    j_dst = out_dir / json_path.name
    m_dst = out_dir / md_path.name
    j_dst.write_text(json_path.read_text())
    m_dst.write_text(md_path.read_text())
    logger.info("Mirrored meta-analysis to %s", out_dir)
    return j_dst, m_dst


__all__ = [
    "FailurePattern",
    "Intervention",
    "MetaAnalysis",
    "META_ANALYSIS_FILENAME",
    "META_ANALYSIS_MARKDOWN_FILENAME",
    "META_ANALYSIS_SCHEMA_VERSION",
    "META_ANALYSIS_SYSTEM_PROMPT",
    "DEFAULT_META_ANALYSIS_MODEL",
    "RootCauseHypothesis",
    "copy_to_journal",
    "run_meta_analysis",
    "write_meta_analysis",
]
