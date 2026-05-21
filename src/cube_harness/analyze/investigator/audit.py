"""Unified self-evaluation pass for the investigator.

Replaces the previous separate `self_investigator` recipe and `post_investigator_survey` —
one pass, one file (`audit.json`), one cost overhead. The audit consumes the
primary finding plus the same context the investigator had, and produces:

- a `verdict` on whether the finding is sound;
- per-axis scores (reasoning, ease-of-analysis, context quality, 0-5);
- enumerated gaps and missed evidence;
- alternative blame attributions worth considering.

The audit is opt-in (`recipe.audit=True` or `--audit`); ~25-30% cost overhead
when on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cube.core import TypedBaseModel
from pydantic import Field, ValidationError

from cube_harness.analyze.investigator.parse import extract_json_block

if TYPE_CHECKING:
    from cube_harness.analyze.investigator.agent_driver import AgentDriver
    from cube_harness.analyze.investigator.recipe import InvestigatorRecipe

logger = logging.getLogger(__name__)

AUDIT_FILENAME = "audit.json"
AUDIT_SCHEMA_VERSION = 1

AUDIT_SYSTEM_PROMPT = """You are an auditor for a post-hoc trajectory investigator.

A prior finding about an agent episode has already been written. Your job
is not to redo that finding — it is to critique it. Read the prior
finding, re-examine the transcript and source it had access to, and flag
where the reasoning is weak, where evidence is thin, where a different
blame attribution would be defensible, and what tooling gaps the original
investigator ran into.

You have the same read-only tools the investigator had (Read, Glob, Grep, Bash).
Do not produce a new finding. Reply with a single JSON object matching
the AuditOutput schema specified in the user prompt — nothing else."""


AUDIT_PROMPT = """You just produced a JSON finding for an episode. Now audit your own work.

Re-examine the transcript and source you read. Look for:
- weaknesses in your reasoning chain (logical leaps, unsupported claims),
- evidence you missed or under-weighted,
- alternative blame attributions you considered too quickly,
- ways the context (transcript, source paths) limited what you could conclude,
- tooling gaps (things you wished you could read or run but couldn't).

Be honest. The point of this pass is not to defend the prior finding — it is
to flag the parts that are weakest so a human reviewer knows where to look.

Reply with a single JSON object inside ```json ... ``` fences, matching this schema exactly:

```json
{
  "verdict": "sound | questionable | wrong",
  "reasoning_quality": 0,
  "ease_of_analysis": 0,
  "context_quality": 0,
  "tooling_gaps": [],
  "missed_evidence": [],
  "alternative_blames": [{"blame": "...", "rationale": "..."}],
  "notes": null
}
```

Score axes (0-5):
  reasoning_quality  — how rigorously did the prior finding chain from evidence to conclusion?
  ease_of_analysis   — how easy was it to form a clear finding given what you had?
  context_quality    — how complete was the transcript + source context?

verdict:
  sound          — finding is well-grounded; minor quibbles only
  questionable   — finding may be right, but the evidence chain is thin
  wrong          — re-reading the transcript suggests a different blame is more likely

No prose outside the JSON fence."""


class BlameAlternative(TypedBaseModel):
    """An alternative blame attribution the auditor would consider."""

    blame: str = Field(description="Blame category (one of the BlameCategory enum values, or free text).")
    rationale: str = Field(description="Why this alternative is worth considering.")


class AuditOutput(TypedBaseModel):
    """Self-evaluation of a primary finding.

    Written to `<episode_dir>/audit.json` next to `findings.json`. The schema
    is versioned so downstream consumers can branch on shape changes.
    """

    schema_version: int = AUDIT_SCHEMA_VERSION
    recipe: str = Field(description="Recipe `name` of the finding being audited.")
    driver: str = Field(description="Driver `name` that produced the finding.")
    verdict: Literal["sound", "questionable", "wrong"]
    reasoning_quality: int = Field(ge=0, le=5)
    ease_of_analysis: int = Field(ge=0, le=5)
    context_quality: int = Field(ge=0, le=5)
    tooling_gaps: list[str] = Field(default_factory=list)
    missed_evidence: list[str] = Field(default_factory=list)
    alternative_blames: list[BlameAlternative] = Field(default_factory=list)
    notes: str | None = None


def _build_fallback_run_prompt(findings_json: str) -> str:
    """Wrap the audit prompt with the prior finding for drivers that can't
    continue a session — the auditor still needs the primary output to react to."""
    return (
        "The following JSON is the prior finding you produced for this episode:\n\n"
        f"```json\n{findings_json}\n```\n\n"
        f"{AUDIT_PROMPT}"
    )


async def run_audit_pass(
    *,
    recipe: "InvestigatorRecipe",
    driver: "AgentDriver",
    findings_json: str,
    investigation_trace_path: Path,
    session_id: str | None = None,
    cwd: Path | None = None,
    additional_dirs: list[Path] | None = None,
    verbose: bool = False,
) -> tuple[AuditOutput, float]:
    """Run the audit pass against `driver`. Returns `(audit, cost_usd)`.

    Tries `driver.continue_session(session_id, AUDIT_PROMPT)` first; on
    `NotImplementedError`, falls back to a fresh `driver.run(...)` with the prior
    finding serialised into the prompt.

    `investigation_trace_path` is accepted but not consumed in the fallback path — the
    finding alone is sufficient for the audit (the trace is helpful but optional).
    Future drivers may use it.
    """
    _ = investigation_trace_path  # accepted for API symmetry; see docstring.

    try:
        if session_id is None:
            raise NotImplementedError("no session_id available; falling back to fresh run")
        result = await driver.continue_session(
            session_id=session_id,
            follow_up_prompt=AUDIT_PROMPT,
            verbose=verbose,
        )
    except NotImplementedError as e:
        logger.info("Audit: continue_session unsupported (%s); running fresh audit pass.", e)
        if cwd is None or additional_dirs is None:
            raise RuntimeError(
                "Audit fallback needs cwd and additional_dirs (driver does not support continue_session)."
            ) from e
        # Important: use AUDIT_SYSTEM_PROMPT here, not recipe.system_prompt.
        # The recipe's system prompt frames the model as "the investigator"; the
        # audit pass needs it framed as "the auditor of a prior finding".
        result = await driver.run(
            system_prompt=AUDIT_SYSTEM_PROMPT,
            user_prompt=_build_fallback_run_prompt(findings_json),
            cwd=cwd,
            additional_dirs=additional_dirs,
            model=recipe.model,
            allowed_tools=recipe.allowed_tools,
            permission_mode=recipe.permission_mode,
            verbose=verbose,
        )

    raw = extract_json_block(result.output_text)
    # Provenance is ours, not the model's — override whatever (if anything)
    # the model emitted in these fields. Keeps the on-disk record honest.
    raw["recipe"] = recipe.name
    raw["driver"] = driver.name
    try:
        audit = AuditOutput.model_validate(raw)
    except ValidationError as e:
        raise RuntimeError(f"Audit output failed schema validation: {e}\nRaw: {raw}") from e
    return audit, result.cost_usd


def write_audit(episode_dir: Path, audit: AuditOutput) -> Path:
    """Write the audit output as a sibling of `findings.json`. Returns the path."""
    out = Path(episode_dir) / AUDIT_FILENAME
    out.write_text(json.dumps(audit.model_dump(mode="json"), indent=2))
    return out


__all__ = [
    "AuditOutput",
    "BlameAlternative",
    "AUDIT_FILENAME",
    "AUDIT_PROMPT",
    "AUDIT_SYSTEM_PROMPT",
    "AUDIT_SCHEMA_VERSION",
    "run_audit_pass",
    "write_audit",
]
