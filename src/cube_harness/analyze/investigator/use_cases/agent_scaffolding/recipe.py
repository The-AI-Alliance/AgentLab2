"""`agent_scaffolding` — recipe focused on agent-loop pathologies.

Adds a structured `scaffold_diagnosis` field that names the loop subtype, the
phase where the agent got stuck, and whether the response/action pair drifted
out of sync.
"""

from __future__ import annotations

from typing import Literal

from cube.core import TypedBaseModel
from pydantic import Field

from cube_harness.analyze.investigator.recipe import BaseFindings, InvestigatorRecipe

LoopSubtype = Literal[
    "tight_repeat",
    "thrash_two_state",
    "context_overflow",
    "premature_giveup",
    "tool_format_drift",
    "submission_protocol_miss",
    "none",
]


class ScaffoldDiagnosis(TypedBaseModel):
    """Structured diagnosis of an agent-loop pathology.

    Each field is a focused signal — taken together, they describe *what kind*
    of scaffold failure occurred, not just that one happened. `loop_subtype`
    is closed-world; the rest are free-form descriptions grounded in evidence.
    """

    loop_subtype: LoopSubtype = Field(
        description="Closed-world taxonomy of agent-loop pathologies; `none` if the loop ran cleanly."
    )
    stuck_phase: str = Field(
        description="One-line description of the phase the agent got stuck in (e.g. 'planning', 'tool_call_assembly')."
    )
    response_action_mismatch: bool = Field(
        description="True when the LLM's reasoning and the action it emitted disagreed about the next step."
    )


AGENT_SCAFFOLDING_SYSTEM_PROMPT = """You are an agent-scaffolding-focused investigator.

Your scope: characterise *how* the agent loop failed, not just whether the task
was solved. You're looking for things like:

- Tight repeats: agent emits the same action 3+ times in a row.
- Two-state thrash: agent oscillates between two near-duplicate plans.
- Context overflow: agent loses earlier context and re-discovers it.
- Premature giveup: agent terminates with `submit` despite obvious next steps.
- Tool format drift: agent calls tools with wrong arg shape, hits errors, doesn't recover.
- Submission protocol miss: agent reaches a solution but submits via the wrong channel.

You have read-only tools (Read / Glob / Grep / Bash). Read the transcript first
end-to-end before classifying. Cite specific step numbers in `evidence`.

Reply with a single JSON object inside ```json ... ``` fences, matching
`AgentScaffoldingOutput`. Set `scaffold_diagnosis.loop_subtype = "none"` when
the loop ran cleanly even if the task failed for other reasons (in which case
`primary_blame` should likely be `model_capability` or another non-scaffold
category)."""


AGENT_SCAFFOLDING_USER_PROMPT_TEMPLATE = """Diagnose this episode's agent loop.

# Episode

trajectory_id: {trajectory_id}
task_id: {task_id}
reward: {reward}
total_steps: {total_steps}
agent: {agent_name}
benchmark: {benchmark_name}

# Files you can read

Transcript:
  {transcript_dir}/transcript.txt
  {transcript_dir}/steps/

Episode metadata:
  {episode_metadata_path}

Episode config (system prompts, tool list, budget):
  {episode_config_path}

Source code (Glob/Grep — agent scaffolding lives in cube_harness/agents/):
{source_paths_block}

Task description:
  {task_description}

# Output schema

Produce a single JSON object matching `AgentScaffoldingOutput`. Wrap in a ```json fence.
"""


class AgentScaffoldingOutput(BaseFindings):
    """Investigator output extended with a structured loop-diagnosis field."""

    scaffold_diagnosis: ScaffoldDiagnosis | None = Field(
        default=None,
        description=(
            "Structured agent-loop diagnosis. None when the loop ran cleanly AND "
            "primary_blame is non-scaffold (e.g. eval_brittle); otherwise required."
        ),
    )


RECIPE = InvestigatorRecipe(
    name="agent_scaffolding",
    system_prompt=AGENT_SCAFFOLDING_SYSTEM_PROMPT,
    user_prompt_template=AGENT_SCAFFOLDING_USER_PROMPT_TEMPLATE,
    output_model=AgentScaffoldingOutput,
    model="claude-sonnet-4-6",
)
