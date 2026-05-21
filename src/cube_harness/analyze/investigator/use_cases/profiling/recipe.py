"""`profiling` — narrow recipe for inner-loop pathologies.

Output schema is reduced: only three blame categories are admissible
(`agent_scaffolding`, `model_capability`, `none`). The investigator focuses on token
usage, retry storms, context-window thrashing, and similar profile-shaped
failures, not on task semantics or eval brittleness.

`BashOutput` is added to `allowed_tools` — profiling traces often live in JSON
files the investigator will need to slice with shell pipelines.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from cube_harness.analyze.investigator.recipe import BaseFindings, InvestigatorRecipe

PROFILING_SYSTEM_PROMPT = """You are a profiling-focused investigator for agent episodes.

Your scope is narrow: did the failure look like a scaffold-level inefficiency
(retry storms, context-window thrash, tool-call timeouts, budget exhaustion) or
a genuine model-capability ceiling? Other blame categories (task_unclear,
eval_brittle, etc.) are outside scope — leave them to the general_blame recipe.

Use Bash + BashOutput freely to slice usage statistics in JSON files (jq is
available). Do not modify any files; the investigator is read-only.

Output rules:
- `primary_blame` MUST be one of: agent_scaffolding, model_capability, none.
  Use `none` for episodes where the profile is fine even if the task failed.
- Evidence MUST cite a specific step or stat (e.g. "n_llm_calls=42 with budget=20").
- Be conservative on confidence — profile signals are noisy.

Reply with a single JSON object inside ```json ... ``` fences, matching
`ProfilingOutput`."""


PROFILING_USER_PROMPT_TEMPLATE = """Profile this episode.

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

Episode metadata (look for `summary_stats`, `usage`, `n_llm_calls`):
  {episode_metadata_path}

Episode config (look for `budget`, `model`, `timeout`):
  {episode_config_path}

Source code (use Glob/Grep — do NOT pre-read all of it):
{source_paths_block}

Task description:
  {task_description}

# Output schema

Produce a single JSON object matching `ProfilingOutput` (see system prompt for
the field list and constraints). Wrap it in a ```json fence."""


ProfilingBlame = Literal["agent_scaffolding", "model_capability", "none"]


class ProfilingOutput(BaseFindings):
    """Profiling-focused investigator output.

    Inherits the base fields (`analysis`, `outcome`, `summary`,
    `primary_blame`, `primary_blame_confidence`, `other_blames`, `evidence`,
    `hypothesis`, `hypothesis_confidence`) so cross-recipe aggregation keeps
    working. Adds two narrow profile-shaped fields:
    """

    profile_signal: str = Field(
        description=(
            "One-line summary of the profile-shaped finding "
            "(e.g. '17 retry-loop calls before timeout', 'budget=20, used 19, no progress')."
        )
    )
    suggested_budget_change: str | None = Field(
        default=None,
        description="Free-form: budget / context-size knob the user might tune.",
    )


RECIPE = InvestigatorRecipe(
    name="profiling",
    system_prompt=PROFILING_SYSTEM_PROMPT,
    user_prompt_template=PROFILING_USER_PROMPT_TEMPLATE,
    output_model=ProfilingOutput,
    model="claude-sonnet-4-6",
    allowed_tools=("Read", "Glob", "Grep", "Bash", "BashOutput"),
)
