"""`hinter` — extract task-specific hints from a failed episode.

The default investigator (`general_blame`) attributes blame and proposes a one-line
hypothesis. `hinter` is narrower and more actionable: given a failed
episode, produce concrete hint candidates the user can drop into
`GennyConfig.task_hints[task_id]` to make the agent succeed on the next run.

Hints are NOT a substitute for fixing real bugs — if the failure root cause
is a tool bug or a scaffolding gap, the right fix is in that layer. Hints
exist for cases where the model needed a small nudge: the task description
was under-specified, the success criterion is implicit, or the right
sequence of actions is non-obvious.

Output extends `BaseFindings` with `task_hints: list[TaskHint]`. The base
fields (`analysis`, `evidence`, `summary`, `outcome`, `primary_blame`,
`hypothesis`, …) are still produced — they ground the hint candidates and
let cross-recipe aggregation continue to work.
"""

from __future__ import annotations

from typing import Literal

from cube.core import TypedBaseModel
from pydantic import Field

from cube_harness.analyze.investigator.recipe import BaseFindings, InvestigatorRecipe
from cube_harness.analyze.investigator.schema_prompt import model_to_json_example


class TaskHint(TypedBaseModel):
    """One hint candidate the user can apply to a future run.

    Field order is CoT-deliberate: explain WHAT the agent missed and WHY
    before producing the hint text and rating it.
    """

    rationale: str = Field(
        description="Why this hint would have helped THIS episode succeed. Cite specific transcript steps."
    )
    hint_type: Literal["clarification", "task_specific", "general_guidance"] = Field(
        description=(
            "clarification = the task description was ambiguous or missing context. "
            "task_specific = a tip that applies to this single task_id. "
            "general_guidance = a tip that applies across many tasks in the same family."
        )
    )
    task_id: str = Field(description="trajectory's task_id this hint targets, or 'general' for cross-task guidance.")
    text: str = Field(
        description="The hint text, ready to inject as `GennyConfig.task_hints[task_id]`. Short and concrete."
    )
    confidence: int = Field(
        ge=0, le=5, description="0=guess, 5=very likely to fix this episode in a re-run with the hint applied."
    )


class HinterOutput(BaseFindings):
    """Investigator output extended with harvested hint candidates.

    Inherits the closed-world blame taxonomy from `BaseFindings` so
    aggregations (cross-experiment CSVs, meta-analysis) can still group hint-
    harvest runs alongside other recipes.
    """

    task_hints: list[TaskHint] = Field(
        default_factory=list,
        description=(
            "Hint candidates extracted from this episode. Empty when the failure root cause "
            "is a real bug (tool / scaffolding / eval) — hints would only mask the bug."
        ),
    )


HINTER_SYSTEM_PROMPT = """You are a hint-harvesting investigator for failed agent episodes.

Your job: read a failed trajectory and extract concrete, narrow hints that would
plausibly let the agent solve THIS task on a re-run. Hints are short prompt
fragments injected as `GennyConfig.task_hints[task_id]`.

Hard rules — DO NOT produce hints when:

- The failure is a tool bug. The right fix is in the tool wrapper. A hint that
  works around a tool bug masks the bug; flag it in `analysis`/`hypothesis` and
  set `task_hints=[]`.
- The failure is an evaluation bug (the agent did the right thing, eval
  rejected). The right fix is in the eval function. Set `task_hints=[]`.
- The failure is a fundamental capability gap that no short hint can fix.
  Set `task_hints=[]`.

When you DO produce hints, prefer:

- `clarification` when the task description was under-specified — the hint
  text supplies the missing context.
- `task_specific` when the task is well-described but a non-obvious workflow
  is needed (e.g. "use the funnel icon, not column header clicks").
- `general_guidance` when the same hint would help several sibling tasks in
  the family.

Each hint must be (a) short — one or two sentences, (b) actionable — name a
concrete tool / UI element / API to use, (c) grounded in evidence — the
`rationale` cites specific transcript steps.

You have read-only tools (Read / Glob / Grep / Bash). Read the transcript
end-to-end first; cite step numbers when they support a hint.

Reply with a single JSON object inside ```json ... ``` fences, matching the
schema in the user prompt."""


_HINTER_OUTPUT_JSON = model_to_json_example(HinterOutput).replace("{", "{{").replace("}", "}}")


HINTER_USER_PROMPT_TEMPLATE = f"""Harvest hints from this failed episode.

# Episode

trajectory_id: {{trajectory_id}}
task_id: {{task_id}}
reward: {{reward}}
total_steps: {{total_steps}}
agent: {{agent_name}}
benchmark: {{benchmark_name}}

# Files you can read

Transcript:
  {{transcript_dir}}/transcript.txt
  {{transcript_dir}}/steps/

Episode metadata (reward_info, action_schemas, summary_stats):
  {{episode_metadata_path}}

Episode config (agent prompts, model, budget, current task_hints):
  {{episode_config_path}}

Source code (Glob / Grep — useful for understanding what the tool exposed):
{{source_paths_block}}

Task description:
  {{task_description}}

# Output schema

Produce a single JSON object matching `HinterOutput`. Wrap in a ```json
fence. Field order is deliberate — describe and cite first, then commit and
score. Each leaf is annotated `<type — description>`:

```json
{_HINTER_OUTPUT_JSON}
```

Reminders:
- `task_hints` may be EMPTY when the failure root cause belongs in tool /
  scaffolding / eval. That is the correct answer in those cases.
- Prefer `task_id`-scoped hints over `general` ones; the former rarely
  regresses other tasks.
- The `hint_text` you emit will be appended to the agent's user prompt
  verbatim — phrase it like an instruction the agent would read."""


RECIPE = InvestigatorRecipe(
    name="hinter",
    system_prompt=HINTER_SYSTEM_PROMPT,
    user_prompt_template=HINTER_USER_PROMPT_TEMPLATE,
    output_model=HinterOutput,
    model="claude-sonnet-4-6",
)
