"""`general_blame` — the default investigator recipe.

Output schema is identical to `BaseFindings`, preserving on-disk
compatibility for the default investigator path. Subclassing without adding fields
gives the recipe a class identity (so it can declare `output_model`) without
changing the JSON shape.

Prompts are defined inline rather than in a sibling `prompt.py` module —
this matches the `profiling/` and `agent_scaffolding/` pattern (each use case
is a self-contained directory) and lets the user-prompt template derive its
JSON example directly from `BaseFindings` via `model_to_json_example`.
That single-source-of-truth pattern is the same one the meta-analysis prompt
uses; renaming or reordering a field on `BaseFindings` updates the prompt
automatically.
"""

from __future__ import annotations

from cube_harness.analyze.investigator.recipe import BaseFindings, InvestigatorRecipe
from cube_harness.analyze.investigator.schema_prompt import model_to_json_example


class GeneralBlameOutput(BaseFindings):
    """The default investigator output. Identical fields to `BaseFindings` —
    a class identity so the recipe can carry an `output_model` reference."""


INVESTIGATOR_SYSTEM_PROMPT = """You are a post-hoc investigator for agent episodes. Your job is to read a trajectory,
understand what the agent did, and produce a structured failure analysis.

You have read-only access to the trajectory transcript and to the source code that
ran the experiment. You will use Read / Glob / Grep / Bash to navigate them.

Hallucination rules (strict):
- Every blame attribution must be backed by a verbatim quote from the transcript.
- Categories are closed-world. Pick from the taxonomy or use `none`. Do not invent.
- Confidence is on a 0-5 scale. Use 2 or below when the evidence is thin and say so.
- Write `analysis` first as a scratchpad. Your structured fields must be consistent with it.

Your final response MUST be a single JSON object inside ```json ... ``` fences.
No other text after the closing fence."""


# Auto-derive the JSON example from `BaseFindings` so the prompt and the
# Pydantic class stay in sync. Braces are pre-escaped (`{` → `{{`) so the
# example survives `str.format()` later — `_build_user_prompt` substitutes
# `{trajectory_id}` and friends; the JSON's own braces must not collide.
_FINDINGS_JSON = model_to_json_example(BaseFindings).replace("{", "{{").replace("}", "}}")


INVESTIGATOR_USER_PROMPT_TEMPLATE = f"""Investigate this episode.

# Episode

trajectory_id: {{trajectory_id}}
task_id: {{task_id}}
reward: {{reward}}
total_steps: {{total_steps}}
agent: {{agent_name}}
benchmark: {{benchmark_name}}

# Files you can read

Transcript (one file per step):
  {{transcript_dir}}/steps/

Consolidated transcript (full text):
  {{transcript_dir}}/transcript.txt

Episode metadata (reward_info, action_schemas, summary_stats):
  {{episode_metadata_path}}

Episode config (agent prompts, model, budget):
  {{episode_config_path}}

Task description:
  {{task_description}}

Source code (use Glob / Grep — do NOT pre-read all of it):
{{source_paths_block}}

# Output schema

Produce a single JSON object with these fields, in this order. The order is
deliberate — emit your reasoning and evidence first, then commit to the
structured fields, then score them. Models reason better when they think
before they classify. Each leaf is annotated `<type — description>`:

```json
{_FINDINGS_JSON}
```

Outcome semantics:
  success                   — solved correctly
  success_lucky             — reward=1 but reached by accident or wrong approach
  almost                    — right strategy, failed on a minor technical detail
  failure                   — task not solved
  should_have_been_rewarded — agent did the right thing but eval rejected it

Blame semantics:
  task_unclear              — task description ambiguous or missing context
  model_capability          — agent understood task, lacked reasoning to solve
  tool_failure              — tool wrapper bug / unexpected exception
  env_failure               — container crash, network timeout, infra
  agent_scaffolding         — system prompt, budget, context mgmt, submission protocol
  action_space_limited      — required action does not exist in tool set
  insufficient_observation  — observation missing crucial info (truncation, pruning)
  eval_brittle              — evaluator rejected a correct solution
  submission_format         — agent reached solution but submitted wrong way
  none                      — clean success or too ambiguous to attribute

Invariants:
- Evidence MUST be non-empty when primary_blame != "none".
- other_blames MUST NOT repeat primary_blame.
- On clean success, primary_blame must be "none"."""


RECIPE = InvestigatorRecipe(
    name="general_blame",
    system_prompt=INVESTIGATOR_SYSTEM_PROMPT,
    user_prompt_template=INVESTIGATOR_USER_PROMPT_TEMPLATE,
    output_model=GeneralBlameOutput,
    model="claude-sonnet-4-6",
)


__all__ = ["RECIPE", "GeneralBlameOutput", "INVESTIGATOR_SYSTEM_PROMPT", "INVESTIGATOR_USER_PROMPT_TEMPLATE"]
