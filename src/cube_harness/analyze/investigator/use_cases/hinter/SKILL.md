# investigator-hint-harvest

**When to pick this recipe**

You have a failed agent episode and want to know whether a small, targeted
hint would have let it succeed — and if so, what that hint should say.
Output is a list of concrete `task_hints[task_id]` candidates ready to drop
into `GennyConfig`.

Common triggers:

- The task description is under-specified (the goal is implicit, the
  success criterion is hidden in the eval function).
- The agent took a "looks-right" action that doesn't actually count
  (e.g. clicked a column header where the eval requires the filter UI).
- A small piece of domain knowledge would obviously help (an API quirk,
  a non-obvious workflow step).

**When NOT to pick this**

The hint catalogue should stay small. Hints mask deeper bugs. If the root
cause is one of these, fix it in the right layer instead:

| Root cause | Right place to fix |
|---|---|
| Tool gives incomplete or wrong information | The tool wrapper |
| Observation missing required state | `obs_postprocess` in the cube task |
| Eval function rejects a correct answer | The eval function |
| Agent loops / thrashes / runs out of context | Agent scaffolding (Genny) |
| Capability ceiling | A different model |

If you suspect any of those, run `general_blame` or `agent_scaffolding`
first. `hinter` is the last resort, used when the failure is
"the model didn't know what we expected."

**Output**

`findings.json` with the standard `BaseFindings` fields PLUS:

- `task_hints[]` — list of `TaskHint(rationale, hint_type, task_id, text,
  confidence)`. May be empty when the recipe concludes that no hint should
  be added.

`task_hints` may be empty even on a failed episode — that is the correct
answer when the right fix is upstream.

**Applying the harvested hints**

Hints are not auto-applied. The user reads the harvest output, picks the
high-confidence candidates, and edits `GennyConfig.task_hints` (or the
appropriate config) by hand. Re-run the same episode with the hint to
confirm the hypothesis before committing.

**Scope of authorship**

A hint's `text` is appended verbatim to the agent's user prompt. Keep it
short (one or two sentences), name a concrete tool / UI element / API,
and ground it in the rationale.
