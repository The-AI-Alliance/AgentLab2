# investigator-profiling

**When to pick this recipe**

Pick this when the failure smell is *quantitative* — token spend, retry counts,
context-window thrash, budget exhaustion — not semantic. Common triggers:

- Agent looped on the same diagnostic for many steps.
- LLM call count is suspiciously high relative to budget.
- Many timeouts or empty completions in `summary_stats`.

**Output**: `findings.json` with the narrowed `ProfilingOutput` schema:
the standard base fields PLUS `profile_signal` and `suggested_budget_change`.
`primary_blame` is constrained to `agent_scaffolding`, `model_capability`, or
`none` — other categories are out of scope for this recipe.

**When NOT to pick this**

- Semantic failures (wrong tool choice, misread task): use `general_blame`.
- Agent-loop pathologies that aren't budget-shaped (e.g. response/action
  mismatches): use `agent_scaffolding`.
