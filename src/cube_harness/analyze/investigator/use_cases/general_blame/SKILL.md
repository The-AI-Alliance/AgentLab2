# investigator-general-blame

**When to pick this recipe**

Default fallback. Pick this when you want a single-pass blame attribution per
episode — what happened, who's at fault, what would fix it. The output schema
is the legacy `Findings` (V1) so any consumer that already reads
`episode_record.findings` keeps working unchanged.

**Output**: `findings.json` (and `findings` field on
`episode_record.json`) with the standard `BaseFindings` fields:
`analysis`, `outcome`, `summary`, `primary_blame`, `primary_blame_confidence`,
`other_blames`, `evidence`, `hypothesis`, `hypothesis_confidence`.

**When NOT to pick this**

- If you suspect a tight inner-loop pathology (token budget, context-window
  thrashing, retry storms): use `profiling`.
- If you suspect a scaffold-level issue (system prompt leaks, action-space
  mismatch, response/action drift): use `agent_scaffolding`.
