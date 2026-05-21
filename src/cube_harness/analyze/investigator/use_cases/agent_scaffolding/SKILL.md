# investigator-agent-scaffolding

**When to pick this recipe**

Pick this when the failure mode is loop-shaped — the agent didn't lack
capability or context, it just got stuck in a pathological pattern. Common
triggers:

- Same action emitted 3+ times in a row.
- Plan oscillates between two near-duplicate states.
- Agent terminates with `submit` despite obvious next steps.
- LLM reasoning and emitted action disagree.
- Tool calls hit format errors and the agent doesn't recover.

**Output**: `findings.json` with the standard base fields PLUS
`scaffold_diagnosis` — a structured `ScaffoldDiagnosis` (loop_subtype,
stuck_phase, response_action_mismatch). The closed-world `loop_subtype`
taxonomy lets aggregate analysis bucket failures.

**When NOT to pick this**

- Quantitative inner-loop issues (budget, retries): use `profiling`.
- Semantic blame attribution (eval_brittle, task_unclear, ...): use
  `general_blame`.
