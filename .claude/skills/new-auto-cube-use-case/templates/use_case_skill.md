# Auto-CUBE — <TODO: name> use-case

You are running the **<TODO: name>** use-case of Auto-CUBE. Auto-CUBE
is the iterate-and-fix outer loop; it owns the methodology, dispatches
the **Investigator** sub-agent per trajectory, and (optionally) ships
Fix Report PRs. This use-case differs from the default (`debug`) in:
<TODO: 1–2 sentences naming the key difference — goal, dispositions,
whether it ships PRs, etc.>

## Posture

<TODO: One short paragraph naming the agent's posture for this
use-case. Examples: "Curious scientist building a sparse coverage map"
(debug), "Patient profiler establishing what this model can do"
(capability), "Sweeping engineer optimising a config" (optimization).>

## Goal

<TODO: 1–2 sentences. What does this use-case produce, and what
long-horizon goal does it contribute to across sessions?>

## When to use this use-case

<TODO: 2–4 bullets. The conditions under which a user would invoke
this use-case rather than another.>

- ...

## Methodology

<TODO: How does the agent approach a session? Prose, not numbers.
Describe the discipline — what to do, what to avoid, how to reason
about choices. Let the agent pick specific numbers (task counts,
budgets) based on context.>

## Sampling & rounds

<TODO: How does the agent decide what to investigate in each round?
Broad-cheap zoom-out then zoom-in? Single-pass sweep? Pure profile?
Describe the structure of a round and how rounds compose.>

## Investigator dispatch

<TODO: Which Investigator use_case(s) does this dispatch by default
(general_blame / profiling / agent_scaffolding / hinter / fix_audit)?
Reference the local `investigator_extra.md` if one is present.
Describe when the agent should pass round-specific `--extra-prompt`
overrides.>

## Dispositions

<TODO: List the disposition labels the agent uses to classify
tasks/cells. Example for debug:
- PASS
- model-ceiling-done
- infra-suspect
- scaffold-suspect
- benchmark-suspect
- interesting / pending

For other use-cases, the ontology can be entirely different
(pass/fail/timeout/refused, or optimum/dominated/incomplete).>

## Stopping criterion

<TODO: When is a session done? Concrete enough that the agent can tell
itself "okay, stop now" without ambiguity. Examples:
- "Every target axis-point has a covered or model-ceiling disposition."
- "The best config has been compared against N alternatives with
  consistent ranking."
- "Capability of the model is profiled across all task categories."

If the criterion is open-ended, name a fallback (budget exhausted,
N rounds, etc.).>

## Outputs

<TODO: What does this use-case produce? Mark each as yes/no:
- `REPORT.md` (always yes)
- Fix Report PRs
- `design-debt` issues
- Updates to `coverage.json`
- Use-case-specific artefacts (custom ledger, scoreboard, …)>

## Cross-session state

<TODO: Does this use-case read/write the shared
`~/auto_cube/coverage.json` ledger? Does it have its own
ledger (e.g. `~/auto_cube/capability.json`)? Or is it
purely single-session? Document the format here so future sessions
can interoperate.>

## Intervention discipline (if shipping fixes)

<TODO: Keep this section if the use-case files Fix Report PRs;
otherwise replace with "n/a — this use-case does not modify code."

If kept, reference the auto-fix methodology:
[`openspec/specs/auto-fix/spec.md`](../../../../../openspec/specs/auto-fix/spec.md).
Standard text from the debug use-case works; tweak as needed.>

## Templates

The shared Auto-CUBE templates live in
[`src/cube_harness/auto_cube/templates/`](../../templates/) — use the
ones that fit:

- `session.md` — session scope + live tracker
- `notes.md` — per-round hypothesis → result
- `exp_config.py` — copy-and-edit experiment recipe
- `fix_report.md` — PR body for fixes (only if shipping PRs)
- `report.md` — final REPORT.md rollup
