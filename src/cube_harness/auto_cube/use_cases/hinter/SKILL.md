# Auto-CUBE — hinter use-case

You are running the **hinter** use-case of Auto-CUBE. Auto-CUBE is the
iterate-and-fix outer loop; it owns the methodology, dispatches the
**Investigator** sub-agent per trajectory, and ships PRs. This use-case
differs from the default (`debug`) in its **goal**: `debug` chases bugs and
attributes blame; **hinter raises performance** on a benchmark by adding
knowledge — better prompts, clearer task wording, a sharper action space —
and it ships those improvements to the right place rather than filing bug
fixes.

## Posture

A coach who is also a compression engineer. You intervene freely to discover
*what knowledge an agent is missing*, then you spend most of your judgment
deciding where that knowledge belongs. The downstream goal is not "make this
task pass" — it is **collect information through interventions and understand
which patterns generalize**, so the right, durable edit can be made. Hold the
goal firmly and the procedure loosely.

## Goal

Turn observed failures into knowledge placed at the **right level of
regularization**, so a generalist agent does better on this benchmark — and,
where the pattern generalizes, on others too. Each session leaves behind
re-tested improvements (mostly PRs) and a clearer map of what steering each
task needs.

## When to use this use-case

- You want to **improve a benchmark score**, not diagnose a regression.
- Failures look like *"the agent didn't know what we expected"* — under-specified
  wording, a non-obvious workflow, a tool whose description undersells it.
- You suspect the **action space** is the bottleneck (a missing action, or one
  the model misuses because of its description).
- **Not** for tool/eval/scaffolding bugs or a capability ceiling — those go to
  `debug` (and the per-episode Investigator will tell you when you've hit one).

## The regularization ladder (the core of this use-case)

Every intervention is knowledge added somewhere. *Where* trades off speed
against generality, and that trade-off is the whole game: think
**overfitting → compression**. Climb only as far as the evidence supports.

| Reg | Destination | Mechanism | Write path |
|-----|-------------|-----------|------------|
| **Low** (overfit OK) | Task hint "cheat" | `GennyConfig.task_hints[task_id]` (per round, in `exp_config.py`) | Local to the session — **no PR** |
| **Mid** (cross-task) | Generalizable memory | *(deferred — JEF-Hinter; not yet in Genny)* | n/a yet |
| **High** (must generalize) | Task clarification | `benchmark_clarifications.py` → `TASK_CLARIFICATION` | PR to the cube |
| **High** | Benchmark prompt | `benchmark_clarifications.py` → `BENCHMARK_HINT` | PR to the cube |
| **High** | Action description | `AgentConfig.description_overrides` to test → tool docstring | PR to the tool |
| **High** | New / fixed action | `@tool_action` on the tool | PR to the tool |
| **High** | Generalist agent prompt | `GennyConfig.system_prompt` | PR to cube-harness |

**Regularization level = review rigor.** Low-reg writes freely; high-reg lands
as a reviewed PR through the auto-fix methodology
([`openspec/specs/auto-fix/spec.md`](../../../../../openspec/specs/auto-fix/spec.md)).

**High-reg content stays professional and general.** No imperative "YOU MUST"
phrasing. A *task clarification* is only for genuinely ambiguous wording and
**must not reveal how to solve the task** — it fixes brittle phrasing, it does
not coach. A *benchmark prompt* orients a first-time reader on conventions, not
on any single task. Anything that smells like overfitting belongs in the
low-reg task-hint cheat, not high-reg.

## Methodology

Two phases. Phase 1 stages cheap, overfit interventions to *learn the
landscape*; Phase 2 reflects across that landscape and *promotes* what
generalizes. This is a disposition for the work, not a script — let the
benchmark and the evidence set the pace.

**Phase 1 — steer & understand (low-reg).** For a batch of failing tasks,
find the smallest hint that lets the agent succeed and drop it into
`GennyConfig.task_hints[task_id]` in the round's `exp_config.py`. Re-run to
confirm the steer works. The win here is not the pass — it is *mapping what
knowledge each task actually needs*. Anything goes; this is the staging ground.

**Phase 2 — reflect, promote, compress (climb the ladder).** When a steer
recurs across several tasks — the same convention, the same misread action, the
same missing step — it is a candidate to **compress** into a higher-reg
destination (see the ladder). Lift it to the right level, then **re-test that
the generalized form still works** (and didn't regress others). Promotion is
the deliverable; the cheat was scaffolding. A steer that *doesn't* generalize
stays a task hint — that is a valid outcome, not a failure.

## Sampling & rounds

- **Zoom out:** run a broad, cheap batch on the target benchmark; dispatch the
  Investigator (below) per episode to harvest hint candidates and separate
  *steerable* failures from *real bugs*.
- **Phase-1 rounds:** apply the harvested `task_hints` for a cluster of tasks;
  re-run; confirm which are genuinely steerable.
- **Phase-2 rounds:** group the confirmed steers by the pattern they share;
  promote each group to its ladder destination; re-run the affected tasks
  (plus a small held-out slice) to confirm the generalized edit holds.

Keep each round's `exp_config.py`, `notes.md`, and trajectories together under
the session (see Templates / Outputs).

## Investigator dispatch

Default: the **`hinter`** Investigator recipe
([`../../../analyze/investigator/use_cases/hinter/SKILL.md`](../../../analyze/investigator/use_cases/hinter/SKILL.md)),
which emits `task_hints[task_id]` candidates (`rationale`, `hint_type`,
`text`, `confidence`) and stays silent when the right fix is upstream. The
local [`investigator_extra.md`](investigator_extra.md) biases it toward this
use-case's framing (steerable-vs-bug, and which ladder rung a candidate
suggests). When a failure is attributed to a tool/eval/scaffold bug, switch
that episode to `general_blame` / `agent_scaffolding` — that is a `debug`
finding, not a hint.

## Dispositions

- **steered** — a low-reg task hint makes the task pass; landscape mapped.
- **promoted** — the steer generalized and was lifted to a high-reg
  destination via a re-tested PR.
- **cheat-only** — steering works but is genuinely task-specific; it stays a
  low-reg task hint (not promoted).
- **not-a-hint** — the failure is a real bug (tool / eval / scaffolding); hand
  off to `debug`.
- **unsteerable** — no hint helps (capability ceiling / fundamental gap); use a
  stronger model or move on.

## Stopping criterion

A session is done when every target task carries one of the dispositions above
— each steerable failure is either *promoted* or consciously left *cheat-only*,
and the rest are *not-a-hint* / *unsteerable*. Fallback: the per-session budget
is exhausted, or a Phase-2 round produces no new generalizable pattern.

## Outputs

- `REPORT.md` — **yes** (scope, the steer→promote arc, ladder placements,
  before/after scores, shipped vs open PRs, cost).
- Fix Report PRs — **yes**, one per promotion (cube / tool / cube-harness).
- `design-debt` issues — when a recurring steer points at a deeper gap (e.g. a
  whole class of tasks needs an action that doesn't exist).
- `coverage.json` updates — **read** for context; this use-case's primary
  ledger is its own (below).
- Use-case-specific — the session's confirmed `task_hints` (in each round's
  `exp_config.py`) and the promotion ledger.

## Cross-session state

Reads the shared `~/auto_cube/coverage.json` for prior context. Its own ledger,
`~/auto_cube/hints.json`, records per-`task_id` steering outcomes so later
sessions don't re-derive the same hints and can see which steers were promoted:

```json
{
  "<cube>|<task_id>": {
    "disposition": "steered | promoted | cheat-only | not-a-hint | unsteerable",
    "hint_text": "<the confirmed low-reg steer, if any>",
    "promoted_to": "task_clarification | benchmark_prompt | action_description | new_action | system_prompt | null",
    "promotion_pr": "<url or null>",
    "last_session": "<session-id>"
  }
}
```

## Intervention discipline (shipping promotions)

Phase-1 cheats are local and unreviewed — that is fine, they are scaffolding.
**Phase-2 promotions ship through the auto-fix methodology**
([`openspec/specs/auto-fix/spec.md`](../../../../../openspec/specs/auto-fix/spec.md)):
a Fix Report PR whose invariant is the *generalization claim* ("this
clarification / description / prompt helps across these tasks without
revealing solutions or regressing others"), with the before/after evidence the
re-test produced. Promotions to a cube or tool are PRs to that package; a
system-prompt change is a PR to cube-harness. The low-reg cheat that confirmed
the pattern is not the deliverable — the promoted, re-tested edit is.

## Templates

Shared Auto-CUBE templates live in
[`../../templates/`](../../templates/) (`session.md`, `notes.md`,
`fix_report.md`, `report.md`). This use-case ships its own
[`templates/exp_config.py`](templates/exp_config.py), pre-wired with
`with_benchmark_clarifications(...)` and `description_overrides` so a round can
apply both the curated overlay and an experimental action-wording change. Keep
`is_official=False`: hinter rounds are iteration, never submittable evaluations.
