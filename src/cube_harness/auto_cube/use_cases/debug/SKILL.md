# Auto-CUBE — debug use-case

You are running the **debug** use-case of Auto-CUBE. Auto-CUBE is the
iterate-and-fix outer loop; it owns the methodology, dispatches the
**Investigator** sub-agent per trajectory, and ships Fix Report PRs.
The debug use-case is for **hardening a cube against real LLMs**: find
what breaks, classify why, fix what's fixable, surface design rot.

If you're invoked as `/auto-cube` with no use-case suffix, you're
running the debug use-case (the default). Other Auto-CUBE use-cases
plug into the same outer-loop skeleton with different goal functions
and dispositions — but `debug` is the right choice when in doubt.

## Your posture: curious scientist

Each session is one chapter in a long-running investigation. You are
**not** trying to fix one task in isolation; you're building a
**sparse but informative coverage map** across the axes that matter:

  task × infra × tool × LLM provider × agent config

across many sessions. Each session picks the highest-value gap on
that map — an uncovered axis-point, a region with conflicting
results, a model × infra combination nobody has stressed yet — and
resolves it.

**Sparse, not dense.** Don't try the cross-product (combinatorial
explosion, wasteful budget). Sample enough on each axis to detect
axis-specific issues — "Daytona+swe broke but EAI+swe worked → likely
infra-specific" is the kind of signal you want. Skipping a cell is OK
if other cells exercise the same axis.

**The downstream goal across sessions:** enough coverage that a new
benchmark, a new model, or a new infra change can be slotted in with
confidence that systemic issues won't go undetected.

## Cross-session ledger

The source of truth across sessions is
**`~/auto_cube/coverage.json`** (loose schema; evolve as
you learn):

```json
{
  "<cube>|<task_id>|<model>|<agent_config>|<infra>|<tool>": {
    "outcome": "success | success_lucky | almost | failure | should_have_been_rewarded",
    "primary_blame": "<one of the 10 BlameCategory values, or 'none'>",
    "primary_blame_confidence": 0,
    "coverage_state": "covered | model-ceiling | zoom-in | shipped-fix",
    "last_session": "<session-slug>",
    "last_seen_utc": "2026-05-21T12:00:00Z",
    "finding_summary": "<one line from the Investigator's summary>"
  }
}
```

The first three fields mirror the Investigator's `BaseFindings`
directly (don't reinterpret). `coverage_state` is the orchestrator's
roll-up — see "Dispositions" below.

Read the ledger at **session start** to pick what to investigate
next. Update it as you classify cells. **"Done" = the cell is covered
well enough**, not that the task passes. A clean failure with
`primary_blame=model_capability` at confidence ≥ 4 is just as "done"
for this model as a pass.

## The loop

### 1. Session start

- **Pick a focus**, motivated by a gap in `coverage.json`. Often: "a
  specific cube against an under-tested infra/model" or "a known-fishy
  cluster of tasks that needs deeper investigation".
- **Set up the session worktree** off `origin/dev` with its own
  `.venv` (per §6 of
  [`openspec/specs/auto-fix/spec.md`](../../../../../openspec/specs/auto-fix/spec.md)).
  The session lives at `~/auto_cube/<session-id>/` — pick a unique
  session id (e.g. `swebench-verified-daytona-r0`) — holding both the
  journal (`session.md`, `round_<N>/`, `REPORT.md`) under `journal/`
  and all trajectory output under `experiments/`. **Export
  `CH_EXP_DIR=~/auto_cube/<session-id>/experiments`** at session start
  so every run in this session lands there instead of polluting
  `~/cube_harness_results/`.
- **Copy `session.md`** from
  [`src/cube_harness/auto_cube/templates/session.md`](../../templates/session.md)
  and fill in scope, target axes, ledger gaps you intend to fill.
- **Scan `~/cube_harness_results/` for reusable experiments** matching
  this cube. For any without an Investigator run (no `meta_analysis.{json,md}`
  next to the experiment dir), dispatch the Investigator now —
  that's free baseline data straight into the ledger. Skip experiments
  too old to be informative (use directory mtime + the `experiment_config.json`
  dump as fuzzy filters; git commit capture is a known gap in the
  framework).
- **Read `coverage.json`** and identify the uncovered or conflicting
  axis-points relevant to your focus.

### 2. Zoom out (broad-cheap)

The first scan of the session. Goal: classify a broad set of tasks
fast and cheaply, then narrow.

- **Default to a cheap model** (e.g. `claude-haiku-4-5`).
- **Single agent config, single infra** for zoom-out — breadth, not
  depth.
- **Broad task slice.** How broad depends on cube size and budget.
  Use judgment: ~20–100 tasks for a typical cube is the right order
  of magnitude, but adapt. Wider when the cube is large and cheap;
  narrower when each task burns budget.
- Experiments land under `~/auto_cube/<session-id>/experiments/`
  automatically (the `CH_EXP_DIR` exported at session start), so a
  session is self-contained — easy to archive or delete as one unit —
  and never pollutes `~/cube_harness_results/`.
- Run the experiment, then dispatch the Investigator on the output
  directory with **`--journal-dir ~/auto_cube/<session-id>/journal`**
  so its synthesis mirrors into the session. **Point `ch-investigate
  --context-dir` at the session dir** (`~/auto_cube/<session-id>/`) so the Opus codebase-map
  agent runs **once per (session, benchmark)** and the map is reused
  across all rounds — per-session keying keeps the map matched to this
  worktree's installed code. The Investigator emits `BaseFindings` per
  episode (canonical 10-category `primary_blame` + `outcome` +
  evidence — see "Dispositions" below).
- **Aggregate each episode into a coverage decision** (covered /
  model-ceiling done / zoom-in candidate — see "Dispositions"). Write
  the per-episode `BaseFindings` into `done.json` along with the
  derived coverage state, and update `coverage.json` for cells that
  are now decisively covered.

### 3. Zoom in (focused-deep)

Take only the **zoom-in-candidate** subset from zoom-out. Now you
can spend.

- Sweep the relevant axes — model × agent config × infra × tool
  variant — informed by the Investigator's `primary_blame` /
  3-bucket signal from zoom-out (agent-side blame → vary
  agent/model; tool-side → vary tool/scaffold; benchmark-side →
  Fix Report against the cube). Vary one axis at a time when
  possible so the next attribution is unambiguous.
- Dispatch the **`general_blame`** Investigator use_case by default
  (`ch-investigate <exp_dir>` — `general_blame` is the framework
  default). Switch via `--recipe <name>` only when a different
  blame ontology fits the round (e.g. `fix_audit` after a
  Fix Report PR). Each dispatch picks up `investigator_extra.md`
  from this directory as a biasing fragment; add round-specific
  bias via `--extra-prompt "..."` or
  `--extra-prompt @<path/to/fragment.md>`.
- Once a root cause is confirmed, follow **intervention discipline**
  (below) — hack to confirm, then ship the principled fix as a Fix
  Report PR.

### 4. Conclude

- **Update `coverage.json`** with newly-covered cells and freshly
  classified fails.
- **Update `done.json`** with this session's final dispositions.
- **Write `REPORT.md`** from
  [`templates/report.md`](../../templates/report.md): scope, the arc
  across rounds, findings ledger with dispositions, shipped vs open
  PRs, consolidated design signals, cost.
- For design-rot signals (a band-aid fix you'd want consolidated
  later), open `design-debt` issues per the spec.

A session is "done" when every (cube × axis-point) you intended to
cover has either landed in the ledger as covered or model-ceiling,
or has shipped a Fix Report for a confirmed agent/tool/benchmark
root cause; or when you've exhausted independent failure modes.
Don't over-iterate on the same five tasks — go broad first, then
deep where the signal is.

## Dispositions — from `BaseFindings` to coverage decisions

The Investigator's structured output **is** the per-episode report.
You don't invent new categories; you read `BaseFindings`
(`src/cube_harness/eval_log.py`) and aggregate. Schema:

- `outcome` ∈ `{success, success_lucky, almost, failure, should_have_been_rewarded}`
- `primary_blame` ∈ closed 10-category taxonomy (Appendix in the CUBE
  paper / `BlameCategory` enum): `task_unclear`, `model_capability`,
  `tool_failure`, `env_failure`, `agent_scaffolding`,
  `action_space_limited`, `insufficient_observation`, `eval_brittle`,
  `submission_format`, `none`
- `primary_blame_confidence` ∈ 0..5
- `evidence` — verbatim transcript quotes (required when
  `primary_blame != "none"`)
- `summary`, `hypothesis`, `hypothesis_confidence`

For coverage decisions, map the Investigator's output onto these
three orchestration buckets:

| Coverage state | Trigger | What to do |
|---|---|---|
| **covered** | `outcome ∈ {success, success_lucky, almost}` | Record cell as covered in `coverage.json`. Move on. |
| **model-ceiling done** | `outcome=failure` + `primary_blame=model_capability` + `confidence ≥ 4` | Record as done-for-this-model; a future session with a more capable model can revisit. |
| **zoom-in candidate** | anything else | Carry forward to zoom-in. Use the `primary_blame` category to decide which axis to vary first. |

For higher-level signal, use the paper's **3-bucket aggregation**:

- **agent-side**: `model_capability`, `agent_scaffolding` → vary
  agent config / model
- **tool-side**: `tool_failure`, `action_space_limited`,
  `insufficient_observation` → vary tool / scaffold
- **benchmark-side**: `task_unclear`, `env_failure`, `eval_brittle`,
  `submission_format` → likely a Fix Report against the cube; a
  high benchmark-side rate is also the wrapper-faithfulness signal
  the paper calls out

`evidence` and `hypothesis` are what you actually read to decide
*how* to zoom in. The Investigator already enforces evidence-grounded
attribution; trust the structured output, don't second-guess
categories without re-reading transcripts.

## Intervention discipline (auto-fix)

While **finding the root cause**, anything goes: hacky one-line
patches, print statements, throwaway side experiments, blunt hints
that mask a bug just to confirm the hypothesis. Speed of understanding
wins here.

Once the **root cause is confirmed**, the committed fix follows the
**auto-fix methodology** —
[`openspec/specs/auto-fix/spec.md`](../../../../../openspec/specs/auto-fix/spec.md).
In brief:

- **Classify** L0–L3 (local-correct → layer → symptom-of-design →
  Auto-CUBE / Investigator defect). Nothing blocks the loop: L2/L3
  still ship a temp PR + a kept-open `design-debt` issue.
- **Fix Report** is the PR body
  ([`templates/fix_report.md`](../../templates/fix_report.md)).
- **fix-audit** independently tries to break the Fix Report's
  generalization claims; reviewers read the verdict, not the diff.
- **Provenance**: `# auto-fix(N)↓ … # /auto-fix(N)` markers + a
  machine-readable footnote.
- **Multi-PR**: every PR branches from `dev` directly. The session's
  integration worktree is the test substrate.
- The diagnostic hack is reverted; only the principled fix is
  committed.

The diagnostic hack is a successful experiment, not a deliverable.
The deliverable is the right fix, its Fix Report, and the journal
entry that explains *why*.

## Templates

All in [`src/cube_harness/auto_cube/templates/`](../../templates/):

- [`session.md`](../../templates/session.md) — session scope + live tracker
- [`notes.md`](../../templates/notes.md) — per-round hypothesis → result
- [`exp_config.py`](../../templates/exp_config.py) — copy-and-edit experiment recipe. Keep `is_official=False`: Auto-CUBE runs are iteration, never submittable evaluations.
- [`fix_report.md`](../../templates/fix_report.md) — PR body for fixes
- [`report.md`](../../templates/report.md) — final REPORT.md rollup

Refine these as the methodology matures.
