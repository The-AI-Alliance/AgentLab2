# RFC: Trajectory Judge — Post-Hoc Failure Analysis for Agent Episodes

**Status:** DRAFT  
**Author:** Alexandre Lacoste  
**Reviewer:** @NicolasAG  
**Date:** 2026-05-04

---

## Problem

cube-harness generates trajectories and measures reward, but produces no structured answer
to the most important question in agent research: *why* did a given episode fail?

Today, diagnosing a failure requires opening the trajectory in XRay, reading the full
conversation, and manually forming a hypothesis. At scale — dozens of runs, hundreds of
episodes — this is not tractable. Worse, human judgements are inconsistent across
reviewers and over time; the same trajectory reviewed twice may yield different root-cause
attributions.

Three concrete gaps this RFC addresses:

1. **No blame attribution.** Summary statistics (pass rate, cost, steps) tell you *how
   much* a system fails, but not where the budget to fix it should go — agent prompt,
   scaffolding design, environment brittleness, task ambiguity, or evaluation harness.

2. **No structured evidence.** Free-form researcher notes in journals or PRs rot. A
   trajectory-linked evidence record stays coherent as the codebase evolves.

3. **No hypothesis tracking.** Improvements are shipped without a testable hypothesis.
   If the hypothesis is recorded alongside the evidence, subsequent experiments can
   confirm or refute it quantitatively.

---

## Scope

- New module `cube_harness/analyze/judge.py` with a single public entry point.
- New Pydantic model `JudgeOutput` (V1 schema defined below).
- CLI runner and batch runner for post-hoc analysis of experiment output directories.
- Populate `judge_output` in `EpisodeRecord` (atlas-eval-log RFC, PR #297) when a judge
  has been run.
- No changes to the episode loop, trajectory format, storage protocol, or existing
  agent/benchmark contracts.

---

## Design

### Approach: LLM-as-judge via Claude Code Python API

The judge is a Claude Code agent invoked programmatically using the `claude` Python SDK.
It receives:

- **A path to the trajectory directory** — the judge reads files directly rather than
  having a serialized log injected into the prompt. This matters for large trajectories:
  observation dumps (axTree, DOM, screenshots) at individual steps can be fetched
  on-demand rather than pre-loaded into a single massive context.
- **A codebase map** — see [Codebase map](#codebase-map) below.
- **Optionally, paths to related trajectories** on the same task — enabling contrastive
  analysis ("this agent solved it in 12 steps, this one looped for 80 — why?").

The judge outputs a JSON object conforming to `JudgeOutput`. The structured fields enable
aggregation and statistical analysis; the free-form `analysis` field captures reasoning
that doesn't fit in a taxonomy.

### Why Claude Code specifically?

Claude Code's agentic loop with file-reading tools enables the judge to do things that
prompt-stuffing cannot:

- **Navigate large trajectory logs.** A trajectory directory contains per-step observation
  files (axTree snapshots, DOM dumps, screenshots). The judge can `cat` step 12's axTree
  and step 17's axTree side-by-side to see what changed — rather than receiving a
  pre-compressed summary.
- **Inspect specific screenshots.** For computer-use tasks, the judge can read the actual
  screenshot at the step where the agent got confused, not a description of it.
- **Cross-reference related trajectories.** Given paths to N related episodes, the judge
  can grep for specific patterns across all of them to identify systematic failure modes.
- **Read actual source files.** When attributing blame to the scaffolding or task
  description, the judge reads the real agent prompt or benchmark task file — not a
  summary injected by the caller. This is the primary hallucination-resistance mechanism:
  blame must be grounded in something the judge actually read.

### Codebase map

The judge needs enough context to distinguish a scaffolding failure from a model
capability failure. Two designs were considered:

**(a) Per-cube `codebase_map.json` produced by a claude-code skill.** The map records
primary source files (agent prompt template, tool definitions, benchmark task
description, reward function), key symbols to grep for, and a pointer to the
git-cloned cube source (preferred over pip-installed wheels, which strip context).
Triggered once per (cube, agent config) pair and cached alongside the experiment
directory.

**(b) Resolve source paths from the venv at judge time** via `importlib.util.find_spec`.
The judge reads `experiment_config.json`, extracts the dotted `_type` for the agent
config, benchmark config, and infra config, and resolves each to a directory on
disk against whatever's installed in the judge's venv. Packages that aren't
importable (e.g. an old run referenced `cube_harness.agents.genny2` which has since
been renamed, or an agent class defined in a `__main__` script) are silently
skipped — the judge falls back to whatever resolved successfully plus the
trajectory transcript itself.

V1 ships **(b)** because it works for any experiment without per-cube setup, and
because in practice the `_type` strings in `experiment_config.json` are sufficient
to point the judge at the right packages. The map (a) remains a useful follow-up
for cubes where the on-disk source is too large for naive grep — at that point the
map adds curated entry points on top of (b)'s resolved roots, rather than
replacing them.

### Hallucination resistance

- Evidence is required: `evidence` must quote specific steps and transcript excerpts when
  `primary_blame != "none"`.
- Blame categories are closed-world: the judge picks from a fixed taxonomy and must
  assign `none` rather than inventing a plausible-sounding cause.
- Confidence is explicit: `primary_blame_confidence` and `hypothesis_confidence` are 0–5
  scores, forcing the model to express uncertainty rather than hide it.
- The `analysis` field is placed **first** in the output schema and acts as a scratchpad:
  the judge reasons through the evidence before committing to the structured fields.

---

## V1 Schema: `JudgeOutput`

Fields are ordered to follow the judge's reasoning process: free-form thinking first,
then structured conclusions.

```python
class BlameCategory(str, Enum):
    task_unclear            = "task_unclear"
    model_capability        = "model_capability"
    tool_failure            = "tool_failure"
    env_failure             = "env_failure"
    agent_scaffolding       = "agent_scaffolding"
    action_space_limited    = "action_space_limited"
    insufficient_observation = "insufficient_observation"
    eval_brittle            = "eval_brittle"
    submission_format       = "submission_format"
    none                    = "none"

class Outcome(str, Enum):
    success                  = "success"
    success_lucky            = "success_lucky"
    almost                   = "almost"
    failure                  = "failure"
    should_have_been_rewarded = "should_have_been_rewarded"

class EvidenceItem(TypedBaseModel):
    step: int     # Step index in the trajectory
    quote: str    # Verbatim excerpt from agent or environment output

class JudgeOutput(TypedBaseModel):
    # Reasoning scratchpad — filled first; grounds all structured fields below
    analysis: str

    # What happened
    outcome: Outcome
    summary: str              # 1–3 sentences

    # Why it happened
    primary_blame: BlameCategory
    primary_blame_confidence: int   # 0 (pure guess) – 5 (certain)
    other_blames: list[BlameCategory] = []
    evidence: list[EvidenceItem]

    # What would fix it
    hypothesis: str           # 1–2 sentences
    hypothesis_confidence: int  # 0 (pure guess) – 5 (certain)
```

### Outcome taxonomy

| Outcome | Meaning |
|---|---|
| `success` | Agent solved the task correctly. |
| `success_lucky` | Task marked as solved but agent reached it by accident or via a wrong approach. Worth less than a clean success as a training signal. |
| `almost` | Agent clearly understood the task and made meaningful progress; failed on a minor technical detail. Worth more than `success_lucky` as a quality signal — the agent's strategy was sound. |
| `failure` | Task not solved. |
| `should_have_been_rewarded` | Agent did what the task asked, but was not rewarded — because the task description was ambiguous, the ground truth was stale, or the evaluation function was too brittle to accept a valid solution. Pairs naturally with `eval_brittle` or `task_unclear` as the primary blame. |

### Blame taxonomy

| Category | Use when |
|---|---|
| `task_unclear` | The task description is ambiguous, contradictory, or missing necessary context. |
| `model_capability` | The agent understood the task but lacked the reasoning ability, domain knowledge, or multi-step planning to solve it. |
| `tool_failure` | A tool raised an exception or returned an unexpected error — a bug or limitation in the tool wrapper itself (e.g. a bash tool that truncates output, a browser tool that crashes on a specific element). |
| `env_failure` | The underlying environment or infrastructure failed outside the agent's and tool's control: container crash, network timeout, VM restart, port binding failure. |
| `agent_scaffolding` | The agent loop, system prompt design, budget limits, context window management, or submission protocol caused the failure — not the underlying LLM capability. |
| `action_space_limited` | The agent could not complete the task because a required action does not exist in its action space. A correct solution is impossible with the current tool set. |
| `insufficient_observation` | The observation presented to the LLM was missing crucial information needed to take the right decision — e.g. a pruned axTree that hid the target element, a truncated tool output, or a screenshot at too low a resolution. |
| `eval_brittle` | The agent produced a correct or acceptable solution but the evaluator rejected it (e.g. wrong whitespace, order-sensitive string match, stale ground truth). |
| `submission_format` | The agent reached a correct solution but failed to submit it through the required channel (e.g. never called `final_step`, submitted to the wrong tool). |
| `none` | Assign on clean success, or when the episode is too ambiguous to assign a blame without speculation. |

**Multi-blame:** `primary_blame` is the dominant cause. `other_blames` captures secondary
contributing factors. `submission_format` and `agent_scaffolding` frequently co-occur:
the agent didn't submit because the prompt never mentioned the submission tool.

### Confidence score (0–5)

| Score | Meaning |
|---|---|
| 5 | Certain — the evidence is unambiguous and directly supports the attribution. |
| 4 | High — strong evidence, one minor alternative interpretation. |
| 3 | Medium — plausible reading of the evidence; another interpretation is credible. |
| 2 | Low — the attribution is a best guess; evidence is thin. |
| 1 | Very low — mostly speculation. |
| 0 | No basis — the judge cannot form a coherent attribution. |

---

## Implementation

### Module layout

```
src/cube_harness/analyze/
├── xray.py              (existing)
├── inspect_results.py   (existing)
├── xray_utils.py        (existing)
└── judge.py             ← NEW
```

### Public API

```python
def judge_episode(
    episode_dir: Path,
    *,
    experiment_dir: Path | None = None,
    model: str = "claude-opus-4-7",
    verbose: bool = False,
) -> tuple[JudgeOutput, JudgeMetadata]:
    """Run a post-hoc judge on a single episode trajectory directory.

    `experiment_dir` defaults to `episode_dir.parent.parent`. Source paths
    (cube package, agent package, cube-harness, cube-standard) are resolved
    via `importlib.util.find_spec` against the current venv — see
    "Codebase map" below.

    Returns both the judgment and its billing/provenance metadata.
    """
    ...

def judge_experiment(
    experiment_dir: Path,
    *,
    model: str = "claude-opus-4-7",
    ids: list[str] | None = None,
    sample: float | None = None,
    n: int | None = None,
    failures_only: bool = False,
    overwrite: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> dict[str, tuple[JudgeOutput, JudgeMetadata]]:
    """Batch judge selected episodes in an experiment output directory.

    Selection: `ids` is an explicit override (returned verbatim, ignoring
    other filters). Otherwise the pool is filtered by `failures_only` and
    already-judged status (skipped unless `overwrite=True`), then narrowed
    by `sample` (random fraction) or `n` (random count). With no selector
    the default is `sample=0.10` (set by the CLI).

    Writes judge_output and judge_metadata into each episode_record.json.
    Writes experiment_judge_summary.json with aggregate cost.
    Returns a mapping from trajectory_id to (JudgeOutput, JudgeMetadata).
    """
    ...
```

The CLI (`ch-judge` / `python -m cube_harness.analyze.judge`) wires these
arguments through and adds `--summary` (aggregate table to stdout) and
`--verbose` (stream per-tool-call progress to stderr while the judge runs).
Parallel execution (`n_parallel`) is **not** in V1 — `judge_experiment` is
sequential — and is tracked as a follow-up.

### Prompt structure

The judge prompt is a Python string constant in `judge.py`, assembled with `.format()`.
It contains three sections:

1. **Context** — paths to relevant source files from the codebase map, task description.
2. **Trajectory pointer** — the path to the trajectory directory; the judge uses file
   tools to navigate step files, fetch observations at specific steps, and read
   screenshots as needed.
3. **Instructions** — taxonomy definitions, output schema, evidence requirements,
   confidence calibration guidelines. The judge is instructed to write `analysis` first
   as a scratchpad before filling the structured fields.

The judge is asked to return a single JSON block; the response is parsed with
`json.loads` with fallback to a regex extractor for common LLM wrapping patterns.

### Judge cost and token tracking

Each judge invocation has its own LLM cost, which must be tracked separately from the
agent episode's cost — they are different LLM calls, run at different times, and billed
to different purposes. The runner reports this to the caller and persists it alongside
the judgment.

**Where to store it.** Three options were considered:

| Option | Verdict |
|---|---|
| Fields inside `JudgeOutput` | ✗ — mixes billing metadata into the judgment schema; `JudgeOutput` should be a pure analytical record, not a receipt |
| Separate `judge_record.json` sidecar per episode | ✗ — proliferates files; the existing `episode_record.json` already aggregates per-episode metadata |
| Sibling `judge_metadata` field in `episode_record.json` | ✓ — mirrors how the agent's own `usage` field sits alongside agent output; one file per episode, clean separation between content and provenance |

**Schema.** A `JudgeMetadata` model is written alongside `judge_output` in
`episode_record.json`:

```python
class JudgeMetadata(TypedBaseModel):
    model: str                  # e.g. "claude-opus-4-7"
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_s: float
    timestamp: float            # wall-clock time the judge ran (Unix)
    judge_schema_version: str   # e.g. "v1" — for forward-compatibility of stored records
```

**`episode_record.json` with both fields populated:**

```jsonc
{
  "judge_output": {
    "analysis": "The agent located the correct file at step 8 and produced a valid patch...",
    "outcome": "should_have_been_rewarded",
    "summary": "Agent produced a valid patch but the evaluator rejected it due to trailing whitespace.",
    "primary_blame": "eval_brittle",
    "primary_blame_confidence": 4,
    "other_blames": [],
    "evidence": [{"step": 42, "quote": "diff --git a/..."}],
    "hypothesis": "Normalizing trailing whitespace in the evaluator would fix this class of rejection.",
    "hypothesis_confidence": 4
  },
  "judge_metadata": {
    "model": "claude-opus-4-7",
    "prompt_tokens": 18400,
    "completion_tokens": 1240,
    "cost_usd": 0.087,
    "duration_s": 34.2,
    "timestamp": 1746400000.0,
    "judge_schema_version": "v1"
  }
}
```

**Aggregate cost in `experiment_judge_summary.json`.** `judge_experiment()` also writes
a per-experiment summary file that aggregates costs across all judged episodes:

```jsonc
{
  "n_judged": 50,
  "total_judge_cost_usd": 4.35,
  "avg_judge_cost_usd": 0.087,
  "total_judge_prompt_tokens": 920000,
  "total_judge_completion_tokens": 62000,
  "model": "claude-opus-4-7",
  "judge_schema_version": "v1",
  "timestamp": 1746400000.0
}
```

**CLI output.** `--summary` prints the aggregate cost alongside blame distribution:

```
Judged 50/50 episodes  |  total judge cost: $4.35  |  avg: $0.087/ep
Outcomes:  success=28  almost=4  failure=16  should_have_been_rewarded=2
Primary blame:  model_capability=11  agent_scaffolding=9  eval_brittle=3  ...
```

### CLI

```bash
# Judge a single experiment
uv run python -m cube_harness.analyze.judge path/to/output_dir

# With model override
uv run python -m cube_harness.analyze.judge path/to/output_dir --model claude-sonnet-4-6

# Print aggregated blame counts and outcome distribution
uv run python -m cube_harness.analyze.judge path/to/output_dir --summary
```

---

## Downstream goals

1. **Automated improvement loop.** The meta-agent (`meta_agent/`) can consume
   `JudgeOutput.hypothesis` from a batch run as structured input to generate targeted
   agent config changes — closing the eval → analyse → fix loop without human
   transcription.

2. **Benchmark-level diagnostics.** Aggregating `primary_blame` distributions across a
   run identifies systematic weaknesses: "60% of failures on this benchmark are
   `insufficient_observation` — fix the observation pipeline before tuning the agent."

3. **Cross-run hypothesis tracking.** If `hypothesis` and `hypothesis_confidence` are
   persisted and linked to the experiment that tested the fix, the improvement is
   quantifiable: `H: adding anti-loop instruction → +6pp` becomes a stored,
   reproducible record.

---

## Alternatives considered

**Human annotation pipeline.** Accurate but doesn't scale; two annotators disagree ~20%
of the time on failure attribution. LLM judgement at scale is less accurate per-instance
but provides consistent, reproducible signals across runs.

**Rule-based classifiers.** Too brittle: failure mode boundaries are semantic, not
syntactic. "Agent looped" can be `model_capability` (couldn't find the fix) or
`agent_scaffolding` (no anti-loop instruction) depending on context.

**Separate judge per benchmark.** Benchmark-specific prompts would be more accurate for
known tasks but require ongoing maintenance and don't generalize. The taxonomy is
benchmark-agnostic; benchmark-specific context is injected via the codebase map and task
description rather than hardcoded.

**Embedding-based clustering.** Useful as a complement for large-scale pattern discovery,
but doesn't produce the structured blame + hypothesis needed for the improvement loop.

---

## Open questions

1. How many related trajectory directories to pass before the judge's navigation overhead
   exceeds the signal from contrastive analysis?
2. Confidence calibration: should scores 0–1 be treated as `none` for aggregation
   purposes?
3. **Schema duplication.** The taxonomy and output schema currently live in four
   places: this proposal, `deltas.md`, the `JUDGE_SYSTEM_PROMPT` /
   `JUDGE_USER_PROMPT_TEMPLATE` constants in `analyze/judge.py`, and the
   `.claude/commands/judge-traces.md` slash command. They are aligned in V1 but will
   drift the moment the taxonomy evolves. Post-V1, factor the prompt content into a
   single source — most likely a Markdown asset under `openspec/specs/` that the
   Python prompt template `read_text()`s and the slash command includes. Until then,
   any change to the taxonomy must update all four sites.

### Resolved

- ~~`judge_experiment()` summary output~~ → V1 writes both per-episode
  `judge_output` + `judge_metadata` into `episode_record.json` AND an aggregate
  `experiment_judge_summary.json`.
- ~~Codebase map: pinned git ref vs HEAD~~ → V1 sidesteps the question: source
  paths are resolved at judge time via `importlib.util.find_spec` against the
  judge's venv, with a graceful skip when the package isn't installed. A curated
  per-cube map remains a useful follow-up but is not required for V1.

---

## Simulation: 6 real episodes judged by sub-agents

To validate the schema and uncover design gaps, we ran the judge as sub-agents on 6
real trajectories from a haiku-4-5 / thought-workflow / HAL-50 run
(`20260503_231245`). Each sub-agent read the trajectory log from disk and produced a
`JudgeOutput` JSON without access to the ground-truth evaluator internals.

### Results

**Episode 1 — `django__django-9296` · reward=1.0 · 110 steps**

```json
{
  "outcome": "success",
  "summary": "Agent correctly implemented Paginator.__iter__. After ~25 wasted steps finding the test runner invocation (pytest not installed, no manage.py), the fix was clean and all tests passed.",
  "primary_blame": "none",
  "primary_blame_confidence": 5,
  "other_blames": ["agent_scaffolding"],
  "evidence": [
    {"step": 17, "quote": "python -m pytest tests/pagination/tests.py\nNo module named pytest"},
    {"step": 63, "quote": "def __iter__(self):\n    for page_num in self.page_range:\n        yield self.page(page_num)"}
  ],
  "hypothesis": "none",
  "hypothesis_confidence": 5
}
```

**Episode 2 — `django__django-11790` · reward=0.0 · 230 steps**

```json
{
  "outcome": "failure",
  "summary": "Agent diagnosed and fixed the AuthenticationForm maxlength regression. All 79 tests passed and final_step was called — but reward=0.0. Evaluator likely expected a simpler one-line patch.",
  "primary_blame": "eval_brittle",
  "primary_blame_confidence": 3,
  "other_blames": ["model_capability"],
  "evidence": [
    {"step": 206, "quote": "Ran 79 tests in 0.122s\n\nOK"},
    {"step": 228, "quote": "[228 DONE] reward=0.0"}
  ],
  "hypothesis": "Evaluator expected widget.attrs['maxlength'] = ... directly; agent's fix called widget_attrs() which re-sets other attributes, producing a different patch shape.",
  "hypothesis_confidence": 3
}
```

**Episode 3 — `sphinx-doc__sphinx-7590` · reward=0.0 · 482 steps**

```json
{
  "outcome": "failure",
  "summary": "Agent identified the C++ UDL parsing bug, then entered a ~350-step loop trying to count characters on a source line via an ever-growing awk/sed/xargs pipeline. Never made a code change.",
  "primary_blame": "model_capability",
  "primary_blame_confidence": 5,
  "other_blames": ["agent_scaffolding"],
  "evidence": [
    {"step": 151, "quote": "awk '...' | od -c | tail -1 | awk '{print $1}' | xargs printf '%d\\n' | xargs -I {} echo \"Line 4667 has {} characters\""},
    {"step": 152, "quote": "Line 4667 has 67 characters"},
    {"step": 391, "quote": "[same pipeline + additional xargs stages, still returns 67]"}
  ],
  "hypothesis": "A scaffold-level guard detecting N consecutive near-identical commands would interrupt this loop. The anti-loop prompt clause did not trigger because the commands always returned a result — just always the same one.",
  "hypothesis_confidence": 4
}
```

**Episode 4 — `django__django-12262` · reward=0.0 · 76 steps**

```json
{
  "outcome": "failure",
  "summary": "Agent correctly identified the bug AND the exact fix (change `param not in unhandled_kwargs` to `param not in kwonly`) but never wrote it to any file. It looped through analysis scripts for ~30 steps then the budget ran out.",
  "primary_blame": "model_capability",
  "primary_blame_confidence": 5,
  "other_blames": ["agent_scaffolding"],
  "evidence": [
    {"step": 54, "quote": "The fix should check:\n  param not in kwonly: False\n  Overall: False"},
    {"step": 73, "quote": "[073 BASH id=toolu_013xZN]\n\n\n[074 DONE] reward=0.0"}
  ],
  "hypothesis": "Explore-to-fix phase transition failed: agent simulated the fix in a Python script instead of writing it to the source file. Explicit scaffolding ('once confirmed in a script, apply to the file immediately') would help.",
  "hypothesis_confidence": 5
}
```

**Episode 5 — `sphinx-doc__sphinx-10435` · reward=0.0 · 106 steps**

```json
{
  "outcome": "failure",
  "summary": "Agent confirmed the failing test and the newline symptom, then entered a ~60-step loop analyzing a hardcoded LaTeX string instead of opening latex.py. No source file was ever modified.",
  "primary_blame": "model_capability",
  "primary_blame_confidence": 5,
  "other_blames": ["agent_scaffolding"],
  "evidence": [
    {"step": 40, "quote": "tests/test_build_latex.py::test_latex_code_role FAILED"},
    {"step": 58, "quote": "0: '\\n'\n1: '\\\\'\n2: 'P'"},
    {"step": 103, "quote": "[103 BASH id=toolu_012kqY]\n\n\n[104 DONE] reward=0.0"}
  ],
  "hypothesis": "A codebase map pointing to sphinx/writers/latex.py:visit_literal as the entry point would have shortcut the exploration failure that led to the loop.",
  "hypothesis_confidence": 4
}
```

**Episode 6 — `django__django-12304` · reward=1.0 · 406 steps**

```json
{
  "outcome": "success",
  "summary": "Agent applied the correct one-line fix (do_not_call_in_templates = True on ChoicesMeta) at step 95 and verified it immediately. Then wasted ~300 steps: ~140 on irrelevant git history and ~50+ repetitions of the same git diff check before calling final_step at step 403.",
  "primary_blame": "agent_scaffolding",
  "primary_blame_confidence": 4,
  "other_blames": ["model_capability"],
  "evidence": [
    {"step": 95, "quote": "class ChoicesMeta(enum.EnumMeta):\n    do_not_call_in_templates = True"},
    {"step": 97, "quote": "Result: 'FR'\nExpected: 'FR'\nStatus: PASS"},
    {"step": 395, "quote": "cd /testbed && git diff | grep -E \"^diff\""}
  ],
  "hypothesis": "Submission instructions should say 'call final_step immediately after confirming the patch' — the current wording leaves the agent free to verify indefinitely.",
  "hypothesis_confidence": 4
}
```

---

### Interpretation and design implications

**Dominant failure mode: degenerate loops.** 5 of 6 episodes contained a loop — either
in the explore phase (episodes 3, 4, 5), the verify phase (episode 6), or both.
Crucially, all of these loops violated the spirit of the anti-loop prompt clause ("avoid
retrying if no apparent effect") without triggering it, because the commands *did*
return output — just always the same output. The prompt clause only guards against
actions with no result; it does not guard against actions with a stale, non-progressing
result.

**Two distinct loop subtypes identified:**

| Subtype | Description | Episodes |
|---|---|---|
| *Exploration lock* | Agent knows the symptom but loops on diagnostics instead of transitioning to writing a fix | 3, 4, 5 |
| *Verification lock* | Agent has the fix but loops on `git diff` / patch confirmation instead of calling `final_step` | 1 (minor), 6 (severe) |

Both subtypes are invisible to the current scaffolding. A scaffold-level identical-command
detector (independent of the LLM) would catch both.

**`eval_brittle` is hard to judge without evaluator access.** Episode 2 produced the
only uncertain attribution (confidence 3): all tests passed, final_step was called,
reward=0.0. The judge correctly suspected `eval_brittle` but could not confirm without
seeing the expected patch or the evaluator's `evaluate()` method. The codebase map
should include a pointer to the evaluator source so the judge can reason about whether
a valid solution was unfairly rejected.

**Successes reveal inefficiency that reward alone misses.** Episodes 1 and 6 both
succeeded but wasted 25 and 300 steps respectively. The blame taxonomy captures this via
`agent_scaffolding` as secondary blame, and the `hypothesis` field surfaces actionable
fixes (test-runner in codebase map; tightened submission instruction). The judge is
therefore useful on *successes* too — not just failures.

**Phase-transition failures are a distinct pattern.** Episodes 4 and 5 share a specific
failure mode: the agent diagnosed the problem correctly but never transitioned from
analysis to editing a file. This is distinct from `model_capability` in the general sense
— the agent had the right answer but couldn't act on it. A future `stuck_phase` field
(values: `reproduce`, `explore`, `fix`, `verify`, `submit`) would make this pattern
queryable across runs.

**New design ideas from this simulation:**

1. **Scaffold-level loop detector as a pre-judge input.** Run a cheap heuristic before
   the LLM judge: count consecutive near-identical tool calls (edit distance < threshold)
   and report the longest run, its start step, and the repeated command. This gives the
   judge structured evidence to quote rather than having it re-discover the loop by
   reading the full log.

2. **Evaluator source in codebase map.** For `eval_brittle` attribution to be
   high-confidence, the judge needs to see how the evaluator decides reward. The codebase
   map should include the path to the benchmark's `evaluate()` method.

3. **`stuck_phase` field (V2).** Values: `reproduce | explore | fix | verify | submit`.
   Captures the explore-to-fix and verify-to-submit transition failures as a first-class
   queryable field, enabling aggregate queries like "what fraction of failures got stuck
   in verify?".

4. **`steps_before_fix` and `steps_after_fix` (V2).** For successful episodes, these
   measure scaffold efficiency independently of reward. Episodes 1 and 6 both have
   reward=1.0 but very different efficiency profiles.

5. **Prompt wording for submission.** Episode 6's 300-step verification loop traces
   directly to the submission instruction wording. The instruction "verify with
   `git diff > patch.txt && cat patch.txt`, then call `final_step`" should be sharpened
   to "call `final_step` immediately after the patch looks correct — do not re-verify."
