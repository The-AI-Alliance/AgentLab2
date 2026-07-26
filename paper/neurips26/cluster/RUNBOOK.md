# Cluster runbook — CUBE NeurIPS 2026 rebuttal experiments

**Audience:** a coding agent with cluster access and a checkout of `cube-harness`.
**Deadline:** the rebuttal is due in days. Partial results beat no results; wrong
results are worse than none, because they go into a public rebuttal under our names.

---

## 0. What this is for

We submitted *"CUBE: A Unified Standard and Benchmark Suite for Evaluating
Generalist Agents"* to NeurIPS 2026 D&B. Three reviewers (scores 3 / 4 / 2) and the
metareview raised four asks. Two are marked **critical for publication**:

1. **Are the wrapped benchmarks faithful to the originals?** — partially answered.
   The *static* half is done and in the paper (task-set identity 368/368 and
   152/152 byte-identical instructions; evaluator AST-level semantic identity
   90.3% OSWorld / 74.5% WAA; residual diffs published). The *dynamic* half —
   does our evaluator actually behave like the original when you run it? — is
   **J1 and J2 below**. This is the single most important thing on this list.
2. **Is the LLM trajectory judge trustworthy?** — being handled off-cluster
   (human annotation + judge-backbone swap). Nothing for you here.
3. **Does the picture survive a change of scaffold?** — done (ReAct vs Genny on
   MiniWoB, in the paper).
4. **How does CUBE compare to Harbor / NeMo Gym / OpenEnv / …?** — done (comparison
   table in the paper).

Separately, reviewer 3's strongest objection is that **RL post-training is the
paper's stated motivation and is never demonstrated**, and that every number uses
closed-weight models. **J3 and J4 answer both at once** with a self-served
open-weight policy. That is the biggest available score-mover after J1/J2.

Paper source: `paper/neurips26/main.tex`. Per-reviewer drafts and a status
table: `paper/neurips26/REBUTTAL.md`.

---

## 1. Setup

```bash
git clone <cube-harness remote> && cd cube-harness

# The rebuttal work is NOT on dev. Use the branch:
git checkout paper/neurips26-rebuttal

# Fallback if that branch has not reached your remote — apply the bundle on dev:
#   git checkout dev && git apply /path/to/cluster-bundle.patch

make install                    # uv sync --all-extras
uv pip install -e cubes/swebench-verified-cube
```

The branch (equivalently, the bundle) carries everything the jobs below need
that is not on `dev`:

| Path | Why you need it |
|---|---|
| `cubes/swebench-verified-cube/.../task.py` | **records the agent's `model_patch` in `reward_info`** — without this, J2 is impossible |
| `scripts/paper/oracle_parity.py` | summarises J1 |
| `scripts/paper/swebench_cross_harness.py` | drives J2 |
| `scripts/paper/compare_runs.py` | task-level A/B with exact McNemar + Spearman |
| `recipes/paper/miniwob_scaffold_model_matrix.py` | J4 cells |
| `src/cube_harness/analyze/investigator/*`, `tests/` | judge fixes (not needed on-cluster; included for a clean tree) |

**Verify before running anything expensive:**

```bash
.venv/bin/python -c "from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmarkConfig as C; print(C(oracle_mode=True).oracle_mode)"
.venv/bin/pytest tests/test_investigator.py -q          # expect: 63 passed, 1 skipped
```

**Infra.** Recipes name an infra from `~/.cube/infra.py`; `"local"` (bare Docker on
the current host) always exists with no config. Start from
`recipes/infra_template.py`. Confirm which named infras resolve:

```bash
.venv/bin/python -c "from cube_harness.infra import INFRA_CONFIGS; print(list(INFRA_CONFIGS))"
```

**Ray launch gotcha (this will bite you).** Never use bare `uv run` when
`VIRTUAL_ENV` is set — uv silently builds an ephemeral env whose `.pth` files point
at deleted paths and Ray workers die with `ImportError`. Use `.venv/bin/python
<recipe>.py` or `uv run --active`. `exp_runner.py` warns when it detects this.

---

## 2. Guardrails

- **Never run repo-wide `make lint`.** It reformats the *vendored upstream
  evaluators* that `scripts/paper/parity_audit.py` measures, which silently
  invalidates the paper's parity numbers. Lint only files you author:
  `uvx ruff check --fix <file> && uvx ruff format <file>`.
- **Do not edit `main.tex`.** Hand back numbers and artifacts (§5); we patch the paper.
- **Never tune a benchmark to make a number look better.** For J1/J2 a
  *disagreement is the result*. Our evaluator makes three deliberate, documented
  relaxations against upstream (pre-existing `PASS_TO_PASS` failures are not
  charged to the agent; known network-dependent tests are skipped; pytest
  "no tests collected" is not a failure), so we **expect** a small,
  one-directional disagreement. Measuring its size and direction is the
  contribution. Hiding it would be misconduct.
- **Report attempted vs completed.** If 500 tasks are launched and 471 produce
  episodes, say so and say why. Never report a rate over a silently shrunk
  denominator.
- **Keep every run directory** and report its absolute path. We re-derive numbers from it.
- Do not `git push` or open PRs. Leave changes in the working tree and report them.

---

## 3. Jobs, in priority order

### J1 — SWE-bench Verified gold-patch oracle *(critical; Docker, no GPU; ~2–5 h)*

**Question:** if a benchmark ships its own answer key, does our wrapper award full
credit for it? A wrapper that fails its own gold patches is broken regardless of
what agents score. This is layer 3 of the paper's four-layer parity argument
(`\label{sec:parity}` in `main.tex`).

The oracle applies each task's gold patch with no LLM in the loop — free apart from compute.

```bash
# All 500 tasks. Pick your infra; scale --ray to your Docker capacity.
.venv/bin/python -m swebench_verified_cube.gold_patch.recipe --experiment local --ray 32

# Smoke first (3 tasks, in-process) — do this before the 500-task run:
.venv/bin/python -m swebench_verified_cube.gold_patch.recipe --experiment local --limit 3

# Summarise (pass several run dirs to separate stable from flaky resolutions):
.venv/bin/python scripts/paper/oracle_parity.py ~/cube_harness_results/gold-patch-verified-* \
    --expected 500 --label "SWE-bench Verified" --dump-solvable solvable_verified.json
```

**Reference magnitudes** (so you can sanity-check): on SWE-bench Live our oracle
resolved 275/300, and 223/300 on a non-root-capable infra — infra capability
moves this number a lot, which is itself a finding worth reporting.

**Ideally run it twice** (or pass two run dirs). A task that resolves only
sometimes is an *infrastructure* signal, not an evaluator signal, and
`oracle_parity.py` separates the stable set from the flaky one. If time allows
only one pass, say so.

**Report:** resolved/attempted, the stable-vs-flaky split, the failing instance
ids **with one line of diagnosis each** (patch didn't apply? tests errored?
container OOM? timeout?), the infra used, and whether containers had root.
Artifacts: `oracle_parity.tex`, `oracle_parity.json`, the solvable-id list, run dir paths.

---

### J2 — Cross-harness regrade *(critical; Docker, no GPU; ~3–6 h)*

**Question:** hold the *artefact* constant, vary the *grader*. Take patches our
agent produced, grade them with the **official upstream SWE-bench harness**, and
compare verdicts task by task. This is the sharpest parity test that exists for a
wrapped benchmark, and it is the number reviewers will weigh most.

**Precondition:** the episodes must carry `model_patch` in `reward_info`. That
capture is **new** (in the bundle patch) — *older runs on the cluster do not have
it*. So J2 needs a **fresh agent run**, or at minimum a fresh evaluate pass.

```bash
# 1. Agent run (edit the recipe for model/infra; it IS the config, not a CLI).
#    A 100-task subset is enough if a full 500 is too expensive — say which you used.
.venv/bin/python recipes/swe_agent_recipe.py --ray 32

# 2. Export the recorded patches in upstream predictions format
.venv/bin/python scripts/paper/swebench_cross_harness.py export <run_dir> --out preds.jsonl

# 3. Grade them upstream (either path)
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path preds.jsonl --run_id cube_parity --max_workers 16
#   or: sb-cli submit swe-bench_verified test --predictions_path preds.jsonl --run_id cube_parity

# 4. Compare verdicts
.venv/bin/python scripts/paper/swebench_cross_harness.py compare <run_dir> --upstream cube_parity.json
```

`compare` prints and writes: n graded by both, resolved-by-both, resolved-by-neither,
**CUBE-only (we were lenient)**, **upstream-only (we were strict)**, and overall
agreement. Given our three documented relaxations, expect a small **CUBE-only**
skew and near-zero upstream-only. If you see a large *upstream-only* count, that
is a real bug in our evaluator — stop and report it loudly rather than
massaging it; we would rather fix it than publish a wrong claim.

**Report:** the full `cross_harness.json`, the instance ids in both disagreement
directions, and for **each disagreeing instance a one-line cause** (which
relaxation explains it, or "unexplained"). Unexplained disagreements are the
most important thing you can hand back.

---

### J3 — Open-weight RL rollout through the rollout service *(GPU; ~4–8 h)*

**Question:** reviewer 3 says the RL motivation is never demonstrated. `cube-harness`
already ships a rollout service (`src/cube_harness/rl/`) that streams token-level
training records over HTTP/SSE against an OpenAI-compatible endpoint. Serving an
open-weight policy on your GPUs and driving real rollouts through it converts that
objection into a result — and simultaneously closes the "all models are
closed-weight" complaint.

**We are not claiming a trained policy.** We claim: the loop runs end-to-end
against a self-served open-weight model, at a measured throughput, emitting
token-ID training records a trainer can consume. Do not overclaim beyond that.

```bash
# 1. Serve the policy (any OpenAI-compatible server; vLLM assumed).
#    MiniWoB observations are html+screenshot, so a vision-capable model keeps the
#    tool stack identical to the paper's runs. Note the exact model + revision.
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8000 --tensor-parallel-size <N>

# 2. Point the harness at it
export CUBE_HARNESS_LLM_BASE_URL=http://localhost:8000/v1
export CUBE_HARNESS_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct
export CUBE_HARNESS_TOKENIZER_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct   # or a local path
export CUBE_HARNESS_LLM_API_KEY=EMPTY

# 3. Smoke the rollout path with no HTTP server in the way
.venv/bin/python recipes/rl/hello_miniwob_local.py \
    --task-ids form-sequence,click-button-sequence,click-checkboxes-large,email-inbox-star-reply

# 4. Full loop: service + mock trainer consuming SSE events
.venv/bin/python recipes/rl/hello_miniwob_service.py --num-rollouts 32 --num-groups 8
```

The run writes token-ID SFT records (one per trainable LLM call) to
`$CUBE_HARNESS_ROLLOUT_OUTPUT_DIR/training_examples.jsonl`.

**Measure and report** (these are the numbers that go in the paper):

- **episodes/hour** at your worker count, and the worker count and GPU count/type.
  Reference point: ~1,200 MiniWoB episodes/hour on an 8-core laptop with a hosted
  model — we need the cluster figure with a self-served policy.
- mean wall-clock per episode, and the split between env time and LLM time if the
  profiler is available (`ProfileConfig` on `RolloutConfig`).
- aggregate generation throughput (tokens/s) as reported by vLLM.
- number of training records emitted, and a sanity check that token IDs
  round-trip through the tokenizer.
- MiniWoB pass rate of the open-weight policy (this doubles as a J4 data point).
- **anything that broke.** A plumbing bug found here is worth reporting; we would
  rather fix it than have a reviewer find it.

---

### J4 — Open-weight MiniWoB matrix cells *(GPU, reuses J3's server; ~1 h)*

The paper's scaffold experiment (Genny 66.1% vs ReAct 69.4% on 125 MiniWoB tasks)
currently has open-weight cells missing: the OpenRouter key in `.env` returns 401.
With J3's vLLM endpoint already up, the same recipe fills them with a self-served
model instead — a strictly better answer, since we control the weights.

```bash
export CUBE_HARNESS_OPENWEIGHT_BASE_URL=http://localhost:8000/v1
export CUBE_HARNESS_OPENWEIGHT_MODEL=hosted_vllm/Qwen/Qwen3-VL-30B-A3B-Instruct

.venv/bin/python recipes/paper/miniwob_scaffold_model_matrix.py -e genny-qwenlocal --ray 8
.venv/bin/python recipes/paper/miniwob_scaffold_model_matrix.py -e react-qwenlocal --ray 8

.venv/bin/python scripts/paper/compare_runs.py <genny_run> <react_run> --label-a genny-qwenlocal --label-b react-qwenlocal
```

`compare_runs.py` reports both pass rates, per-task agreement, an **exact McNemar
test** on the discordant pairs, Spearman rank correlation across tasks, and the
discordant task list. Report all of it, including a non-significant p — the
paper's claim is that scaffold choice moves results, and an honest p is part of it.

**Report:** the two pass rates, n tasks, McNemar p, Spearman ρ, task agreement %,
and the exact model id + revision served.

---

### J5 — Terminal-Bench budget-lift ablation *(optional; no GPU; ~2 h)*

Reviewer 1 asks whether Terminal-Bench failures are a real capability ceiling or
just our cost cap. Same agent, same tasks, `max_cost_usd` raised (2.0 → 10.0 via
`--set max_cost_usd=10.0`), and report the pass-rate delta. If it barely moves,
the ceiling is real and that strengthens the paper. Only do this if J1–J4 are done.

---

## 4. Priority and budget

| Job | Blocks what | Needs | Rough time | Cost driver |
|---|---|---|---|---|
| J1 oracle | metareview #1 (critical) | Docker | 2–5 h | compute only, no LLM |
| J2 regrade | metareview #1 (critical) | Docker + agent run | 3–6 h | LLM for the agent run |
| J3 RL rollout | R3's main objection | 1 GPU node | 4–8 h | GPU hours |
| J4 open-weight cells | R3 secondary | J3's server | ~1 h | GPU hours |
| J5 budget lift | R1 minor | Docker | 2 h | LLM |

If you can only do one thing: **J1**. If two: **J1 + J2** — together they are the
whole dynamic parity argument. J3 is the highest-*upside* single job but does not
substitute for J1/J2.

---

## 5. Hand-back contract

Write `paper/neurips26/cluster/RESULTS.md` and include, per job attempted:

```markdown
## J<n> — <name>
Status: complete | partial | failed
Run dirs: <absolute paths>
Infra: <named infra, node type, worker count, root-capable containers y/n>
Model: <exact id + revision, if any>
Attempted: <n>   Completed: <n>   Reason for the gap: <...>

### Numbers
<the exact figures the job section asks for>

### What broke
<every failure mode you hit, even ones you worked around, with the fix you applied>

### Caveats
<anything that would make a reviewer distrust the number if they knew it>
```

Also hand back the raw artifacts, not just the summary:

- `oracle_parity.{tex,json}` + solvable-id list (J1)
- `cross_harness.{tex,json}` + `preds.jsonl` + the upstream report JSON (J2)
- `training_examples.jsonl` (head + line count is enough if it is large) and the
  rollout event log (J3)
- `compare_runs` stdout and both run dirs (J4)

The "What broke" and "Caveats" sections are not boilerplate. The reviewers who
gave this paper a 2 and a 3 did so because they judged its central claims
under-evidenced. The way we lose the rebuttal is by handing them a clean number
that turns out to have a hole in it.
