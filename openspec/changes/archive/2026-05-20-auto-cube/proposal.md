# RFC: Auto-CUBE — Use-Case-Driven Investigator + Coding-Agent Drivers

**Status:** DRAFT (revision 4 — implementation-ready, folds in design feedback)
**Author:** Alexandre Lacoste
**Date:** 2026-05-13

**Companions:** `trajectory-investigator` (PR #366, the investigator this builds on), `agent-owns-loop` (PR #386, orthogonal).

---

## Problem

The trajectory investigator (PR #366) is monolithic: one prompt, one model, one tool surface, one transport (`claude-agent-sdk`), one output schema. The meta-agent has no typed contract with it. Auto-CUBE adds the seams to:

- Run the investigator **per use case** (general blame vs profiling vs scaffolding) without forking the investigator module.
- Run the investigator **without an API key** (subscription holders) or on top of Codex.
- Inject **experiment-specific context** automatically via a sub-agent.
- Get a **self-quality signal** per judgment (audit pass).
- Get a **cross-experiment view** the outer loop can grep.

Post-hoc only — episode loop and storage formats are untouched.

---

## Design

### 1. One directory per use case

```
src/cube_harness/analyze/investigator/use_cases/<name>/
├── __init__.py
├── recipe.py         # exports RECIPE: InvestigatorRecipe (prompts inline, OutputModel inline)
├── SKILL.md          # meta-agent skill description
└── scripts/          # optional helpers (e.g. profiling/summarise_profile.py)
```

`use_cases/__init__.py` walks subdirectories on import and assembles `RECIPE_CATALOG: dict[str, InvestigatorRecipe]`. Adding a use case is a one-directory PR.

Initial catalog: **`general_blame`**, **`profiling`**, **`agent_scaffolding`**.

**Why `recipe.py` and not `recipe.json`:** constitution PS-Python-is-config. The recipe owns its `OutputModel` (a type, not data) and its prompts (long strings best edited as Python, not embedded JSON). `.py` gives Go-to-Definition. The runtime *records* which recipe was used in `investigation_metadata.json` — that's the JSON side; recipes themselves stay Python.

### 2. `InvestigatorRecipe` shape

```python
class InvestigatorRecipe(TypedBaseModel):
    name: str
    system_prompt: str          # describes the use case + ontology + output schema
    user_prompt_template: str
    output_model: type[TypedBaseModel]  # per-recipe output schema
    model: str = "claude-sonnet-4-6"
    allowed_tools: tuple[str, ...] = ("Read", "Glob", "Grep", "Bash")
    permission_mode: Literal["bypassPermissions", "ask"] = "bypassPermissions"
    audit: bool = False         # run the audit pass
```

**Each recipe owns its output schema.** `general_blame.OutputModel` is the current `Findings`; `agent_scaffolding.OutputModel` extends it with scaffold-specific fields; `profiling.OutputModel` is narrower. Aggregation across recipes works on the intersection (`primary_blame`, `outcome`, `confidence`) declared as a base class all `OutputModel`s extend.

**No `driver` or `selector` field on the recipe.** Both are call-time arguments — choice of transport is orthogonal to what the investigator is asked to do.

### 3. `AgentDriver` Protocol — our wrapper, not LiteLLM

LiteLLM is a model gateway, not a coding-agent abstraction. To swap Claude SDK ↔ terminal `claude -p` ↔ `codex exec`, we need our own thin Protocol:

```python
class AgentDriver(Protocol):
    name: str
    max_parallelism: int
    async def run(...) -> DriverResult
    async def continue_session(...) -> DriverResult   # for audit pass
```

| Driver | Auth | Parallelism per host | Cost reported |
|---|---|---|---|
| `claude-code-sdk` | API key | ~8 | yes |
| `claude-terminal` | subscription (`claude -p`) | ~2 | no |
| `codex-cli` | OpenAI key (follow-up) | ~3 | yes |

Drivers are call-time arguments. Underneath, model calls route through LiteLLM when `LITELLM_PROXY_URL` is set (PS-002 satisfied at the token-routing layer).

### 4. Benchmark-context sub-agent — automatic context-file generation

When `investigate_experiment` runs and `<experiment_dir>/investigation_context.md` is missing, it spawns a **benchmark-context-agent** (Opus by default, runs through the same `AgentDriver`) that reads `experiment_config.json`, walks the relevant source trees, and writes the file. Output format is the existing `\`\`\`paths` fenced block already understood by `validate_context_file`.

The investigator then calls `validate_context_file(path)` — extracts paths via the existing `_PATHS_FENCE_RE` regex, verifies each path exists locally, raises `FileNotFoundError` on any miss. Validation is cheap and catches the failure mode where the sub-agent hallucinates a path; the audit pass catches deeper context-quality issues.

Replaces the old `collect_source_paths` venv-walk heuristic. `ch-investigate init-context <experiment_dir>` becomes a thin CLI wrapper around the same sub-agent for ad-hoc bootstrap.

### 5. Audit pass — single unified self-evaluation

Replaces both `self_judge` and the previous separate `post_judge_survey`. One pass, one file, one cost overhead.

When `--audit` is set (or `recipe.audit=True`), the investigator issues `driver.continue_session(...)` after the primary judgment with an audit prompt. Output: `audit.json` with:

- `verdict`: `sound` / `questionable` / `wrong`
- `reasoning_quality`: 0-5
- `ease_of_analysis`: 0-5 (was: survey)
- `context_quality`: 0-5 (was: survey)
- `tooling_gaps`: list[str] (was: survey)
- `missed_evidence`: list[str]
- `alternative_blames`: list[{blame, rationale}]
- `notes`: str | None

Off by default; ~25-30% cost overhead when on. Drivers without `continue_session` fall back to a fresh `run` with the prior judgment in the prompt — same `audit.json` output.

### 6. Batch outputs — both CSV and JSON

After `investigate_experiment` finishes, two aggregate files are written next to the experiment dir:

- `experiment_investigation_report.csv` (already in PR #366) — flat, one row per episode, for grep/spreadsheet.
- `experiment_investigation_report.json` (new) — preserves the per-recipe `OutputModel` shape for downstream consumers that need typed access.

### 7. Cross-investigator agreement + joint CSV — runners ship in this PR

The user explicitly asked these be in this PR rather than deferred.

- **Cross-investigator runner.** `investigate_experiment(..., n_seeds=N)` re-investigates each selected episode N times with the **same recipe** at varying seeds. Writes `cross_investigation_agreement.csv` keyed by `(trajectory_id, recipe)` with modal blame + agreement fraction across seeds. Cross-recipe comparison stays out of scope (well-defined only on shared base fields; not the most useful signal).
- **Joint CSV runner.** `cube_harness.analyze.cross_experiment.write_joint_csv(sweep_dir)` walks a sweep directory, reads each experiment's `experiment_investigation_report.csv` plus its `cross_investigation_agreement.csv` if present, writes one row per `(experiment, episode)` to `<sweep_dir>/joint_investigation_report.csv`. Columns include `experiment_id`, `family_id`, `agent_dotted`, `benchmark_dotted`, `driver`, `recipe`, `litellm_proxy_url`, plus the per-experiment columns and joined cross-investigator columns.

### 8. Investigator script = parallel runner, meta-agent passes a subset

`investigate_experiment` already supports `n_parallel`, `ids`, `sample`, `n`, `failures_only` via PR #366. The meta-agent uses these to scope a investigating run; default is "all episodes, single recipe, n_parallel=driver.max_parallelism." Auto-CUBE adds `recipe`, `driver`, `selector`, `audit`, `n_seeds` — no other API change.

---

## Outputs (full list)

Per episode (`<episode_dir>/`):
- `findings.json` (PR #366), `investigation_trace.json` (PR #366)
- `audit.json` — when `--audit` is set

Per experiment (`<experiment_dir>/`):
- `investigation_context.md` — auto-generated by benchmark-context-agent if missing
- `experiment_investigation_report.csv` (PR #366)
- `experiment_investigation_report.json` (new — typed-shape mirror)
- `cross_investigation_agreement.csv` — when `n_seeds > 1`

Per sweep (when meta-agent dispatches one):
- `joint_investigation_report.csv`

---

## Scope

**In (this PR):** `AgentDriver` Protocol + `claude-code-sdk` and `claude-terminal` drivers; four use-case directories (`general_blame`, `profiling`, `agent_scaffolding`, `hinter`); benchmark-context-agent; required + auto-generated context file; merged audit pass; cross-investigator runner; joint-CSV runner; selector Protocol with three built-in selectors; `experiment_investigation_report.json` aggregate; post-batch meta-analysis synthesis with auto-derived JSON template + retry-on-validation-error; CLI flags (`--recipe`, `--driver`, `--audit`, `--n-seeds`, `--investigator-model`, `--synthesis-model`, `--journal-dir`).

**Out (signposted follow-ups):** Codex driver; real similarity selector (`TopKBySimilarityStub` ships as a placeholder); additional use-case directories (`auto_verified`); typed meta-agent loop replacing the slash command; dedicated meta-meta investigator-quality synthesis (human-triggered for now via the artefacts that `hinter` and audit produce).

**Not touched:** cube-standard, episode loop, trajectory format, storage Protocol, `Episode`, `Experiment`, `exp_runner`, `LLM` wrapper. The previous `meta_agent/` directory was removed entirely; the slash-command workflow was repurposed as the `hinter` use_case under `analyze/investigator/use_cases/`.

---

## Constitution alignment

- **PS-002 (LiteLLM).** Drivers honour `LITELLM_PROXY_URL`. Agent-runtime abstraction is ours because LiteLLM does not cover it.
- **PS-Python-is-config.** Recipes are `.py` files exporting Pydantic instances; runtime artifacts are JSON.
- **EX-002 (no global state).** No registries — `RECIPE_CATALOG` is assembled on import; drivers and selectors are arguments.
- **CC-005 (minimalism).** Three use cases, two drivers, one audit flag. Larger surfaces are separate PRs.
- **SR-002 (trace-first).** Driver calls wrapped in `auto_cube.investigator.driver_run` OTel spans.
- **SR-003 (escape hatch).** Both drivers expose `.raw` on `DriverResult`.

---

## Open questions

1. **`SKILL.md` registration.** Symlink `use_cases/<name>/SKILL.md` into `.claude/skills/investigator-<name>` (lightweight, one source) or copy at build time (no symlink dep)? Recommend symlink.
2. **`continue_session` behaviour across the audit pass.** Verify against the SDK before merging — if continuation across a closed-and-reopened session doesn't work, the audit fallback is a fresh `run` with prior context serialised in.

Everything else is implementer's call.

---

## References

- `openspec/changes/trajectory-investigator/proposal.md` — PR #366.
- `openspec/changes/agent-owns-loop/proposal.md` — PR #386.
- `src/cube_harness/analyze/investigator/` — seam PR landed `InvestigatorRecipe`, `validate_context_file`, `_PATHS_FENCE_RE`.
- LiteLLM Agent SDKs: https://docs.litellm.ai/docs/agent_sdks
- Claude Code headless: https://code.claude.com/docs/en/headless
- Codex non-interactive: https://developers.openai.com/codex/noninteractive
