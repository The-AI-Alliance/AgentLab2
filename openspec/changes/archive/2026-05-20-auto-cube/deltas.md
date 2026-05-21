# Deltas: Auto-CUBE

Applies to: `openspec/specs/analyze/spec.md` (primary), `openspec/specs/storage/spec.md` (minor).

The seam PR (forward-compat to PR #366) already landed `InvestigatorRecipe`, `PostJudgeSurvey`, `validate_context_file`, and the `related_trajectories` parameter as no-op extensions. These deltas widen those types into a first-class surface and add the coding-agent driver Protocol, audit pass, benchmark-context sub-agent, and cross-experiment runners.

---

## ADDED — `cube_harness/analyze/investigator/driver.py`

`AgentDriver` Protocol with async `run(...)` and `continue_session(...)` methods returning a `DriverResult` (output text, token counts, cost, duration, observed tool actions, optional LiteLLM proxy URL, optional session id).

Two concrete drivers: `ClaudeCodeSDKDriver` (`max_parallelism=8`, wraps `claude-agent-sdk`) and `TerminalClaudeDriver` (`max_parallelism=2`, wraps `claude -p` via subprocess; reports `cost_usd=0`). Codex driver is a follow-up.

Both honour `LITELLM_PROXY_URL` / `LITELLM_PROXY_AUTH_TOKEN` by exporting `ANTHROPIC_BASE_URL` to the underlying SDK/CLI. The URL (no credentials) is recorded on `DriverResult`.

### Invariants

- `run` is `async`. `continue_session` raises `NotImplementedError` when unsupported; callers fall back to a fresh `run` with prior context serialised into the prompt.
- `max_parallelism` is advisory — `investigate_experiment` clamps and logs.
- `DriverResult.cost_usd == 0.0` is a sentinel for "driver does not meter."

---

## ADDED — `cube_harness/analyze/investigator/use_cases/`

Each subdirectory is a self-contained use case: `recipe.py` (exports `RECIPE: InvestigatorRecipe`, with prompts and `OutputModel` declared inline), `SKILL.md` (meta-agent skill), optional `scripts/`. `__init__.py` walks subdirectories on import and assembles `RECIPE_CATALOG: dict[str, InvestigatorRecipe]`.

Initial directories: `general_blame`, `profiling`, `agent_scaffolding`. Adding a use case is a single PR.

A small script (`scripts/sync_investigator_skills.py`) symlinks each `SKILL.md` into `.claude/skills/investigator-<name>`.

### Invariants

- Recipe `name` equals directory name.
- Importing a use-case module performs no I/O beyond Pydantic field validation.
- Each `OutputModel` extends a shared `BaseFindings` carrying the cross-recipe core fields (`primary_blame`, `outcome`, `primary_blame_confidence`).

---

## ADDED — `cube_harness/analyze/investigator/recipe.py` widening

`InvestigatorRecipe` (seam PR landed `name`, `system_prompt`, `model`, `allowed_tools`) gains:

- `user_prompt_template: str`
- `output_model: type[TypedBaseModel]` — per-recipe output schema (extends `BaseFindings`)
- `audit: bool = False`
- `permission_mode: Literal["bypassPermissions", "ask"]`

`allowed_tools` tightens from `list[str] | None` to `tuple[str, ...]`.

**No `driver` or `selector` field.** Both are call-time arguments. **No `post_judge_survey` field** — survey is folded into the audit pass.

---

## ADDED — `cube_harness/analyze/investigator/audit.py`

Single unified self-evaluation pass (replaces both the previous `self_judge` recipe and the separate `post_judge_survey`).

`AuditOutput`:
- `schema_version: int = 1`
- `recipe: str`, `driver: str`
- `verdict: Literal["sound", "questionable", "wrong"]`
- `reasoning_quality: int` (0-5)
- `ease_of_analysis: int` (0-5)
- `context_quality: int` (0-5)
- `tooling_gaps: list[str]`
- `missed_evidence: list[str]`
- `alternative_blames: list[BlameAlternative]`
- `notes: str | None`

`run_audit_pass(...)` calls `driver.continue_session` with the audit prompt; falls back to `driver.run` when continuation is unsupported. Writes `audit.json` next to `findings.json`.

The previous separate `post_judge_survey.json` file is **removed from scope** — its fields move into `audit.json`.

---

## ADDED — `cube_harness/analyze/investigator/benchmark_context_agent.py`

Sub-agent that auto-generates `<experiment_dir>/investigation_context.md` when missing.

- Default model: `"claude-opus-4-7"`.
- Runs through the same `AgentDriver` interface as the investigator itself.
- System prompt instructs the agent to walk `experiment_config.json`, identify the cube package, agent package, and `cube_harness` source dir, and emit a `\`\`\`paths` fenced block matching the format `validate_context_file` already parses.
- Called automatically from `_investigate_episode_impl` when `find_default_context_file(experiment_dir)` raises.
- Surfaced via `ch-investigate init-context <experiment_dir>` for explicit/ad-hoc use.

### Invariants

- The sub-agent's output is parsed by the existing `_PATHS_FENCE_RE` regex; no new marker format introduced.
- Every path in the emitted block is verified to exist locally before the investigator proceeds; missing paths raise `FileNotFoundError`.

---

## ADDED — `cube_harness/analyze/investigator/selection.py` (Selector extensions)

`Selector` Protocol with `select(*, main_episode, experiment_dir, all_refs) -> list[Path]`. Three built-in selectors: `SameTaskDifferentAgent`, `SameAgentPreviousIteration`, `TopKBySimilarityStub` (real similarity scorer is a follow-up).

Selectors are call-time arguments. Existing `EpisodeRef` / `discover_episodes` / `select_episodes` helpers are unchanged.

### Invariants

- A selector never returns the main episode's own path.
- Returned paths exist on disk; list length is bounded by `k`.

---

## ADDED — `cube_harness/analyze/cross_experiment/`

**Runners ship in this PR**, not just schemas (per design feedback).

### `joint_csv.py`

`write_joint_csv(sweep_dir: Path, *, experiment_dirs: Sequence[Path] | None = None) -> Path`

Walks experiment directories, reads each `experiment_investigation_report.csv` plus `cross_investigation_agreement.csv` if present, writes one row per `(experiment, episode)` to `<sweep_dir>/joint_investigation_report.csv`. Atomic write.

`JOINT_REPORT_COLUMNS` includes per-experiment columns plus `experiment_id`, `family_id`, `agent_dotted`, `benchmark_dotted`, `driver`, `recipe`, `litellm_proxy_url`, and joined cross-investigator columns.

### `cross_investigation_agreement.py`

Implementation invoked by `investigate_experiment(..., n_seeds=N)`. Re-runs the investigator N times per selected episode with the **same recipe** at varying seeds, then writes `cross_investigation_agreement.csv` keyed by `(trajectory_id, recipe)` with columns `seeds`, `primary_blame_modal`, `primary_blame_agreement`, `outcome_modal`, `outcome_agreement`, `confidence_mean`.

### Invariants

- Joint CSV: one row per `(experiment_id, trajectory_id)`; atomic write.
- Agreement CSV: one row per `(trajectory_id, recipe)` with `n_judgments >= 2`; agreements in `[0, 1]`.
- Cross-investigator runs **same recipe, different seeds** — cross-recipe comparison is out of scope for this artifact.

---

## ADDED — `experiment_investigation_report.json`

Aggregate report alongside the existing `experiment_investigation_report.csv` (PR #366). Same per-episode rows but preserves the per-recipe `OutputModel` shape (the CSV flattens to the base schema). Written by the same code path that writes the CSV.

---

## MODIFIED — `cube_harness/analyze/investigator/core.py`

`investigate_episode` and `investigate_experiment` gain `recipe`, `driver`, `selector`, `audit`, `n_seeds` keyword arguments; lose `model` (now part of recipe). `n_parallel` clamps to `driver.max_parallelism` with a log line.

`_investigate_episode_impl`:
- Calls `find_default_context_file(experiment_dir)`. If missing, invokes `benchmark_context_agent.generate(experiment_dir, driver=...)` to create it. Then calls `validate_context_file(...)`. Raises if any path doesn't resolve.
- Calls `selector.select(...)` when a selector is provided.
- Calls `driver.run(...)` instead of `_run_claude_code(...)`.
- When `audit or recipe.audit`, calls `audit.run_audit_pass(...)` and writes `audit.json`. **No separate survey pass.**
- Parses output against `recipe.output_model`, not a hardcoded `Findings`.

`investigate_experiment` runs the cross-investigator loop when `n_seeds > 1` and writes `cross_investigation_agreement.csv`. Always writes both `experiment_investigation_report.csv` and `experiment_investigation_report.json`.

Aggregate cost in `experiment_investigation_summary.json` excludes audit cost by default; an additive `total_audit_cost_usd` field is appended.

---

## MODIFIED — `cube_harness/analyze/investigator/context.py`

`find_default_context_file(experiment_dir: Path) -> Path` returns `experiment_dir / "investigation_context.md"` and raises if absent. Used by `_investigate_episode_impl` to decide whether to invoke the benchmark-context sub-agent.

`collect_source_paths` is **removed**. Its behaviour is superseded by the benchmark-context sub-agent for the auto-generation path and is no longer called at investigator time.

`validate_context_file` (already seamed) is now the only context resolver in the investigator's hot path. No signature change.

---

## MODIFIED — `cube_harness/analyze/investigator/sdk.py`

`_run_claude_code` is deleted; its body moves into `ClaudeCodeSDKDriver.run`. The module retains `_extract_json_block`, `_summarise_tool_input`, and re-exports `TraceMode` / `INVESTIGATOR_ALLOWED_TOOLS` for backwards compatibility.

---

## MODIFIED — `cube_harness/analyze/investigator/cli.py`

New flags: `--recipe NAME`, `--driver {claude-code-sdk,claude-terminal}`, `--audit`, `--n-seeds N`. The `--model` flag is removed; backwards-compatibility shim accepts it with a deprecation warning.

`--no-survey` is **not added** — survey is folded into the audit pass; toggle via `--audit` instead.

New subcommand: `ch-investigate init-context <experiment_dir>` invokes the benchmark-context sub-agent explicitly (same code path as the auto-generation hook).

---

## MODIFIED — `openspec/specs/analyze/spec.md`

Adds a "Investigator subsystem" section documenting: the use-case directory layout (`recipe.py` + `SKILL.md` + optional `scripts/`), `InvestigatorRecipe` (with per-recipe `OutputModel`, no driver/selector fields), the `AgentDriver` Protocol and its two concrete drivers, LiteLLM proxy routing, the benchmark-context sub-agent, the merged audit pass, the cross-investigator runner, the joint-CSV runner, and the CLI surface. Also notes that batch results are written in both CSV and JSON.

---

## MODIFIED — `openspec/specs/storage/spec.md`

One additive sentence under the experiment-root artifacts table covering `experiment_investigation_report.json`, `cross_investigation_agreement.csv`, the per-episode `audit.json` when audit is enabled, and the sweep-level `joint_investigation_report.csv`. No change to `EpisodeRecord`.

---

## NOT CHANGED

`InvestigationMetadata`, `EpisodeRecord`, `Storage` Protocol, `Episode`, `Experiment`, `exp_runner`, `Trajectory`, `LLM`, `meta_agent/recipes/`, `.claude/commands/meta-agent.md`, cube-standard.

`Findings` is renamed to `BaseFindings` and becomes the parent class of each recipe's `OutputModel`. The `general_blame` recipe's `OutputModel` is identical to today's `Findings` (preserving on-disk compatibility for the default path).
