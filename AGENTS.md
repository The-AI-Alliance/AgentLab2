# Agents guide — cube-harness

*(Served as both `AGENTS.md` and `CLAUDE.md` — the latter is a symlink.)*

You are working in **cube-harness**, the runtime that executes agents against CUBE
benchmarks and records trajectories. This file is your map; it is deliberately short.
Read the relevant spec in `openspec/specs/` before modifying any layer.

## What this repo is

cube-harness runs experiments. It consumes the contracts defined by **cube-standard**
(`Task`, `Benchmark`, `Tool`, `Resource`) and adds: agents, episode loops,
trajectory storage, OTel tracing, parallel execution (Ray), the XRay viewer, and
MCP server bridges.

It does NOT define the task/benchmark/tool protocol — that's cube-standard. If you're
tempted to change base class signatures (`Task.step`, `Benchmark.setup`, etc.), you're
in the wrong repo; go to cube-standard and start with an openspec change proposal.

## Package layout

```
src/cube_harness/
├── core.py                     # AgentOutput, Trajectory, TrajectoryStep, ActionSpace
├── agent.py                    # AgentConfig, Agent (abstract)
├── llm.py                      # LLM, LLMConfig, Prompt, LLMCall, Usage (LiteLLM wrapper)
├── episode.py                  # Episode, EpisodeConfig, MAX_STEPS
├── experiment.py               # Experiment, ExpResult
├── exp_runner.py               # run_sequentially, run_with_ray
├── storage.py                  # Storage Protocol, FileStorage (V2 + V1 fallback)
├── summary.py                  # SummaryProcessor, ExperimentSummary
├── episode_logs.py             # Per-episode stdout/stderr redirection
├── utils.py                    # parse_actions, HTML pruning, misc
├── results.py                  # Higher-level result types
├── agents/
│   ├── react.py                # ReAct agent (primary)
│   ├── genny.py                # Genny agent (context-aware, rolling summaries)
│   └── legacy_generic_agent.py # Deprecated XML-tag agent — see DEPRECATED.md
├── action_spaces/              # Protocol definitions for action sets
├── benchmarks/                 # Legacy in-tree benchmarks (miniwob, workarena) — most now live in cubes/
├── metrics/tracer.py           # OpenTelemetry tracer, Ray env-var propagation
├── analyze/
│   ├── xray.py                 # Gradio-based XRay viewer
│   ├── inspect_results.py      # CLI-ish inspection helpers
│   └── xray_utils.py
└── mcp/                        # Serve harness tools AS an MCP server
    ├── server.py
    └── convert.py

cubes/                          # External benchmark packages (arithmetic, osworld, swebench-*, terminalbench, webarena-verified, workarena, miniwob)
recipes/                        # Example experiment scripts
tests/                          # pytest suite
```

## Spec index

Each spec is the authoritative contract for its layer.

| Layer | Module | Spec |
|-------|--------|------|
| Core types (Trajectory, AgentOutput) | `cube_harness.core` | [core/spec.md](openspec/specs/core/spec.md) |
| Agent | `cube_harness.agent` | [agent/spec.md](openspec/specs/agent/spec.md) |
| LLM wrapper | `cube_harness.llm` | [llm/spec.md](openspec/specs/llm/spec.md) |
| Episode | `cube_harness.episode` | [episode/spec.md](openspec/specs/episode/spec.md) |
| Experiment + runners | `cube_harness.experiment`, `cube_harness.exp_runner` | [experiment/spec.md](openspec/specs/experiment/spec.md) |
| Storage | `cube_harness.storage`, `cube_harness.summary` | [storage/spec.md](openspec/specs/storage/spec.md) |
| Metrics / OTel | `cube_harness.metrics` | [metrics/spec.md](openspec/specs/metrics/spec.md) |
| XRay viewer | `cube_harness.analyze` | [analyze/spec.md](openspec/specs/analyze/spec.md) |
| MCP server | `cube_harness.mcp` | [mcp/spec.md](openspec/specs/mcp/spec.md) |

**External contracts (cube-standard):** Any field typed as `cube.task.Task`,
`cube.benchmark.Benchmark`, `cube.tool.Tool`, `cube.core.*`, or `cube.resource.*`
is governed by cube-standard's specs. Don't subclass those here — consume them.

## Engineering principles

- **Read the spec first.** Before touching any layer, read its spec in `openspec/specs/`. Specs are the authoritative design intent — but they can be stale or wrong; flag discrepancies rather than silently working around them.
- **Fix in the right place.** A quick local experiment to understand a problem is fine. But the committed fix must address the root cause in the correct layer — not a workaround scoped to a single call site or context.
- **Understand before fixing.** Many bad fixes come from acting too fast. Make sure you understand the broader design before proposing a change. A fix that misses the bigger picture is worse than no fix.
- **Lean diffs.** Make the minimal change that solves the problem. Avoid verbose additions, unnecessary abstractions, and duplicated logic that already exists elsewhere. If existing code can be reused or consolidated, do it. A hard-to-review diff is a liability.
- **Think long-term.** Every change should age well. Ask whether today's shortcut becomes tomorrow's debt — and whether the design could evolve cleanly if requirements change.

## Code review

**Default branch is `dev`** — base all PRs off it, not `main`.

**Sign your commits.** Every commit needs a `Signed-off-by` line (`git commit -s`). DCO is enforced by CI — unsigned commits will be blocked.

PRs are reviewed with `/code-review` ([plugin docs](https://github.com/anthropics/claude-code/blob/main/plugins/code-review/README.md)), which audits changes against these guidelines. Write PRs as if a reviewer will check each principle above against the diff.

**Auto-fix provenance.** Auto-CUBE-produced fixes carry `# auto-fix(N)↓ … # /auto-fix(N)`
markers + a one-line machine-readable footnote at module bottom (`N` = PR number
for L0/L1, design-debt issue number for L2/L3). When a diff touches an `auto-fix`
region or its footnote, treat it as a possibly-rotten marker (review rule AF-001).
Methodology (Fix Report, L0–L3 tiers, rot lint):
[`openspec/specs/auto-fix/spec.md`](openspec/specs/auto-fix/spec.md). Human
entry point for running the loop:
[`.claude/skills/auto-cube/README.md`](.claude/skills/auto-cube/README.md).

## Workflow for code changes

1. **Find the relevant spec** — which layer? Start there.
2. **Check "Invariants" and "Gotchas"** — these are the traps.
3. **Check `openspec/changes/`** — someone may already be proposing your change.
4. **For substantive contract changes**, write a delta spec
   (`openspec/changes/<name>/deltas.md` with ADDED / MODIFIED / REMOVED sections)
   before coding. Archive to `openspec/changes/archive/YYYY-MM-DD-<name>/` when done.
5. **Constitution alignment:** every change is reviewed against the
   [constitution](.claude/rules/constitution.md) and [review rules](.claude/rules/review-rules.md).

## Key conventions (already enforced in code)

- **Python is the configuration** — no YAML/Hydra. `AgentConfig`, `LLMConfig`,
  `Experiment` are all Pydantic `TypedBaseModel`.
- **LiteLLM is the only LLM gateway** — never import `openai`, `anthropic`, etc. directly.
- **Module-level imports only** — no function-scoped imports (EX-001).
- **Type hints required everywhere**, including tests (CC-001).
- **Serialization boundary:** Workers receive `TaskConfig` + `EpisodeConfig` (pickled).
  Live `Task`, `Tool`, `Benchmark`, `Agent` objects never cross process boundaries.
- **Trajectory steps alternate** env → agent → env → agent in persistence order.
- **Trace-first:** every new long-running operation should get a `tracer.span()`.
- **CLIs use Typer** — new scripts/recipes that need a CLI should use `typer.run(main)` with `typer.Option`-annotated args (FastAPI-style: type hints + docstring become `--help`). `scripts/experiments_report.py` is the canonical example. Don't add new `argparse` boilerplate.

## Development commands

```bash
make install            # uv sync --all-extras
make test               # full pytest
make debug              # small end-to-end run
make xray               # open the trajectory viewer
make report             # markdown table of experiments in ~/cube_harness_results/ (forward args with ARGS="--last 10")
make lint               # uvx ruff check --fix && uvx ruff format  (auto-fixes in place)
make lint-check         # uvx ruff check --diff && uvx ruff format --diff  (read-only, what CI runs)
make review PR=<n>      # check out a PR and wire up any cross-repo cube-standard dependency
uv run recipes/hello_miniwob.py   # example run
```

### Test categories

| Type | When | Where |
|---|---|---|
| Unit (`pytest tests/`) | every iteration | `tests/` — fast, no external deps. What CI runs by default (`-m "not slow and not live_api"`). |
| `slow` (`pytest -m slow`) | when touching the marked area | `tests/` with `@pytest.mark.slow`. Excluded from CI by default — Ray-based timing tests are flaky on shared GitHub Actions runners; reliable locally. |
| `integration` (`pytest -m integration`) | when touching the marked area | `tests/` with `@pytest.mark.integration`. Setup details (Playwright install, etc.) live in the marker's docstring in `pyproject.toml`. |
| `live_api` (`pytest -m live_api`) | when touching the marked area | `tests/` with `@pytest.mark.live_api`. Hits a real LLM provider; costs money; auto-skips without `ANTHROPIC_API_KEY`; never runs in CI. |
| Smoke (`scripts/smoke/*.py`) | when a PR touches plumbing unit tests can't reach | Standalone scripts a coding agent runs to verify end-to-end behavior. Never CI. May stand up real infrastructure or call external APIs; minutes-long runs are fine. Each prints `SMOKE OK/FAIL/SKIP: <name>` (exit 0/1/2). Discover with `find . -path '*/scripts/smoke/*.py'`. |

Smokes are the coding agent's judgment call — for a PR that touches a marked area, pick the relevant smokes, adapt the environment (auth, credentials, profiles), and iterate until green. **Reflex:** when adding complex new code, drop a smoke alongside it; a green end-to-end run is the strongest signal the change actually works as intended.

Always run `make lint` before finishing a task. `ruff check` and `ruff format` are
**separate passes** — running only one is not enough for CI.

Environment vars go in `.env` (loaded by pyproject). `OPENAI_API_KEY`
is the only required one for the baseline recipes; see individual cubes for others.

**Ray launch**: use `.venv/bin/python recipe.py` or `uv run --active recipe.py` — never bare
`uv run` when `VIRTUAL_ENV` is set. uv will silently create an ephemeral env whose `.pth` files
can point to deleted paths, causing `ImportError` on Ray workers. `exp_runner.py` warns when it
detects this.

## Cross-repo PRs (cube-harness ↔ cube-standard)

When a PR depends on an unreleased cube-standard branch, do **not** commit
`path = "..."` local sources to any `pyproject.toml` (root or under `cubes/`).
The pre-commit hook (`.githooks/pre-commit`) will block this — local paths break
for anyone with a different folder structure.

**Authoring a cross-repo PR:**

1. Keep `pyproject.toml` pointing at PyPI (or a git ref) — do **not** commit the local path.
2. Add a line starting with `Depends-on: cube-standard/<branch-name>` to the PR description body. The line must start at column 0 (no leading whitespace, not inside a list or quote block).

**Reviewing a cross-repo PR:**

```bash
make review PR=<n>
```

This checks out the PR branch, reads `Depends-on:` from the PR description, clones
`cube-standard` into the repo root (gitignored), checks out the correct branch, and
installs all workspace packages with `uv pip install -e cube-standard --all-packages --all-extras`.

## What lives elsewhere

- **cube-standard** — protocol and base classes. Never subclass cube-standard ABCs
  here without first updating cube-standard's openspec if needed.
- **cube-registry** — public metadata registry; `cube registry add` submits entries.
- **cubes/\*** — individual benchmark packages. Each has its own `debug.py` that
  `cube test <name>` runs. Changes to a cube are usually local to its directory.

## Investigator use cases

`src/cube_harness/analyze/investigator/use_cases/<name>/` is the investigator recipe
catalog. Each subdirectory is one use case:

- **`general_blame`** — default. Closed-world blame attribution per episode.
- **`profiling`** — narrower taxonomy aimed at scaffold-level inefficiency.
- **`agent_scaffolding`** — deep loop-pathology diagnosis.
- **`hinter`** — extract `task_hints[task_id]` candidates from failed
  episodes.

Each use_case has a `recipe.py` (Pydantic `InvestigatorRecipe`) and a `SKILL.md`
(skill description for the Auto-CUBE orchestrator that dispatches it).
`scripts/sync_investigator_skills.py` symlinks SKILL.md files into
`.claude/skills/investigator-<name>` so Claude Code picks them up.

Per-batch synthesis (`meta_analysis.json` + `.md`) is mirrored into
`~/cube_auto_cube_journal/<experiment>/` for cross-iteration narrative —
the only artefact the Investigator writes outside the experiment dir.
