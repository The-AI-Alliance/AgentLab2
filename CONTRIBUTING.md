# Contributing to cube-harness

For contribution philosophy, DCO requirements, RFC process, and community guidelines, see the canonical [CONTRIBUTING.md in cube-standard](https://github.com/The-AI-Alliance/cube-standard/blob/main/CONTRIBUTING.md).

## Changing cube-harness

Before proposing a change to a shared surface (`core.py` / `agent.py` / `llm.py`, the `openspec/specs/`, or any cross-vertical API), read the project-wide [Design Philosophy](https://the-ai-alliance.github.io/cube-standard/design-philosophy) and the harness [Constitution](.claude/rules/constitution.md) (the 5 pillars). Most "change the harness" needs are better served by your own recipe, agent/infra config, or a new use-case — not a shared-surface change.

- **Check yourself first with `/gatekeep-rfc`** (Claude Code) — run it locally and iteratively on your draft. It separates the real need from the mechanism, leads with the escape hatch that fits, and points you at the smallest version before anyone else reads it. Converge locally before opening a PR.
- **A need that requires changing a cube-standard contract** (`Task` / `Benchmark` / `Tool` / `Action` / `Observation`) goes **upstream first** — open an `openspec/changes/` proposal in cube-standard (run its `/gatekeep-rfc` there); the harness change, if any, follows once the contract lands. cube-harness *consumes* those contracts, it doesn't extend them.
- **Otherwise**, breaking changes get an `openspec/changes/<name>/` proposal (below); additive, backward-compatible changes just keep the living spec accurate.

## OpenSpec — how we manage contracts

We follow the [OpenSpec](https://github.com/Fission-AI/OpenSpec) methodology. Each layer of
cube-harness has a living spec in `openspec/specs/<layer>/spec.md` that defines its public
API, invariants, and gotchas. Before modifying a layer, read its spec. After a PR that
changes a public contract, run `/update-openspec` in Claude Code to sync the spec.

For breaking changes, write a short delta proposal in `openspec/changes/<name>/` before
coding — this makes the contract change visible to the team before code lands.

Full workflow, delta format, and examples: [`openspec/README.md`](openspec/README.md).  
The methodology reference is in [cube-standard's openspec/README.md](https://github.com/The-AI-Alliance/cube-standard/blob/main/openspec/README.md).

## Setup

```bash
git clone https://github.com/The-AI-Alliance/cube-harness.git
cd cube-harness
make install           # uv sync --all-extras
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

```bash
make lint    # ruff check + format (auto-fix)
make test    # pytest tests/
```

All commits need a [DCO sign-off](https://developercertificate.org/): `git commit -s -m "..."`. Running `make install` sets up a git hook that adds this automatically.

## Repo Layout

```
src/cube_harness/
  agent.py          # Agent protocol and AgentConfig base
  benchmark.py      # Benchmark interface for task collections
  core.py           # Data structures: Action, Observation, Trajectory, Task
  environment.py    # Environment and EnvConfig abstractions
  episode.py        # Episode execution and trajectory persistence
  experiment.py     # Experiment configuration and statistics
  exp_runner.py     # Sequential and Ray-based parallel execution
  llm.py            # LLM wrapper using LiteLLM
  storage.py        # Trajectory storage backends
  tool.py           # Tool abstraction for action spaces
  agents/           # Agent implementations (ReAct, Genny, …)
  tools/            # Tool implementations (Playwright, BrowserGym, …)
  benchmarks/       # Benchmark wrappers (MiniWob, WorkArena, …)
  metrics/          # Telemetry and tracing (OpenTelemetry-based)
  action_spaces/    # Browser action space protocols
  analyze/          # Trajectory analysis and XRay inspection utilities
  mcp/              # MCP server for exposing tools via Model Context Protocol
recipes/            # Example experiment scripts
tests/              # Test suite
```

## Releases

Releases are tag-driven and cross-repo (cube-harness packages publish in tiers 4–5, after cube-standard). The full runbook — promoting `dev`→`main`, the dependency tiers, and the `scripts/release.py` driver — is canonical in cube-standard's [`RELEASING.md`](https://github.com/The-AI-Alliance/cube-standard/blob/main/RELEASING.md).

## Licenses

- **Code** — Apache 2.0 ([LICENSE.Apache-2.0](LICENSE.Apache-2.0))
- **Documentation** — CC BY 4.0 ([LICENSE.CC-BY-4.0](LICENSE.CC-BY-4.0))
- **Data** — CDLA Permissive 2.0 ([LICENSE.CDLA-2.0](LICENSE.CDLA-2.0))

## Community

- [GitHub Issues](https://github.com/The-AI-Alliance/cube-harness/issues) — bug reports and feature requests
- [GitHub Discussions](https://github.com/The-AI-Alliance/cube-harness/discussions) — design conversations and RFCs
- [Apply as a core contributor](https://forms.gle/JFiBi4ynfVLMghAH8) — if you want to help shape priorities

See also the AI Alliance [community repo](https://github.com/The-AI-Alliance/community/) for cross-project guidelines and the [Code of Conduct](https://github.com/The-AI-Alliance/community/blob/main/CODE_OF_CONDUCT.md).
