# The cube-harness Constitution

> **📝 To update this constitution**, use the Claude command `/update-constitution`.
> This ensures all dependent files (review-rules.md) are updated automatically.

**Version**: 0.2 — reconciled with code on 2026-04-20

**Mission**: Empower the open-source community with high-throughput data generation for training and standardized benchmark evaluation by providing a modular, scalable, and efficient agent execution platform.

---

## Preamble: The "One Team" Mindset

We're scaling up to a distributed engineering effort. To prevent architectural entropy, every contributor agrees to uphold the following principles. We value **explicitness over magic**, **composition over inheritance**, and **protocols over implementations**.

This constitution is reviewed against actual code patterns. Anything listed here either **is** current practice or **is** actively migrating toward it — aspirational rules are flagged so you can tell the difference.

---

## Pillar I: The Team Contract & Ownership

*How we organize, communicate, and own our work.*

### Explicit Ownership *(aspirational — ownership map pending)*

**Directive**: Every file and feature should have a clear owner. We distinguish between:
- **Horizontal Ownership (Infrastructure)**: Core capabilities used by everyone (parallelism, protocols, tracing).
- **Vertical Ownership (Features)**: End-to-end features (a specific benchmark + agents).

**Mechanism**: Ownership table lives at the top of `ROADMAP.md` (to be populated).

**Rule**: If you need to modify a component, consult its Owner.

### The RFC Process

**Directive**: Any change that alters the Core API or affects multiple verticals requires a written proposal before implementation.

**Process**:
1. Create a folder in `openspec/changes/<short-name>/`.
2. Write `proposal.md` (problem, change, alternatives) and `deltas.md` (ADDED / MODIFIED / REMOVED requirements against the affected spec). Keep both concise and skimmable — favor schemas/signatures over prose, show each thing once; see the proposal-writing guidance in `openspec/README.md`.
3. Post to the team channel, tag relevant owners, async review, decision.
4. On merge: apply deltas to `openspec/specs/`, move the folder to `openspec/changes/archive/YYYY-MM-DD-<name>/`.

Additive, backward-compatible changes (a new method, a new optional field) skip this — just keep the living spec accurate and describe the change in the PR.

Cross-repo changes (cube-standard ↔ cube-harness) require a proposal in the **upstream** repo first. cube-harness is a consumer of cube-standard's contracts.

---

## Pillar II: The Principle of Explicitness

*The codebase should be readable like a book. We reject "magic" configuration and hidden state.*

### Python is the Configuration

We reject complex YAML hierarchies, opaque Hydra overrides, and massive bash scripts.

**Directive**: All configurations are defined as strictly typed Pydantic `TypedBaseModel` subclasses. Recipes (in `recipes/`) are Python files that instantiate and run experiments directly — they ARE the config files.

**Rule**: If you can't Go-to-Definition on a parameter in your IDE, it is forbidden.

### Composition Over Inheritance

We avoid deep inheritance trees where a subclass inherits dozens of methods it doesn't use.

**Directive**: Build complex agents/benchmarks by nesting standard components, not by subclassing a "god object."

**Pattern**:
- ❌ Bad: `class MyAgent(BaseAllKnowingAgent): ...`
- ✅ Good: `class MyAgent(Agent): def __init__(self, planner: Planner, memory: Memory): ...`

### No Global State

**Directive**: No global variables, singletons, or module-level state that cannot be reset. Module-level loggers are fine.

**Test**: Instantiate two agents with different configs in the same process — they must not interfere.

**Exception (documented):** `cube_harness.metrics.tracer` configures a global OTel `TracerProvider` via `get_tracer()`. This is required by OpenTelemetry's design and is safe because the provider is idempotent.

---

## Pillar III: The "Scalable Research" Philosophy

*We build for massive scale and efficiency, while keeping the developer experience friendly.*

### Local-Dev, Cloud-Scale

**Directive**: The system is designed for massive parallelism (Ray) from day one, but agent logic must remain debuggable on a laptop.

**Mechanism**:
- Agent logic is testable via `run_sequentially(exp, debug_limit=1)`.
- Infrastructure features that require a cluster should provide local mocks when feasible (local Docker backend, in-memory tool stubs).

**Rule**: You should be able to `pdb` through an agent's decision step on your laptop, even if you can't run the full distributed RL loop locally.

### The Inner Loop is Sacred (Efficiency)

**Directive**: The core agent↔environment loop must be optimized for high-throughput sampling.

**Constraint**: Avoid blocking calls and heavy serialization on the critical path. Support asynchronous execution to maximize GPU and environment utilization.

**Goal**: Samples per second is a first-class metric. Features introducing overhead should be discussed (file an openspec change proposal).

See [`openspec/changes/core-extensions/`](../openspec/changes/core-extensions/) (streaming obs, async core) in cube-standard for active work on this front.

### The Escape Hatch (Raw Access)

**Directive**: Abstractions must never prevent a user from inspecting the underlying raw object when necessary.

**Example**: `McpServer.raw → FastMCP` exposes the underlying MCP server instance for advanced clients (see `cube_harness/mcp/server.py`).

Recipe authors and advanced users get `.raw` escape hatches wherever practical.

### Trace-First Engineering

**Directive**: Telemetry is not an afterthought. Logs, screenshots, tool outputs, and reasoning steps are a first-class data product.

**Standard**: We emit OpenTelemetry spans following the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (`gen_ai.tool.*`, `gen_ai.agent.*`). See [metrics spec](../openspec/specs/metrics/spec.md).

**ADP compatibility** is an ongoing goal — the current GenAI attribute surface is the stepping stone.

---

## Pillar IV: The Protocol Strategy

*We define standards to play nice with the ecosystem (NeMo, LangChain, MCP clients).*

### Interfaces over Implementations

**Directive**: Core interactions (Agent ↔ Env) are defined via Protocols / abstract base classes, not concrete classes.

**Goal**: Swap backends (local Docker → Modal → Daytona) without changing agent code. See `cube.resource.InfraConfig` and `cube.container.ContainerBackend` for the pattern.

### Embrace Standards

**Directive**: We do not invent new standards when a working one exists.

- **LLM**: `cube_harness.llm.LLM` uses LiteLLM only. Direct `openai` / `anthropic` SDK calls are forbidden (rule PS-002).
- **Tools**: We support MCP both as a client (consuming external MCP servers via `tools/mcp.py`) and as a server (exposing our tools via `cube_harness.mcp.McpServer`).
- **Data**: OpenTelemetry GenAI semantic conventions; full ADP migration is an open goal.

### Hermetic Reproducibility *(partial — see gaps below)*

**Directive**: Every experiment run should capture:
- The exact git commit hash. **Gap:** not yet automated. Add to `save_config()` when it lands.
- The full Configuration object. **Adopted** — `experiment_config.json` dumps the full `Experiment` with `serialize_as_any=True`.
- The Docker container ID / image hash of the environment. **Gap:** container hash capture is benchmark-specific; no standard yet.

Track these gaps in `DEPRECATED.md` / `openspec/changes/` when work picks up.

---

## Pillar V: The Craft of Code

*We maintain a lean, high-quality codebase. Code is a liability, not an asset.*

### The Minimalist Imperative

**Directive**: Prefer a smaller, simpler codebase over one that supports every edge case. If a feature adds significant complexity but is rarely used, reject or remove it.

**Action**: Refactoring to delete code is prioritized over adding non-critical features. If you can delete 100 lines by refactoring the core, raise a proposal.

### Function Atomicity

**Directive**: Break long functions into logical sub-functions. A function should fit on a standard screen (~50–80 lines).

**Goal**: Self-documenting code. Prefer named helpers like `_parse_observation()` over inline comments explaining a 50-line block.

### AI-Assisted, Human-Architected (No "Vibe Coding")

**Directive**: We use AI tools to generate snippets and search for solutions, but we never blindly paste large blocks of code.

**Risk**: "Vibe coding" pollutes the codebase with verbose, hallucinated, or unoptimized logic.

**Rule**: You must understand and curate every line you commit. If the AI wrote it, refactor and tighten it before merging.

### The Testing Pyramid

**Directive**: We prioritize high coverage with simple, fast unit tests. Slow integration tests go to nightly builds.

**CI Rule**: The core test suite must run in under 5 minutes.

**Style**: `ruff` for formatting and lint. Type hints required on all functions and tests (rule CC-001).

---

## Alignment with cube-standard

cube-harness consumes cube-standard's contracts. Any change that would require altering a cube-standard ABC or Pydantic model must first land as a change proposal in that repo. See [cube-standard's openspec](https://github.com/The-AI-Alliance/cube-standard/tree/main/openspec).
