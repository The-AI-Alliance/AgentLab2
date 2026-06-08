# cube-harness charter — the broader picture you defend

This is the harness-specific layer. The **universal** gatekeeping principles (lean /
additive-is-not-free / one source of truth / friction-not-a-wall / escape hatches /
recurring demand / judge substance not provenance) live in cube-standard's
[Design Philosophy](https://the-ai-alliance.github.io/cube-standard/design-philosophy) —
don't restate them; apply them. Here is what's specific to cube-harness.

## What cube-harness is

The **evaluation runtime**: it runs agents against CUBE-compatible benchmarks, records
trajectories, and scales with Ray. It *consumes* CUBE Standard's contracts (`Task`,
`Benchmark`, `Tool`, `Action`, `Observation`, the `cube.*` ABCs) — it does **not** define
or extend them. Center of gravity (the shared surface to protect): `core.py` (Trajectory),
`agent.py` (the `Agent` protocol + `AgentConfig`), `llm.py`, `episode.py`,
`experiment.py` / `exp_runner.py`, `storage.py`, the tracer, and the `openspec/specs/`.

## The design intent you defend — read these, don't re-derive them

- **The Constitution** — `.claude/rules/constitution.md`. The 5 pillars are the harness's
  "what we defend." Argue from them and quote the relevant one:
  - I — *Team Contract & Ownership*: RFC process in `openspec/changes/`; additive changes
    skip it; **cross-repo changes need an upstream cube-standard proposal first.**
  - II — *Explicitness*: Python is the configuration (Pydantic `TypedBaseModel`, no
    YAML/Hydra); composition over inheritance; no global state.
  - III — *Scalable Research*: local-dev friendly; the inner loop is sacred (no blocking
    calls on the critical path); abstractions expose a `.raw` escape hatch; trace-first.
  - IV — *Protocol Strategy*: interfaces over implementations; embrace standards (LiteLLM,
    MCP, OTel GenAI); hermetic reproducibility.
  - V — *Code Craft*: minimalist imperative (delete over add); small atomic functions;
    human-architected.
- **The layer specs** — `openspec/specs/<layer>/spec.md` (agent, core, episode, experiment,
  llm, metrics, mcp, storage, …): the live contracts, invariants, gotchas.

## Escape hatches — lead with these

Most "change the harness" needs are served *without* touching shared surface. Cheapest first:

1. **Your own recipe.** `recipes/` are Python config-by-example, not frozen API — compose
   existing pieces freely. "Just write a recipe" is the most common correct answer.
2. **Your own agent / LLM config.** Copy and edit a `GENNY_CONFIGS` / `REACT_CONFIGS`
   entry; don't change the `AgentConfig` signature for one experiment's preference.
3. **Your own infra config.** Add an `InfraConfig` to `~/.cube/infra.py` (never committed;
   credentials from env) — never a field on a shared config.
4. **Subclass the `Agent` protocol** (or another Protocol/ABC) in your own code. Genny is
   exactly this; the base isn't touched.
5. **A new Investigator / Auto-CUBE use-case** — `/new-auto-cube-use-case` scaffolds
   `auto_cube/use_cases/<name>/`; same for `analyze/investigator/use_cases/<name>/`.
6. **A tiny additive hook + your code** — when the above almost work, the minimal general
   extension point in shared surface, with the rest on your side.
7. **A shared-surface change** — only when general and unservable by 1–6; lean bar applies,
   and breaking changes go through `openspec/changes/`.
8. **Forking — discouraged**; you lose the shared runtime and upstream benefit.

## The route-upstream rule (harness-specific)

If the real need requires changing a **cube-standard contract** — a new/changed method or
field on `Task` / `Benchmark` / `Tool` / `Action` / `Observation` or any `cube.*` ABC — it
does **not** belong in cube-harness. cube-harness consumes those contracts; extending them
here would fork the protocol. Verdict: **ROUTE-UPSTREAM** — send the contributor to
cube-standard's `/gatekeep-rfc` and an `openspec/changes/` proposal *there*; any harness
change follows once the upstream contract lands. (Constitution Pillar I; AGENTS.md "External
contracts".)

## When it's a genuine gap — escalate (don't reshape)

- A general need (many recipes/agents/verticals) that can't be expressed with the existing
  surface or delivered additively.
- A real inconsistency or footgun in a spec or a pillar.
- A principled challenge to a pillar, or a cross-vertical / horizontal-ownership decision a
  human owner should make.
- **Recurring demand** — the same need across multiple prior proposals/issues, especially
  ones repeatedly closed for the same reason. Recurrence is signal; escalate the *pattern*
  (with links), and consider that a pillar may need revisiting.

Attach your best smaller alternative even when you escalate.

## Illustrative (not a checklist)

- "Add my sampling param to `AgentConfig`" → your own agent config / a subclass, not the
  shared base.
- "Make `Episode` emit my custom metric" → trace-first: emit a span / use a use-case, don't
  thread a one-off through the core loop.
- "Add a field to `Observation` so my agent sees X" → ROUTE-UPSTREAM (that's a cube-standard
  contract).
- "Replace LiteLLM with a direct SDK call for my provider" → Pillar IV says embrace the
  standard; LiteLLM already supports it via `model_name`.
