# RFC: Multi-Episode Rollouts for Cross-Episode Memory Agents

**Status:** DRAFT
**Author:** Oleksiy Ostapenko
**Date:** 2026-05-25
**Base branch:** `dev`

---

## RFC scope vs. additive changes

Per the constitution's RFC process (post-`068a718f`): "Additive, backward-compatible changes (a new method, a new optional field) skip this — just keep the living spec accurate and describe the change in the PR."

| Part | RFC-bound? |
|---|---|
| New `Rollout` layer (new spec) | **Yes** — net-new architectural layer |
| `Agent.reflect()` default no-op | No — additive method with default impl |
| `Episode.run_with(task, agent)` | No — additive method on existing layer |
| Storage `rollout_id` / `episode_index_in_rollout` fields | No — additive optional fields |

This document describes all four because they're co-designed; only the Rollout layer requires async review. Class signatures live in `deltas.md` — this doc covers placement and rationale.

---

## Design rationale (one paragraph)

**Rollout stays a distinct class** (not a loop folded into `Episode`) — keeps the "one Episode = one Trajectory" invariant intact, preserves the existing Episode contract for all current callers, and code duplication is eliminated via the additive `Episode.run_with(task, agent)` (composition, not subclassing). **Reflection lives in `Agent.reflect()` for V1** — a default-no-op hook called by Rollout between non-final episodes. The cleaner alternative (drop the hook; agent reflects inside `step()` based on observed `reward` / `done`) is deferred: it requires an upstream cube-standard RFC to add those fields to `Observation`. Worth revisiting once that lands; the additive hook is the interim mechanism. See Alternatives for what was rejected and why.

---

## Problem

cube-harness today runs one episode at a time: `Episode` constructs a fresh `Agent` via `AgentConfig.make()`, plays it against a `Task`, exits. There's no place for an agent instance to survive across episodes, no inter-episode hook, no cross-episode reward aggregate. Meta-RL methods that *adapt across multiple episodes against the same task* by carrying memory — e.g. **LaMer** ([arxiv 2512.16848](https://arxiv.org/abs/2512.16848)), +11–19pp over RL baselines on Sokoban / MineSweeper / Webshop — can't be expressed in the current API.

The new abstraction is generic ("rollout of N episodes with persistent agent + reward discount"), not LaMer-specific — LaMer is the first concrete consumer, not the only one.

## Scope

### In scope (V1)

- **New layer** `Rollout` (`cube_harness/rollout.py`) — N episodes against the same task with a long-lived `Agent`.
- **Agent API** — additive `reflect(trajectory, final_reward) -> None` default no-op on the base `Agent` ABC.
- **Reward aggregation** — `RolloutResult.discounted_reward = Σ γ^k · r_k`.
- **Storage** — trajectories carry optional `rollout_id` / `episode_index_in_rollout`.
- **Validation** — MiniWob via cube-browser-tool's `SyncPlaywrightTool`. No cube changes.
- **POC agent** — `LaMerAgent` (ReactAgent + memory + reflection) shipped as `recipes/lamer_miniwob.py`, not as a permanent agent.

### Out of scope (V2+)

- Ray parallelism over rollouts (sequential first; each Ray task = one whole rollout preserves the agent-doesn't-cross-process-boundaries invariant).
- Cross-rollout / global memory persistence (requires external service — separate proposal).
- Reward-model integration / meta-RL training loop (backend consumes `discounted_reward` like it consumes per-episode rewards today).
- New cubes — MiniWob suffices.

## Design

### Layer placement

```
Experiment           ← unchanged: collection of tasks
  └─ Rollout         ← NEW: N episodes against one task with persistent agent
       └─ Episode    ← unchanged: one task attempt
            └─ Agent.step() loop
```

`Rollout` sits *between* `Experiment` and `Episode`. It **uses** `Episode` internally — `Episode` is not extended, because that conflates "one task attempt" with "N attempts with shared memory" and breaks the existing episode spec.

Composition is realized via a small additive method on `Episode`: **`Episode.run_with(task, agent, extra_metadata=None) -> Trajectory`**. The existing `Episode.run()` keeps its current contract (constructs task + agent locally, closes the task at the end) — it's now a thin wrapper that calls `run_with()` with the constructed objects. `Rollout` calls `run_with()` directly with its own long-lived `task` and `agent`, threading rollout metadata (`rollout_id`, `episode_index_in_rollout`) through `extra_metadata`. This eliminates the duplicate episode loop that the first cut of `Rollout._run_episode` had to copy from `Episode._run_loop`.

### Invariants (the architectural decisions)

- **Same `Agent` instance handles all N episodes** in one rollout. That's the entire mechanism.
- **`Task.reset()` between episodes** — already part of cube-standard's `Task` contract; no cube-standard change.
- **`agent.reflect()` runs between non-final episodes** (`N-1` times). No reflect call after the last episode — nothing reads it.
- **Agent never crosses a process boundary** inside `Rollout.run()`. Each rollout lives entirely in one worker. V2 Ray parallelism parallelizes whole rollouts.

### Cube-standard alignment

`MiniWobTask.reset()` (`cubes/miniwob/src/miniwob_cube/task.py:37`) and `SolveArithmeticTask.reset()` (`cubes/arithmetic-cube/src/arithmetic_cube/task.py:24`) already satisfy what `Rollout` needs. Cubes whose `reset()` is expensive (Docker re-spin) pay that cost N times per rollout — per-cube concern.

Env note: needs `cube-standard>=0.1.0rc9` (`ConfigRegistry`) and `cube-browser-tool>=0.3.0` from cube-standard dev's `cube-tools/cube-browser-tool` subdirectory until it ships to PyPI.

### Storage and provenance

Trajectories produced inside a Rollout carry two new optional metadata fields (`rollout_id`, `episode_index_in_rollout`), both `None` outside a rollout. Existing files remain readable. XRay and the trajectory judge work unchanged on standalone trajectories; rollout-aware grouping in XRay is a follow-up.

### POC agent

See `recipes/lamer_miniwob.py`. ~100 lines on top of `ReactAgent`: `reflect()` writes to `self.memory: list[str]`, `choose_steps_to_render` splices the memory in just after the system prompt. Recipe-only — graduates to `src/cube_harness/agents/` only after validation.

## Alternatives considered

- **Extend `Episode` with an `n_episodes` loop (no separate `Rollout` class).** One concept handles both cases; smaller surface. Rejected: conflates "one task attempt" with "N attempts," requires changing `Episode.run()`'s return type or inventing fake merged trajectories, breaks XRay / eval log / trajectory judge assumptions (one Episode = one Trajectory), and forces every existing caller to think about `n_episodes=1`. Composition (Rollout uses `Episode.run_with`) keeps both layers single-purpose.
- **Delegate reflection entirely to the agent; drop `Agent.reflect()` hook.** Agent detects episode boundaries from observation metadata and reflects inside `step()`. Cleaner separation of concerns (framework stays opinion-free; reflection cadence becomes an agent choice — every K steps, on failure, never). Rejected for V1: agent cannot currently see `reward` or `done` (both env-side fields on `EnvironmentOutput`, not on `Observation`), so this design requires an upstream RFC in cube-standard to add those fields to `Observation`. Worth revisiting once that lands — the additive `reflect()` hook is the interim mechanism. Also fuses the reflection LLM call into an act-step's `AgentOutput`, losing the separate-event semantics XRay and cost stats use.
- **File-based cross-episode memory (no framework hook at all).** Agent reads/writes a per-task file at startup and after each attempt. Zero API change. Rejected: file-based concurrency is fragile under parallel rollouts (multiple workers, same task → race on the file), the reflection LLM call has no canonical place in any trajectory (invisible to XRay / cost / training-data extraction), and the path-keying scheme is left to each agent (no shared convention).
- **MCP service for memory from day one.** Adds operational complexity (run a service, eventual consistency under parallelism) for no V1 value since LaMer's memory is rollout-scoped. The right design for *cross-rollout* persistence (V2). Rejected for V1.
- **Carry memory in `EpisodeConfig`.** Threads mutable state through pickled configs — breaks the worker-serialization boundary. Rollout instead keeps the agent alive inside one worker. Rejected.
- **Separate `MetaAgent` interface.** Over-engineering. Default no-op on `Agent` ABC adds one method and lets any existing agent opt in. Rejected.

## Open questions

1. **Reflection input granularity.** V1: latest `(trajectory, final_reward)` only; agent accumulates prior context in `self.memory`. Forward-compatible to expand signature.
2. **Reward-aggregation location.** V1: `Rollout` computes `discounted_reward`. Backends wanting raw rewards still get `per_episode_rewards`.
3. **Episode budget.** V1: `max_actions` is per-episode (existing semantics). Per-rollout meta-budget is a config-level extension.
4. **XRay visualization.** V1: independent trajectories (no rollout grouping). Rollout-aware view is a follow-up — see `.notes_oo/todo.md`.

## References

- LaMer paper: [arxiv 2512.16848](https://arxiv.org/abs/2512.16848)
- Schemas + ADDED/MODIFIED sections: `deltas.md`
- Recipe template: `recipes/hello_miniwob.py`
- Validation cube: `cubes/miniwob/src/miniwob_cube/task.py`
- POC agent: `recipes/lamer_miniwob.py`
