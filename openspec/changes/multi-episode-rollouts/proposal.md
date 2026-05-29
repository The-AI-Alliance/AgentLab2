# RFC: `Agent.finalize(reward)` — End-of-Episode Hook for Agents

**Status:** DRAFT
**Author:** Oleksiy Ostapenko
**Date:** 2026-05-29
**Base branch:** `dev`
**Note on folder name:** the change folder is `multi-episode-rollouts/` for git
history continuity; the design has narrowed away from a dedicated `Rollout`
class to the single minimal hook described below.

---

## RFC scope vs. additive changes

Per the constitution (Pillar I, post-`068a718f`): "Additive, backward-compatible changes (a new method, a new optional field) skip this — just keep the living spec accurate and describe the change in the PR." The change here is one additive method on the `Agent` ABC; no RFC review strictly required by that rule. This document remains for traceability and to record the alternatives considered.

| Part | Layer | RFC-bound? |
|---|---|---|
| `Agent.finalize(reward: float) -> AgentOutput \| None` default no-op | cube-harness | No — additive method with default impl |
| `Episode` invokes `agent.finalize(reward)` after the per-turn loop exits; appends non-None return as a synthetic trajectory step | cube-harness | No — additive; default agents inherit a no-op |

No cube-standard changes. No `Observation` changes. No new fields anywhere. No new orchestration class. Just one method on `Agent` plus the matching call site in `Episode`.

---

## Design rationale (one paragraph)

**Add the smallest possible signal so agents can react to outcomes — and stop there.** An agent that wants to do anything outcome-conditioned (reflect, persist memory, log metrics, decide whether to retry on the next invocation) needs to know two things the framework currently doesn't expose to it: that the episode has ended, and what the final reward was. Episode-end is signaled by *the framework calling `finalize` at all*; the reward is the single parameter. Memory, retry policy, reflection cadence, and cross-episode coordination are all agent or recipe concerns — file-based persistence handles cross-episode state without framework help. No `Rollout` class, no `Observation` change, no per-step reward/done propagation. The framework grows by exactly one method.

---

## Problem

Agents in cube-harness see only `Observation` via `step()`. They have no signal that an episode has ended (Episode's loop simply stops calling them), and no access to the final reward (env-side on `EnvironmentOutput`, consumed by the framework, discarded before the agent's view).

This blocks anything outcome-conditioned:

- **Meta-RL** methods (LaMer — arxiv 2512.16848) that reflect on outcomes after each attempt.
- **Memory persistence** (file flushes at episode end).
- **Per-episode logging / metrics** that the agent wants to record alongside its own state.
- **Conditional retry behavior** an agent might want to express across episodes when invoked sequentially.

The smallest mechanism that unblocks all four is a single hook: the framework tells the agent "the episode is over, here's the reward."

## Scope

### In scope

- **`Agent.finalize(reward: float) -> AgentOutput | None`** — additive default-no-op method on the base `Agent` ABC.
- **Episode invokes it** after the per-turn loop exits (whether via env `done=True` or `max_steps`), passing the final reward from the last `EnvironmentOutput`.
- **Non-None returns** become synthetic trajectory steps — preserves XRay / cost stats / training-extraction visibility for any LLM call the agent makes inside `finalize()`.
- **Documented recipe-side convention** for on-disk layout of multi-rollout experiments (`<output_dir>/rollouts/<rollout_id>/{episodes,agent_state}/`). Not framework-enforced; recipes construct paths directly. See "Output directory layout" below.

### Out of scope

- **Anything on `Observation`.** No `reward` / `done` fields, no upstream cube-standard change.
- **A `Rollout` class** or any framework-level multi-episode orchestration. Sequential N-episode runs are recipe-level for-loops over `Episode.run()` if needed.
- **`Agent.reflect()` or any other inter-episode hook.** Reflection cadence is an agent concern — happens inside `finalize()` or persisted to file and read by the next invocation's `__init__`.
- **`Episode.run()` signature changes** (no `extra_metadata` kwarg, no other additions).
- **Reward aggregation across episodes.** Recipes compute whatever aggregate they want from per-episode trajectory rewards.
- **Trajectory metadata grouping** (no `rollout_id` etc.). Recipes that need to group trajectories use one experiment dir per group (the existing pattern).

## Design

### The signature

```python
class Agent(ABC):
    def finalize(self, reward: float) -> AgentOutput | None:
        """Called once at the end of each episode. Default: returns None (no-op).

        ``reward`` is the final reward from the last EnvironmentOutput
        produced by the task (the env-side authoritative value).

        Implementations may use this hook to persist memory (file writes),
        flush logs, record metrics, or run end-of-episode reflection LLM
        calls.

        When the agent runs an LLM call inside finalize, it SHOULD return
        an AgentOutput carrying that LLMCall (with a tag of the agent's
        choice — e.g. "reflection" or "finalize"). Episode then appends
        the returned AgentOutput as a synthetic trajectory step on the
        just-finished trajectory — making the finalize-time LLM call
        observable in XRay, cost stats, and training-data extraction.

        Returning None means "nothing to record" — agents that only
        persist state (memory file writes, log flushes) without LLM calls
        return None and rely on their own side effects.

        Standalone Episodes always invoke this method (after the per-turn
        loop exits, whether via env done=True or max_steps).
        """
        _ = reward
        return None
```

Three properties to note:

- **Default returns `None`.** Existing agents (ReAct, Genny, legacy) inherit the no-op and are byte-identical in behavior.
- **One scalar parameter.** No `Observation`, no `info` dict, no `Trajectory`. The agent has been tracking its own state through step() calls; `finalize` only adds the one piece the agent never saw (the final reward).
- **`AgentOutput | None` return.** Preserves trajectory observability for finalize-time LLM work — same mechanism XRay and cost stats already use for per-turn agent steps. Agents that only do side-effect work (file writes) return `None`.

### Episode flow

```python
# cube_harness/episode.py (sketch)
obs, info = task.reset()
env_output = EnvironmentOutput(obs=obs, info=info)
trajectory.append(env_output)

while not env_output.done and turns < max_steps:
    agent_output = agent.step(env_output.obs)
    trajectory.append(agent_output)
    env_output = task.step(agent_output.actions)
    trajectory.append(env_output)
    turns += 1

# After loop: agent learns the outcome
finalize_output = agent.finalize(env_output.reward)
if finalize_output is not None:
    trajectory.append(finalize_output)
```

Three invariants worth highlighting:

- `finalize()` is called **exactly once** per episode, after the per-turn loop exits.
- It receives the final `EnvironmentOutput.reward` (the env's authoritative outcome).
- Always called, including on max-steps termination and (under a `finally` block) on exception.

### Recipe pattern for multi-episode runs

A *rollout* is recipe-level vocabulary for "N sequential episodes against the same task with cross-episode memory." The framework has no notion of a rollout — it's a for-loop the recipe writes. Within one rollout, episodes are sequential (each depends on the prior's memory file). Across rollouts, recipes parallelize via `@ray.remote`.

#### Single rollout (sequential)

```python
# 5 sequential attempts at the same task, cross-episode memory via file
benchmark_config = MINIWOB_CONFIGS["default"]
task_config = next(iter(benchmark_config.get_task_configs()))

rollout_dir = experiment_dir / "rollouts" / "rollout_000"
rollout_dir.mkdir(parents=True, exist_ok=True)

agent_config = LaMerAgentConfig(
    llm_config=LLMConfig(model_name="openai/gpt-4o"),
    memory_path=rollout_dir / "agent_state" / "lamer.json",   # ← recipe controls the path
)

with benchmark_config.make() as benchmark:
    for k in range(5):
        Episode(
            id=k, output_dir=rollout_dir, agent_config=agent_config,
            task_config=task_config, exp_name="lamer", max_steps=10,
            storage=None, runtime_context=benchmark._runtime_context,
        ).run()
```

#### Many rollouts in parallel (the typical evaluation shape)

```python
import ray
from cube_harness import make_experiment_output_dir

experiment_dir = make_experiment_output_dir("lamer", "miniwob")
benchmark_config = MINIWOB_CONFIGS["default"]
task_configs = list(benchmark_config.get_task_configs())

@ray.remote
def run_rollout(task_config, experiment_dir, rollout_idx, n_episodes=5):
    rollout_dir = experiment_dir / "rollouts" / f"rollout_{rollout_idx:03d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    agent_config = LaMerAgentConfig(
        llm_config=LLMConfig(model_name="openai/gpt-4o"),
        memory_path=rollout_dir / "agent_state" / "lamer.json",
    )
    with benchmark_config.make() as benchmark:
        for k in range(n_episodes):
            Episode(
                id=k, output_dir=rollout_dir, agent_config=agent_config,
                task_config=task_config, exp_name="lamer", max_steps=10,
                storage=None, runtime_context=benchmark._runtime_context,
            ).run()

futures = [
    run_rollout.remote(tc, experiment_dir, i)
    for i, tc in enumerate(task_configs)
]
ray.get(futures)
```

Sequentiality is inner (the for-loop, where each episode reads the memory the prior wrote). Parallelism is outer (different rollouts run in different Ray workers, each with its own `rollout_dir` — no contention).

#### What the agent does

- `__init__`: read `self.config.memory_path` from disk into `self.memory`. If the file doesn't exist (first episode of a rollout), start empty.
- `step(obs)`: normal action selection; uses `self.memory` to inject context into prompts.
- `finalize(reward)`: append a reward-conditioned reflection to `self.memory`, write `self.memory` back to `memory_path`. Optionally return an `AgentOutput` carrying the reflection's `LLMCall` for trajectory observability.

The framework has zero involvement in cross-episode coordination. Each `Episode.run()` constructs a fresh agent; the *file* is what survives across episodes within a rollout.

### Output directory layout (recipe convention, not framework-enforced)

The recipe is responsible for the on-disk layout. The convention this proposal *recommends* (but does not enforce):

```
~/cube_harness_results/
└── <timestamp>_<exp_name>_<hash>/                ← one experiment dir per `make_experiment_output_dir()` call
    └── rollouts/
        ├── rollout_000/                          ← one rollout = N sequential episodes
        │   ├── episodes/                         ← FileStorage V2 layout, written by Episode
        │   │   ├── <traj_id_ep0>/
        │   │   │   ├── metadata.json
        │   │   │   └── steps/...
        │   │   ├── <traj_id_ep1>/
        │   │   └── ...
        │   └── agent_state/                      ← agent's own state files (recipe convention)
        │       └── lamer.json                    ← cross-episode memory; agent reads on __init__, writes on finalize
        ├── rollout_001/
        │   ├── episodes/...
        │   └── agent_state/...
        └── ...
```

Three properties of this layout:

- **One experiment dir, many rollouts.** `make_experiment_output_dir()` creates the top-level dir once per recipe invocation; the recipe constructs `rollouts/rollout_NNN/` subdirs for each rollout. Conceptually parallel to "an experiment evaluates an agent across many rollouts."
- **`agent_state/` is recipe convention.** The framework doesn't reference `agent_state/`; agents pick paths under `memory_path` (a `LaMerAgentConfig` field). The convention exists so recipes coalesce on one location and downstream tools (analysis scripts, judge runs) can find agent state predictably.
- **No framework helper for paths.** The recipe constructs `rollout_dir`, `memory_path`, and any other paths directly. Adding a `make_rollout_dir()` framework helper was considered and rejected — recipes need flexibility (different scope-keying, different conventions for non-LaMer agents), and the path construction is two lines.

### Lifecycle and cleanup

- The experiment dir lives in `~/cube_harness_results/` until the user cleans it up — same as non-rollout experiments today. Nothing auto-deletes.
- Memory files in `agent_state/` are not special — they're part of the experiment artifact, alongside trajectories. Analysis tooling can read them as part of the per-rollout record.
- Fresh start on next invocation: just call `make_experiment_output_dir()` again. New top-level dir → new `rollouts/` subtree → agent's `__init__` finds no `lamer.json` → starts cold.

## Alternatives considered

- **Add `reward` and `done` to `cube.core.Observation`.** Would let the agent observe outcomes per-step (not just at end). Rejected: requires upstream cube-standard RFC (cross-repo), and unnecessary for the use cases on the table — every outcome-aware behavior identified so far reacts at *end of episode*, not mid-episode. `finalize` covers them all with one call site, zero upstream changes.
- **Pass `terminal_obs: Observation` to `finalize`.** Gives the agent the env's final state. Rejected: the agent has been receiving Observations via `step()` all along and has its own internal state. The one thing missing — the reward — is a scalar that doesn't need a whole `Observation` wrapper. Tighter signature, less overhead.
- **Pass `info: dict` to `finalize`.** Would give the agent task-specific metadata at episode end. Rejected: agents that need this can stash it during `step()` (since they can see `Observation.contents` and the task already routes info into obs). Keeping `finalize` to one scalar is the minimal contract.
- **A dedicated `Rollout` class + `Agent.reflect(trajectory, reward)` hook.** Earlier design iteration in this same RFC folder. Rejected: bakes meta-RL-specific orchestration into the framework. `Rollout` is one recipe-level for-loop; `reflect`'s `trajectory` parameter is information the agent already has (it received every obs via `step()`).
- **File-based memory as a framework-provided helper (path-construction, cleanup).** Considered as a framework affordance (e.g., `make_rollout_dir(experiment_dir, rollout_id) -> Path`). Rejected: the path construction is two lines of recipe code and different agents/recipes need different keying schemes (per-task, per-rollout, per-config-variant). The proposal documents a *recommended on-disk layout* (`<output_dir>/rollouts/<rollout_id>/{episodes,agent_state}/`) so recipes coalesce on one convention, but the framework provides no code — recipes construct paths directly.
- **Drop `finalize` too; agent infers episode end from step-count heuristics.** Rejected: agent has no reliable signal — the env's `done` flag is invisible to it, and step-count thresholds are fragile across tasks with variable lengths. `finalize` is the minimum signal that's actually reliable.

## Open questions

1. **`finalize()` on errors.** If the episode terminates via exception in `step()` or `task.step()`, should `finalize()` still be called? Proposed: yes — called in a `finally` block, so cleanup runs. The reward passed in that case is whatever `EnvironmentOutput.reward` was at the time of the failure (typically `0.0`).
2. **Return type: `AgentOutput | None` vs strict `None`.** Proposed: `AgentOutput | None`. Strict `None` would be one fewer concept on the ABC but would make finalize-time LLM calls invisible to the trajectory (no observability for cost stats, XRay, training extraction). The `AgentOutput | None` shape preserves observability at the cost of two extra concepts (return type + the framework's append behavior).
3. **Whether to pass `done` or `truncated` alongside reward.** Could be useful for an agent that wants to distinguish "env signaled completion" from "max_steps hit." Proposed: no for V1 — keep the signature minimal; agents that need the distinction can check whether their own action history reached `max_steps`.
4. **XRay nested-layout support.** XRay currently scans `<output_dir>/episodes/<traj_id>/` directly. Under the recommended nested layout (`<output_dir>/rollouts/<rollout_id>/episodes/<traj_id>/`), XRay would need a small update to traverse the rollout subdir level. Not in scope for this RFC (XRay change lands as a follow-up); the layout works for everything else (analysis scripts, judge, training extraction) since they walk file paths directly.
5. **Folder name.** The change folder is still `multi-episode-rollouts/` for git continuity. Optional cosmetic rename to `agent-finalize-hook/` or similar later — not load-bearing.

## References

- LaMer paper: [arxiv 2512.16848](https://arxiv.org/abs/2512.16848) — meta-RL via cross-episode reflection. Motivating use case; `finalize(reward)` is sufficient for an agent implementing LaMer.
- `cube-harness/.ai_docs/cube-harness-architecture.md` — repo-level architecture reference.
- Schemas + ADDED/MODIFIED sections: `deltas.md`.
