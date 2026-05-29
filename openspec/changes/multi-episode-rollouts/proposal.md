# RFC: Outcome-Aware Agents — Reward/Done in Observation + `Agent.finalize()`

**Status:** DRAFT
**Author:** Oleksiy Ostapenko
**Date:** 2026-05-25
**Base branch:** `dev`
**Note on folder name:** the change folder is `multi-episode-rollouts/` for git
history continuity; the design has since pivoted away from a dedicated `Rollout`
class to the more general outcome-awareness approach below.

---

## RFC scope vs. additive changes

Per the constitution (Pillar I, post-`068a718f`): "Additive, backward-compatible changes (a new method, a new optional field) skip this — just keep the living spec accurate and describe the change in the PR." All four parts of this change are additive; none are RFC-bound by that rule. The doc remains for traceability and because part #1 is an **upstream** cube-standard change that needs its own review there.

| Part | Layer | RFC-bound? |
|---|---|---|
| `Observation.reward: float \| None` and `Observation.done: bool \| None` | **cube-standard** (upstream) | Additive — but lands upstream; needs cube-standard review |
| `Episode` populates `obs.reward` / `obs.done` when calling `agent.step()` | cube-harness | No — internal change, contract unchanged |
| `Agent.finalize(terminal_obs: Observation) -> AgentOutput \| None` default no-op | cube-harness | No — additive method with default impl |
| `Episode` calls `agent.finalize(terminal_obs)` after the loop exits; if it returns a non-None `AgentOutput`, appends it as a synthetic trajectory step | cube-harness | No — additive; default agents inherit a no-op |
| `Episode.run(extra_metadata=None)` optional kwarg merged into `trajectory.metadata` | cube-harness | No — additive optional parameter |

---

## Design rationale (one paragraph)

**Add the minimum mechanism that makes agents outcome-aware; let everything else (memory, reflection cadence, multi-episode orchestration) be the agent's or recipe's concern.** Two pieces of information are env-internal today: `reward` and `done`. Exposing them in `Observation` is broadly useful — meta-RL methods (LaMer-style reflection), online learning, retry-on-failure, reward-shaped behaviour, fail-fast logic. Adding `Agent.finalize(terminal_obs)` gives the agent one hook to see the *terminal* observation (the one it never receives via `step()` because the env returns it after the agent's last action). With those two additions, agents that want cross-episode adaptation manage their own memory (typically file-based, keyed however they like) and run their own reflection cadence inside `step()` or in `finalize()`. The framework adds **zero** orchestration classes — multi-episode runs are recipe-level for-loops over `Episode.run()`. That keeps the framework opinion-free and the agent-side complexity proportional to how sophisticated the agent wants to be.

---

## Problem

Agents in cube-harness can't currently see two pieces of information the environment produces:

- **`reward`** — the outcome signal. Lives on `EnvironmentOutput`. The framework consumes it (for trajectory recording, episode termination logic); the agent never receives it.
- **`done`** — the env's episode-end signal. Same shape: env-side only.

This blocks a whole class of agent designs:

- **Meta-RL methods** that condition reflection on outcomes (LaMer — *arxiv 2512.16848*). The motivating example.
- **Online learning** — agents that want to update behaviour based on reward signals within a run.
- **Retry-on-failure** — an agent that's about to call `final_step` but sees `reward < threshold` could try one more thing.
- **Outcome-aware logging** — agents that want to record their own per-task metrics.

There's also no clean way for an agent to know "this is the end of the episode" — the framework breaks the episode loop, but the agent isn't called again to react to the terminal state.

## Scope

### In scope (V1)

- **cube-standard (upstream RFC):** add `Observation.reward: float | None` and `Observation.done: bool | None` as additive optional fields. Default `None` (existing Observations unchanged; existing agents that don't read them are unaffected).
- **cube-harness Episode:** when calling `agent.step(obs)`, populate `obs.reward` and `obs.done` from the *prior* `env_output`. First step of an episode (post-reset): both `None`.
- **cube-harness Agent:** add `Agent.finalize(terminal_obs: Observation) -> AgentOutput | None` as a default-no-op method on the base ABC. Default returns `None`. Implementations that run an LLM call inside finalize (e.g. for end-of-episode reflection or persistence) return an `AgentOutput` carrying that call's `LLMCall` so it's observable in XRay / cost stats / training extraction. Existing agents (ReAct, Genny, legacy) inherit the no-op.
- **cube-harness Episode:** after the per-turn loop exits (whether via `done=True` or `max_steps`), call `agent.finalize(terminal_obs)` once. `terminal_obs` is the post-final-action observation with `reward` and `done` populated. If finalize returns a non-None `AgentOutput`, Episode appends it as a synthetic trajectory step on the just-finished trajectory before persisting.
- **cube-harness Episode:** add optional `extra_metadata: dict | None = None` kwarg to `Episode.run()`. When provided, merged into `trajectory.metadata` at trajectory-creation time. Lets recipes inject `rollout_id`, ablation labels, seeds, etc. into per-trajectory metadata for cross-trajectory filtering by downstream tools (training backends, analysis, XRay grouping).
- **Demonstration recipe:** `recipes/lamer_miniwob.py` shows a LaMer-style agent using these primitives — file-based cross-episode memory, in-step reflection triggered by `obs.done == True`, finalize hook for memory persistence.

### Out of scope (orthogonal / deferred concerns, not design alternatives)

- **Reward aggregation across episodes.** Recipes compute whatever aggregate they want from per-episode trajectory rewards (discounted sum, mean, max, custom). Not a framework concern.
- **Ray parallelism over multi-episode runs.** Existing `run_with_ray(exp)` over an Experiment whose Benchmark yields N copies of the same task covers this. No new parallelism primitive needed.
- **Reward-model integration / meta-RL training loop.** Training backend's concern — consumes per-episode trajectory rewards as it does today.

(Design alternatives — including `Rollout` class, `Agent.reflect()` hook, file-based memory as a framework concern — are in *Alternatives considered* below, with rejection reasons.)

## Design

### Observation augmentation

```python
# cube-standard
class Observation:
    contents: list[Content]
    reward: float | None = None     # NEW — additive, optional
    done: bool | None = None        # NEW — additive, optional
```

Both fields default `None`. An agent that doesn't read them is byte-identical in behaviour to today.

### Episode flow

```python
# cube-harness Episode._run_loop (simplified)
obs, info = task.reset()
env_output = EnvironmentOutput(obs=obs, info=info)   # reward=0.0, done=False default
trajectory.append(env_output)

while not env_output.done and turns < max_steps:
    obs_for_agent = _augment(env_output.obs, reward=env_output.reward, done=env_output.done)
    agent_output = agent.step(obs_for_agent)         # ← agent sees prior reward/done
    trajectory.append(agent_output)
    env_output = task.step(agent_output.actions)
    trajectory.append(env_output)
    turns += 1

# After loop: agent sees terminal state once
terminal_obs = _augment(env_output.obs, reward=env_output.reward, done=env_output.done)
finalize_output = agent.finalize(terminal_obs)   # ← default returns None
if finalize_output is not None:
    trajectory.append(finalize_output)            # ← LLM calls in finalize end up in the trajectory
```

Two invariants worth highlighting:

- The agent observes reward/done from the **prior** step's `env_output`, not the current step. So at step k, the agent sees what happened at step k-1.
- `finalize()` is called exactly once per episode (after the loop exits), with the actual terminal observation that the agent never sees via `step()`. Always called, including on max-steps-hit.

### `Agent.finalize()` semantics

```python
class Agent(ABC):
    def finalize(self, terminal_obs: Observation) -> AgentOutput | None:
        """Called once at the end of each episode. Default: returns None (no-op).

        terminal_obs carries the final reward and (typically) done=True. Agents
        may use this hook to persist memory, flush logs, record metrics, write
        reflections, etc.

        When the agent runs an LLM call as part of finalize (e.g. end-of-episode
        reflection), it should return an AgentOutput carrying that LLMCall (tag
        of the caller's choice — e.g. "reflection" or "finalize"). Episode
        appends the returned AgentOutput as a synthetic trajectory step on the
        just-finished trajectory — making any finalize-time LLM call observable
        in XRay, cost stats, and training-data extraction.

        Returning None means "nothing to record" — agents that only persist
        state (memory file writes, log flushes) without LLM calls can return
        None and rely on their own side effects.
        """
```

Three properties:

- Default returns `None` — existing agents unaffected.
- Receives the terminal observation (with reward + done) — the one thing the agent never gets via `step()`.
- When non-None: the returned `AgentOutput` becomes a synthetic trajectory step. Same mechanism XRay and cost stats already use for per-turn agent steps; works for free.

### `Episode.run(extra_metadata=...)` semantics

```python
class Episode:
    def run(self, extra_metadata: dict | None = None) -> Trajectory:
        ...
        # Inside _run_loop, when constructing the Trajectory:
        trajectory.metadata = {
            "task_id": ...,
            "agent_name": ...,
            "action_schemas": ...,
            **(extra_metadata or {}),  # ← caller's keys merged in
        }
```

Optional kwarg. Defaults to `None` (no extra metadata, behaviour unchanged). When provided, the dict's keys land in `trajectory.metadata` and persist to `episodes/<traj_id>/metadata.json`. Lets recipes inject:

- `rollout_id` to group N trajectories produced by the same recipe loop.
- Ablation / variant labels (`{"variant": "with_reflection"}`).
- Seeds, experiment IDs, anything recipe-side worth filtering by later.

The framework doesn't interpret these keys — they're recipe convention. Conflict resolution: caller's keys override built-in keys (recipe wins; allows e.g. overriding `agent_name` if needed, though discouraged).

### Recipe pattern for multi-episode runs

```python
# recipes/lamer_miniwob.py (sketch)
benchmark = MINIWOB_CONFIGS["default"].make()
task_config = next(iter(benchmark.get_task_configs()))
agent_config = LaMerAgentConfig(llm_config=LLMConfig(model_name="openai/gpt-4o"))

with benchmark:
    rollout_id = str(uuid.uuid4())
    for k in range(n_episodes):
        Episode(
            id=k, output_dir=output_dir, agent_config=agent_config,
            task_config=task_config, exp_name="lamer", max_steps=10,
            storage=None, runtime_context=benchmark._runtime_context,
        ).run(extra_metadata={"rollout_id": rollout_id, "episode_index": k})
        # Memory persists across iterations via the agent's file I/O.
        # extra_metadata threads rollout grouping into each trajectory for
        # downstream filtering (training backend, XRay, analysis).
```

The agent (LaMerAgent) reads memory from a file in `__init__`, reflects on `obs.done == True` in `step()`, persists in `finalize()`. The framework has zero involvement in any of that — the loop is recipe code.

## Alternatives considered

- **Add a dedicated `Rollout` class and `Agent.reflect()` hook (the prior V1 design).** Rejected: bakes meta-RL-specific orchestration into the framework. The Rollout class duplicates Episode's loop or requires an `Episode.run_with(task, agent)` helper just to share it. `reflect()` imposes a specific reflection cadence ("between episodes") on every agent. The current design generalizes: the same `reward`/`done`/`finalize` primitives serve meta-RL, online learning, retry-on-failure, and anything else outcome-aware — without specialized framework abstractions.
- **Pass `EnvironmentOutput` to `step()` instead of `Observation`.** Rejected: bigger contract change (every existing agent's `step()` signature would need updating). The additive-fields-on-Observation approach keeps the existing contract intact.
- **Episode.run_with(task, agent)** (composing rollout-style loops with a long-lived agent). Considered when a `Rollout` class was on the table. Dropped along with `Rollout`: with file-based memory, the agent doesn't need to live across episodes — each `Episode.run()` constructs a fresh agent which reads its memory from disk. The "agent persistence" requirement disappears, and so does the need for `run_with`.
- **File-based memory as a framework concern.** Rejected: agents handle their own persistence. Different agents will key memory differently (per-task, per-rollout-id, per-config); a framework helper would either be too generic to be useful or too specific to be general. The framework provides `finalize()` as the hook; what to do with it is agent-side.
- **Pass `terminal_obs` to a special last `step()` call instead of a separate `finalize()`.** Rejected: muddies `step()`'s purpose (action selection). A separate `finalize()` makes the "observe-only, no action" semantics explicit and frees the agent from returning a meaningless `AgentOutput`.

## Open questions

1. **First-step semantics for `obs.reward` / `obs.done`.** At the very first `step()` of an episode (post-reset), there's no prior `env_output`. Should the agent see `reward=None, done=None`, or `reward=0.0, done=False`? V1: `None` for both, signaling "no prior step." Agents that want a default of `0.0` can normalise themselves.
2. **`finalize()` on errors.** If the episode terminates via exception in `step()` or `task.step()`, should `finalize()` still be called? V1: yes — called in a `finally` block, so cleanup runs. Agents that want to distinguish should check `terminal_obs.done`.
3. **Trajectory recording of `finalize()`-time LLM calls.** Resolved: `finalize()` returns `AgentOutput | None`. Non-None returns are appended as synthetic trajectory steps; pure side-effects (memory file writes, log flushes) return `None`. Agents pick per call.
4. **Naming.** The change folder is `multi-episode-rollouts/` for git continuity; the design's scope has narrowed and broadened in different ways. Renaming the folder is optional cleanup.

## References

- LaMer paper: [arxiv 2512.16848](https://arxiv.org/abs/2512.16848) — meta-RL via cross-episode reflection. The motivating use case; the current design's `reward`/`done`/`finalize` primitives let an agent implement LaMer without framework support.
- cube-standard `Observation`: `cube.core.Observation` — the upstream type that gains `reward` and `done` fields.
- Schemas + ADDED/MODIFIED sections: `deltas.md`.
- POC agent: `recipes/lamer_miniwob.py` (file-based memory, in-step reflection, finalize hook).
