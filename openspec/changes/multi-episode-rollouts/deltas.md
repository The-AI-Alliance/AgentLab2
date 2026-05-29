# Deltas: Outcome-Aware Agents — Reward/Done on Observation + `Agent.finalize()`

Applies to:
- **`cube-standard`** (upstream): `cube.core.Observation` gains optional `reward` and `done` fields. Additive; needs its own RFC in cube-standard.
- `openspec/specs/agent/spec.md` — modified (additive: new default-no-op `finalize()` method).
- `openspec/specs/episode/spec.md` — modified (Episode populates `obs.reward` / `obs.done` when calling `agent.step()`; calls `agent.finalize(terminal_obs)` after the loop; new optional `extra_metadata` kwarg on `Episode.run()`).

All cube-harness changes are additive backward-compatible per the constitution's Pillar I exception — no RFC required for them in cube-harness itself. The upstream cube-standard change requires its own RFC there. This document is the cube-harness-side spec delta; the cube-standard RFC is a precondition.

---

## DEPENDS ON — `cube-standard` (upstream RFC)

### `cube.core.Observation` — two additive optional fields

```python
class Observation:
    contents: list[Content]
    reward: float | None = None     # NEW — additive optional
    done: bool | None = None        # NEW — additive optional
```

Both default `None`. Existing Observations (no caller setting these) are unchanged. Agents that don't read them are byte-identical in behavior to today.

This change is **the foundational primitive** the cube-harness changes below build on. It is broadly useful beyond meta-RL: any outcome-aware agent (online learning, retry-on-failure, reward-shaped policies, fail-fast heuristics) benefits. The cube-standard RFC should be motivated on its own terms, not as "for LaMer."

---

## MODIFIED — `openspec/specs/agent/spec.md`

### `Agent` base class — new optional method

```python
def finalize(self, terminal_obs: Observation) -> AgentOutput | None:
    """Called once at the end of each episode. Default: returns None (no-op).

    ``terminal_obs`` is the post-final-action observation carrying the final
    reward and (typically) ``done=True``. This is the one piece of state the
    agent never sees via ``step()`` — the env's response to the agent's last
    action.

    Implementations may use this hook to persist memory (file writes), flush
    logs, record metrics, or run end-of-episode reflection LLM calls.

    When the implementation runs an LLM call as part of finalize (e.g. a
    reflection), it SHOULD return an ``AgentOutput`` carrying that ``LLMCall``
    (with a tag of the agent's choice — e.g. ``"reflection"`` or
    ``"finalize"``). ``Episode`` then appends the returned ``AgentOutput`` as
    a synthetic trajectory step on the just-finished episode's trajectory,
    making the finalize-time LLM call observable in XRay, cost stats, and
    training-data extraction. Returning ``None`` means "nothing to record" —
    pure side-effect agents (memory file writes only) return ``None``.

    Standalone Episodes always invoke this method (after the per-turn loop
    exits, whether via ``done=True`` or ``max_steps``).
    """
    _ = terminal_obs
    return None
```

### Invariants

- Default implementation returns `None`; subclasses opt in by overriding.
- `finalize()` is invoked exactly once per episode by `Episode`, after the per-turn loop exits (whether via env `done=True` or `max_steps`).
- `terminal_obs.reward` and `terminal_obs.done` are populated from the final `env_output`.
- When `finalize()` returns a non-None `AgentOutput`, the framework appends it as a `TrajectoryStep` on the just-finished trajectory before persistence.
- When `finalize()` raises, the exception propagates (treated as an episode-level error); agents that want to suppress LLM failures should catch internally and return `None`.

### Not changed

- `Agent.step()` signature and contract.
- `AgentConfig.make()` contract.
- ReAct, Genny, legacy_generic_agent — they inherit the default-no-op `finalize()` and require no source changes.

---

## MODIFIED — `openspec/specs/episode/spec.md`

### Episode populates `obs.reward` / `obs.done` when calling `agent.step()`

When `Episode._run_loop` invokes `agent.step(obs)`, the Observation it passes has `reward` and `done` populated from the **prior** `env_output`. First call after `task.reset()`: both `None`.

```python
# Inside Episode._run_loop (sketch):
obs, info = task.reset()
env_output = EnvironmentOutput(obs=obs, info=info)

while not env_output.done and turns < max_steps:
    obs_for_agent = _augment(env_output.obs, reward=env_output.reward, done=env_output.done)
    agent_output = agent.step(obs_for_agent)
    env_output = task.step(agent_output.actions)
    turns += 1
```

The `_augment` helper constructs a new `Observation` with reward/done set, leaving the env-side `EnvironmentOutput.obs` untouched.

### Episode calls `agent.finalize(terminal_obs)` after the loop

After the per-turn loop exits, Episode invokes the agent's finalize hook once:

```python
# After the loop:
terminal_obs = _augment(env_output.obs, reward=env_output.reward, done=env_output.done)
finalize_output = agent.finalize(terminal_obs)
if finalize_output is not None:
    trajectory.steps.append(TrajectoryStep(output=finalize_output, ...))
```

### `Episode.run(extra_metadata: dict | None = None)` — new optional kwarg

```python
class Episode:
    def run(self, extra_metadata: dict | None = None) -> Trajectory:
        ...
```

When provided, the dict's keys are merged into `trajectory.metadata` at trajectory-construction time:

```python
trajectory.metadata = {
    "task_id": ...,
    "agent_name": ...,
    "action_schemas": ...,
    **(extra_metadata or {}),  # ← caller's keys merged in
}
```

The framework doesn't interpret these keys. Recipes use this to inject:
- `rollout_id` to group N trajectories produced by the same recipe loop.
- Ablation / variant labels (`{"variant": "with_reflection"}`).
- Seeds, experiment IDs, anything recipe-side worth filtering by later.

**Conflict resolution:** caller's keys override built-in keys (recipe wins; allows overriding e.g. `agent_name` if needed, though discouraged).

### Invariants (additions)

- The observation passed to `agent.step()` carries `reward` and `done` from the **prior** `env_output` (not the current one). First step of an episode: both `None`.
- `agent.finalize(terminal_obs)` is invoked exactly once per episode, after the per-turn loop exits.
- `finalize()` is called in a `finally` block (or equivalent) so it runs even when the episode terminates via exception.
- Non-None `finalize()` return values are appended to the trajectory before persistence.
- `extra_metadata` keys are merged into `trajectory.metadata`; on collision with built-in keys (`task_id`, `agent_name`, `action_schemas`), the caller-provided value wins.

### Not changed

- `Episode.run()` return type — still `Trajectory`.
- `Episode._run_loop` core structure — only the obs construction and finalize call are new.
- `EpisodeConfig` fields — unchanged.
- Episode's persistence behavior — same `FileStorage` layout.

---

## NOT CHANGED

- `Trajectory`, `TrajectoryStep`, `AgentOutput` structural shape — only `Observation` gets new fields (upstream).
- `Experiment`, `ExpResult`, `exp_runner` — Episode-level changes; runners unaffected.
- `Task`, `Benchmark`, `Tool`, `Resource` (cube-standard contracts) — only `Observation` adds fields.
- Concrete cubes (`cubes/miniwob/`, `cubes/arithmetic-cube/`, etc.) — no changes. Cubes can optionally start populating new `Observation.reward` / `done` fields when they return obs, but they're not required to.
- `LLM`, `LLMConfig`, `Prompt`, `LLMCall` — no changes.
- ReAct, Genny, legacy_generic_agent — unchanged source. They inherit the default-no-op `finalize()`.
- XRay viewer — finalize-time AgentOutput appears as a regular trajectory step; XRay reads it via the existing mechanism.
- MCP server, metrics/tracer — no changes.
- Storage layout / `FileStorage` — no changes. Trajectory metadata can grow (via `extra_metadata`) but the file layout is unchanged.
