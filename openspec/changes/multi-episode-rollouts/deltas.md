# Deltas: `Agent.finalize(reward)` — End-of-Episode Hook

Applies to:
- `openspec/specs/agent/spec.md` — modified (additive: new default-no-op `finalize()` method).
- `openspec/specs/episode/spec.md` — modified (Episode invokes `finalize(reward)` after the per-turn loop; non-None returns appended as synthetic trajectory step).

Both changes are additive backward-compatible per the constitution's Pillar I exception — no RFC required. This document is the spec delta for traceability.

No cube-standard changes. No new types, no new fields, no new orchestration class.

---

## MODIFIED — `openspec/specs/agent/spec.md`

### `Agent` base class — new optional method

```python
def finalize(self, reward: float) -> AgentOutput | None:
    """Called once at the end of each episode. Default: returns None (no-op).

    ``reward`` is the final reward from the last ``EnvironmentOutput``
    produced by the task.

    Implementations may use this hook to persist memory (file writes),
    flush logs, record metrics, or run end-of-episode reflection LLM calls.

    When the implementation runs an LLM call as part of finalize, it
    SHOULD return an ``AgentOutput`` carrying that ``LLMCall`` (with a
    tag of the agent's choice — e.g. ``"reflection"`` or ``"finalize"``).
    ``Episode`` then appends the returned ``AgentOutput`` as a synthetic
    trajectory step on the just-finished trajectory, making the
    finalize-time LLM call observable in XRay, cost stats, and
    training-data extraction. Returning ``None`` means "nothing to
    record" — pure side-effect agents (memory file writes only) return
    ``None``.
    """
    _ = reward
    return None
```

### Invariants (new)

- Default implementation returns `None`; subclasses opt in by overriding.
- `finalize(reward)` is invoked exactly once per episode, after the per-turn loop exits (whether via env `done=True` or `max_steps`).
- `reward` is the `EnvironmentOutput.reward` value from the last `task.step()` (or from `task.reset()`'s implicit zero-reward state, if no `step()` ever ran).
- When `finalize()` returns a non-None `AgentOutput`, the framework appends it as a `TrajectoryStep` on the just-finished trajectory before persistence.
- When `finalize()` raises, the exception propagates (treated as an episode-level error). Implementations that want to suppress failures (e.g., a failed reflection LLM call) should catch internally and return `None`.

### Not changed

- `Agent.step()` signature and contract.
- `AgentConfig.make()` contract.
- `AgentOutput` structural shape — finalize-time AgentOutput has the same shape as a per-turn one.
- `Agent` class attributes (`name`, `description`, `input_content_types`, `output_content_types`).
- ReAct, Genny, legacy_generic_agent — they inherit the default-no-op `finalize()` and require no source changes.

---

## MODIFIED — `openspec/specs/episode/spec.md`

### Episode calls `agent.finalize(reward)` after the per-turn loop

```python
# Inside Episode._run_loop, after the existing per-turn loop exits:
finalize_output = agent.finalize(env_output.reward)
if finalize_output is not None:
    trajectory.steps.append(TrajectoryStep(output=finalize_output, ...))
```

The `reward` passed is from the same `env_output` that drove the loop-exit decision — either the env's `done=True` `EnvironmentOutput`, or the last one produced before `max_steps` fired.

### Invariants (new)

- `agent.finalize(reward)` is invoked exactly once per `Episode.run()` call, regardless of how the per-turn loop terminated.
- Invocation is in a `try/finally` block (or equivalent) such that finalize runs even if a `step()` call raised — supports cleanup workloads like memory persistence.
- Non-None `finalize()` return values are appended to the trajectory **before** the trajectory is finalized for persistence (i.e., the synthetic step is part of the saved trajectory, not added post-hoc).
- The append happens whether the episode terminated via `done`, `max_steps`, or exception.

### Not changed

- `Episode.run()` signature — still `() -> Trajectory`.
- `Episode._run_loop` core structure — only the post-loop finalize call is new.
- `EpisodeConfig` fields — unchanged.
- Episode's persistence behavior — same `FileStorage` layout. Per-trajectory metadata (`task_id`, `agent_name`, etc.) is unchanged.
- The per-turn step loop itself — agents receive `Observation` exactly as today; reward and done remain env-side on `EnvironmentOutput` and are NOT propagated into the agent's per-step view.

---

## NOT CHANGED (explicitly)

- **`cube-standard`** — no upstream changes. `Observation`, `EnvironmentOutput`, `Task`, `Benchmark`, `Tool`, `Resource`, `Action`, `ActionSchema` are all unchanged. Agents see exactly what they see today in `step()` — only the end-of-episode notification is new.
- **`Trajectory`, `TrajectoryStep`, `AgentOutput` structural shape** — no field changes.
- **`Rollout` (proposed in an earlier iteration of this RFC) — not added.** Multi-episode runs are recipe-level for-loops.
- **`Experiment`, `ExpResult`, `exp_runner`** — no changes.
- **Concrete cubes (`cubes/miniwob/`, `cubes/arithmetic-cube/`, etc.)** — no changes.
- **`LLM`, `LLMConfig`, `Prompt`, `LLMCall`** — no changes.
- **ReAct, Genny, legacy_generic_agent** — unchanged source; inherit the default-no-op `finalize()`.
- **XRay viewer** — finalize-time `AgentOutput` (when non-None) appears as a regular trajectory step; XRay reads it via the existing mechanism. No XRay code changes.
- **MCP server, metrics/tracer** — no changes.
- **Storage layout / `FileStorage`** — no changes. The synthetic finalize step lands in the same `steps/` directory as the rest.

## Migration

Fully backward-compatible:

- Existing agents inherit `finalize()` from the base ABC; the no-op default means no behavioral change.
- Existing episodes will start calling `agent.finalize(reward)`, but since the default is a no-op returning `None`, nothing visible changes for any current agent or recipe.
- Existing trajectories are byte-identical for agents that don't override `finalize()`.
- No cube changes required. No cube-standard changes required.
