# Deltas: Multi-Episode Rollouts

Applies to:
- `openspec/specs/rollout/spec.md` — **new layer, RFC-bound**
- `openspec/specs/agent/spec.md` — modified (additive: new default-no-op method; no RFC required per constitution Pillar I)
- `openspec/specs/storage/spec.md` — modified (additive: new optional fields; no RFC required)
- `openspec/README.md` — modified (add rollout layer to index)

The agent and storage deltas are listed here for completeness and traceability,
even though both are additive backward-compatible changes that can be applied
directly to their specs in the implementation PR per the constitution.

---

## ADDED — `openspec/specs/rollout/spec.md` (NEW LAYER)

### Module

`cube_harness/rollout.py`

### Public types

- `RolloutConfig` — `TypedBaseModel`:
  - `n_episodes: int` (≥ 1)
  - `gamma: float = 1.0` (in `(0, 1]`; 1.0 = no discount)

- `RolloutResult` — `TypedBaseModel`:
  - `rollout_id: str` — uuid generated at rollout start; links the per-episode trajectories
  - `trajectories: list[Trajectory]` — one per episode, in execution order
  - `per_episode_rewards: list[float]` — raw final reward per episode
  - `discounted_reward: float` — `Σ γ^k · r_k` for `k in [0, n_episodes)`
  - `agent_config_dump: dict` — serialized config of the agent that ran the rollout (for provenance)

- `Rollout` — orchestrator class:
  - `__init__(self, task: Task, agent: Agent, config: RolloutConfig) -> None`
  - `run(self) -> RolloutResult`

### Public methods

- `Rollout.run() -> RolloutResult`

  Executes `n_episodes` against `task` reusing the same `agent` instance. Between
  episodes calls `task.reset()` then `agent.reflect(trajectory, final_reward)`.
  Reflection is **not** called after the final episode.

### Invariants

- The same `Agent` instance handles all `n_episodes` within one `Rollout.run()` call.
- `task.reset()` is invoked before each episode (including the first).
- `agent.reflect()` is invoked after each non-final episode (`n_episodes - 1` times).
- `len(trajectories) == len(per_episode_rewards) == n_episodes` on a normal completion.
- `discounted_reward = sum(gamma**k * r for k, r in enumerate(per_episode_rewards))`.
- `rollout_id` is unique per `Rollout.run()` call (uuid4).

### Gotchas

- `Rollout` keeps the `Agent` alive across episodes. The agent does **not** cross
  process boundaries — `Rollout.run()` executes entirely within one worker. Ray
  parallelism (V2) parallelizes whole rollouts, not episodes within a rollout.
- Per-episode `max_actions` semantics are unchanged. There is no per-rollout action
  budget in V1.
- `Task.reset()` must restore the task to its initial state without recreating its
  `Resource`. Cubes whose reset is expensive pay that cost `n_episodes` times.

---

## MODIFIED — `openspec/specs/agent/spec.md`

### `Agent` base class — new optional method

```python
def reflect(self, trajectory: Trajectory, final_reward: float) -> None:
    """Called between episodes within a Rollout.

    Default implementation: no-op. Implementations may update internal state
    that subsequent step() calls read (e.g., a memory list spliced into prompts).

    Standalone Episode runs never invoke this method — backwards compatible.
    """
```

### Invariants (new)

- Default implementation on `Agent` is a no-op; subclasses opt in by overriding.
- `reflect()` is invoked exactly once per non-final episode within a `Rollout`,
  with the just-finished trajectory and its final scalar reward.
- `reflect()` is **never** called by `Episode` (single-episode execution path).
- Implementations may raise; the rollout treats reflection errors the same as
  episode-level errors (propagated up).

### Not changed

- `Agent.step()` signature and contract.
- `AgentConfig.make()` contract.
- ReAct, Genny, legacy_generic_agent — they inherit the no-op default and require
  no source changes to remain functional inside a `Rollout`.

---

## MODIFIED — `openspec/specs/storage/spec.md`

### Trajectory metadata extension

Trajectories produced inside a `Rollout` carry two new optional fields:

- `rollout_id: str | None` — uuid linking trajectories from the same rollout; `None`
  for trajectories produced by a standalone `Episode`.
- `episode_index_in_rollout: int | None` — 0-indexed position within the rollout;
  `None` outside a rollout.

Both fields are additive and default to `None`. Pre-existing trajectory files remain
readable without migration.

### File layout extension

`Rollout` writes a sibling `rollout_record.json` alongside the per-episode
directories:

```
<experiment_dir>/
└── rollouts/
    └── <rollout_id>/
        ├── rollout_record.json    ← RolloutResult dump (less trajectories)
        └── episodes/
            ├── ep0/...            ← existing Episode storage layout, unchanged
            ├── ep1/...
            └── ...
```

`rollout_record.json` contains `rollout_id`, `per_episode_rewards`,
`discounted_reward`, `gamma`, `n_episodes`, and `agent_config_dump`. Per-episode
trajectories live where `Episode` already writes them; the rollout record points to
them by relative path.

---

## MODIFIED — `openspec/README.md`

Add row to the layer index:

| Layer | Source | Spec |
|-------|--------|------|
| Rollout | `src/cube_harness/rollout.py` | [rollout/spec.md](specs/rollout/spec.md) |

---

## NOT CHANGED

- `Episode`, `EpisodeConfig`, `MAX_STEPS` — `Rollout` uses `Episode` internally;
  no signature or behavior changes to the existing layer.
- `Experiment`, `ExpResult`, `exp_runner` — V1 wires `Rollout` from a recipe; runner
  integration (parallel rollouts via Ray) is V2.
- `Task`, `Benchmark`, `Tool`, `Resource` (cube-standard contracts) — `Task.reset()`
  is already part of the protocol; `Rollout` consumes it as-is.
- Concrete cubes (`cubes/miniwob/`, `cubes/arithmetic-cube/`, etc.) — no changes.
- `LLM`, `LLMConfig`, `Prompt`, `LLMCall` — no changes.
- `Trajectory`, `TrajectoryStep`, `AgentOutput` structural shape — only metadata
  additions on trajectory records (see storage delta).
- ReAct, Genny, legacy_generic_agent — unchanged source; behavior in `Rollout`
  follows from the default no-op `reflect()`.
- XRay viewer — rollout-aware visualization is V2; V1 displays the N trajectories
  independently.
- MCP server, metrics/tracer — no changes.
