# Deltas: Multi-Episode Rollouts

Applies to:
- `openspec/specs/rollout/spec.md` — **new layer, RFC-bound**
- `openspec/specs/agent/spec.md` — modified (additive: new default-no-op method; no RFC required per constitution Pillar I)
- `openspec/specs/episode/spec.md` — modified (additive: new `run_with()` method that lets `Rollout` reuse `Episode`'s loop; no RFC required)
- `openspec/specs/storage/spec.md` — modified (additive: new optional fields; no RFC required)
- `openspec/README.md` — modified (add rollout layer to index)

The agent, episode, and storage deltas are listed here for completeness and
traceability, even though all three are additive backward-compatible changes
that can be applied directly to their specs in the implementation PR per the
constitution.

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
  - `__init__(self, task_config: TaskConfig, agent_config: AgentConfig, config: RolloutConfig, output_dir: Path | None = None, runtime_context: RuntimeContext | None = None, exp_name: str = "rollout") -> None`
  - `run(self) -> RolloutResult`

  Takes **configs**, not live `Task` / `Agent` instances — `Rollout.run()` constructs both
  once internally (preserving the "Python is the config" idiom and letting `output_dir`
  remain optional for in-memory rollouts).

### Public methods

- `Rollout.run() -> RolloutResult`

  Constructs `task` and `agent` once, then executes `n_episodes` by repeatedly
  delegating to `Episode.run_with(task, agent, extra_metadata=...)`. Between
  non-final episodes calls `agent.reflect(trajectory, final_reward)`. If
  `reflect()` returns a non-None `AgentOutput`, the rollout appends it as a
  synthetic trajectory step on the just-finished episode's trajectory before
  moving on. `task.close()` is called once at rollout end (not per episode) —
  the task survives across attempts and `Task.reset()` (invoked at the start
  of each `Episode.run_with`) returns it to its initial state.

### Invariants

- The same `Agent` instance handles all `n_episodes` within one `Rollout.run()` call.
- `task.reset()` is invoked before each episode (including the first).
- `agent.reflect()` is invoked after each non-final episode (`n_episodes - 1` times).
- When `reflect()` returns a non-None `AgentOutput`, that output is appended as a
  `TrajectoryStep` on the just-finished episode's trajectory — making the reflection
  LLMCall (`tag="reflection"`) observable in XRay, cost stats, and training-data extraction.
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
def reflect(self, trajectory: Trajectory, final_reward: float) -> AgentOutput | None:
    """Called between episodes within a Rollout.

    Default implementation: no-op (returns None). Implementations may update
    internal state that subsequent step() calls read (e.g., a memory list spliced
    into prompts).

    When the implementation runs an LLM call as part of reflecting, it SHOULD
    return an AgentOutput carrying that LLMCall (with tag="reflection") and the
    reflection text in `thoughts`. The Rollout appends that output as a synthetic
    trajectory step so the reflection is observable in XRay / cost stats /
    training-data extraction. Returning None means "nothing to record".

    Standalone Episode runs never invoke this method — backwards compatible.
    """
```

### Invariants (new)

- Default implementation on `Agent` is a no-op returning `None`; subclasses opt in by overriding.
- `reflect()` is invoked exactly once per non-final episode within a `Rollout`,
  with the just-finished trajectory and its final scalar reward.
- `reflect()` is **never** called by `Episode` (single-episode execution path).
- The return value, when non-None, is an `AgentOutput` whose first `LLMCall` carries
  `tag="reflection"`. The Rollout appends it to the trajectory as a synthetic
  step (see Rollout invariants above).
- Implementations may raise; the rollout treats reflection errors the same as
  episode-level errors (propagated up). Implementations that want to suppress
  LLM failures should catch internally and return `None`.

### Not changed

- `Agent.step()` signature and contract.
- `AgentConfig.make()` contract.
- ReAct, Genny, legacy_generic_agent — they inherit the no-op default and require
  no source changes to remain functional inside a `Rollout`.

---

## MODIFIED — `openspec/specs/episode/spec.md`

### `Episode` — new optional method

```python
def run_with(self, task, agent, extra_metadata: dict | None = None) -> Trajectory:
    """Run one episode against a pre-constructed task and agent.

    Like ``run()``, but skips the internal ``task_config.make()`` and
    ``agent_config.make()`` calls — the caller owns the task and agent
    lifecycle. The caller is also responsible for calling ``task.close()``
    afterwards; ``run_with`` does NOT close the task.

    ``extra_metadata`` is merged into ``Trajectory.metadata`` alongside the
    standard fields (``task_id``, ``agent_name``, ``action_schemas``).
    ``Rollout`` uses this to thread ``rollout_id`` and
    ``episode_index_in_rollout`` into the trajectory.
    """
```

### `Episode.run()` — refactored to compose on `run_with`

The existing ``run()`` keeps its current public contract (construct task,
construct agent, run the loop, close the task). Internally it is now a thin
wrapper:

```python
def run(self) -> Trajectory:
    task = self.config.task_config.make(runtime_context=self._runtime_context)
    agent = self.config.agent_config.make(action_set=task.action_set, task_id=...)
    try:
        return self.run_with(task=task, agent=agent)
    finally:
        task.close()
```

### Invariants (additions)

- `Episode.run()` constructs task + agent, calls ``run_with``, closes the task.
  Behavior unchanged from prior versions for all existing callers.
- `Episode.run_with()` does NOT close the task — the caller (typically
  ``Rollout``) owns the task lifecycle so it can survive across episodes.
- `extra_metadata` keys are merged into `trajectory.metadata`; on key collision
  with the standard fields, the caller-provided value wins (allows overriding
  e.g. `agent_name` if needed, though that's discouraged).
- `Trajectory.steps[0]` is still the env reset output; ``run_with`` calls
  ``task.reset()`` itself, so the caller passes a task that may have been
  reset zero or more times before.

### Not changed

- `Episode.run()` signature and existing callers (experiment, exp_runner,
  recipes, tests).
- `EpisodeConfig` fields.
- `Episode._run_loop` (the inner per-turn loop).
- Episode's persistence behavior — both ``run()`` and ``run_with()`` write the
  same files via the same ``FileStorage``.

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

`Rollout._persist` writes trajectories using the canonical `FileStorage` V2 layout
(so XRay and existing trajectory tooling discover them unchanged) **plus** a
rollout-level aggregate as a sibling directory:

```
<experiment_dir>/
├── episodes/                          ← canonical FileStorage V2 (XRay reads this)
│   └── <trajectory_id>/
│       ├── metadata.json
│       └── steps/NNN_*.json, ...
└── rollouts/
    └── <rollout_id>/
        └── rollout_record.json        ← rollout-level aggregate
```

The aggregate `rollout_record.json` contains `rollout_id`, `per_episode_rewards`,
`discounted_reward`, `gamma`, `n_episodes`, `agent_config_dump`, and
`trajectory_ids: list[str]` (the trajectory IDs of the episodes that ran, in
execution order). Per-episode content lives entirely under `episodes/<trajectory_id>/`;
the rollout record only carries the IDs and aggregates, so the two layers don't
duplicate trajectory data.

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
