# Episode

**Module:** `cube_harness.episode`

## Purpose

An `Episode` runs one agent against one task and produces a `Trajectory`. It owns the
main loop (reset → step*  → close), incremental trajectory persistence, OpenTelemetry
tracing, and error recovery. Workers receive an `EpisodeConfig` (serializable) and
materialize the `Episode` locally.

## Public API

### `MAX_STEPS = 1000`
Module-level upper limit. An episode also accepts a lower `max_steps` argument;
whichever is smaller wins.

### `EpisodeConfig` (serializable)
```python
class EpisodeConfig(TypedBaseModel):
    id: int                          # per-experiment episode number
    agent_config: AgentConfig
    exp_name: str
    output_dir: Path
    max_steps: int
    task_config: TaskConfig          # cube.task.TaskConfig
```

Saved to disk at `{output_dir}/episodes/{trajectory_id}/episode_config.json` before
the episode runs, so experiments can resume after crashes.

### `Episode`
```python
class Episode:
    def __init__(
        self, id: int, output_dir: Path, agent_config: AgentConfig, task_config: TaskConfig,
        exp_name: str = "default", max_steps: int = MAX_STEPS,
        storage: Storage | None = None,        # defaults to FileStorage(output_dir)
        runtime_context: RuntimeContext | None = None,    # from Benchmark._setup()
        container_backend: ContainerBackend | None = None,
    )

    @classmethod
    def load_episode_from_config(cls, config_path: Path, benchmark: Benchmark | None = None) -> "Episode"
    # Accepts both V2 (episodes/<id>/episode_config.json) and V1 layouts.
    # If benchmark provided, forwards runtime_context and container_backend.

    def run(self) -> Trajectory
    # Main loop. Creates the task via task_config.make(...), runs reset → step*, persists
    # every step incrementally, closes the task in finally.

    allow_overwrite: bool = False   # when True, archives existing trajectory before saving
```

### `_compute_summary_stats(traj) -> dict` (module-level)
Computed at end-of-episode and stored in `Trajectory.summary_stats`. Includes
`n_env_steps`, `n_agent_steps`, `total_actions`, `total_llm_calls`, token counts,
`cost`, `duration`, `final_reward`.

## Main loop semantics

1. Enter `tracer.episode(task_id, experiment=exp_name)` span
2. `_open_status` writes RUNNING `status.json` with `current_step=0`. This is the
   **phase signal** the runner's `_kill_stale_workers` reads — `current_step == 0`
   means setup hasn't completed yet, so the setup-phase budget applies.
3. `task_config.make(runtime_context=..., container_backend=...)` → live Task
4. `setup_fn()` → first `EnvironmentOutput` from `task.reset()`
5. Save initial trajectory + episode_config on disk
6. While not done and turns < max_steps:
   - Heartbeat: write `status.json` with `current_step = turns + 1` and fresh
     `last_heartbeat_at`. Heartbeat write is **fail-open** (logs warning, does
     not abort the run on transient I/O errors).
   - `agent.step(obs)` → `AgentOutput`
     - On exception: save the failed agent step, re-raise (trajectory is preserved)
   - Append agent step to trajectory + save incrementally
   - If empty actions and no error → log and break
   - `step_fn(agent_output.actions)` → `EnvironmentOutput`
     - On exception: save failed env step (with prior obs), re-raise
   - Append env step + save
7. `finally`: call `task.close()` and `tracer.shutdown()`
8. Compute `summary_stats`, persist final trajectory, return

Final episode status is `OK` if `final_reward > 0`, else `ERROR` (sets OTel span status).

## Invariants

1. Every step is persisted incrementally — no in-memory-only state that can be lost.
2. `task.close()` is always called (finally block), even on exceptions.
3. Agent and env exceptions are caught, written as a step with `error` populated, then
   re-raised. Callers see the exception; the trajectory remains on disk.
4. Empty actions + no error → graceful break (agent says "done").
5. `trajectory.id = f"{task_id}_ep{episode_id}"` — the episode directory layout
   relies on this convention.

## Storage layout (V2)

```
<output_dir>/episodes/<trajectory_id>/
├── episode_config.json
├── episode.metadata.json      # Trajectory minus steps
├── steps/
│   ├── 000_obs.msgpack.zst
│   ├── 001_act.msgpack.zst
│   └── ...
├── episode_summary.jsonl      # written by SummaryProcessor
├── failure.txt                # stack trace if run crashed before completion
└── logs/...                   # redirected stdout/stderr
```

Steps are msgpack + zstd compressed. V1 JSONL layout under `trajectories/` is still
loadable but no longer written.

## Contracts for implementers

- Agents that need to recover from partial episodes can reload via
  `Episode.load_episode_from_config()`. Pass the `Benchmark` if the task needs
  `runtime_context` or `container_backend`.
- Storage backends must implement the `Storage` protocol (`save_trajectory`,
  `save_step`, `save_episode_config`, `update_experiment_summary`). See
  [storage spec](../storage/spec.md).

## Gotchas

- `Episode.__init__` does NOT call `task_config.make()` — the task is created inside
  `run()` so long-lived resources are owned by the worker, not the scheduler.
- Ray workers share `benchmark._runtime_context` by reference — treat it as
  read-only after `setup()` returns (see cube-standard benchmark spec).
- `_compute_summary_stats` walks the full trajectory; for very long trajectories
  (thousands of steps) this can be slow. Currently acceptable; revisit if needed.
- Episode timeouts are enforced by `run_with_ray` at the scheduler level, not inside
  the episode. Sequential runs have no timeout.
