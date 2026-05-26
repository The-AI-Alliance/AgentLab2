# Stream trajectory steps to disk (stop accumulating them in RAM)

## Why

The episode loop builds the full `Trajectory.steps` list in memory, then at episode end
re-writes it (`save_trajectory`) and re-summarizes it (`_compute_summary_stats`). Ray
workers return the full trajectory to the driver, which holds **every** episode's steps at
once. On image-heavy benchmarks (OSWorld, web) a single trajectory is 15–400 MB, so the
driver balloons to ~20 GB over a run — and hundreds of GB of screenshots are serialized over
the Ray object store for nothing, since the steps are already on disk.

Nothing in the agent↔env loop reads the accumulated steps: `agent.step(obs)` gets only the
current observation. The step list is a persistence/reporting byproduct, and both consumers
already stream — `storage.save_step` writes each step incrementally and `SummaryProcessor`
accumulates stats per step. Full step *content* is only needed by xray/investigator, which
load it from disk on demand.

## What

- `SummaryProcessor` becomes the single source of `summary_stats` (adds cached /
  cache-creation tokens and the first step `error_type`; exposes `summary_stats(...)`,
  `has_error`, `final_reward`).
- The episode loop streams each step (`save_step` + `SummaryProcessor.on_step` via a counter)
  and **stops retaining `Trajectory.steps`**. The returned `Trajectory` carries metadata +
  `summary_stats` + `reward_info`, with `steps == []`; step content loads lazily via
  `storage.load_trajectory`.
- `_compute_summary_stats` is removed (subsumed by `SummaryProcessor`); the redundant
  end-of-run re-write and re-summary are gone.
- `EpisodeRecord.from_trajectory`, `Experiment.print_stats`, and the Ray completion log read
  `summary_stats`/`reward_info` instead of walking `steps`.

Net: driver/worker RAM is flat regardless of step size, and the pointless Ray serialization
of screenshots is eliminated.

## Breaking change

`Episode.run()` / `run_with_ray` / `run_sequentially` now return **step-less** trajectories.
Callers needing step content must `FileStorage(output_dir).load_trajectory(id)`. No in-repo
consumer relied on the returned steps (xray already loads from disk); the eval-log
`EpisodeRecord` is a summary record and no longer reads steps.

## Alternatives considered

- **Clear steps after `print_stats`** (band-aid): frees driver RAM but still serializes and
  ships every screenshot over Ray. Rejected — treats the symptom, not the cause.
- **Keep `_compute_summary_stats`, just clear before return:** keeps the redundant double
  summary and the per-worker full materialization. Rejected.
