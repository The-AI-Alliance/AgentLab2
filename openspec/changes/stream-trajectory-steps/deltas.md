# Deltas

## episode/spec.md

### MODIFIED
- **Main loop:** each step is streamed (`save_step` + `SummaryProcessor.on_step`, indexed by
  a counter) and **not** appended to `Trajectory.steps`; the list stays empty throughout.
  `summary_stats` comes from `SummaryProcessor.summary_stats(...)`; `final_reward` from the
  last `EnvironmentOutput` (`reward_info`).
- **`Episode.run() -> Trajectory`** returns a step-less trajectory (metadata + `summary_stats`
  + `reward_info`). Step content is loaded lazily from disk via `storage.load_trajectory`.
- **Invariant 1** strengthened: steps are persisted incrementally **and never accumulated in
  memory** — the returned/worker-returned trajectory carries no steps.

### REMOVED
- `_compute_summary_stats(traj)` — subsumed by `SummaryProcessor.summary_stats(...)`.
- Gotcha "`_compute_summary_stats` walks the full trajectory; for very long trajectories this
  can be slow" — no longer applicable (stats are accumulated incrementally).

## core/spec.md

### MODIFIED
- `Trajectory.steps` is a lazy, on-disk-backed view, **not** an in-memory accumulator. A
  trajectory produced by the runner has `steps == []`; use `summary_stats` / `reward_info`
  for aggregates and `FileStorage.load_trajectory(id)` to materialize step content.

## storage/spec.md

### MODIFIED
- `SummaryProcessor` is the authoritative source of per-episode `summary_stats` (accumulated
  incrementally as steps stream in); it now also tracks cached / cache-creation tokens and
  the first step `error_type`, and exposes `summary_stats(...)`, `has_error`, `final_reward`.
