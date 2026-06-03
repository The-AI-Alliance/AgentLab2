# Analyze — XRay Viewer

**Module:** `cube_harness.analyze`

## Purpose

Gradio-based web UI for exploring experiment outputs. Browse agents → tasks → seeds,
step through trajectories, inspect observations (screenshots, AXTree, HTML, reward),
view agent reasoning, and compare runs across experiments.

## Public API

### Entry point
```bash
make xray                          # Makefile target
# or
uv run python -m cube_harness.analyze.xray --results-dir <path>
```

### `XRayState` (dataclass)
Holds all mutable viewer state. Captured by Gradio handler closures. Not a
serializable model — it's UI-only state that lives for the duration of a viewer
session.

Key fields:
- `trajectories: list[Trajectory]` — loaded metadata stubs (drive the tables)
- `current_trajectory` — the selected episode's metadata stub
- `current_events: EpisodeEvents | None` — the selected episode's event stream
- `selected: int` — index of the selected event card
- `_storages: list[FileStorage]` — one per loaded experiment dir
- `_traj_storages: list[FileStorage]` — index-aligned with trajectories
- `_exp_tags` — timestamp tag per storage (for disambiguation)
- `_bg_loading_done` / `_bg_gen` — background loading coordination

### `inspect_results` (`cube_harness.analyze.inspect_results`)
CLI-style inspection helpers used by the viewer and exported for ad-hoc scripts.

### `xray_utils` (`cube_harness.analyze.xray_utils`)
Formatting and data-extraction helpers (HTML rendering, trace fragments, step
summaries), plus `_promote_ghost_episodes(exp_dir)` — best-effort sweep run on
every UI refresh:

  - RUNNING + ray (or no exp_status) → promote when per-episode heartbeat is older
    than `GHOST_TIMEOUT` (`should_sweep_running_to_stale` predicate).
  - RUNNING + sequential + driver_dead → promote immediately (driver IS the
    worker; both dead).
  - QUEUED + driver_dead → promote (no worker will ever pick it up if the
    scheduler is gone). QUEUED is **never** promoted when the driver is alive —
    in a large parallel batch, tasks legitimately wait hours for a slot.

The "is the driver alive?" decision lives with the type it queries: see
`is_driver_alive(exp_status, exp_dir, *, timeout_s)` in
`cube_harness.experiment_status` for the mode-aware logic. Same shape as
`should_sweep_running_to_stale` for episode statuses — predicate over the
status object, callable from any consumer (viewer, monitoring, reports).

## UI model — event stream

The detail view consumes the trajectory as a **flat, ordered event stream**
(`LLMCallEvent` / `ToolCallEvent` / `EvaluationEvent` / `AgentErrorEvent`),
loaded via `FileStorage.load_episode → TrajectoryView` and wrapped by
`analyze.xray_events.EpisodeEvents`. There is **no "turn"**: the only link is
`ToolCallEvent.parent_event_id` → the `LLMCallEvent` that produced it.

- **Event-card rail** (`render_event_rail_html`) — a vertical, scrollable column,
  one `.xray-event-card` per event, coloured by kind (LLM / observation /
  evaluation / error). Card height scales with the event's wall-clock duration;
  it replaces the old horizontal timeline and doubles as the profiler.
- **Dependency-graph grouping** (`EpisodeEvents.group_for`) — selecting any card
  marks it active and highlights its logical group: the LLM call + the
  observation(s) it produced + their step-wise evaluations + any error in the
  chain. The detail tabs render that whole group (Chat / Observation /
  AXTree / Evaluation / Error / Debug), so one selection answers "why did the
  agent act, what did it observe, what reward, any error".
- **Parallel tool calls** render as stacked sibling observations within the
  group.

Legacy V1/V2 trajectories are adapted into this same event stream by the
storage loader (`TrajectoryView._step_to_event`); the viewer never sees the old
`EnvironmentOutput | AgentOutput` step shape.

## Invariants

1. Read-only for *trajectory* data — the viewer never modifies trajectories,
   logs, or configs. The single exception is `_promote_ghost_episodes` writing
   `STALE` into `status.json` files for in-flight episodes whose driver is
   provably dead (see `xray_utils` above). This is gated by
   `experiment_status.json` so the viewer cannot accidentally kill live work.
2. Consumes events only; V1/V2 legacy layouts are adapted to events in the
   loader (`TrajectoryView`), never in the viewer.
3. Background loading: a worker thread populates `trajectories` incrementally;
   stale threads self-abort by comparing `_bg_gen`.
4. Displays `_missing=True` stub trajectories (planned but never ran) distinctly.
5. Injects `_failure_text` from `failure.txt` into metadata when a trajectory has
   no `end_time` — so failed episodes show their stack trace in the UI.

## Gotchas

- Gradio state is per-tab. Closing and reopening the browser resets the view; the
  server keeps running.
- Large trajectories (thousands of events) are loaded lazily via `TrajectoryView`
  — switching trajectories may have noticeable latency on first open.
- `EpisodeEvents` decodes the whole event stream into memory for random card
  access; very long sessions with many open trajectories can grow memory use.
