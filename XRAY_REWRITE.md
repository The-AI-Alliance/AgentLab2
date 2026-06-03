# XRay event-stream rewrite — design brief

Follow-up to PR #386 (`openspec/agent-owns-loop`). PR #386 replaced the binary
`EnvironmentOutput | AgentOutput` trajectory with a flat **event stream**
(`LLMCallEvent` / `ToolCallEvent` / `EvaluationEvent` / `AgentErrorEvent`, read
lazily via `TrajectoryView`) and shrank `AgentOutput` to `{actions, error}`.
That silently blanked XRay's LLM/thoughts/token panels behind `getattr`
band-aids. This PR rewrites XRay to consume the event stream natively.

## Principles (decided with the maintainer)

1. **Flat events, no "turn".** XRay sees an ordered event list. The only link is
   `ToolCallEvent.parent_event_id` → the `LLMCallEvent` that produced it.
2. **Dependency-graph grouping.** Selecting any card surfaces its whole logical
   group: the LLM call + the observation(s) it produced + their step-wise
   evaluations + any error in the chain. Derived at view time from
   `parent_event_id` — the data model stays flat. See `analyze/xray_events.py`
   (`EpisodeEvents.group_for`, already built + unit-tested).
3. **Card rail = navigation + profiler.** One card per event, coloured by kind
   (LLM blue / observation green / evaluation purple / error red). Selecting a
   card marks it *active* (solid) and its group-mates *accompanying* (muted).
   Vertical, left of the tabs, scrollable (scales to 100s of events); replaces
   the old horizontal timeline strip. Card height ∝ duration, clamped
   `[min,max]`; profiling label (if any) is a left-edge stripe.
4. **Legacy isolated in the loader.** Old V1/V2 trajectories are adapted to
   events by `storage.TrajectoryView._step_to_event`. XRay only ever consumes
   events — no `EnvironmentOutput | AgentOutput`, no `Trajectory.steps`, no
   `getattr` band-aids. OK to lose minor legacy-only features.

## View-model API (done — `analyze/xray_events.py`)

- `EpisodeEvents.from_view(view)` — flat list from a `TrajectoryView`.
- `.cards() -> list[EventCard]` — `(index, kind, icon, color, title, subtitle, is_error, accompanying)`.
- `.group_for(i) -> EventGroup` — `(root, selected, members, llm_index, observation_indices, evaluation_indices, error_indices)`.
- `.llm_call(i)`, `.observation(i)`, `.action(i)`, `.error(i)` — None-safe typed extractors.
- Colours/icons: `KIND_COLORS`, `KIND_ICONS`.

## Detail panes (render the resolved group, not a lone event)

- **Chat** — `_render_llm_call_html(group.llm_index's LLMCall)` (reuse existing helper).
- **Observation** — for each `group.observation_indices`: screenshot + text contents
  (fold the old Screenshot tab in here); multiple = parallel siblings stacked.
- **AXTree** — axtree content of the selected/!first observation.
- **Evaluation** — reward + info from `group.evaluation_indices`.
- **Error** — `group.error_indices`.
- **Debug** — raw event JSON for the group's events.

## XRayState changes

- Keep `trajectories: list[Trajectory]` (metadata stubs) for the agent/traj
  tables and header/status/stats/logs — unchanged.
- Add `current_events: EpisodeEvents | None`, `selected: int`.
- `select_trajectory`: keep the metadata stub as `current_trajectory`; load
  events via `storage.load_episode(id)` → `EpisodeEvents.from_view`. Drop the
  legacy `load_trajectory` + steps-eviction path.
- Delete `_env_step_indices`, `get_env_output`, `get_agent_output`,
  `get_env_traj_step`, `get_agent_traj_step`, `_build_env_indices`.
- Navigation (`selected`) indexes the flat event list; the rail's click input
  reuses the existing hidden `Number` + JS pattern from the old timeline.

## Keep untouched

Experiment/agent/trajectory tables, background loading, ghost promotion
(`_promote_ghost_episodes`), config tabs, global/error reports, logs, retries.

## Verify

- `pytest tests/test_xray_events.py` (+ a new integration test loading the
  fixture through `load_episode`).
- `scripts/smoke/xray_event_ui_playwright.py` → inspect `/tmp/xray_shots`.
- `scripts/smoke/xray_loads_event_trajectory.py` still green.
- `make lint`.

## Flagged upstream to the #386 agent

- Drop `ToolCallEvent.turn_id` + `TrajectoryView.events_of_turn()` (redundant
  with `parent_event_id`).
- Stamp `parent_event_id` on the terminal `EvaluationEvent` (→ last tool call)
  and on `AgentErrorEvent` (→ originating event) so the final reward and crashes
  group exactly instead of by stream position.
