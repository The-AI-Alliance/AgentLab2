# Tasks: Agent Owns the Loop

Implementation breakdown for the RFC. Spec changes are in
[proposal.md](proposal.md) and [deltas.md](deltas.md). All work lands in
the same PR as the spec.

---

## Implementation status (this PR)

Phases A–L shipped in the first pass (commits on this branch). The
2026-06-01 scope expansion adds Phases M–R: replace `Trajectory` with
`TrajectoryMetadata` + `TrajectoryView`, write metadata at episode start, do
the full XRay rewrite, migrate Investigator + EpisodeRecord + tests, and
delete the dead code the original phases couldn't touch.

| Phase | Status | Validation |
|---|---|---|
| A: Core event types | ✅ shipped | `tests/test_event_types.py` (11/11) |
| B: MonitoredTool + install_monitoring | ✅ shipped | `tests/test_monitored_tool_compat.py` (17/17) |
| C: EventStreamer | ✅ shipped | `tests/test_recorder_dual_api.py` (14/14, incl. record_external_run) |
| D: Default Agent.run | ✅ shipped | `tests/test_default_agent_run.py` (4/4) |
| E: Episode.run rewrite (async) | ✅ shipped | `tests/test_episode.py` + `test_cube_episode.py` + `test_experiment.py` (all pass) |
| F: Storage event-file layout | ✅ shipped | `tests/test_storage_event_layout.py` (10/10) |
| G: Structural parity | ✅ arithmetic-cube + miniwob-cube `cube test` pass | (SWE-bench / TerminalBench need Docker — verified plumbing works, full runs deferred to Phase L) |
| H: Parallel-dispatch mode in Genny (`GennyConfig.parallel_actions`) | ✅ shipped | `tests/test_genny_parallel.py` (3/3) — sibling ToolCallEvents, parallel speedup, budget enforcement |
| I: XRay event compat (shim) | ✅ legacy steps view materialized from events at load time | superseded by Phase Q below |
| J: Connector seam (record_external_run) | ✅ shipped + tested | covered by Phase C tests |
| K: End-to-end smokes | ✅ 3 smokes in `scripts/smoke/` | `agent_owns_loop_events.py`, `genny_parallel_recorder.py`, `xray_loads_event_trajectory.py` all SMOKE OK |
| L: Reference experiments | ✅ TerminalBench-2 reproduced at parity (gpt-5.4-mini) | see *Reference baseline reproduction* below |
| M: TrajectoryMetadata pydantic type | ✅ shipped | `tests/test_episode_view.py::TestTrajectoryMetadata` |
| N: TrajectoryView lazy loader + per-view cache | ✅ shipped | `tests/test_episode_view.py` (12/12, V2 + V2-steps + crashed-mid-run) |
| O: Write metadata-at-start + finalize_episode(meta) | ✅ shipped | episode.py rewrites |
| P: Storage migration (load_episode, list_episodes, V1 upgrade) | ✅ shipped | covered by test_episode_view + test_storage |
| P2: Episode + MonitoredTool + EventStreamer + EpisodeRecord migration | ✅ shipped | 1022 unit tests pass |
| Q (scoped): load_trajectory wraps load_episode (XRay/utils/inspect_results unchanged) | ✅ shipped | smokes + cube debug suites |
| T (scoped): slim Trajectory; drop events / streaming / event-stream helpers from Trajectory | ✅ shipped | core/storage diff = -200 LOC |
| Smokes | ✅ green | agent_owns_loop_events, genny_parallel_recorder, xray_loads_event_trajectory, streaming_trajectory |
| Cube debug suites | ✅ green | arithmetic-cube, miniwob-cube, workarena-cube |
| Q-full: XRay event-card rewrite (drop legacy shim) | ⏭ follow-up PR `agent-owns-loop-xray` | — |
| R: Investigator migration to TrajectoryView | ⏭ follow-up PR | — |
| S: make_fake_episode test helper | ⏭ follow-up PR (needed when test_xray_utils migrates) | — |

**Final aggregate this PR: 1022 unit tests pass + 4 smokes + 3 cube
debug suites green. Net: 1019 starting → 1022 (3 new TrajectoryView /
metadata tests; legacy event-stream-on-Trajectory tests deleted as
their assertions moved to test_episode_view).**

### Cube-standard companion (PR #152)

No further changes needed. The companion proposal already says the
right thing — Trajectory was never in cube-standard. Mergeable, all
CI green; mark ready alongside this PR when reviewed.

### Design adjustments made during implementation

These are the calls the RFC didn't specify and that emerged from the
code:

1. **Two MonitoredTool variants** (`MonitoredTool` for sync, `AsyncMonitoredTool`
   for async) instead of the single async wrapper the RFC drafted —
   real cubes are mostly sync (`ArithmeticTool`, `BgymTool`,
   `ComputerBase` …) and `Toolbox.execute_action` asserts
   `isinstance(tool, AbstractTool)`, so a sync-Toolbox-compatible
   wrapper is necessary.
2. **`Budget.turns` bookkeeping in `EventStreamer`**, not just
   `MonitoredTool`. Some tests use a hand-rolled `task.step` that
   bypasses `tool.execute_action` entirely — without recorder-side
   bumping + `BudgetExceeded` raise, the loop runs forever on such
   tasks.
3. **`task.tool` attribute name** for the toolbox in Genny's
   parallel-dispatch path (cube-standard's canonical Task attribute),
   with `task.toolbox` fallback for downstream tasks that aliased it.
4. **EvaluationEvent re-raise instead of swallow** — the legacy
   `test_episode_captures_env_error` expects `task.evaluate` exceptions
   to propagate. Episode's `finally` block still runs in all cases.
5. **XRay legacy-steps materialization at load time** instead of a
   full UI rewrite. ~10 call sites in `xray_utils.py` and
   `inspect_results.py` walk `trajectory.steps`; rather than touching
   each one, `FileStorage._load_trajectory` synthesizes a legacy view
   from events. The full event-card timeline is Phase 2 work.
6. **`record_failure` does NOT bump `budget.turns`** — failure events
   are metadata, not turns the agent took. Without this distinction
   `n_agent_steps` would over-count by one whenever budget hit.
7. **`MonitoredTool.__getattr__` delegates non-monitored attribute
   access to the inner tool.** Surfaced by running the TerminalBench-2
   reference baseline on this branch: terminalbench2's `Task.evaluate`
   / `Task.reset` / `Task.close` call `self.tool.bash(...)` directly
   (and `bash_unlimited`, `read`, …) for setup, verification, and the
   oracle path. Those calls are NOT agent tool calls; they're
   task-internal driver methods. Without delegation,
   `monitored_tool.bash(...)` raised `AttributeError` the moment
   Episode installed monitoring, breaking every task that uses the
   direct method pattern (terminalbench2, swebench-verified, …).
   Fix: `__getattr__` forwards anything Python didn't find on the
   wrapper to the inner tool. `execute_action` / `action_set` /
   `reset` / `close` still go through the monitored path. Direct
   `@tool_action` method calls reach the underlying tool unchanged
   and are NOT recorded as `ToolCallEvent`s (right semantics: those
   calls aren't agent-driven). Regression test:
   `test_monitored_tool_forwards_direct_method_calls_to_inner`.

---

## Reference baseline reproduction (Phase L)

Reproduced the team's reference baseline for **gpt-5.4-mini on
TerminalBench-2** on this branch end-to-end (Daytona infra, Genny[swe]
agent, cost_limit=$0.5/task, max_actions=100, identical config to the
reference: `~/dev/sandbox/reference-baselines-r0/mini_tbench2.py`):

| Metric | Reference (2026-05-29) | This branch (2026-05-31) |
|---|---|---|
| Solved | 26/89 | 24/89 |
| Scored | 88 | 82 |
| Solve-rate (scored) | 29.5% | **29.3%** |
| Solve-rate (raw) | 29.2% | **27.0%** |
| env_failure exclusions | 1 (prove-plus-comm flaky) | 7 (Daytona timeouts + Azure ContentPolicy) |

**Verdict: reference parity within the noise of the env-failure rate.**
Solve-rate on scored episodes is essentially identical (29.3% vs 29.5%).
The raw rate is 2pp lower because we had more transient
Daytona/Azure infra exclusions this run — same pre-existing noise
modes the reference RESULTS.md flagged (`Daytona/container env issues`
and `Azure ContentPolicy`), just a slightly worse draw of them.

Run dir: `~/cube_harness_results/20260531_120149_Genny-azure_gpt-5.4-mini_terminalbench2-cube_ref-baseline-r0-mini-tbench2_412f770b/`
Wallclock: ~4h with `--ray 8` on Daytona.

---

## Parity strategy (no mock LLM needed)

cube-harness has no mock LLM, but every cube ships a debug suite
(`debug.py`) whose scripted deterministic agents are equivalent to a
mocked LLM. **The structural-parity test is: `cube test <name>` for every
in-tree cube must still pass with reward==1.0 via the new `Agent.run`
path.** Behavioral parity for real LLMs is checked through smokes against
the four reference experiments the team is launching in parallel.

Baselines for the reference experiments are captured in a separate
thread — not our task here.

---

## Phase A — Core event types (`cube_harness.core`)

- [ ] A1. `AgentEvent`, `ToolCallEvent`, `EvaluationEvent` pydantic models.
- [ ] A2. `TrajectoryEvent` replaces `TrajectoryStep`; `output` union expands.
- [ ] A3. `Trajectory.events: list[TrajectoryEvent]`; keep `steps` as a
  computed read-only alias that exposes the legacy view (one release).
- [ ] A4. Helpers: `last_env_output`, `n_agent_events`,
  `n_tool_calls`, `n_evaluations`. (No `events_of_turn`: parallel
  siblings are queried directly via shared `parent_event_id`.)
- [ ] A5. Update `core/spec.md` invariants (drop alternation; add
  back-reference and ordering invariants).
- [ ] A6. Unit tests: `tests/test_event_types.py` (serialization, back-ref
  validation, parent_event_id sibling grouping).

**Validation**: `pytest tests/test_event_types.py` green.

---

## Phase B — MonitoredTool + monitoring install (`cube_harness.tool`)

- [ ] B1. Re-create `src/cube_harness/tool.py` (deleted in `e760f9e5`)
  containing `MonitoredTool(AsyncTool)`.
- [ ] B2. `class BudgetExceeded(BaseException)` with `action: Action`.
- [ ] B3. `Budget` model with `max_turns` (enforced), plus accepted but
  unused field names `max_tool_calls`, `max_cost_usd`, `max_wallclock_s`
  (open question #1: enforce later).
- [ ] B4. `install_monitoring(task, trajectory, budget, storage, summary)`
  helper that walks `task.toolbox` and wraps each member as `MonitoredTool`
  in place.
- [ ] B5. Re-create `openspec/specs/tool/spec.md` (per ADDED section of
  deltas).
- [ ] B6. Unit tests: `tests/test_monitored_tool_compat.py` — drop-in
  compatibility, mixed monitored/unmonitored dispatch, budget enforcement.

**Validation**: `pytest tests/test_monitored_tool_compat.py` green.

---

## Phase C — EventStreamer (`cube_harness.summary` or new `cube_harness.streamer`)

- [ ] C1. `EventStreamer` with `record(output)`, `begin_turn() → Turn`.
- [ ] C2. `Turn` context manager with `add_llm_call`, `add_thought`,
  `add_response_text`, `add_profile`, `add_error`; `__exit__` flushes one
  `AgentEvent`.
- [ ] C3. Episode-only helpers on `EventStreamer`: `record_reset`,
  `record_failure`, `record_evaluation`.
- [ ] C4. `record_external_run(final_text, usage, raw_events)` — lossy
  capture path for connectors (Phase 2 callers; ship now for the spec).
- [ ] C5. Internal: `record(output)` is implemented as a wrapper over
  `begin_turn()` for single-code-path guarantee.
- [ ] C6. Unit tests: `tests/test_recorder_dual_api.py` — coarse and
  granular paths produce byte-equal `AgentEvent`s for the same payload.

**Validation**: `pytest tests/test_recorder_dual_api.py` green.

---

## Phase D — Default Agent.run (`cube_harness.agent`)

- [ ] D1. Add `async def run(initial_obs, env_tool)` to `Agent` base
  as a thin dispatcher selecting between `_run` (sync body) and
  `_arun` (async body) by `AgentConfig.parallel_actions: bool`.
- [ ] D2. `_run` calls `self.step(obs)` directly and dispatches via
  `env_tool.execute_action(action)` (sync, no `await`) — debuggable.
- [ ] D3. `_arun` uses `asyncio.gather` over `env_tool.async_execute_action(action)`
  for parallel dispatch; `Agent._merge_results` is the default merge
  (subclasses override for bespoke merging).
- [ ] D4. `Agent.step` stays as-is — backward compatible.
- [ ] D5. Update `agent/spec.md` per deltas. (Async `task.astep` open
  question resolved as deferred — agents that want async LLM override
  `_arun` directly.)

**Validation**: structural parity test — Phase G below.

---

## Phase E — Episode.run rewrite (`cube_harness.episode`)

- [ ] E1. `Episode.run` becomes async. Body: `task.reset()` →
  `install_monitoring` → `recorder.record_reset(initial)` →
  `await agent.run(initial.obs, task, recorder)` → `try/except
  BaseException/finally`.
- [ ] E2. `finally`: `task.evaluate()` (with `record_evaluation` capturing
  failures), `storage.finalize`, `summary.on_episode_complete`,
  `task.close()`.
- [ ] E3. Preserve today's OTel handling: `tracer.episode()` around the
  `try`, `tracer.shutdown()` in outer finally.
- [ ] E4. Update `exp_runner.run_sequentially` and `run_with_ray` to
  `await` the now-async `Episode.run`.
- [ ] E5. Update `episode/spec.md` per deltas.
- [ ] E6. Unit tests: `tests/test_episode_finalization.py` — finalize on
  `BudgetExceeded`, `BaseException`, normal return, and `task.evaluate`
  failure.

**Validation**: `pytest tests/test_episode_finalization.py` green.

---

## Phase F — Storage event-file layout (`cube_harness.storage`)

- [ ] F1. `Storage` protocol adds `save_event(event, trajectory_id, n)`,
  `load_event`, `finalize(trajectory)`.
- [ ] F2. `FileStorage`: write `events/{nnn:03d}_{agent|tool_call|eval}.msgpack.zst`.
- [ ] F3. `save_step` / `load_step` remain as deprecated aliases that
  read/write the new event files.
- [ ] F4. Migration shim: `load_trajectory` auto-detects `events/` vs
  `steps/` and converts old `_obs` / `_act` files to synthetic event types
  in memory.
- [ ] F5. `SummaryProcessor` counter renames: `n_agent_events`,
  `n_tool_calls`, `n_evaluations` (with backward JSON aliases).
- [ ] F6. Update `storage/spec.md` per deltas.
- [ ] F7. Unit tests: `tests/test_storage_event_layout.py` — both layouts
  load to equivalent in-memory model.

**Validation**: `pytest tests/test_storage_event_layout.py` green.

---

## Phase G — Structural parity via debug suites

- [ ] G1. Run `cube test <name>` for every in-tree cube on the refactored
  branch. ReactAgent + Genny (via default `Agent.run`) must complete with
  reward==1.0 wherever they did before.
- [ ] G2. Drive parity through `make debug` and `make test`.
- [ ] G3. Fix regressions until all cubes' debug suites pass.

**Validation**: every `cube test <name>` returns success. Equivalent to
the structural parity test the user requested.

---

## Phase H — New parallel-tool agent

- [ ] H1. `GennyConfig.parallel_actions: bool` flag on `Genny` that dispatches
  tool calls via `asyncio.gather(*(task.toolbox.execute_action(a) for a in actions))`
  when set (selected via `Agent` base's dual `_run` / `_arun`).
- [ ] H2. Reuse Genny's prompt + tool-call parser; only the loop differs.
- [ ] H3. Result aggregation: `_merge_observations(results)` for the next
  `obs`.
- [ ] H4. Document the `parallel_actions` mode in `agent/spec.md`.
- [ ] H5. Smoke: `scripts/smoke/parallel_tool_arithmetic.py` — runs
  `Genny(parallel_actions=True)` on one arithmetic episode; asserts ≥1 turn with
  ≥2 ToolCallEvents sharing the same `parent_event_id`. Skip if no `OPENAI_API_KEY`.

**Validation**: smoke passes.

---

## Phase I — XRay event-card UI (`cube_harness.analyze`)

- [ ] I1. Timeline renders per-event cards coloured by kind.
- [ ] I2. Parallel `tool_call` siblings render in horizontal lanes within
  a turn group.
- [ ] I3. Selection model: `selected_event` + computed
  `last_agent_event` / `last_observation_event`.
- [ ] I4. Tabs: Reasoning/Chat, Observation (folds screenshots), Turn
  observations (new), Profiling. Header strip.
- [ ] I5. Storage shim: load both `events/` and legacy `steps/` (via the
  layer from Phase F).
- [ ] I6. Smoke: `scripts/smoke/xray_dual_format.py` — load a legacy
  experiment dir AND a new one through the viewer's data layer; assert
  tabs render content. No browser required (data-layer test).
- [ ] I7. Manual: `make xray` against both layouts.

**Validation**: smoke passes; manual UI check.

---

## Phase J — Connector seam (deferred bodies, Phase-1 plumbing only)

- [ ] J1. `EventStreamer.record_external_run` is implemented in Phase C
  already; this phase verifies the contract works.
- [ ] J2. Unit test: `tests/test_external_run_record.py` — calling it
  produces a single `AgentEvent` carrying `final_text` + `usage` +
  `raw_events`.

No connector implementations land in this PR.

---

## Phase K — End-to-end smokes (real LLM)

- [ ] K1. `scripts/smoke/default_agent_arithmetic.py` — ReactAgent via
  `Agent.run` on one arithmetic episode; reward==1.0. Skip if no API key.
- [ ] K2. `scripts/smoke/genny_arithmetic.py` — Genny via `Agent.run` on
  one arithmetic episode. Reward==1.0.
- [ ] K3. `scripts/smoke/parallel_tool_swebench_single.py` —
  `Genny(parallel_actions=True)` on the cheapest SWE-bench Lite task. Trajectory
  loadable; verifier runs cleanly (pass or fail OK; what matters is the
  plumbing). Daytona OR local Docker.
- [ ] K4. `scripts/smoke/parallel_tool_terminalbench_single.py` —
  `Genny(parallel_actions=True)` on a TerminalBench debug task. Daytona OR local Docker.
- [ ] K5. `scripts/smoke/episode_done_via_env.py` — fixture task with
  `env_output.done=True` mid-loop; default agent terminates, finalization
  happens once. No LLM needed.
- [ ] K6. `scripts/smoke/budget_exceeded.py` — `max_turns=2` on a fixture
  task; `BudgetExceeded` propagates and Episode finalizes cleanly. No LLM
  needed.

Smokes exit 0/1/2 with `SMOKE OK/FAIL/SKIP: <name>`.

**Validation**: all K-smokes return OK or SKIP. No FAIL.

---

## Phase L — Final acceptance (reference experiments)

- [ ] L1. After Phases A–K green, signal the user.
- [ ] L2. User's 4 reference experiments run on this branch.
- [ ] L3. Acceptance: reward distribution per benchmark ≥ baseline within
  stat noise.

---

## Phase M — `TrajectoryMetadata` pydantic type

- [ ] M1. Add `class TrajectoryMetadata(TypedBaseModel)` to `cube_harness.core`
  with fields `id`, `metadata`, `start_time`, `end_time`, `summary_stats`,
  `reward_info`.
- [ ] M2. Round-trip JSON test: serialize, write to disk, read, validate.
- [ ] M3. Update `openspec/specs/core/spec.md` to declare TrajectoryMetadata
  in place of Trajectory.

**Validation**: pytest tests/test_episode_metadata.py green.

---

## Phase N — `TrajectoryView` lazy loader

- [ ] N1. Add `class TrajectoryView` to `cube_harness.storage`. Holds
  `(storage, id, metadata, _index, _cache)`. Methods: `__len__`,
  `__getitem__(i)`, `__iter__`, `iter_events`, `is_complete`,
  properties for `n_agent_events` / `n_tool_calls` / `n_evaluations`,
  shortcuts to `metadata.summary_stats` / `metadata.reward_info`.
  (No `events_of_turn`: callers query parallel siblings via shared
  `parent_event_id`.)
- [ ] N2. Internal cache is a plain `dict[int, TrajectoryEvent]`,
  populated on access, scoped to view lifetime. No LRU.
- [ ] N3. `_index` is a list of (event_num, kind, Path) tuples built
  from `events/` directory listing — no event decode at construction.
- [ ] N4. Open-time detection: events/ + episode.metadata.json → V2;
  events/ alone → V2 crashed-mid-run with stub metadata; steps/ alone
  → V1 legacy-upgrade view that synthesizes events on iteration.
- [ ] N5. Unit tests: `tests/test_episode_view.py` — lazy access, cache
  hits, iteration order, len, V1 upgrade path, crashed-mid-run path.

**Validation**: pytest tests/test_episode_view.py green.

---

## Phase O — Write metadata-at-start + finalize_episode

- [ ] O1. `Episode.run` calls `storage.save_metadata(meta)` immediately
  after `task.reset()` (or just before — bikeshed minimally; see code).
  Stub `meta` has `end_time=None`, `summary_stats=None`, `reward_info={}`.
- [ ] O2. Episode `finally` block constructs the final `TrajectoryMetadata`
  (filled end_time + summary_stats + reward_info) and calls
  `storage.finalize_episode(meta)` which rewrites `episode.metadata.json`.
- [ ] O3. Remove `Trajectory` construction from `Episode.run`. No
  `trajectory.streaming = True` setter — gone with the class.
- [ ] O4. `Episode.run` returns `storage.load_episode(self.id)` —
  TrajectoryView onto what was just written.
- [ ] O5. Crash-load smoke: force an exception mid-episode, then
  `storage.load_episode(id)` and assert `view.is_complete == False`,
  events decoded so far are visible, `status.json` failure summary is
  readable via metadata or sibling load.

**Validation**: smoke `scripts/smoke/crashed_episode_loads.py` SMOKE OK.

---

## Phase P — Storage migration (load_episode, list_episodes, V1 upgrade)

- [ ] P1. `FileStorage.save_metadata(meta)` — writes
  `episode.metadata.json` (creates `episodes/<id>/` dir if needed,
  fails on overwrite without `allow_overwrite=True` *unless* this is
  the same `id` already in `_saved_ids` — the start-write counts as
  saving).
- [ ] P2. `FileStorage.load_episode(id) -> TrajectoryView`. Detects V2 vs
  V1 layout; returns view bound to this storage handle.
- [ ] P3. `FileStorage.finalize_episode(meta)` — same code path as
  `save_metadata` but allows overwrite of the start-write file.
- [ ] P4. `FileStorage.list_episodes() -> list[TrajectoryMetadata]` —
  walks episodes/ dir, reads each `episode.metadata.json` (or
  `trajectory.json` for V1). Tolerant: malformed/missing metadata file
  returns a stub with id from dir name.
- [ ] P5. Delete `save_trajectory`, `load_trajectory`, `finalize`
  (old). Delete `_events_to_legacy_steps`.
- [ ] P6. Update `cube_harness.summary` / `cube_harness.episode_logs` /
  any internal callers that previously used the removed APIs.
- [ ] P7. Unit tests: `tests/test_storage_episode_view.py` — V2 fresh
  load, V2 crashed-mid-run load (no metadata.json), V1 legacy load
  (steps/), list_episodes round-trip.

**Validation**: pytest tests/test_storage_episode_view.py green.

---

## Phase Q — XRay full event-card rewrite

- [ ] Q1. `xray.py` and `xray_utils.py`: every `trajectory.steps[i]`
  access becomes `view[i]`. Element type changes from `TrajectoryStep`
  to `TrajectoryEvent`; consumers pattern-match `event.output`.
- [ ] Q2. Timeline: per-event cards coloured by kind. Drop the legacy
  obs/act pair UI.
- [ ] Q3. Parallel `tool_call` siblings render in horizontal lanes
  within a turn group (grouped by shared `parent_event_id` from `view`).
- [ ] Q4. Selection model: `selected_event_index` + computed
  `last_agent_event_index` / `last_observation_event_index` from the
  cheap kind table.
- [ ] Q5. Tabs: Reasoning/Chat, Observation (folds screenshots), Turn
  observations (new), Profiling. Header strip.
- [ ] Q6. Crashed-episode banner: shows `status.json` failure summary
  when `view.is_complete == False`.
- [ ] Q7. Delete dead code in `xray_utils.py`: any function that
  walked `trajectory.steps` to produce a paired obs/act view.
- [ ] Q8. Update `scripts/smoke/xray_loads_event_trajectory.py` to
  assert the full event-card UI renders content from both events/ and
  legacy steps/ layouts.
- [ ] Q9. Manual: `make xray` against both layouts; eyeball the
  parallel-sibling lane rendering.

**Validation**: smoke green; manual UI check.

---

## Phase R — Investigator migration

- [ ] R1. `InvestigatorContext.trajectory: Trajectory` →
  `InvestigatorContext.view: TrajectoryView`. Update every consumer
  (`use_cases/general_blame`, `profiling`, `agent_scaffolding`,
  `hinter`, `fix_audit`).
- [ ] R2. Investigator pipeline functions that iterate steps
  (`per_step_blame`, etc.) iterate `view` instead.
- [ ] R3. Any pre-aggregation that depended on the materialized steps
  view rewrites against `view.iter_events()` with kind dispatch.
- [ ] R4. Integration test: `tests/test_investigator_event_view.py` —
  run a use case against a fake event-stream view, assert blame output
  is reachable.

**Validation**: investigator integration tests green; one in-tree
investigator use case smokes against a real episode dir.

---

## Phase S — Test helper + migrate hand-built tests

- [ ] S1. Add `make_fake_episode(events: list[TrajectoryEvent], *,
  storage: TmpStorage | None = None, **meta_fields) -> TrajectoryView` to
  `tests/_helpers.py` (new module). Writes events to a TmpStorage and
  returns a real `TrajectoryView` so tests exercise the actual load path.
- [ ] S2. Migrate every test that built `Trajectory(id=..., steps=[...])`
  or `Trajectory(id=..., events=[...])` directly. ~30 sites.
- [ ] S3. Drop `streaming=True` from any test constructor that set it
  — gone with the class.

**Validation**: full `pytest tests/` green.

---

## Phase T — Delete Trajectory + dead helpers

This is the cleanup pass after Phases M–S are green. It must be the
last commit of the PR — no in-tree caller can reference Trajectory
when this lands.

- [ ] T1. Delete `class Trajectory` from `cube_harness.core`.
- [ ] T2. Delete public-API `TrajectoryStep` export (the legacy
  reader's internal version stays inside storage.py).
- [ ] T3. Delete `Trajectory.streaming` plumbing in `MonitoredTool`,
  `EventStreamer`, `tool.py`, `recorder.py` — events are always streamed.
- [ ] T4. Delete `Trajectory.last_env_step` / `last_env_output` /
  `n_agent_steps` / `n_env_steps` / `events_of_turn` (the
  Trajectory-method versions). `TrajectoryView` has the live versions.
- [ ] T5. Delete `_events_to_legacy_steps` from storage.py.
- [ ] T6. Delete every `if trajectory.events: ... else: ...` defensive
  branch in core/storage/summary.
- [ ] T7. Delete `EpisodeRecord.from_trajectory` (kept replaced by
  `from_view`).
- [ ] T8. Grep audit: `grep -rn "class Trajectory\|Trajectory(" src/`
  returns zero hits (excluding the V1 legacy-reader internals if any).
- [ ] T9. Update `cube_harness/CLAUDE.md` package-layout block: drop
  references to Trajectory, add TrajectoryMetadata / TrajectoryView.

**Validation**: full `pytest tests/`, `make lint`, all smokes green.

---

## Iteration discipline

- For each phase, mark a TodoWrite item `in_progress` before starting,
  `completed` immediately after.
- Run `make lint` before any commit.
- Per-smoke fix budget: 3 attempts. If still red, escalate.
- Session-wide budget: 6 hours of autonomous iteration before checking
  in.
- Commits stay scoped to one phase where possible — easier review later.
- Phase T (deletion) is the **last** commit of the PR; no earlier
  commit may leave the tree without a working Trajectory if T1–T8
  haven't all landed.
