# Tasks: Agent Owns the Loop

Implementation breakdown for the RFC. Spec changes are in
[proposal.md](proposal.md) and [deltas.md](deltas.md). All work lands in
the same PR as the spec.

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
- [ ] A4. Helpers: `last_env_output`, `events_of_turn`, `n_agent_events`,
  `n_tool_calls`, `n_evaluations`.
- [ ] A5. Update `core/spec.md` invariants (drop alternation; add
  back-reference and ordering invariants).
- [ ] A6. Unit tests: `tests/test_event_types.py` (serialization, back-ref
  validation, turn_id grouping).

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

## Phase C — TurnRecorder (`cube_harness.summary` or new `cube_harness.recorder`)

- [ ] C1. `TurnRecorder` with `record(output)`, `begin_turn() → Turn`.
- [ ] C2. `Turn` context manager with `add_llm_call`, `add_thought`,
  `add_response_text`, `add_profile`, `add_error`; `__exit__` flushes one
  `AgentEvent`.
- [ ] C3. Episode-only helpers on `TurnRecorder`: `record_reset`,
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

- [ ] D1. Add `async def run(initial_obs, task, recorder)` to `Agent` base
  with the default gym-style implementation using `asyncio.to_thread(self.step, obs)`
  and `task.astep(actions)`.
- [ ] D2. `Agent.step` stays as-is — backward compatible.
- [ ] D3. `task.astep` doesn't exist in cube-standard yet; if not added by
  then, wrap `task.step` in `asyncio.to_thread` inside the default `run`.
- [ ] D4. Update `agent/spec.md` per deltas.

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

- [ ] H1. `src/cube_harness/agents/genny_parallel.py` — `GennyParallel(Genny)`
  that overrides `Agent.run` to dispatch tool calls via
  `asyncio.gather(*(task.toolbox.execute_action(a) for a in actions))`.
- [ ] H2. Reuse Genny's prompt + tool-call parser; only the loop differs.
- [ ] H3. Result aggregation: `_merge_observations(results)` for the next
  `obs`.
- [ ] H4. Add to `agents/__init__.py`. Document in `agent/spec.md`.
- [ ] H5. Smoke: `scripts/smoke/parallel_tool_arithmetic.py` — runs
  `GennyParallel` on one arithmetic episode; asserts ≥1 turn with
  `len(events_of_turn) > 1`. Skip if no `OPENAI_API_KEY`.

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

- [ ] J1. `TurnRecorder.record_external_run` is implemented in Phase C
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
  `GennyParallel` on the cheapest SWE-bench Lite task. Trajectory
  loadable; verifier runs cleanly (pass or fail OK; what matters is the
  plumbing). Daytona OR local Docker.
- [ ] K4. `scripts/smoke/parallel_tool_terminalbench_single.py` —
  `GennyParallel` on a TerminalBench debug task. Daytona OR local Docker.
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

## Iteration discipline

- For each phase, mark a TodoWrite item `in_progress` before starting,
  `completed` immediately after.
- Run `make lint` before any commit.
- Per-smoke fix budget: 3 attempts. If still red, escalate.
- Session-wide budget: 6 hours of autonomous iteration before checking
  in.
- Commits stay scoped to one phase where possible — easier review later.
