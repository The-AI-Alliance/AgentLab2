# Deltas: Agent Owns the Loop

Applies to (cube-harness specs):

- `openspec/specs/core/spec.md` (primary — trajectory model)
- `openspec/specs/agent/spec.md` (primary — `Agent.run`)
- `openspec/specs/tool/spec.md` (primary — `MonitoredTool` replaces `ToolWithTelemetry`)
- `openspec/specs/episode/spec.md` (primary — loop refactor + defensive finalization)
- `openspec/specs/storage/spec.md` (event-file layout)
- `openspec/specs/analyze/spec.md` (XRay event-card UI)

See companion: `cube-standard/openspec/changes/agent-owns-loop/deltas.md`.

---

## MODIFIED — `openspec/specs/core/spec.md`

### Trajectory becomes an event list

`TrajectoryStep` is renamed to `TrajectoryEvent`. Its `output` union expands
from `EnvironmentOutput | AgentOutput` to `AgentEvent | ToolCallEvent | EvaluationEvent`.

```python
class AgentEvent(TypedBaseModel):
    id: str                            # references target for ToolCallEvent.agent_event_id
    actions: list[Action]              # intended tool calls; each Action.id is the link
    llm_calls: list[LLMCall]
    thoughts: str | None
    response_text: str | None          # assistant prose alongside tool calls
    profiling: dict[str, tuple[float, float]]
    error: StepError | None

class ToolCallEvent(TypedBaseModel):
    agent_event_id: str                # parent AgentEvent.id
    action_id: str                     # AgentEvent.actions[i].id
    output: EnvironmentOutput          # obs / reward / done / info / error
    turn_id: str                       # groups parallel siblings
    step_eval: StepEval | None = None  # populated iff task.requires_step_eval

class StepEval(TypedBaseModel):
    reward: float                      # per-tool-call evaluation reward
    info: dict                         # per-tool-call evaluation info

class EvaluationEvent(TypedBaseModel):
    reward: float                      # terminal eval reward
    info: dict                         # terminal eval info

class TrajectoryEvent(TypedBaseModel):
    output: AgentEvent | ToolCallEvent | EvaluationEvent
    start_time: float
    end_time: float

class Trajectory(TypedBaseModel):
    id: str
    events: list[TrajectoryEvent]      # was: steps: list[TrajectoryStep]
    metadata: dict
    start_time: float | None
    end_time: float | None
    reward_info: dict                  # kept for back-compat; mirrors final EvaluationEvent
    summary_stats: dict | None
```

### Helpers

- `Trajectory.last_env_output() -> EnvironmentOutput | None` (replaces
  `last_env_step()`): walks events in reverse, returns the most recent
  `ToolCallEvent.output`, or `None` if none.
- `Trajectory.events_of_turn(turn_id) -> list[TrajectoryEvent]`: all sibling
  events sharing a `turn_id`.
- `Trajectory.n_agent_events`, `n_tool_calls`, `n_evaluations` properties.
- `Trajectory.steps` and `n_agent_steps`/`n_env_steps` remain as deprecated
  read-only aliases for one release. `steps` is a computed view that pairs
  `AgentEvent` + first `ToolCallEvent` for each turn so legacy callers don't
  break in-tree.

### Invariants (replaces old #1, #2)

1. Events are ordered by `start_time` and represent the agent's interaction
   with the task. They are **not** required to alternate.
2. Every `ToolCallEvent.agent_event_id` and `ToolCallEvent.action_id` must
   resolve to an `AgentEvent` and one of its `actions` entries appearing
   earlier in the trajectory.
3. At most one `EvaluationEvent` per trajectory; if present, it is the last
   event.
4. `AgentEvent.id` is unique within a trajectory.
5. `turn_id` is unique per `AgentEvent`; all `ToolCallEvent`s spawned from one
   `AgentEvent` share the same `turn_id`.

### Gotchas

- The legacy `EnvironmentOutput | AgentOutput` union is removed from
  `TrajectoryEvent.output`. Callers that pattern-match on those types must
  switch to the new event types.
- `AgentOutput` itself remains as the return type of `Agent.step()` — it is
  internally converted into an `AgentEvent` by `TurnRecorder.record()`.

---

## MODIFIED — `openspec/specs/agent/spec.md`

### `Agent.run` (new, default impl provided)

```python
class Agent(ABC):
    def step(self, obs: Observation) -> AgentOutput        # unchanged
    async def run(
        self,
        initial_obs: Observation,
        toolbox: "MonitoredToolbox",
        recorder: "TurnRecorder",
    ) -> None
```

Default implementation in the base class:

1. `obs = initial_obs`
2. Loop forever (termination is driven by exceptions raised out of
   `toolbox(action)`, not by checking flags):
   1. `agent_output = await asyncio.to_thread(self.step, obs)` (sync agents)
      or `agent_output = await self.astep(obs)` if subclass defines `astep`.
   2. `recorder.record(agent_output)`.
   3. If `not agent_output.actions and not agent_output.error`: return.
   4. For each `action` in `agent_output.actions`: `await toolbox(action)`.
      The last `EnvironmentOutput`'s `obs` becomes the next `obs`. The call
      may raise `EpisodeDone` or `BudgetExceeded` (both `BaseException`
      subclasses) — those propagate to `Episode`.
3. Returns normally on empty actions. Raises propagate.

### `TurnRecorder`

Agent-facing telemetry sink, constructed by `Episode` per-episode.

```python
class TurnRecorder:
    # Coarse — one call per LLM cycle, all-at-once. Default agent uses this.
    def record(self, output: AgentOutput) -> None
    # Granular — for streaming agents that emit incrementally.
    def begin_turn(self) -> "Turn"

class Turn:
    # Context manager. __exit__ flushes one AgentEvent built from
    # accumulated fields.
    def add_llm_call(self, call: LLMCall) -> None
    def add_thought(self, text: str) -> None
    def add_response_text(self, text: str) -> None
    def add_profile(self, label: str, start: float, end: float) -> None
    def add_error(self, err: StepError) -> None
```

`record(output)` is internally implemented as a thin wrapper around
`begin_turn()` — one underlying code path, two surfaces.

### Semantics

- `Agent.run` is the **canonical entry point** invoked by `Episode`.
- `Agent.step` remains as the unit of one LLM turn for sync agents. Agents
  that want parallel/async behavior override `run` and may ignore `step`.
- `EpisodeDone` and `BudgetExceeded` (both `BaseException` subclasses) raised
  out of `toolbox(action)` should propagate. Catching them is forbidden by
  convention. `MonitoredTool` re-raises immediately on subsequent calls
  (sticky-done), so swallowing once still terminates the agent.
- Agents must not capture `recorder` or `toolbox` references that outlive
  `run()`.
- Agents do not read budget, `is_done`, or trajectory state. Those concerns
  belong to `toolbox` (which raises) and `Episode` (which finalizes). The
  agent's job is to think and call tools.

### Updated invariants

1. `AgentConfig.make(action_set)` returns an `Agent` subclass. (unchanged)
2. Either `Agent.step` (sync turn) or `Agent.run` (async loop) must be
   implementable; agents that override `run` only must still provide a
   trivial `step` that raises `NotImplementedError` for clarity.
3. Every LLM call inside `step` or `run` must be captured in the resulting
   `AgentEvent.llm_calls` (via `recorder.record(...)` or `Turn.add_llm_call(...)`).

### Contracts for implementers

- For parallel tool calls in an overridden `run`:
  `await asyncio.gather(*(toolbox(a) for a in actions))`.
- For early termination from inside the agent: return from `run` (don't raise).
- For "the env said done" handling: catch `EpisodeDone` only if you need the
  final `EnvironmentOutput` carried on the exception; otherwise let it propagate.
- For streaming agents: use `recorder.begin_turn() as turn:` and call
  `turn.add_*` as data arrives. Avoid the coarse `record(output)` for
  streaming use cases — it loses the partial-emit advantage.

---

## MODIFIED — `openspec/specs/tool/spec.md`

### `MonitoredTool` replaces `ToolWithTelemetry`

```python
class MonitoredTool(AsyncTool):
    def __init__(
        self,
        inner: AsyncTool | Tool,
        trajectory: Trajectory,
        budget: Budget,
        storage: Storage,
        summary: SummaryProcessor,
        tracer: Tracer,
    )
    # State the toolbox owns internally (not exposed to agents):
    #   _is_done: bool                   (sticky)
    #   _last_env_output: EnvironmentOutput | None

    async def __call__(self, action: Action) -> EnvironmentOutput
    # 1. If self._is_done: raise EpisodeDone(self._last_env_output).
    # 2. If self.budget.exhausted: set self._is_done; raise BudgetExceeded.
    # 3. Open OTel span; await inner.aexecute_action(action) wrapping sync as needed.
    # 4. If self.task.requires_step_eval: call self.task.evaluate(env_output.obs);
    #    build StepEval(reward, info). Catch exceptions and record
    #    StepEval(0.0, {"step_eval_failed": str(exc)}) instead of propagating.
    # 5. Append a ToolCallEvent (with step_eval if produced) to trajectory;
    #    storage.save_event; summary.on_event.
    # 6. self._last_env_output = env_output.
    # 7. If env_output.done: set self._is_done; raise EpisodeDone(env_output).
    # 8. Return env_output.

    @property
    def last_env_output(self) -> EnvironmentOutput | None
    # Read-only accessor for Episode (to compute final_obs after agent.run returns).

class MonitoredToolbox(AsyncToolbox):
    def __init__(
        self,
        inner: Toolbox | AsyncToolbox,
        task: Task,                    # for step-wise evaluate() if requires_step_eval
        trajectory: Trajectory,
        budget: Budget,
        storage: Storage,
        summary: SummaryProcessor,
        tracer: Tracer,
    )
    # Wraps each member tool as MonitoredTool sharing the same _is_done /
    # _last_env_output / budget / task state across the toolbox instance.

class EpisodeDone(BaseException):
    output: EnvironmentOutput | None

class BudgetExceeded(BaseException):
    output: EnvironmentOutput | None
```

### Contract

- `MonitoredTool` is the **only** place where storage / summary / tracer hooks
  fire for tool execution. Subclasses of `cube.tool.Tool` / `AsyncTool` must
  NOT add storage calls.
- `EpisodeDone` / `BudgetExceeded` subclass `BaseException` (not `Exception`)
  so an agent's `try / except Exception` does not swallow them. Bare `except:`
  in agents is forbidden by review (CC-003 vibe-coding rule).
- Sticky-done: the toolbox's internal `_is_done` flag is checked on entry to
  every `__call__`. Once set, no further env state mutates.
- The agent does not see `_is_done`, `_last_env_output`, or `budget`
  directly — only the raised exceptions and the `EnvironmentOutput`
  returned from a successful call. The toolbox is the agent's window on
  termination signals.

### Removed

- `ToolWithTelemetry` and `AsyncToolWithTelemetry` are removed.
  In-tree tools currently subclassing them migrate to `cube.tool.Tool` /
  `AsyncTool` directly; telemetry now comes from `MonitoredTool` wrapping at
  the harness boundary.

### Gotchas

- `MonitoredToolbox` must be constructed per-episode — its internal state
  (`_is_done`, `_last_env_output`, budget counters) is per-episode. Re-using
  a `MonitoredToolbox` across episodes is a bug.
- The OTel span attribute `gen_ai.tool.call.result` still uses the
  string-coerced result body, same as today.

---

## MODIFIED — `openspec/specs/episode/spec.md`

### Loop becomes `await agent.run(...)` with defensive finalization

`Episode.run` is now `async` and follows this shape:

```python
async def run(self) -> Trajectory:
    task = self.task_config.make(runtime_context=..., container_backend=...)
    trajectory = Trajectory(id=..., events=[])
    budget = Budget(max_turns=self.max_steps, ...)
    toolbox = MonitoredToolbox(
        task.toolbox, task, trajectory, budget,
        self.storage, self.summary, self.tracer,
    )
    recorder = TurnRecorder(trajectory, self.storage, self.summary, self.tracer)
    initial = task.reset()
    recorder.record_reset(initial)                # Episode-only helper on recorder
    try:
        await self.agent.run(initial.obs, toolbox, recorder)
    except EpisodeDone:
        pass
    except BudgetExceeded as e:
        recorder.record_failure(e)
    except BaseException as e:
        recorder.record_failure(e)
    finally:
        final_obs = toolbox.last_env_output.obs if toolbox.last_env_output else None
        try:
            reward, info = task.evaluate(final_obs)
            recorder.record_evaluation(reward, info)
        except Exception as e:
            recorder.record_evaluation(0.0, {"evaluate_failed": str(e)})
        await self.storage.finalize(trajectory)
        self.summary.on_episode_complete(trajectory, self.storage)
        try:
            task.close()
        finally:
            self.tracer.shutdown()
    return trajectory
```

The `trajectory` lives on `Episode`. `toolbox` and `recorder` are both bound
to it at construction. Only `toolbox` and `recorder` are passed to the agent;
the agent never sees the trajectory directly.

`TurnRecorder` exposes Episode-only helpers (`record_reset`,
`record_failure`, `record_evaluation`) on the same object as the agent-facing
methods, to keep event construction in one place. Agents should not call
these; the convention is documented but not actively prevented in v1.

### Invariants

1. Final finalization (record_evaluation, storage.finalize, summary, task.close)
   runs even if `agent.run` raises an arbitrary exception. (replaces old #1, #2, #3)
2. `MonitoredToolbox` is the only object passed to the agent for env
   interaction; the agent does not see `task.step` directly.
3. `task.reset()` is always called by `Episode`, never by the agent.
4. `task.evaluate()` is always called by `Episode` after `agent.run` returns,
   exactly once.
5. `Trajectory.events` is persisted incrementally — every `MonitoredTool` call
   appends a `ToolCallEvent`; every `recorder.record()` / `Turn.__exit__`
   appends an `AgentEvent`; each append triggers a corresponding
   `storage.save_event`.

### Updated `EpisodeConfig`

```python
class EpisodeConfig(TypedBaseModel):
    id: int
    agent_config: AgentConfig
    exp_name: str
    output_dir: Path
    budget: Budget                   # replaces max_steps
    task_config: TaskConfig
```

`Budget` is a new model with `max_turns`, `max_tool_calls`, `max_cost_usd`,
`max_wallclock_s`. `max_steps` field is accepted as a deprecated alias that
maps to `max_turns`.

### Storage layout impact

The `episodes/<trajectory_id>/steps/` directory is renamed to `events/`.
Step files become `{nnn:03d}_{agent|tool_call|eval}.msgpack.zst`. Old `steps/`
directories remain loadable via the migration shim (see storage delta).

### Gotchas

- `Episode.run` becoming `async` means callers using
  `asyncio.run(episode.run())` or running inside an existing loop must adapt.
  `exp_runner.run_sequentially` / `run_with_ray` are updated to await.

---

## MODIFIED — `openspec/specs/storage/spec.md`

### Event-file naming

V2 layout's `steps/` directory is renamed `events/`. Files become
`{nnn:03d}_{kind}.msgpack.zst` where `kind ∈ {agent, tool_call, eval}`.
`status.json`, `episode.metadata.json`, `episode_config.json`, `failure.txt`,
`logs/` are unchanged.

### New protocol methods

```python
class Storage(Protocol):
    def save_event(self, event: TrajectoryEvent, trajectory_id: str, event_num: int) -> None
    def load_event(self, trajectory_id: str, event_num: int) -> TrajectoryEvent
    def finalize(self, trajectory: Trajectory) -> None
    # ... existing methods ...
```

`save_step` and `load_step` remain as deprecated aliases that read/write the
new event files using a kind inference rule (kind embedded in filename).

### Migration

`FileStorage.load_trajectory` auto-detects layout:

1. If `events/` exists: load `_agent`, `_tool_call`, `_eval` files in order.
2. Else if `steps/` exists: load `_obs` / `_act` files, then convert in-memory:
   - `_act` (AgentOutput) → `AgentEvent` (with synthetic `id`, empty
     `response_text`).
   - `_obs` (EnvironmentOutput) → `ToolCallEvent` (with synthetic
     `agent_event_id` referencing the prior `AgentEvent`, synthetic
     `action_id`, synthetic `turn_id`).
3. New writes are always `events/`.

### Invariants (additions)

- Event files are immutable after write. `finalize` writes a
  `trajectory.metadata.json` that summarises the event count and final
  `EvaluationEvent`.
- The conversion in step 2 above is best-effort and round-trippable for
  trajectories that obeyed the old alternation invariant.

### Summary

`SummaryProcessor` keeps `n_agent_events`, `n_tool_calls`, `n_evaluations`
counters in place of `n_agent_steps` / `n_env_steps`. Existing
`episode_summary.jsonl` line format is otherwise unchanged. The deprecated
field names remain as JSON aliases.

---

## MODIFIED — `openspec/specs/analyze/spec.md`

### Event-card timeline

The viewer renders one card per `TrajectoryEvent`. Card colour by kind:

- `agent` — assistant turn (thoughts, response_text, LLM calls, intended actions)
- `tool_call` — one env interaction
- `eval` — final task.evaluate output

`ToolCallEvent`s sharing a `turn_id` render in horizontal lanes within a turn
group (parent `AgentEvent` above, siblings below).

### Selection model

- `selected_event` is the user's last click.
- `last_agent_event` = the most recent `AgentEvent` with
  `start_time <= selected_event.start_time`.
- `last_observation_event` = the most recent `ToolCallEvent` matching the same rule.

### Tabs

- **Reasoning / Chat** — `last_agent_event.thoughts`, `response_text`,
  `llm_calls`, intended `actions`.
- **Observation** — if `selected_event` is a `tool_call`, render
  `selected_event.output`; else render `last_observation_event.output`.
  Screenshots, AXTree, and HTML are all surfaces inside the obs renderer; no
  separate screenshot tab.
- **Turn observations** — list of all `tool_call` events sharing
  `selected_event`'s `turn_id` (empty when no turn). Useful for parallel tool
  calls.
- **Profiling** — per-event `profiling` timing breakdown.
- Header always reads: `Event X / N — kind={agent|tool_call|eval}, turn=<id>, t=<s>s`.

### Invariants

- Read-only (unchanged).
- Loads both `events/` (new) and `steps/` (legacy) layouts via storage shim.
- Stale background-loader generations still self-abort (unchanged).

### Removed

- The standalone screenshot tab. Screenshot rendering moves inside the
  Observation tab (screenshots are obs content, not a separate concern).
- "UI step" pairing logic (env+agent paired into a single navigation unit) is
  removed — navigation is per-event.

### Gotchas

- Trajectories with very wide parallel tool calls (e.g. 20+ siblings in one
  turn) will overflow the horizontal lane layout. Out of scope for v1;
  acceptable to fall back to a vertical list above some threshold.

---

## REMOVED

- `cube_harness.tool.ToolWithTelemetry`, `AsyncToolWithTelemetry`.
- `Trajectory.steps` direct field — replaced by `Trajectory.events`. The
  deprecated `steps` property remains as a computed alias for one release.
- "Trajectory steps alternate" invariant in `core/spec.md`.
- Standalone screenshot tab in XRay.

---

## Tests landed in the same PR

- **Integration**: same task, default `Agent.run` (gym-style) and overridden
  `Agent.run` (parallel tool calls) both run to completion, produce loadable
  trajectories, and pass the verifier.
- **Integration**: agent that catches `Exception` mid-loop — verify
  `EpisodeDone(BaseException)` and sticky-done both fire, and that
  `Episode.run` still finalizes everything.
- **Smoke**: existing experiment dir (steps/ layout) loads through XRay and
  renders all tabs.
- **Smoke**: a new experiment dir (events/ layout) loads through XRay and
  renders all tabs.
- **Unit**: `TurnRecorder.record()` and `TurnRecorder.begin_turn()` produce
  equivalent `AgentEvent`s; both paths preserve `AgentEvent.id`,
  back-references, and field set.
- **Unit**: `MonitoredTool.__call__` honours sticky-done, budget exhaustion,
  and `EnvironmentOutput.done`. The internal `_is_done` / `_last_env_output`
  state is per-toolbox-instance.
- **Unit**: when `task.requires_step_eval` is True, every `MonitoredTool.__call__`
  attaches a `StepEval(reward, info)` to the resulting `ToolCallEvent`.
  When False, `ToolCallEvent.step_eval` is None. A `task.evaluate()` that
  raises is captured as `StepEval(0.0, {"step_eval_failed": ...})` without
  failing the call.

---

## Open questions

1. **Budget granularity.** Phase 1 ships `max_turns` and a deprecated alias
   for `max_steps`. Do we ship `max_tool_calls`, `max_cost_usd`,
   `max_wallclock_s` now or later? Recommendation: ship the field names but
   only enforce `max_turns` in `MonitoredTool.__call__` until we wire up cost
   accounting end-to-end.
2. **`Agent.step` deprecation timeline.** Keep one release? Two?
   Recommendation: one release, then `step` becomes optional (only required
   if `run` is not overridden).
3. **Async `step` (`astep`)** as a first-class method on sync agents wanting
   coroutine semantics without overriding `run` — worth adding or YAGNI?
   Recommendation: YAGNI for Phase 1.
