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

class EvaluationEvent(TypedBaseModel):
    reward: float
    info: dict

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
  internally converted into an `AgentEvent` by `LoopContext.record_agent_event`.

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
        ctx: "LoopContext",
    ) -> None
```

Default implementation in the base class:

1. `obs = initial_obs`
2. While not `ctx.is_done` and `ctx.turn < ctx.budget.max_turns`:
   1. `agent_output = await asyncio.to_thread(self.step, obs)` (sync agents)
      or `agent_output = await self.astep(obs)` if subclass defines `astep`.
   2. `ctx.record_agent_event(agent_output)`.
   3. If `not agent_output.actions and not agent_output.error`: return.
   4. For each `action` in `agent_output.actions`: `await toolbox(action)`.
      The last `EnvironmentOutput`'s `obs` becomes the next `obs`.
3. Returns normally on budget exhaustion or empty actions. Raises propagate.

### Semantics

- `Agent.run` is the **canonical entry point** invoked by `Episode`.
- `Agent.step` remains as the unit of one LLM turn for sync agents. Agents
  that want parallel/async behavior override `run` and may ignore `step`.
- `EpisodeDone` and `BudgetExceeded` (both `BaseException` subclasses) raised
  out of `toolbox(action)` should propagate. Catching them is forbidden by
  convention. `MonitoredTool` re-raises immediately on subsequent calls
  (sticky-done), so swallowing once still terminates the agent.
- Agents must not capture `ctx` references that outlive `run()`.

### Updated invariants

1. `AgentConfig.make(action_set)` returns an `Agent` subclass. (unchanged)
2. Either `Agent.step` (sync turn) or `Agent.run` (async loop) must be
   implementable; agents that override `run` only must still provide a
   trivial `step` that raises `NotImplementedError` for clarity.
3. Every LLM call inside `step` or `run` must be captured in the resulting
   `AgentEvent.llm_calls`.

### Contracts for implementers

- For parallel tool calls in an overridden `run`:
  `await asyncio.gather(*(toolbox(a) for a in actions))`.
- For early termination from inside the agent: return from `run` (don't raise).
- For "the env said done" handling: catch `EpisodeDone` only if you need the
  final `EnvironmentOutput` carried on the exception; otherwise let it propagate.

---

## MODIFIED — `openspec/specs/tool/spec.md`

### `MonitoredTool` replaces `ToolWithTelemetry`

```python
class MonitoredTool(AsyncTool):
    def __init__(self, inner: AsyncTool | Tool, ctx: "LoopContext")

    async def __call__(self, action: Action) -> EnvironmentOutput
    # 1. If ctx.is_done: raise EpisodeDone(ctx.last_env_output).
    # 2. If ctx.budget.exhausted: set ctx.is_done; raise BudgetExceeded.
    # 3. Open OTel span; await inner.aexecute_action(action) wrapping sync as needed.
    # 4. ctx.record_tool_call(action, env_output).
    # 5. If env_output.done: set ctx.is_done; raise EpisodeDone(env_output).
    # 6. Return env_output.

class MonitoredToolbox(AsyncToolbox):
    def __init__(self, inner: Toolbox | AsyncToolbox, ctx: "LoopContext")
    # Wraps each member tool as MonitoredTool sharing ctx.

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
- Sticky-done: `ctx.is_done` is checked on entry to every `__call__`. Once
  set, no further env state mutates.

### Removed

- `ToolWithTelemetry` and `AsyncToolWithTelemetry` are removed.
  In-tree tools currently subclassing them migrate to `cube.tool.Tool` /
  `AsyncTool` directly; telemetry now comes from `MonitoredTool` wrapping at
  the harness boundary.

### Gotchas

- `MonitoredToolbox` must be constructed per-episode — `ctx` is per-episode
  state. Re-using a `MonitoredToolbox` across episodes is a bug.
- The OTel span attribute `gen_ai.tool.call.result` still uses the
  string-coerced result body, same as today.

---

## MODIFIED — `openspec/specs/episode/spec.md`

### Loop becomes `await agent.run(...)` with defensive finalization

`Episode.run` is now `async` and follows this shape:

```python
async def run(self) -> Trajectory:
    task = self.task_config.make(runtime_context=..., container_backend=...)
    ctx = LoopContext(
        task=task, storage=self.storage, summary=self.summary,
        tracer=self.tracer, trajectory=Trajectory(id=..., events=[]),
        budget=Budget(max_turns=self.max_steps, ...),
    )
    toolbox = MonitoredToolbox(task.toolbox, ctx)
    initial = task.reset()
    ctx.record_reset(initial)                     # synthetic event capturing the initial obs
    try:
        await self.agent.run(initial.obs, toolbox, ctx)
    except EpisodeDone:
        pass
    except BudgetExceeded:
        ctx.record_failure(BudgetExceeded(...))
    except BaseException as e:
        ctx.record_failure(e)
    finally:
        final_obs = ctx.last_env_output.obs if ctx.last_env_output else None
        try:
            reward, info = task.evaluate(final_obs)
            ctx.record_evaluation(reward, info)
        except Exception as e:
            ctx.record_evaluation(0.0, {"evaluate_failed": str(e)})
        await self.storage.finalize(ctx.trajectory)
        self.summary.on_episode_complete(ctx.trajectory, self.storage)
        try:
            task.close()
        finally:
            self.tracer.shutdown()
    return ctx.trajectory
```

### Invariants

1. Final finalization (record_evaluation, storage.finalize, summary, task.close)
   runs even if `agent.run` raises an arbitrary exception. (replaces old #1, #2, #3)
2. `MonitoredToolbox` is the only object passed to the agent for env
   interaction; the agent does not see `task.step` directly.
3. `task.reset()` is always called by `Episode`, never by the agent.
4. `task.evaluate()` is always called by `Episode` after `agent.run` returns,
   exactly once.
5. `Trajectory.events` is persisted incrementally — each `record_*` call
   triggers a corresponding `storage.save_event`.

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
- **Unit**: `LoopContext.record_*` methods produce the right events with the
  right back-references (`AgentEvent.id` / `ToolCallEvent.action_id` /
  `turn_id`).
- **Unit**: `MonitoredTool.__call__` honours sticky-done, budget exhaustion,
  and `EnvironmentOutput.done`.

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
