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
        task: Task,                                # cube-standard Task; task.toolbox is monitored
        recorder: "TurnRecorder",
    ) -> None
```

Default implementation in the base class:

1. `obs = initial_obs`
2. Loop:
   1. `agent_output = await asyncio.to_thread(self.step, obs)` (sync agents)
      or `agent_output = await self.astep(obs)` if subclass defines `astep`.
   2. `recorder.record(agent_output)`.
   3. If `not agent_output.actions and not agent_output.error`: return.
   4. `env_output = await task.astep(agent_output.actions)`. May raise
      `BudgetExceeded` (propagates to `Episode`).
   5. If `env_output.done`: return.
   6. `obs = env_output.obs`.

Agents that want parallel tool calls override `run` and call
`task.toolbox.execute_action(action)` directly, each returning
`Observation | StepError`. Done detection then comes from the agent
inspecting obs or from a done-signaling tool whose obs triggers
`task.finished()` for any subsequent `task.astep` call.

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

#### Lossy capture path (for connectors)

External-framework connectors (LangGraph, Codex CLI, A2A, …) often cannot
break the run into per-turn `AgentEvent`s — they only expose a final
output and aggregate usage. For these, `TurnRecorder` provides a lossy
shortcut:

```python
class TurnRecorder:
    def record_external_run(
        self,
        final_text: str | None,
        usage: Usage | None,                  # tokens, cost — best-effort
        raw_events: list[dict] | None = None, # framework-specific stream, opaque blob
    ) -> None:
        """Emit one synthetic AgentEvent summarizing an external agent's run.

        Use when the connector cannot decompose the agent's execution into
        per-turn events. The synthetic AgentEvent carries the final response
        text and the aggregated usage; `raw_events` is preserved verbatim
        on the event for post-hoc inspection but is not interpreted by the
        recorder or summary.

        Connectors that CAN observe per-turn events (Pydantic AI, LangGraph,
        OpenAI Agents SDK, Inspect AI) should use record() or begin_turn()
        instead — record_external_run is the fallback for opaque frameworks.
        """
```

The trade-off is documented per-connector: full-visibility frameworks use
the standard path; opaque ones use this. See *Connector taxonomy* in
[proposal.md](proposal.md).

### Semantics

- `Agent.run` is the **canonical entry point** invoked by `Episode`.
- `Agent.step` remains as the unit of one LLM turn for sync agents. Agents
  that want parallel/async behavior override `run` and may ignore `step`.
- `BudgetExceeded` (`BaseException` subclass) raised out of any monitored
  tool call should propagate. Catching it is forbidden by convention.
- Done detection comes from `EnvironmentOutput.done` (returned by
  `task.astep`) or from the agent's own logic — `MonitoredTool` does not
  raise on done.
- Agents must not capture `task` or `recorder` references that outlive
  `run()`. Conventionally they call only `task.astep`, `task.toolbox.*`,
  and `task.aevaluate`; not `reset` / `close` (Episode's).
- Agents do not read budget or trajectory state. Those concerns belong to
  monitored tools (which raise) and `Episode` (which finalizes).

### Updated invariants

1. `AgentConfig.make(action_set)` returns an `Agent` subclass. (unchanged)
2. Either `Agent.step` (sync turn) or `Agent.run` (async loop) must be
   implementable; agents that override `run` only must still provide a
   trivial `step` that raises `NotImplementedError` for clarity.
3. Every LLM call inside `step` or `run` must be captured in the resulting
   `AgentEvent.llm_calls` (via `recorder.record(...)` or `Turn.add_llm_call(...)`).

### Contracts for implementers

- For parallel tool calls in an overridden `run`:
  `await asyncio.gather(*(task.toolbox.execute_action(a) for a in actions))`.
  Each returns `Observation | StepError`. Monitoring happens inside each
  `MonitoredTool.execute_action`.
- For early termination from inside the agent: return from `run` (don't raise).
- For "the env said done" in gym-style: check
  `env_output.done` after each `task.astep` and return.
- For streaming agents: use `recorder.begin_turn() as turn:` and call
  `turn.add_*` as data arrives. Avoid the coarse `record(output)` for
  streaming use cases — it loses the partial-emit advantage.

---

## MODIFIED — `openspec/specs/tool/spec.md`

### `MonitoredTool` replaces `ToolWithTelemetry`

`MonitoredTool` subclasses `cube.tool.AsyncTool` and has the **same**
`execute_action` signature. It is a transparent decorator — mixable in a
`Toolbox` alongside unmonitored tools. Agents call it identically.

```python
class MonitoredTool(AsyncTool):
    def __init__(
        self,
        inner: Tool | AsyncTool,
        trajectory: Trajectory,
        budget: Budget,
        storage: Storage,
        summary: SummaryProcessor,
        tracer: Tracer,
    )

    @property
    def action_set(self) -> list[ActionSchema]
    # Delegates to inner.action_set — transparent.

    async def execute_action(self, action: Action) -> Observation | StepError
    # 1. If self.budget.exhausted: raise BudgetExceeded(action=action).
    # 2. Open OTel span; invoke inner.execute_action(action) — await
    #    directly when inner is AsyncTool, asyncio.to_thread when sync.
    # 3. Append a ToolCallEvent (with the action and result) to trajectory;
    #    storage.save_event; summary.on_event; budget.tool_calls += 1.
    # 4. Return result (Observation | StepError) unchanged.

class BudgetExceeded(BaseException):
    action: Action                       # the call that pushed over budget
```

`MonitoredToolbox` is a `Toolbox` whose member tools are each wrapped in
`MonitoredTool`. Since `Toolbox` is-a `Tool`, the wrapping is
straightforward and recursive: a toolbox may contain monitored tools,
unmonitored tools, and nested toolboxes. Dispatch by action name routes
each call to the right member, monitored or not. There is no separate
`MonitoredToolbox` class — `Toolbox(members=[MonitoredTool(inner=t1), t2, ...])`
is sufficient.

### Contract

- `MonitoredTool.execute_action` returns `Observation | StepError` —
  **same** as any cube-standard `Tool`. It does NOT return
  `EnvironmentOutput` and does NOT detect `done`. Those remain
  `Task.step`'s responsibility, which calls into the toolbox and wraps the
  result.
- `MonitoredTool` is the only place where storage / summary / tracer hooks
  fire for tool execution. Subclasses of `cube.tool.Tool` / `AsyncTool`
  must NOT add storage calls.
- `BudgetExceeded` subclasses `BaseException` so `try / except Exception`
  does not swallow it. Bare `except:` in agents is forbidden by review
  (CC-003 vibe-coding rule).
- Step-wise evaluation is NOT a `MonitoredTool` concern. cube-standard's
  `Task.step()` already invokes `self.evaluate(obs)` internally when
  `self.validate_per_step` is `True` (see
  [cube-standard task/spec.md](../../../../cube-standard/openspec/specs/task/spec.md)
  and [task.py:346](../../../../cube-standard/src/cube/task.py#L346)).
  The resulting `reward` and `info` flow back through `EnvironmentOutput.reward`
  / `info`, captured automatically in `ToolCallEvent.output` when the
  monitored tool is invoked by `task.step`. No harness-side step-eval logic
  needed.

### Removed

- `ToolWithTelemetry` and `AsyncToolWithTelemetry` are removed.
  In-tree tools currently subclassing them migrate to `cube.tool.Tool` /
  `AsyncTool` directly; telemetry now comes from `MonitoredTool` wrapping
  at the harness boundary.

### Gotchas

- `MonitoredTool` and its budget counter are per-episode — re-using a
  `MonitoredTool` across episodes is a bug.
- The OTel span attribute `gen_ai.tool.call.result` still uses the
  string-coerced result body, same as today.

---

## MODIFIED — `openspec/specs/episode/spec.md`

### Loop becomes `await agent.run(...)` with Episode-owned finalization

`Episode.run` is now `async` and follows this shape. The harness no longer
has its own loop — there is only `agent.run`. Sync agents use the
base-class default implementation; agents that want parallel/streaming
behavior override it.

```python
async def run(self) -> Trajectory:
    task = self.task_config.make(runtime_context=..., container_backend=...)
    trajectory = Trajectory(id=..., events=[])
    budget = Budget(max_turns=self.max_steps, ...)

    # Wrap each member of task.toolbox with MonitoredTool, sharing trajectory + budget.
    # task.toolbox is mutated in place so task.step also goes through monitored wrappers.
    install_monitoring(task, trajectory, budget,
                       self.storage, self.summary, self.tracer)

    recorder = TurnRecorder(trajectory, self.storage, self.summary, self.tracer)
    initial = task.reset()
    recorder.record_reset(initial)
    try:
        await self.agent.run(initial.obs, task, recorder)
    except BudgetExceeded as e:
        recorder.record_failure(e)
    except BaseException as e:
        recorder.record_failure(e)
    finally:
        try:
            reward, info = task.evaluate()        # cube-standard: obs optional
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

The `trajectory` lives on `Episode`. The agent receives `task` and
`recorder` — the monitoring wrappers are already installed onto
`task.toolbox`, so any path through tools (gym-style `task.astep` or
tool-level `task.toolbox.execute_action`) emits monitoring uniformly.

`TurnRecorder` exposes Episode-only helpers (`record_reset`,
`record_failure`, `record_evaluation`) on the same object as the agent-facing
methods, to keep event construction in one place. Agents should not call
these; the convention is documented but not actively prevented in v1.

The `except BaseException` (not `Exception`) catches `BudgetExceeded` and
any agent-side crash without swallowing programmer-intent signals like
`KeyboardInterrupt`. Finalization runs unconditionally so trajectories on
disk survive any agent misbehavior.

### Invariants

1. Final finalization (record_evaluation, storage.finalize, summary, task.close)
   runs even if `agent.run` raises an arbitrary exception. (replaces old #1, #2, #3)
2. The agent receives `task` (with monitoring already installed on its
   toolbox) and `recorder`. The agent uses `task.astep`, `task.toolbox.*`,
   or `task.aevaluate` for tool / step / eval calls — all routed through
   monitored wrappers.
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
  `BudgetExceeded(BaseException)` propagates regardless and that
  `Episode.run` still finalizes everything (evaluate, storage, task.close,
  recorder.record_failure).
- **Smoke**: existing experiment dir (steps/ layout) loads through XRay and
  renders all tabs.
- **Smoke**: a new experiment dir (events/ layout) loads through XRay and
  renders all tabs.
- **Unit**: `TurnRecorder.record()` and `TurnRecorder.begin_turn()` produce
  equivalent `AgentEvent`s; both paths preserve `AgentEvent.id`,
  back-references, and field set.
- **Unit**: `MonitoredTool.execute_action` returns `Observation | StepError`
  unchanged, records a `ToolCallEvent` per call, raises `BudgetExceeded`
  when budget is exhausted. Drop-in compatibility: a Toolbox with mixed
  monitored + unmonitored tools dispatches correctly by action name.
- **Integration**: a task with `validate_per_step=True` produces
  `ToolCallEvent.output.reward` and `info["profiling"]["evaluate"]` populated
  on every event (validated against cube-standard's existing per-step eval
  in `Task.step()` — no harness-side logic).

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
