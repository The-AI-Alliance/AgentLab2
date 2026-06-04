# Deltas: Agent Owns the Loop

Applies to (cube-harness specs):

- `openspec/specs/core/spec.md` (primary — trajectory event model)
- `openspec/specs/agent/spec.md` (primary — `Agent.run`)
- `openspec/specs/tool/spec.md` (**ADDED** — spec layer was deleted with
  `ToolWithTelemetry` in commit e760f9e5; this RFC re-creates it for
  `MonitoredTool`)
- `openspec/specs/episode/spec.md` (primary — loop refactor; builds on
  streaming foundation from `stream-trajectory-steps`, already merged)
- `openspec/specs/storage/spec.md` (event-file layout, evolving the
  streamed-step model from `stream-trajectory-steps`)
- `openspec/specs/analyze/spec.md` (XRay event-card UI)

See companion: `cube-standard/openspec/changes/agent-owns-loop/deltas.md`.

### Relationship to recent dev changes

- `stream-trajectory-steps` already lands the "stream every step to
  disk, never accumulate in memory" pattern. This RFC adopts that
  pattern wholesale and changes both **what** is streamed and **who
  streams it**: the event union expands from
  `EnvironmentOutput | AgentOutput` to `LLMCallEvent | ToolCallEvent |
  EvaluationEvent | AgentErrorEvent`, and the prior `SummaryProcessor`
  is folded into `EventStreamer` (no separate aggregator class; counter
  folding happens inline as events flow through).
- `e760f9e5` (cleanup: delete unused tool implementations + telemetry
  wrapper) already removed `ToolWithTelemetry`, `AsyncToolWithTelemetry`,
  and the `openspec/specs/tool/` spec layer. The tool spec is **created
  fresh** by this RFC, not modified.
- `cube-standard:2dfcdeb` (Task generic over tool type) gives `Task` the
  signature `Task[TMeta, TTool]`. Our additions in
  `cube-standard/openspec/changes/agent-owns-loop/deltas.md` are additive
  on top of that generic form.

---

## MODIFIED — `openspec/specs/core/spec.md`

### Trajectory class removed; replaced by TrajectoryMetadata + TrajectoryView

`Trajectory` (the Pydantic class) is **deleted** from `cube_harness.core`.
Two new abstractions replace it: `TrajectoryMetadata` (pure metadata,
persisted as `episode.metadata.json`) lives in `cube_harness.core`;
`TrajectoryView` (lazy reader over a storage handle) lives in
`cube_harness.storage`.

`TrajectoryStep` (the legacy `EnvironmentOutput | AgentOutput` union) is
removed from the public API. The type remains internally in the V1
load-upgrade path but is not exported from `cube_harness.core`.

```python
# In cube_harness.core
class LLMCallEvent(TypedBaseModel):
    id: str                            # parent_event_id for child ToolCallEvents (groups parallel siblings)
    call: LLMCall | None               # full prompt/response/usage (None on legacy decode only)
    profiling: dict[str, tuple[float, float]]
    metadata: dict                     # free-form bag — producers attach sink-specific data
                                       # (e.g. RL: prompt_token_ids, logprobs, trainable_call_index)
                                       # without coupling the core event model to a consumer.
    error: StepError | None

class ToolCallEvent(TypedBaseModel):
    id: str                            # for step-wise EvaluationEvent.parent_event_id back-ref
    parent_event_id: str               # parent LLMCallEvent.id (or RESET sentinel); shared by parallel siblings
    action_id: str | None              # echoes Action.id
    action: Action | None              # full action payload — self-contained trajectory
    obs: Observation                   # what came back to the agent (empty when error)
    error: StepError | None            # set when execute_action returned a StepError

class EvaluationEvent(TypedBaseModel):
    reward: float
    info: dict
    is_terminal: bool
    parent_event_id: str | None        # ToolCallEvent.id for step-wise; for terminal, id of the most recent ToolCallEvent

class AgentErrorEvent(TypedBaseModel):
    id: str
    error: StepError                   # Episode-level failure not tied to a call
    parent_event_id: str | None        # id of the most recent LLMCallEvent (or last ToolCallEvent if none), or None

class TrajectoryEvent(TypedBaseModel):
    output: LLMCallEvent | ToolCallEvent | EvaluationEvent | AgentErrorEvent
    start_time: float
    end_time: float

class TrajectoryMetadata(TypedBaseModel):
    """Persisted at episode.metadata.json. Written at episode start with
    end_time=None and stub fields; updated at episode end with the
    final summary."""
    id: str
    metadata: dict                     # task_id, agent_config dict, infra, …
    start_time: float | None
    end_time: float | None             # None until the episode finalizes
    summary_stats: dict | None
    reward_info: dict                  # mirrors the final EvaluationEvent
```

### `TrajectoryView` (in `cube_harness.storage`)

```python
class TrajectoryView:
    """Lazy reader for an episode directory.

    Holds .metadata eagerly (one JSON read). Events live on disk; the
    view decodes them on demand and caches in a per-view dict. Iteration,
    random access, and len() are all supported.
    """
    storage: Storage
    id: str
    metadata: TrajectoryMetadata          # eager
    # private: _cache: dict[int, TrajectoryEvent], _index: list[Path]

    def __len__(self) -> int           # from events/ directory listing
    def __getitem__(self, i: int) -> TrajectoryEvent
    def __iter__(self) -> Iterator[TrajectoryEvent]
    def iter_events(self) -> Iterator[TrajectoryEvent]   # alias for __iter__
    # tool calls of one turn are queried directly from parent_event_id:
    # [e for e in view if isinstance(e.output, ToolCallEvent)
    #                  and e.output.parent_event_id == target_id]
    @property
    def summary_stats(self) -> dict | None
    @property
    def reward_info(self) -> dict
    @property
    def is_complete(self) -> bool      # metadata.end_time is not None
    @property
    def n_agent_events(self) -> int    # one directory scan, no decode
    @property
    def n_tool_calls(self) -> int
    @property
    def n_evaluations(self) -> int
```

### Invariants (replaces old #1, #2)

1. Events are ordered by `start_time` and represent the agent's interaction
   with the task. They are **not** required to alternate.
2. Every `ToolCallEvent.parent_event_id` must resolve to either a preceding
   `LLMCallEvent.id` or the `RESET` sentinel (for tool calls fired before
   any LLM call, e.g. the synthetic reset event).
3. `EvaluationEvent` may appear step-wise (after `MonitoredTool` dispatch
   when `task.validate_per_step=True`) AND once terminal (`is_terminal=True`,
   emitted by Episode in `finally`).
4. `LLMCallEvent.id` is unique within a trajectory.
5. All `ToolCallEvent`s spawned from one LLM turn share the same
   `parent_event_id` (= the originating `LLMCallEvent.id`); that is how
   parallel siblings are grouped.
6. `TrajectoryView` never accumulates the full event list in memory; the
   per-view cache is bounded by accessed events and is freed when the
   view is GC'd.

### Gotchas

- The legacy `EnvironmentOutput | AgentOutput` union is removed from
  `TrajectoryEvent.output`. Callers that pattern-match on those types must
  switch to the new event types.
- `AgentOutput` is now a minimal `{actions, error}` payload — what
  `Agent.step()` returns. LLM calls auto-emit `LLMCallEvent` via
  `LLM.attach_recorder`/`LLM.call(...)` — `AgentOutput` no longer
  bundles `llm_calls` / `thoughts` / `response_text` / `profiling`.
- Migrating from `traj.steps[i]` to `view[i]` changes element types: the
  former gave `TrajectoryStep`, the latter gives `TrajectoryEvent`. Inspect
  `event.output` to discriminate `LLMCallEvent` / `ToolCallEvent` /
  `EvaluationEvent` / `AgentErrorEvent`.
- The view's internal cache is unbounded *within one episode*. Callers
  that iterate every event of a SWE-bench-scale episode and then keep
  the view alive will hold all events in memory — drop the view when
  done.

---

## MODIFIED — `openspec/specs/agent/spec.md`

### `Agent.run` (new, default impl provided)

```python
class Agent(ABC):
    def step(self, obs: Observation) -> AgentOutput        # unchanged shape: {actions, error}

    def attach_recorder(self, recorder: "EventStreamer") -> None:
        """Stash on self._recorder; subclasses override to propagate
        to held LLMs so LLM.call(...) auto-emits LLMCallEvent."""

    async def run(
        self,
        initial_obs: Observation,
        env_tool: AbstractAsyncTool,               # always async-shaped; see "async-uniform" below
    ) -> None
```

**No recorder parameter.** `Episode` calls
`agent.attach_recorder(recorder)` BEFORE `run()`. Recording happens
automatically: `LLM.call(prompt, tag)` emits `LLMCallEvent` if its
`_recorder` is set; `MonitoredTool.execute_action` emits `ToolCallEvent`.
Agent code never touches the recorder directly.

**Budget introspection.** `attach_recorder`'s default stashes the
recorder on `self._recorder`, exposing the live `Budget` at
`self._recorder.budget` for two agent-side patterns:

- **Soft self-stop:** check `budget.exhausted` in `step()` and return
  `STOP_ACTION` for clean termination instead of letting
  `MonitoredTool` raise `BudgetExceeded` mid-call.
- **Prompt injection:** `str(budget)` returns a concise human-readable
  summary of configured caps and current usage; inject every K turns
  (Genny does this via `display_budget_every_k`) so the LLM can plan
  against remaining budget.

Agents that override `attach_recorder` must call `super()` to keep
this wiring intact.

**Async-uniform env_tool.** `Agent.run`'s `env_tool` parameter is
narrowed to `AbstractAsyncTool` (NOT `AbstractTool | AbstractAsyncTool`).
Sync underlying tools are wrapped by `as_async(tool)` at the Episode
boundary — the adapter dispatches `execute_action` via
`asyncio.to_thread`. Agent code that overrides `run` therefore has no
sync/async branch; it always `await`s. Sync-only agent authors don't
see this — they override `step()` and inherit the base `run`.

The name `env_tool` (vs. the older draft `toolbox`) makes the
agent/env boundary explicit: this is the tool that drives the
**monitored environment**, distinct from any agent-private tools
(memory, scratchpad) the agent holds on `self`.

Default implementation in the base class: `Agent.run` is a thin
dispatcher that calls `_run` or `_arun` based on
`AgentConfig.parallel_actions: bool = False`.

- `_run(initial_obs, env_tool)` — sync body, sequential action
  dispatch, no `await`. Used for the default case.
- `_arun(initial_obs, env_tool)` — async body; dispatches N actions
  per turn via `asyncio.gather(*(env_tool.async_execute_action(a) for a in actions))`.
- `_merge_results(results)` — static method on `Agent` that combines
  multiple `Observation | StepError` returns into the next obs;
  subclasses override.

Both bodies follow the same shape:

1. `obs = initial_obs`
2. Loop:
   1. `agent_output = self.step(obs)`. LLM calls inside `step()`
      auto-emit `LLMCallEvent` via the attached recorder; the agent
      does NOT bundle them.
   2. If `not agent_output.actions and not agent_output.error`: return.
   3. Dispatch actions (sequential in `_run`, parallel via
      `asyncio.gather` in `_arun`). May raise `TaskDone` (graceful,
      includes STOP_ACTION / `task.finished()` true) or
      `BudgetExceeded` — both propagate to `Episode`.
   4. If any result is `StepError`: return.
   5. `obs = self._merge_results(results)` (or `results[0]` for the
      sequential case).

Agents that want parallel tool calls override `run` and dispatch
N actions via `asyncio.gather(*(env_tool.execute_action(a) for a in actions))`.
The async-uniform shape means parallel-dispatch agents (e.g. `Genny`
with `parallel_actions=True`) don't need their own sync-vs-async
branch either.

### `EventStreamer`

The trajectory's event sink — no longer an agent-facing API.
Built by `Episode` per-episode; producers attach to it.

```python
class EventStreamer:
    def __init__(
        self,
        trajectory_id: str,
        storage: Storage | None,
        budget: Budget | None,
        metadata_updates: dict | None = None,        # back-channel merged into TrajectoryMetadata.metadata
    ): ...

    # Sinks list — `storage` registers as sink-0 in __init__; additional
    # sinks (OTel emitter, RL HTTP pump, ...) append post-construction
    # via `streamer._sinks.append(sink)`. Sinks implement EventSink
    # (Protocol with one method: `save_event(te, trajectory_id) -> None`).
    _sinks: list[EventSink]

    # Sole fan-out entry point. Folds per-episode stats counters and
    # forwards to sinks. All producers (LLM, MonitoredTool, the
    # boundary helpers below) funnel through this method.
    def emit(self, te: TrajectoryEvent) -> str

    # Producer-facing hook called by LLM.call() auto-emit:
    def on_llm_call(
        self,
        call: LLMCall,
        profiling: dict[str, tuple[float, float]] | None = None,
        error: StepError | None = None,
    ) -> str                                       # returns event id (also new current_parent_event_id)

    # Bump `budget.agent_steps` and enforce; called by Agent.run per step.
    def on_step(self) -> None

    # Episode-only boundary helpers:
    def record_reset(self, initial: EnvironmentOutput) -> None
    def record_failure(self, exc: BaseException) -> None        # → AgentErrorEvent
    def record_evaluation(self, reward: float, info: dict | None = None,
                          *, is_terminal: bool = True) -> None

    # Getter consumed by MonitoredTool.parent_event_id_getter:
    def current_parent_event_id(self) -> str

    # Final per-episode stats. Written to TrajectoryMetadata.summary_stats
    # at finalize. Replaces the dropped SummaryProcessor; no separate
    # episode_summary.jsonl is written.
    def summary_stats(self, *, duration: float | None,
                      final_reward: float) -> dict

    @property
    def budget(self) -> Budget                                  # for agent introspection
```

`LLM.attach_recorder(recorder)` wires the LLM so every `.call()`
auto-emits. `Agent.attach_recorder(recorder)` is overridden by agents
that hold LLMs to propagate the wiring (`Genny`, `React`,
`GenericAgent`). The legacy `record(AgentOutput)` / `begin_turn()` /
`Turn` / `add_*` surface is **dropped** — producer auto-emit replaces
it.

#### `EventStreamerConfig` (forward seam)

A pydantic `EventStreamerConfig` field on `EpisodeConfig` reserves the
hook for Phase-2 sinks (OTel, RL HTTP, custom). Phase 1 ships an
empty `EventStreamerConfig` — `FileStorage` is the sole sink
(per-episode stats are folded directly inside the streamer; no
separate sink for them). Future fields like `enable_otel: bool` and
`rl_http_endpoint: str | None` plug in additively.

#### `EventSink` Protocol

```python
@runtime_checkable
class EventSink(Protocol):
    def save_event(self, te: TrajectoryEvent, trajectory_id: str) -> None: ...
```

Structural shape every sink implements. Matches `FileStorage.save_event`
exactly — the first sink conformed by accident; the Protocol declaration
makes future sinks self-documenting. `EventStreamer.emit()` iterates
`self._sinks` and catches per-sink exceptions so a misbehaving
downstream (slow HTTP, full disk) cannot kill the trajectory. Sinks
**must** be cheap and non-blocking — `emit()` runs on the agent loop's
hot path under the stats lock; for I/O, queue inside the sink.

#### Connector path (Phase 2)

External-framework connectors (LangGraph, Codex CLI, A2A, …) that
can't decompose the agent's execution into per-turn events emit
synthetic `LLMCallEvent`s by calling `recorder.on_llm_call(...)`
directly with a hand-rolled `LLMCall` (best-effort prompt/usage).
The old lossy `record_external_run` shortcut was dropped — the
auto-emit hook is the same path for native and connector use.

### Semantics

- `Agent.run` is the **canonical entry point** invoked by `Episode`.
- `Agent.step` remains as the unit of one LLM turn for sync agents. Agents
  that want parallel/async behavior override `run` and may ignore `step`.
- `BudgetExceeded` (`BaseException` subclass) raised out of any monitored
  tool call should propagate. Catching it is forbidden by convention.
- Done detection comes from `EnvironmentOutput.done` (returned by
  `task.step`) or from the agent's own logic — `MonitoredTool` does not
  raise on done.
- Agents must not capture `task` or `recorder` references that outlive
  `run()`. Conventionally they call only `task.step`, `task.toolbox.*`,
  and `task.evaluate`; not `reset` / `close` (Episode's). (An async
  `task.step` variant is out of scope for this PR — agent authors who
  need async LLM override `_arun` and use whatever they want; a
  follow-up RFC may add a first-class async step.)
- Agents do not read budget or trajectory state. Those concerns belong to
  monitored tools (which raise) and `Episode` (which finalizes).

### Updated invariants

1. `AgentConfig.make(action_set)` returns an `Agent` subclass. (unchanged)
2. Either `Agent.step` (sync turn) or `Agent.run` (async loop) must be
   implementable; agents that override `run` only must still provide a
   trivial `step` that raises `NotImplementedError` for clarity.
3. Every LLM call inside `step` or `run` should go through
   `LLM.call(prompt, tag)` (which auto-emits `LLMCallEvent` when a
   recorder is attached). Agents that hold the LLM but skip
   `attach_recorder` propagation deliberately opt those LLM calls out
   of the trajectory.

### Contracts for implementers

- For parallel tool calls in an overridden `_arun`:
  `await asyncio.gather(*(task.toolbox.async_execute_action(a) for a in actions))`.
  Each returns `Observation | StepError`. Monitoring happens inside each
  `MonitoredTool.execute_action`.
- For early termination from inside the agent: return from `run` (don't raise).
- For "the env said done" in gym-style: check
  `env_output.done` after each `task.step` and return.
- For streaming agents: use `recorder.begin_turn() as turn:` and call
  `turn.add_*` as data arrives. Avoid the coarse `record(output)` for
  streaming use cases — it loses the partial-emit advantage.

---

## ADDED — `openspec/specs/tool/spec.md`

The `openspec/specs/tool/` spec layer was deleted in commit `e760f9e5`
together with `ToolWithTelemetry`. This RFC re-creates the layer with a
single class: `MonitoredTool`.

### `MonitoredTool`

`MonitoredTool` is a single class with a **dual API**
(`execute_action` sync + `async_execute_action` async). It collapses
what an earlier draft split across `MonitoredTool(AbstractTool)` +
`AsyncMonitoredTool(AbstractAsyncTool)` (plus the `_SyncToolAsAsync`
shim). It is a transparent decorator — mixable in an `AsyncToolbox`
alongside unmonitored sync or async tools. Agents call it identically.

```python
class MonitoredTool:
    def __init__(
        self,
        inner: AbstractTool | AbstractAsyncTool,
        emit: Callable[[TrajectoryEvent], str],   # streamer.emit
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        task: Any | None = None,
    )

    @property
    def action_set(self) -> list[ActionSchema]
    # Delegates to inner.action_set — transparent.

    def execute_action(self, action: Action) -> Observation | StepError
    # Sync path; sync inner only. Raises TypeError if inner is async.
    # 1. If self.budget.exhausted: raise BudgetExceeded(action=action).
    # 2. Invoke inner.execute_action(action) directly.
    # 3. Append a ToolCallEvent (with the action and result) to trajectory;
    #    storage.save_event; summary.on_event; budget.tool_calls += 1.
    # 4. Return result (Observation | StepError) unchanged.

    async def async_execute_action(self, action: Action) -> Observation | StepError
    # Async path; handles both inner kinds.
    # 1. Budget check as above.
    # 2. Invoke inner: `await inner.async_execute_action(action)` for async
    #    inner, `await asyncio.to_thread(inner.execute_action, action)` for sync.
    # 3+4. Emit + return as above.

class BudgetExceeded(BaseException):
    action: Action                       # the call that pushed over budget
```

No new harness-side toolbox container is added. `cube.tool.AsyncToolbox`
was relaxed (cube-standard PR #152, merged to `dev` as `3e59fd14`) to
accept mixed sync + async leaves, so
`AsyncToolbox(members=[MonitoredTool(inner=t1), t2, ...])` covers all
cases. Dispatch by action name routes each call to the right member,
monitored or not.

### Contract

- `MonitoredTool.execute_action` returns `Observation | StepError` —
  **same** as any cube-standard `Tool`. It does NOT return
  `EnvironmentOutput` and does NOT detect `done`. Those remain
  `Task.step`'s responsibility, which calls into the toolbox and wraps the
  result.
- `MonitoredTool` is the only place where storage / summary hooks fire for
  tool execution. Subclasses of `cube.tool.Tool` / `AsyncTool` must NOT
  add storage calls.
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

### Gotchas

- `MonitoredTool` and its budget counter are per-episode — re-using a
  `MonitoredTool` across episodes is a bug.
- Per-tool-call OTel spans are NOT re-introduced. The previous
  `ToolWithTelemetry`-based emission path was removed by `e760f9e5` with
  no replacement consumer. The trajectory event stream (`ToolCallEvent`
  per call) is the structured per-call observability — strictly richer
  than a span. `cube_harness.metrics.tracer.tool_span(action)` remains
  available for any future wrapper that wants to export spans to an
  external collector.

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
    budget = Budget(max_agent_steps=self.max_steps, ...)   # Budget lives in cube_harness.budget; re-export shim in tool.py

    # Wrap each member of task.toolbox with MonitoredTool, sharing trajectory + budget.
    # task.toolbox is mutated in place so task.step also goes through monitored wrappers.
    install_monitoring(task, trajectory, budget, self.storage, self.summary)

    recorder = EventStreamer(trajectory, self.storage, self.summary)
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
        task.close()
    return trajectory
```

Episode- and benchmark-level OTel handling (today's `tracer.episode(...)`
span around the `try` block and `tracer.shutdown()` in the outer `finally`,
plus `tracer.benchmark(...)` higher up in `exp_runner`) are preserved from
today's `Episode.run` — elided from the pseudo-code above for clarity. This
RFC adds NO new OTel surface: no per-tool-call span, no per-turn span. The
trajectory event stream is the harness's structured observability.

The `trajectory` lives on `Episode`. The agent receives `task` and
`recorder` — the monitoring wrappers are already installed onto
`task.toolbox`, so any path through tools (gym-style `task.step` or
tool-level `task.toolbox.execute_action` /
`task.toolbox.async_execute_action`) emits monitoring uniformly.

`EventStreamer` exposes Episode-only helpers (`record_reset`,
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
   toolbox) and `recorder`. The agent uses `task.step`, `task.toolbox.*`,
   or `task.evaluate` for tool / step / eval calls — all routed through
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

`Budget` lives in `cube_harness.budget` (with a re-export shim in
`cube_harness.tool`). Its fields are `max_agent_steps`, `max_tool_calls`,
`max_cost_usd`, `max_wallclock_s`; corresponding counters are
`agent_steps` / `tool_calls` / ...; `bump_agent_step` is the bump hook.
`max_steps` field is accepted as a deprecated alias that maps to
`max_agent_steps`.

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

### Episode directory layout

```
episodes/<trajectory_id>/
├── episode.metadata.json   # TrajectoryMetadata: id, metadata, start_time,
│                           # end_time (None mid-run), summary_stats,
│                           # reward_info. Written at episode START, updated at END.
├── episode_config.json     # TaskConfig + AgentConfig + EpisodeConfig (input)
├── status.json             # running/completed/failed + retry_count
├── events/                 # one file per TrajectoryEvent
│   ├── 000_agent.msgpack.zst
│   ├── 001_tool_call.msgpack.zst
│   ├── 002_tool_call.msgpack.zst    # parallel sibling, same parent_event_id
│   ├── 003_eval.msgpack.zst
│   └── …
├── failure.txt             # optional: exception text on crash
└── logs/                   # per-episode stdout/stderr
```

V1 archived episodes use `trajectory.json` (the old metadata file name) and
`steps/*_obs.msgpack.zst` / `*_act.msgpack.zst`. Both are kept readable;
all new writes use the V2 layout above.

### New protocol methods

```python
class Storage(Protocol):
    # New write API:
    def save_metadata(self, meta: TrajectoryMetadata) -> None
    def save_event(self, event: TrajectoryEvent, trajectory_id: str, event_num: int) -> None
    def finalize_episode(self, meta: TrajectoryMetadata) -> None

    # New read API:
    def load_episode(self, trajectory_id: str) -> TrajectoryView
    def load_event(self, trajectory_id: str, event_num: int) -> TrajectoryEvent
    def list_episodes(self) -> list[TrajectoryMetadata]    # cheap study scan

    # ... existing methods (save_config, archive_episode, etc.) ...
```

Removed: `save_trajectory(trajectory: Trajectory, …)`, `load_trajectory(id) -> Trajectory`,
`finalize(trajectory: Trajectory)`. Their old in-tree callers move to the
new API in this PR.

### Write-at-start semantics

`save_metadata` is called by `Episode.run` **at episode start** with
`end_time=None` and stub summary fields. This makes crashed runs loadable
by XRay (the file exists; events that did land are renderable; status.json
disambiguates "in-flight" from "failed").

`finalize_episode` is called in `Episode.run`'s `finally` block with the
final `TrajectoryMetadata` (`end_time`, `summary_stats`, `reward_info` filled).
It overwrites the same `episode.metadata.json`.

### Lazy load via `TrajectoryView`

`load_episode(id)` is **cheap**: one JSON read for the metadata,
directory listing of `events/`, no event-payload I/O. The returned view
decodes events on demand via `view[i]` / iteration.

`load_episode` auto-detects layout at open time:

1. If `events/` exists + `episode.metadata.json` exists → standard V2 view.
2. If `events/` exists + only `status.json` exists (mid-run crash before
   metadata first-write) → stub metadata view, `is_complete == False`.
3. If `steps/` exists (V1 archive) → legacy-upgrade view; iteration
   synthesizes events on the fly: `_act` (AgentOutput) → `LLMCallEvent`
   (with `call=None`), `_obs` (EnvironmentOutput) → `ToolCallEvent`
   parented to the most recent LLM call event.

`list_episodes()` reads only `episode.metadata.json` per episode dir.
Used for study aggregation, EpisodeRecord generation, Atlas indexing.

### Invariants (additions)

- Event files are immutable after write.
- `episode.metadata.json` is written **at episode start**; the same file
  is overwritten **at episode end** with the final summary. No other
  callers write to it.
- `TrajectoryView` never materializes the full event list. Its internal
  cache is per-view (GC'd with the view).

### Summary

Per-episode summary stats now live INSIDE the streamer: counters
(`n_llm_calls`, `n_tool_calls`, `n_evaluations`, `total_actions`,
token/cost totals, first-seen `error_type`) are folded incrementally
as events flow through `EventStreamer.emit(te)`. `summary_stats(...)`
returns the final dict, persisted to `TrajectoryMetadata.summary_stats`
at finalize.

Notable simplifications:
- `SummaryProcessor` class deleted; no separate sink for stats.
- `episode_summary.jsonl` per-event log dropped — no production
  consumer was reading it. Final totals are on the metadata file.
- For back-compat with XRay's table columns, the summary dict
  surfaces `n_llm_calls` as both `n_agent_steps` and `total_llm_calls`,
  and `n_tool_calls` as `n_env_steps` (since the prior counts were
  per-LLM-call and per-env-step in the dropped batched model).

---

## MODIFIED — `openspec/specs/analyze/spec.md`

### Data layer: lazy TrajectoryView

XRay's data layer is rebuilt around `TrajectoryView`. When the user opens an
episode, the viewer calls `storage.load_episode(id)` (cheap: metadata
+ directory listing) and iterates once to build a lightweight
`event_index -> kind` table. Card rendering then accesses
`view[selected_index]` (decode on demand, cached in the view).

The `_events_to_legacy_steps` materialization shim from Phase I is
**deleted** — there is no remaining consumer after this PR.

### Event-card timeline

The viewer renders one card per `TrajectoryEvent`. Card colour by kind:

- `agent` — assistant turn (thoughts, response_text, LLM calls, intended actions)
- `tool_call` — one env interaction
- `eval` — final task.evaluate output

`ToolCallEvent`s sharing a `parent_event_id` render in horizontal lanes
within a turn group (parent `AgentEvent` above, siblings below).

### Selection model

- `selected_event_index` is the user's last click.
- `last_agent_event_index` = the largest index <= `selected_event_index`
  pointing at an `AgentEvent`.
- `last_observation_event_index` = the same rule for `ToolCallEvent`.
- Indices are derived from the cheap `event_index -> kind` table — no
  payload I/O.

### Tabs

- **Reasoning / Chat** — payload of `view[last_agent_event_index]`
  (thoughts, response_text, llm_calls, intended actions).
- **Observation** — if `selected_event` is a `tool_call`, render
  `view[selected_event_index].output`; else render
  `view[last_observation_event_index].output`. Screenshots, AXTree, and
  HTML are all surfaces inside the obs renderer; no separate screenshot tab.
- **Turn observations** — list of all `tool_call` events sharing
  `selected_event`'s `parent_event_id` (empty when no turn). Useful for
  parallel tool calls.
- **Profiling** — per-event `profiling` timing breakdown.
- Header always reads: `Event X / N — kind={agent|tool_call|eval}, parent=<id>, t=<s>s`.

### Crashed / in-flight episodes

If `view.is_complete == False` (no `end_time` in metadata), the timeline
renders what's on disk and shows a banner reading the `status.json`
failure summary. No special-case loader path; the same TrajectoryView API
is used.

### Invariants

- Read-only (unchanged).
- Loads both `events/` (V2 fresh + V2 crashed-mid-run) and `steps/`
  (legacy V1) layouts via `TrajectoryView`'s open-time detection.
- Stale background-loader generations still self-abort (unchanged).

### Removed

- The standalone screenshot tab. Screenshot rendering moves inside the
  Observation tab (screenshots are obs content, not a separate concern).
- "UI step" pairing logic (env+agent paired into a single navigation unit) is
  removed — navigation is per-event.
- Any code path that expected `trajectory.steps[i]` — replaced by
  `view[i]` (different element type: `TrajectoryEvent`).

`_events_to_legacy_steps` is **kept** as an on-read materialization shim
until XRay + the investigator finish migrating to `TrajectoryView.events`
directly (planned follow-up `agent-owns-loop-xray`). Marked deprecated
in code; removal moves to that PR.

### Gotchas

- Trajectories with very wide parallel tool calls (e.g. 20+ siblings in one
  turn) will overflow the horizontal lane layout. Out of scope for v1;
  acceptable to fall back to a vertical list above some threshold.
- `TrajectoryView`'s cache is bounded by the events the viewer accessed. If
  the user scrubs through every event on a SWE-bench-scale episode, the
  whole trajectory ends up in RAM for the lifetime of the view. Acceptable
  — switching episodes constructs a new view and GC's the old cache.

---

## MODIFIED — `openspec/specs/analyze/spec.md` (Investigator)

The investigator's per-trajectory blame pipeline walks
`view.iter_events()` instead of `traj.steps`. Each tool call's
`ToolCallEvent.output` carries the same observation + reward + info the
legacy `EnvironmentOutput` step did; each `AgentEvent` carries the same
intended actions + LLM calls + thoughts the legacy `AgentOutput` step
did. The diff is type-level, not behavior-level.

`InvestigatorContext.trajectory: Trajectory` → `InvestigatorContext.view:
TrajectoryView`. Per-step blame uses `view[i]`; cross-step pattern detection
uses `for event in view:`. No new code paths.

## REMOVED

The following are **deleted outright** (no deprecation alias):

- `class Trajectory` (from `cube_harness.core`) — replaced by
  `TrajectoryMetadata` + `TrajectoryView`.
- `class TrajectoryStep` (the `EnvironmentOutput | AgentOutput` union) —
  removed from the public API. The legacy V1 reader uses it internally
  but it is not exported.
- `Trajectory.streaming` flag — no longer needed; events are always
  streamed to disk by `MonitoredTool` / `EventStreamer` and never
  accumulated.
- `Trajectory.steps` field/alias — replaced by `TrajectoryView` iteration.
- `Trajectory.last_env_step`, `last_env_output`, `n_agent_steps`,
  `n_env_steps` (methods) — moved to `TrajectoryView` where they
  belong. Their old form on `Trajectory` is gone.
- `Trajectory.events_of_turn` / `TrajectoryView.events_of_turn` —
  dropped outright. With no separate `turn_id` field, callers query
  parallel siblings directly by `parent_event_id`:
  `[e for e in view if isinstance(e.output, ToolCallEvent) and e.output.parent_event_id == target_id]`.
- `ToolCallEvent.turn_id` field — dropped. Parallel siblings are
  identified by their shared `parent_event_id` (the originating
  `LLMCallEvent.id`).
- `_events_to_legacy_steps` is retained as a deprecated shim until the
  XRay rewrite lands; see deferred-removal note above.
- `Storage.save_trajectory` / `load_trajectory` / `finalize(trajectory)`
  — replaced by `save_metadata` / `load_episode` / `finalize_episode(meta)`.
- `EpisodeRecord.from_trajectory(traj)` — replaced by
  `from_view(view)`.
- "Trajectory steps alternate" invariant in `core/spec.md`.
- Standalone screenshot tab in XRay.
- Every defensive `if trajectory.events: ... else: ...` branch in
  storage and core — the dual-path code went away with the class.

(`ToolWithTelemetry` / `AsyncToolWithTelemetry` were already removed by
`e760f9e5` — listed here for historical context only, not by this RFC.)

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
- **Unit**: `EventStreamer.emit()` fold + sink fan-out is coherent under
  parallel dispatch (lock-guarded counters; see
  `tests/test_summary_concurrency.py`).
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

1. ~~**Budget granularity.**~~ **Resolved.** All caps (`max_turns`,
   `max_tool_calls`, `max_cost_usd`, `max_prompt_tokens`,
   `max_completion_tokens`, `max_wallclock_s`) ship enforced. `Budget.exhausted`
   checks every cap; `EventStreamer.on_llm_call` bumps cost + tokens from
   each `LLMCall.usage`; `EventStreamer.on_step` bumps `turns` once per
   agent step; `MonitoredTool` raises `BudgetExceeded` on tool-call
   ticks. `max_steps` was never an alias — `max_turns` is the only name.
2. **`Agent.step` deprecation timeline.** Decided: keep one release —
   `step` stays required as the canonical sync entry point. Agents
   that override `run` get to omit it; the abstract requirement softens
   in the follow-up release.
3. ~~**Async `step`**~~ **Resolved (out of scope).** No async-step
   method ships in this PR. Agent authors who need async LLM
   semantics override `_arun` and use whatever they want. A
   first-class async-step entry point is deferred to a future RFC.
