# RFC: Agent Owns the Loop — Event-Stream Trajectories & Monitored Tools

**Status:** DRAFT
**Author:** Alexandre Lacoste
**Reviewer:** TBD
**Date:** 2026-05-13

**Cross-repo:** also `cube-standard/openspec/changes/agent-owns-loop/` (small companion).

---

## Problem

cube-harness owns the agent loop today: `Episode._run_loop` alternates
`agent.step(obs)` and `task.step(actions)` and weaves heartbeat, storage,
summary, and tracing between the two. This shape has three costs.

1. **No parallel tool calls.** The loop assumes one batch of actions per turn,
   serialised through `task.step`. Modern agents (Claude Code, Codex, Genny) emit
   N concurrent tool calls per LLM response, and our agents can't follow.
   `LLMConfig.parallel_tool_calls=False` is the documented default.

2. **No async path for agents.** Tool execution, LLM calls, and trajectory I/O
   are sync. We pay sequential latency at every layer. AsyncToolbox exists in
   cube-standard but the harness loop is sync-only.

3. **Trajectory is too rigid.** `core/spec.md` invariant #1 says
   "trajectory steps alternate `EnvironmentOutput` and `AgentOutput`". That
   invariant blocks event-stream trajectories — where parallel tool calls,
   reasoning fragments, and final evaluation all need to be first-class events
   with their own timestamps.

The fix is structural, not incremental: hand the loop to the agent and turn
trajectories into typed event streams.

---

## Scope

### In

- New `Agent.run(initial_obs, task, recorder) async` with a default
  implementation that drives the existing `step()`-based loop. Sync agents
  keep working. The agent receives the live `task` (well-defined
  cube-standard interface) and a telemetry sink; `task.toolbox` is already
  wired with monitored wrappers by `Episode`.
- New `MonitoredTool` / `MonitoredToolbox` wrappers in cube-harness that
  **subclass `cube.tool.Tool` / `AsyncTool` with the same `execute_action`
  signature** — drop-in replacements, mixable in a `Toolbox` alongside
  unmonitored tools. Agents call `execute_action(action) → Observation | StepError`
  without knowing or caring which tools are monitored. The wrappers emit
  trajectory events and OTel spans on every call. They replace
  `ToolWithTelemetry`. Budget enforcement lives in `MonitoredTool` (raises
  `BudgetExceeded`).
- New trajectory event model: `AgentEvent`, `ToolCallEvent`, `EvaluationEvent`,
  replacing the binary `EnvironmentOutput | AgentOutput` union. Alternation
  invariant removed.
- `TurnRecorder` is the agent's outbound telemetry sink. The agent calls
  methods on it (LLM calls, thoughts, response text, profiling, agent
  errors) — the toolbox can't observe these. `TurnRecorder` exposes both a
  coarse API (`record(agent_output)`, one call per turn) and a granular API
  (`begin_turn() / add_*`) for streaming agents. Replaces the original
  `LoopContext`.
- Defensive episode finalization: `Episode` wraps `agent.run` in a
  `try/except BaseException`, then runs `task.evaluate()`, persists final
  trajectory, and updates the experiment summary — regardless of how the agent
  returned. Cross-turn state (trajectory, storage, summary, tracer) is owned
  by `Episode`; agents never see it directly.
- `BudgetExceeded(BaseException)` propagates out of `MonitoredTool.execute_action`
  to terminate runaway loops. Subclassing `BaseException` (not `Exception`)
  sidesteps `except Exception:` swallowing. **Done detection is unchanged from
  today** — it comes from `EnvironmentOutput.done` returned by `task.step`
  (cube-standard's existing gym contract); the agent inspects it or returns
  naturally. `MonitoredTool` does not raise on done.
- One reference agent that overrides `run()` with parallel tool calls
  (`agents/parallel_tool_agent.py` or evolution of Genny).
- XRay rewrite: render each event as a card coloured by event kind, with tabs
  that follow the selected event (see Design).
- Integration test: same task, gym-style agent (default `run`) and
  parallel-tool agent (overridden `run`) both produce valid trajectories that
  load in XRay and pass the verifier.
- Smoke test: replay an existing experiment dir through the new viewer.

### Out (Phase 2)

- External agents over JSON-RPC + `MonitoredToolbox` (the cube-standard
  `cube.server` JSON-RPC layer already exists; per-session monitoring
  attaches in a follow-up).
- `cube_harness/mcp/server.py` migration — current FastMCP wrapper is
  duplicative with `cube.server`; consolidation is its own change.
- WebSocket / streaming transport — covered by the
  `cube-standard/openspec/changes/json-rpc-streaming` change.
- **Pi-style primitive-toolbox support.** Pi (the
  [pi-mono](https://lucumr.pocoo.org/2026/1/31/pi/) agent by Armin Ronacher)
  uses 4 generic tools — `read`, `write`, `edit`, `bash` — and rejects MCP-style
  pre-declared tools. CUBE should support both styles long-term: rich
  per-task action sets (today, MCP-compatible via `cube.server`) and a
  Pi-style primitive toolbox available on shell-accessible cubes. Phase 1
  declares only the protocol seam (see cube-standard companion: new
  optional `Task.primitive_toolbox()` method). Phase 2 ships the concrete
  `cube-shell-tools` package, a `PiStyleAgent` reference that uses it, and
  a `PiCliAgent` that spawns the real Pi CLI as a subprocess inside the
  cube's sandbox. The agent-owns-loop design (`Agent.run` + `MonitoredToolbox`)
  already supports both shapes uniformly — Phase 2 is mostly packaging.

---

## Design

### `Agent.run`

```python
class Agent(ABC):
    def step(self, obs: Observation) -> AgentOutput: ...   # unchanged
    async def run(
        self,
        initial_obs: Observation,
        task: Task,                        # cube-standard Task; task.toolbox is monitored
        recorder: TurnRecorder,
    ) -> None:
        """Default impl reproduces today's gym-style loop.

        Termination: BudgetExceeded raised by monitored tools propagates;
        EnvironmentOutput.done from task.step terminates naturally; agent
        can also return.
        """
        obs = initial_obs
        while True:
            agent_output = await asyncio.to_thread(self.step, obs)
            recorder.record(agent_output)
            if not agent_output.actions and not agent_output.error:
                return  # graceful done
            env_output = await task.astep(agent_output.actions)  # may raise BudgetExceeded
            if env_output.done:
                return
            obs = env_output.obs
```

Agents that want parallel tool calls override `run()` and use
`task.toolbox` directly:
```python
results = await asyncio.gather(*(
    task.toolbox.execute_action(a) for a in actions
))  # each is Observation | StepError; monitoring fires inside each call
```

Agents that don't override get backwards-compatible gym behaviour for free.

The three parameters:
- **`initial_obs`** — the observation from `task.reset()`, supplied by `Episode`.
- **`task`** — the live `cube.task.Task`. `task.toolbox` has been wrapped
  with `MonitoredTool`s by `Episode` before `agent.run` is called, so any
  path through tools (whether `task.astep(...)` or
  `task.toolbox.execute_action(...)`) emits monitoring. Agents are
  conventionally expected to call `astep` / tools / `aevaluate` only; not
  `reset` / `close` (those are `Episode`'s).
- **`recorder`** — what the agent reports out. Telemetry-only. Agents
  emit LLM calls, thoughts, response text, profiling.

### `TurnRecorder`

```python
class TurnRecorder:
    """Agent's outbound telemetry sink. Constructed by Episode, scoped
    to one episode. Two complementary APIs."""

    # --- Coarse API: one call per LLM cycle, all-at-once. ---
    def record(self, output: AgentOutput) -> None:
        """Emit one AgentEvent built from a complete AgentOutput.
        The default Agent.run uses this — matches today's step-style."""

    # --- Granular API: emit data as it arrives (streaming agents). ---
    def begin_turn(self) -> "Turn":
        """Returns a context manager. Use when you want to add events
        incrementally during a turn (partial LLM responses, mid-turn
        profiling, etc.)."""

class Turn:  # __enter__ / __exit__
    def add_llm_call(self, call: LLMCall) -> None: ...
    def add_thought(self, text: str) -> None: ...
    def add_response_text(self, text: str) -> None: ...
    def add_profile(self, label: str, start: float, end: float) -> None: ...
    def add_error(self, err: StepError) -> None: ...
    # __exit__ flushes the accumulated fields as one AgentEvent.
```

Why two surfaces:

- **Coarse `record(output)`** is what today's `Agent.step`-style code wants.
  Structurally enforces "every turn emits a complete `AgentEvent`" — hard to
  forget fields. The simple agent path stays simple.
- **Granular `begin_turn() / add_*`** is what streaming agents (Pi-style,
  Claude Code, Codex) need. LLM responses arrive in chunks; profiling spans
  open and close at different points; the agent emits as data lands.

Internally, `record(output)` is a thin wrapper around `begin_turn()` — one
implementation, two surfaces. No double-maintenance.

Cross-turn state (trajectory, storage, summary, tracer) lives on `Episode`
and is bound into the `TurnRecorder` at construction. Agents never read or
write that state directly.

### `MonitoredTool` / `MonitoredToolbox`

`MonitoredTool` subclasses `cube.tool.AsyncTool` and exposes exactly the
same `execute_action` signature. It is a transparent decorator: agents and
`task.step` call it identically to any other tool.

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
    ): ...

    @property
    def action_set(self) -> list[ActionSchema]:
        return self.inner.action_set                          # transparent

    async def execute_action(self, action: Action) -> Observation | StepError:
        if self.budget.exhausted:
            raise BudgetExceeded(action=action)
        with self.tracer.tool_span(action):
            # Await directly when inner is AsyncTool;
            # wrap with asyncio.to_thread when inner is sync Tool.
            result = await self._invoke_inner(action)
        self._record_tool_call_event(action, result)          # storage + summary + trajectory
        self.budget.tool_calls += 1
        return result                                          # unchanged
```

`MonitoredToolbox` is just a `Toolbox` whose member tools are each wrapped
in `MonitoredTool`. Since `Toolbox` is-a `Tool`, the wrapper is recursive:
a toolbox may contain monitored tools, unmonitored tools, and nested
toolboxes side-by-side. The agent calls `toolbox.execute_action(action)`;
dispatch by action name routes to the right member, monitored or not.

Three things to note about this shape:

1. **`MonitoredTool` returns `Observation | StepError`** — same as any
   `Tool`. It does not return `EnvironmentOutput` and does not detect
   `done`. Those remain the responsibility of `Task.step` (cube-standard's
   gym wrapper), which calls into the toolbox and constructs the
   `EnvironmentOutput` from `obs + finished(obs) + evaluate(obs) + …`.
2. **Done detection is unchanged.** `Task.step` returns
   `EnvironmentOutput.done = True` when `self.finished(obs)` is True. The
   default `Agent.run` inspects that and terminates. Tool-level agents that
   bypass `task.astep` need their own done logic — typically a "submit"
   tool whose obs triggers `task.finished()` for the next gym caller, or
   the agent returning when its own success criterion is met.
3. **Step-wise evaluation is already handled by cube-standard.** `Task.step()`
   invokes `self.evaluate(obs)` internally when `Task.validate_per_step` is
   `True`, and the resulting `reward` / `info` flow back through
   `EnvironmentOutput.reward` / `info`. `MonitoredTool` doesn't need its
   own step-eval path. The terminal `task.evaluate()` call in
   `Episode.run`'s `finally` still happens unconditionally, exactly once,
   and is recorded as an `EvaluationEvent`.

### Responsibility map (vs. today's loop)

Every concern in today's `Episode._run_loop` has a clear new home.

| Concern in today's loop | New owner | How |
|---|---|---|
| Tool dispatch (`tool.execute_action`) | `MonitoredTool` | Wraps inner Tool; same `execute_action` signature. |
| Per-tool-call OTel span | `MonitoredTool` | One span per `execute_action` call. |
| `ToolCallEvent` persistence (env-step save) | `MonitoredTool` | Records to trajectory + storage on every call. |
| Per-call summary update (env-step counter, reward) | `MonitoredTool` | `summary.on_event(tool_call_event)`. |
| Budget enforcement (`max_steps`, etc.) | `MonitoredTool` | Counts calls; raises `BudgetExceeded(BaseException)`. |
| Heartbeat / `status.json` | `MonitoredTool` | Updates `last_heartbeat_at` per call. Cheap. |
| Tool-side error handling (`StepError` from inner) | `MonitoredTool` | Records the `StepError` into the `ToolCallEvent`, returns it (does not raise). |
| `AgentEvent` persistence (agent-step save) | `TurnRecorder` | `record()` or `Turn.__exit__` flushes one AgentEvent. |
| Per-turn summary update (LLM calls, tokens, cost) | `TurnRecorder` | Updates `summary` from `AgentEvent.llm_calls`. |
| Agent output logging | `TurnRecorder` | Logs alongside the event flush. |
| Per-turn OTel span (`tracer.step("turn_N")`) | `TurnRecorder` | Spans an `AgentEvent` lifetime via `begin_turn()`. |
| Agent-side error capture (`agent.step` raised) | `TurnRecorder.record_failure` | Episode's `except` calls it after `agent.run` raises. |
| `done` detection | `Task.step` (cube-standard, unchanged) | Returns `EnvironmentOutput.done`. Default agent reads it. |
| Per-step `reward` | `Task.step` (cube-standard, unchanged) | Comes through `EnvironmentOutput.reward`. |
| Step-wise `evaluate` (`validate_per_step`) | `Task.step` (cube-standard, unchanged) | Built into `task.step` at [task.py:346](../../../src/cube/task.py#L346). |
| `obs_postprocess` | `Task.step` (cube-standard, unchanged) | |
| Action validation (empty → break) | Default `Agent.run` | Convention; not enforced. |
| `task.reset` | `Episode` (before `agent.run`) | Initial obs handed to the agent. |
| Terminal `task.evaluate` | `Episode` (in `finally`) | `recorder.record_evaluation(reward, info)`. |
| `task.close` | `Episode` (in `finally`) | |
| Storage finalize | `Episode` (in `finally`) | |
| Episode-level OTel span | `Episode` | Wraps the whole `try / except / finally`. |

### Defensive `Episode.run`

```python
async def run(self) -> Trajectory:
    task = self.task_config.make(...)
    trajectory = Trajectory(id=..., events=[])
    budget = Budget(max_turns=self.max_steps, ...)

    # Wrap each member of task.toolbox with MonitoredTool, sharing trajectory + budget.
    # task.toolbox is mutated in place so task.step also goes through monitored wrappers.
    install_monitoring(task, trajectory, budget,
                       self.storage, self.summary, self.tracer)

    recorder = TurnRecorder(trajectory, self.storage, self.summary, self.tracer)
    try:
        initial = task.reset()
        recorder.record_reset(initial)            # Episode-only helper on recorder
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

The agent cannot prevent finalization. `trajectory` and `recorder` are
owned by `Episode`; the agent receives `task` and `recorder`. The monitoring
wrappers are installed onto `task.toolbox` once, so any tool invocation —
whether via `task.astep(actions)` or direct `task.toolbox.execute_action(action)`
— routes through monitoring. `record_reset` / `record_failure` /
`record_evaluation` are Episode-only helpers on `TurnRecorder` (not actively
hidden from agents, but conventionally Episode's).

Note: `task.evaluate()` is called with no obs in `finally`. Tasks that need
the final obs to evaluate must track it internally (cube-standard's `Task`
already does for the gym path). This avoids the harness having to chase the
"last good obs" through the agent's loop logic.

### Event-stream trajectory

```python
class AgentEvent(TypedBaseModel):
    id: str                            # for ToolCallEvent.agent_event_id back-reference
    actions: list[Action]              # intended tool calls (each has Action.id)
    llm_calls: list[LLMCall]
    thoughts: str | None
    response_text: str | None          # assistant's prose alongside the tool calls
    profiling: dict[str, tuple[float, float]]
    error: StepError | None

class ToolCallEvent(TypedBaseModel):
    agent_event_id: str                # parent AgentEvent.id
    action_id: str                     # references one of agent_event.actions[i].id
    output: EnvironmentOutput          # obs / reward / done / info / error
    turn_id: str                       # groups sibling parallel calls

class EvaluationEvent(TypedBaseModel):
    reward: float
    info: dict

class TrajectoryEvent(TypedBaseModel):
    output: AgentEvent | ToolCallEvent | EvaluationEvent
    start_time: float
    end_time: float

class Trajectory(TypedBaseModel):
    id: str
    events: list[TrajectoryEvent]      # was: steps
    ...
```

Why this shape:

- **`AgentEvent` carries both the actions list and the assistant's response
  text**, so XRay can display what the agent "said" alongside what it fired —
  even when several tool calls land in parallel.
- **`ToolCallEvent.action_id` references back to the parent
  `AgentEvent.actions[i].id`**, so the event stream is a flat list but the
  parent-child structure is recoverable.
- **`turn_id` groups parallel calls** so XRay can render them as siblings of
  one turn.
- **`EvaluationEvent`** makes the final `task.evaluate()` call a first-class
  event with its own timestamp, instead of being smuggled into `Trajectory.reward_info`.

Storage filenames evolve from `000_obs.msgpack.zst` / `001_act.msgpack.zst` to
`000_agent.msgpack.zst` / `001_tool_call.msgpack.zst` / `002_eval.msgpack.zst`.
Old V2 layouts remain loadable via a migration shim.

### XRay rewrite

- Timeline: one card per `TrajectoryEvent`. Colours by kind: `agent` (LLM /
  thoughts / response text), `tool_call` (one per action), `evaluation`
  (final). Parallel `tool_call` siblings share a `turn_id` and render in
  horizontal lanes within a turn group.
- Selection: clicking a card sets `selected_event`. The viewer derives
  `last_agent_event = max(e for e in events if e.kind == 'agent' and e.start_time <= selected.start_time)`
  and `last_observation_event` similarly.
- Tabs:
  - **Reasoning / Chat** — renders `last_agent_event` (thoughts, response_text,
    intended actions, LLM messages).
  - **Observation** — renders the obs from `selected_event` if it's a
    `tool_call`, else `last_observation_event`. Screenshots are content
    inside the observation, not a separate tab.
  - **Turn observations** — all `tool_call` events sharing the selected
    event's `turn_id`. Empty for non-tool events.
  - **Profiling** — per-event timing breakdown (already supported via
    `AgentEvent.profiling`).
  - Header strip always shows: `Event X / N — kind, turn=…, t=…s`.
- Drop the standalone screenshot tab; it's redundant with Observation.

### RPC layer

cube-standard already has the canonical RPC surface: `cube.server` exposes
`tools/list`, `tools/call`, `cube/step`, etc. as JSON-RPC 2.0 (MCP-compatible).
The Phase 1 PR does **not** change `cube.server`. The companion cube-standard
change (`cube-standard/openspec/changes/agent-owns-loop/`) only clarifies that
`MonitoredToolbox` lives in the harness (it captures harness-side trajectory
state) and that future external-agent connectivity will use the existing
`cube.server` endpoint with a per-session monitoring context attached on the
harness side. The harness's duplicate `cube_harness/mcp/server.py` is left
alone in Phase 1 and slated for retirement in a follow-up.

### Async-first

The new `Agent.run` is `async def`. Sync `step()` is wrapped in
`asyncio.to_thread` by the default `run()`. `MonitoredToolbox` is built on
`AsyncToolbox` (cube-standard already ships it). LLM calls become awaitable
through `cube_harness.llm` — out of scope for this RFC if LiteLLM async is
already available (likely yes, follow-up if not).

---

## Migration

- Old `Trajectory` / `TrajectoryStep` types remain importable as deprecated
  aliases for one release. Loaders convert old `EnvironmentOutput`-as-step
  records into synthetic `ToolCallEvent`s and `AgentOutput`-as-step into
  `AgentEvent`s so existing experiment dirs render in the new XRay.
- Existing agents (`ReactAgent`, `Genny`) keep their `step()` and run via the
  default `Agent.run` — no agent-side changes required for backwards compat.
- The new reference agent that exercises parallel tool calls ships in the
  same PR as a guarantee that the new path actually works.

---

## Risks

- **XRay rewrite is the largest single piece.** Mitigated by shipping the
  migration shim so old data still renders, and by an integration test that
  loads a known trajectory and asserts the expected tabs render content.
- **Behaviour drift in default `Agent.run` vs. today's loop.** Mitigated by
  routing today's loop through the same hook helpers — diff is structural,
  not behavioural.
- **Subclass agents that override `step()` but expect a specific call
  cadence.** No such agents exist in-tree; out-of-tree agents may need a
  trivial update. Documented in the agent spec.
- **Storage format change** — addressed by reading both formats and writing
  only the new one. One release window for tooling to catch up.

---

## Spec changes

See `deltas.md` (cube-harness) and `cube-standard/openspec/changes/agent-owns-loop/deltas.md`.
