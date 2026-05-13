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

- New `Agent.run(initial_obs, toolbox, ctx) async` with a default implementation
  that drives the existing `step()`-based loop. Sync agents keep working.
- New `MonitoredTool` / `MonitoredToolbox` wrappers in cube-harness that emit
  trajectory events (and OTel spans) on every call. They replace
  `ToolWithTelemetry` for in-process runs.
- New trajectory event model: `AgentEvent`, `ToolCallEvent`, `EvaluationEvent`,
  replacing the binary `EnvironmentOutput | AgentOutput` union. Alternation
  invariant removed.
- `LoopContext` (`ctx`) carries the storage, summary, tracer, trajectory,
  budget, last env output, and an `is_done` sticky flag. Both the default
  `Agent.run` and `MonitoredTool` mutate it through the same hook helpers.
- Defensive episode finalization: `Episode` wraps `agent.run` in a
  `try/except BaseException`, then runs `task.evaluate()`, persists final
  trajectory, and updates the experiment summary — regardless of how the agent
  returned.
- `EpisodeDone(BaseException)` and `BudgetExceeded(BaseException)` propagate
  out of `MonitoredTool.__call__` to terminate misbehaving loops. Subclassing
  `BaseException` (not `Exception`) sidesteps `except Exception:` swallowing.
- Sticky-done: every `MonitoredTool.__call__` raises immediately if
  `ctx.is_done` is set, so even an agent that catches once is stopped on its
  next call.
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

---

## Design

### `Agent.run`

```python
class Agent(ABC):
    def step(self, obs: Observation) -> AgentOutput: ...   # unchanged
    async def run(
        self,
        initial_obs: Observation,
        toolbox: MonitoredToolbox,
        ctx: LoopContext,
    ) -> None:
        """Default impl reproduces today's gym-style loop using ctx hooks."""
        obs = initial_obs
        while not ctx.is_done and ctx.turn < ctx.budget.max_turns:
            agent_output = self.step(obs)
            ctx.record_agent_event(agent_output)
            if not agent_output.actions and not agent_output.error:
                return  # graceful done
            for action in agent_output.actions:
                env_output = await toolbox(action)
                obs = env_output.obs
```

Agents that want parallel tool calls override `run()` and call
`asyncio.gather(toolbox(a) for a in actions)`. Agents that don't override get
backwards-compatible behaviour for free.

### `LoopContext`

```python
class LoopContext:
    task: Task
    storage: Storage
    summary: SummaryProcessor
    tracer: Tracer
    trajectory: Trajectory
    budget: Budget                 # max_turns, max_tool_calls, max_cost, etc.
    is_done: bool                  # sticky — set by env or by budget exhaustion
    last_env_output: EnvironmentOutput | None
    turn: int                      # incremented per agent event

    def record_agent_event(self, output: AgentOutput) -> None: ...
    def record_tool_call(self, action: Action, env_output: EnvironmentOutput) -> None: ...
    def record_evaluation(self, reward: float, info: dict) -> None: ...
    def record_failure(self, exc: BaseException) -> None: ...
```

These five `record_*` methods are the single point where storage, summary,
tracer, and trajectory state are updated. Both `MonitoredTool` and the default
`Agent.run` go through them. No more inlined `storage.save_step(...)` calls
scattered through the loop body.

### `MonitoredTool` / `MonitoredToolbox`

```python
class MonitoredTool(AsyncTool):
    def __init__(self, inner: AsyncTool | Tool, ctx: LoopContext): ...

    async def __call__(self, action: Action) -> EnvironmentOutput:
        if self.ctx.is_done:
            raise EpisodeDone(self.ctx.last_env_output)
        if self.ctx.budget.exhausted:
            self.ctx.is_done = True
            raise BudgetExceeded(self.ctx.last_env_output)
        env_output = await self.inner.aexecute_action(action)
        self.ctx.record_tool_call(action, env_output)
        if env_output.done:
            self.ctx.is_done = True
            raise EpisodeDone(env_output)
        return env_output
```

`MonitoredToolbox` is the same wrapper applied to every tool inside a
`Toolbox`, with one shared `ctx`.

### Defensive `Episode.run`

```python
async def run(self) -> Trajectory:
    task = self.task_config.make(...)
    ctx = LoopContext(task, self.storage, self.summary, self.tracer, ..., self.budget)
    toolbox = MonitoredToolbox(task.toolbox, ctx)
    try:
        initial = task.reset()
        ctx.record_tool_call(synthetic_reset_action, initial)
        await self.agent.run(initial.obs, toolbox, ctx)
    except EpisodeDone:
        pass
    except BudgetExceeded:
        ctx.record_failure(BudgetExceeded(...))
    except BaseException as e:
        ctx.record_failure(e)
    finally:
        final_obs = ctx.last_env_output.obs if ctx.last_env_output else None
        reward, info = task.evaluate(final_obs)
        ctx.record_evaluation(reward, info)
        await self.storage.finalize(ctx.trajectory)
        self.summary.on_episode_complete(ctx.trajectory, self.storage)
        task.close()
    return ctx.trajectory
```

The agent cannot prevent finalization. State that needs to survive is in `ctx`,
which `Episode` owns.

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
