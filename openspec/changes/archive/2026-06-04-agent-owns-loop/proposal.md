# RFC: Agent Owns the Loop — Event-Stream Trajectories, Monitored Tools, Lazy Episodes

**Status:** DRAFT
**Author:** Alexandre Lacoste
**Reviewer:** TBD
**Date:** 2026-05-13 (revised 2026-06-01)

**Cross-repo:** also `cube-standard/openspec/changes/agent-owns-loop/` (small companion — no API changes there).

## Scope expansion (2026-06-01)

The original RFC restructured the loop around `Agent.run` + `MonitoredTool` + event-stream
trajectories. Phases A–L of that work shipped on this branch. While running the
reference baseline (TerminalBench-2, gpt-5.4-mini, 29.3% — parity) we discovered
two structural debts. Phases M–T of this same PR pay them down; the XRay
event-card UI rewrite splits to a focused follow-up PR (see "Scope split"
below).

1. **`Trajectory` is no longer a useful in-memory abstraction during runs.**
   Under streaming it carries empty `events: []` lists; the actual data lives
   in `events/*.msgpack.zst` on disk. Every defensive `if events: ... else: ...`
   branch in core/storage was paying upkeep for a vestigial container. Replace
   on the production write-path with `TrajectoryMetadata` (pure metadata) +
   `TrajectoryView` (lazy reader) — modeled on AgentLab's loader pattern.

2. **XRay was not actually rewritten.** Phase I shipped a `_events_to_legacy_steps`
   materialization shim so the *legacy* UI rendered new-format trajectories.
   Parallel `ToolCallEvent` siblings collapsed into a flat sequence. The
   promised event-card timeline with horizontal lanes is owed by a follow-up
   PR (`agent-owns-loop-xray`).

### Scope split

This PR (`agent-owns-loop` final):
- Add `TrajectoryMetadata` + `TrajectoryView` (lazy reader).
- Storage write-at-start; `load_episode` is the canonical lazy entry.
- Production runtime (`Episode`, `MonitoredTool`, `EventStreamer`,
  `EpisodeRecord`) writes/reads through the new API. No in-memory event
  accumulation. Crashed-mid-run episodes loadable.
- Trajectory slimmed: keeps `id`, `metadata`, `steps` (materialized from
  events on load), `summary_stats`, `reward_info`, `start_time`, `end_time`.
  Drops `events` field, `streaming` flag, event helpers
  (`last_env_step`/`events_of_turn`/`n_agent_events`/…), and all dual-path
  code in core + storage.
- `storage.load_trajectory(id)` becomes a thin wrapper over `load_episode(id)`
  that materializes legacy steps via `_events_to_legacy_steps`. Lets XRay
  + investigator + inspect_results keep their current consumers unchanged.

Follow-up PR (`agent-owns-loop-xray`):
- XRay rewrite around `TrajectoryView` (event-card timeline, parallel sibling
  lanes, drop the legacy materialization shim).
- Investigator + inspect_results migration to `view.iter_events()`.
- Delete the legacy `Trajectory` class entirely.

Why split: XRay is ~3.5k lines + ~188 tests that hand-build `Trajectory`.
Bundling that with the runtime refactor would push this PR past safe-review
size. The runtime cleanup IS reviewable today; the UI rewrite deserves its
own focused review.

Net code reduction in THIS PR: ~150 LOC of `streaming` / `events` / dual-path
branches in core + storage + summary + tool + recorder. Net interface gain:
crashed-mid-run loadability; uniform `TrajectoryView` reader for new consumers.

The cube-standard companion needs no further changes — Trajectory was never
in cube-standard.

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

- New `Agent.run(initial_obs, env_tool) async` with a default
  implementation that drives the existing `step()`-based loop. Sync agents
  keep working. The agent receives the env-facing tool (`task.tool` /
  `task.toolbox`, wrapped by `MonitoredTool` and `as_async()` at the
  Episode boundary). The recorder is **not** a parameter — Episode
  calls `agent.attach_recorder(recorder)` before `run()`. Recording
  happens automatically: LLM calls auto-emit `LLMCallEvent`
  (`LLM.attach_recorder` + `LLM.call(prompt, tag)`), tool dispatches
  auto-emit `ToolCallEvent` (MonitoredTool). The agent never touches
  the recorder; `self._recorder.budget` is available for introspection.
- New `MonitoredTool` wrapper in cube-harness that **subclasses
  `cube.tool.AbstractTool` with a dual `execute_action` / `async_execute_action`
  API** — a single class wraps either a sync `AbstractTool` or an async
  `AbstractAsyncTool` inner. Drop-in mixable in `cube.tool.AsyncToolbox`
  alongside unmonitored tools (cube-standard's companion PR relaxed
  `AsyncToolbox` to accept mixed sync + async leaves, so no harness-side
  toolbox wrapper is needed). Agents call
  `execute_action(action) → Observation | StepError` (sync inner only) or
  `await tool.async_execute_action(action)` (both inner kinds) without
  knowing or caring which tools are monitored. The wrapper emits
  trajectory events on every call. The previous `ToolWithTelemetry` shim
  was already deleted in commit `e760f9e5` together with the
  `openspec/specs/tool/` spec layer; this RFC re-creates the layer around
  `MonitoredTool`. Budget enforcement lives in `MonitoredTool` (raises
  `BudgetExceeded`); the `Budget` class itself moved to
  `cube_harness/budget.py`.
- New trajectory event model: `LLMCallEvent` (one event per LLM API call),
  `ToolCallEvent` (one per tool dispatch, carries the full `Action`),
  `EvaluationEvent` (step-wise or terminal), `AgentErrorEvent`
  (Episode-level failure). Replaces the binary `EnvironmentOutput | AgentOutput`
  union AND the prior batched `AgentEvent`. Alternation invariant removed.
- `EventStreamer` is the trajectory's event sink — no longer an
  agent-facing API. Built by Episode; producers (LLM, MonitoredTool)
  emit through their `attach_recorder` hooks. The recorder forwards
  to storage + summary; future sinks (OTel, RL HTTP) plug in via
  `EventStreamerConfig` on `EpisodeConfig`. The legacy
  `record(agent_output)` / `begin_turn()` / `Turn` / `add_*` surface
  was dropped in favor of producer auto-emit (cleaner UX, streaming-
  friendly, no batched flush boundaries).
- Defensive episode finalization: `Episode` wraps `agent.run` in a
  `try/except BaseException`, then runs `task.evaluate()`, persists final
  trajectory, and updates the experiment summary — regardless of how the agent
  returned. Cross-turn state (trajectory, storage, summary) is owned
  by `Episode`; agents never see it directly.
- `BudgetExceeded(BaseException)` propagates out of `MonitoredTool.execute_action`
  to terminate runaway loops. Subclassing `BaseException` (not `Exception`)
  sidesteps `except Exception:` swallowing. **Done detection is unchanged from
  today** — it comes from `EnvironmentOutput.done` returned by `task.step`
  (cube-standard's existing gym contract); the agent inspects it or returns
  naturally. `MonitoredTool` does not raise on done.
- One reference agent that overrides `run()` with parallel tool calls
  (`agents/parallel_tool_agent.py` or evolution of Genny).
- `Trajectory` class removed. Replaced by `TrajectoryMetadata` (pydantic record
  persisted at `episode.metadata.json`) + `TrajectoryView` (lazy reader bound to
  storage + episode id). `Episode.run` returns `TrajectoryView`. The
  per-run object never holds the event list in memory.
- `episode.metadata.json` is now written at episode START with stub fields
  (`end_time=None`), then updated at END with the final summary. Crashed
  runs are loadable: `TrajectoryView` interprets `end_time=None` plus
  `status.json` as "in-flight or failed" and renders what's on disk.
- `EpisodeRecord.from_view(view)` replaces `from_trajectory(traj)` —
  reads only `view.metadata`, no event I/O. Study-aggregation stays O(1)
  per episode.
- XRay rewrite: render each event as a card coloured by event kind, with tabs
  that follow the selected event (see Design). Consume the lazy
  `TrajectoryView` — random-access events decode on demand. Drop the
  `_events_to_legacy_steps` shim entirely.
- Investigator migrated to `view.iter_events()`. The materialization shim
  has no remaining consumer.
- Integration test: same task, gym-style agent (default `run`) and
  parallel-tool agent (overridden `run`) both produce valid trajectories that
  load in XRay and pass the verifier.
- Smoke test: replay an existing experiment dir through the new viewer.

### Out (Phase 2)

- External agents over JSON-RPC with per-session `MonitoredTool` wrappers
  (the cube-standard `cube.server` JSON-RPC layer already exists;
  per-session monitoring attaches in a follow-up).
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
  cube's sandbox.
- **Connectors for existing agent frameworks.** CUBE should evaluate agents
  written against major frameworks without forcing reimplementation. Phase 1
  adds two small seams (`EventStreamer.record_external_run`, doc note on
  `cube.server` as the canonical MCP endpoint for CLI-agent connectors).
  Phase 2 ships reference connector packages (LangGraph, Pydantic AI,
  OpenAI Agents SDK, Inspect AI, Claude Agent SDK, A2A, Codex CLI, Goose,
  Pi CLI). See *Connector taxonomy* below.

---

## Design

### `Agent.run`

```python
class Agent(ABC):
    def step(self, obs: Observation) -> AgentOutput: ...   # unchanged

    def attach_recorder(self, recorder: EventStreamer) -> None:
        """Stash on self; subclasses override to propagate to held LLMs.
        Episode calls this before `run()`, so `self._recorder` is set
        for the duration of the loop (used by Genny.step to read
        `self._recorder.budget` for graceful self-stop / prompt injection)."""
        self._recorder = recorder

    async def run(
        self,
        initial_obs: Observation,
        env_tool: AbstractAsyncTool,       # always async; sync tools wrapped at the Episode boundary
    ) -> None:
        """Default impl drives a one-action-per-call loop on top of self.step.

        From the agent's POV the only environment surface is
        `await env_tool.execute_action(action) -> Observation | StepError`.
        Episode applies `as_async(task.tool)` before invoking this so
        `env_tool` is always `AbstractAsyncTool`-shaped — sync underlying
        tools dispatch via `asyncio.to_thread` inside the wrapper. Agent
        code has no sync/async branch.

        The recorder is NOT a parameter. Episode calls
        `agent.attach_recorder(recorder)` BEFORE `run()`. Recording
        happens automatically: LLM calls auto-emit `LLMCallEvent`
        (via `LLM.attach_recorder`), tool dispatches auto-emit
        `ToolCallEvent` (via `MonitoredTool`). The agent never touches
        the recorder directly.

        Termination:
          - `self.step` returns empty actions (graceful done).
          - `TaskDone` (BaseException) raised by MonitoredTool — task
            finished or agent emitted STOP_ACTION. Propagates to Episode.
          - `BudgetExceeded` (BaseException) raised by MonitoredTool or
            by `recorder.on_llm_call` after the cap is crossed.
        """
        obs = initial_obs
        while True:
            agent_output = await asyncio.to_thread(self.step, obs)
            if not agent_output.actions and not agent_output.error:
                return  # graceful done
            for action in agent_output.actions:
                result = await env_tool.execute_action(action)
                if isinstance(result, StepError):
                    return
                obs = result
```

Sync-only agent authors don't see this — they override `step()` only
and inherit `Agent.run`, never writing async. The async-uniform
`env_tool` is purely for the `run()`-override path.

Agents that want parallel tool calls override `run()` and call
`env_tool` directly:
```python
results = await asyncio.gather(*(
    env_tool.execute_action(a) for a in actions
))  # each is Observation | StepError; ToolCallEvent + budget bump fire inside each call
```

Agents that don't override get the one-at-a-time default for free.

The two parameters:
- **`initial_obs`** — the observation from `task.reset()`, supplied by `Episode`.
- **`env_tool`** — the task's tool (a `cube.tool.Toolbox` or single
  `AbstractTool`), with `MonitoredTool` wrappers installed in place by
  Episode. The name signals what it represents — *the tool that drives
  the monitored environment*, distinct from any agent-private tools.
  The agent calls `env_tool.execute_action(action)`; no `task`
  reference reaches the agent. Done detection, step-wise evaluation,
  and obs_postprocess are absorbed by MonitoredTool — the agent's
  view of the return value is just `Observation | StepError`. Agents
  that want their own private tools (memory, scratchpad, planner) hold
  them as instance fields and compose locally if they want a unified
  dispatch: `combined = Toolbox([env_tool, self.memory])`. The framework
  doesn't have a hook for this — agents have full Python.

The recorder is attached separately via `agent.attach_recorder(recorder)`
before `run()`. Two reasons to override:

1. **Propagate to held LLMs.** Subclasses with one or more `LLM`
   instances call `self.llm.attach_recorder(recorder)` so
   `LLM.call(prompt, tag)` auto-emits `LLMCallEvent`. Multi-LLM
   agents attach selectively — LLMs they don't attach are excluded
   from the trajectory.

2. **Reach the live `Budget`.** The base impl stashes the recorder
   on `self._recorder`, exposing the live `Budget` at
   `self._recorder.budget`. Agent authors use this for two patterns:

   - **Soft self-stop** when `budget.exhausted` is True — return a
     `STOP_ACTION` from `step()` so the episode terminates cleanly
     rather than letting `MonitoredTool` raise `BudgetExceeded`
     mid-call.
   - **Prompt injection** with `str(budget)` (concise summary of
     configured caps + current usage) so the LLM can plan against
     remaining budget. Genny does this every K turns via
     `display_budget_every_k`.

   Read-only fields available: `turns`, `tool_calls`, `cost_usd`,
   `prompt_tokens`, `completion_tokens`, plus the `exhausted`
   property.

### User experience: writing an agent

The contract above is small but it's an **inversion-of-control change**
versus today's `step(obs) -> AgentOutput`:

| Before | After |
|---|---|
| Return `AgentOutput` from `step(obs)` | Either keep `step` (sync path) or own the loop in `run` (async path) |
| Episode called `task.step(actions)` for you | The loop calls `await env_tool.execute_action(a)` — no `task` reference |
| Sync world | `Agent.run` is `async`; sync agents inherit the default and never write `await` |
| Termination: empty actions or `done` | Termination: graceful return from `run`, or `TaskDone` / `BudgetExceeded` raises |

There are exactly two ergonomics paths. Pick by what the agent actually needs.

#### Sync path — override `step()` and inherit `run`

The 90% case. Your agent emits one action per turn, runs against an
LLM, doesn't need parallelism. You write **only** `step()` and never
see async code or an `env_tool`:

```python
class MyAgent(Agent):
    def step(self, obs: Observation) -> AgentOutput:
        call = self.llm.call(self._prompt(obs))
        actions = self._parse(call.output.content)
        return AgentOutput(actions=actions, llm_calls=[call])
```

That's it. The base `Agent.run` wraps `step()` in `asyncio.to_thread`,
records the turn via `EventStreamer.record()`, and dispatches each
action through `await env_tool.execute_action(...)`. `ReactAgent` and
`Genny` work this way. Budget self-stop available via
`self._recorder.budget` if you want to inject a "running low" prompt
or graceful-stop on cap; otherwise `MonitoredTool` raises
`BudgetExceeded` for you.

#### Async path — override `run()` for parallelism or streaming

When you want parallel tool calls, async LLM dispatch, or
fine-grained streaming events, override `Agent.run`. `env_tool` is
**always** `AbstractAsyncTool` — Episode applies `as_async()` to sync
underlying tools at the boundary so your code has no branch.

```python
class ParallelAgent(Agent):
    def attach_recorder(self, recorder):
        super().attach_recorder(recorder)
        self.llm.attach_recorder(recorder)   # so every `self.llm.call()` auto-emits

    async def run(self, initial_obs, env_tool):
        obs = initial_obs
        while True:
            call = await self.llm.acall(self._prompt(obs))   # auto-emits LLMCallEvent
            actions = self._parse(call.output.content)
            if not actions:
                return  # graceful done

            # Parallel dispatch — N concurrent tool calls. Each lands as
            # a ToolCallEvent sharing the parent LLMCallEvent's id as
            # parent_event_id; XRay renders sibling ToolCallEvents with
            # the same parent_event_id as horizontal lanes.
            # Budget + storage hooks fire inside each MonitoredTool.
            # TaskDone / BudgetExceeded propagate through asyncio.gather
            # to Episode's outer except.
            results = await asyncio.gather(*(
                env_tool.execute_action(a) for a in actions
            ))
            obs = self._merge(results)
```

`Genny` with `parallel_actions=True` follows this shape. ~10 lines for the whole loop. No
`task` reference, no `task.reset` / `task.evaluate` / `task.close`,
no `recorder` reference inside the loop — Episode owns lifecycle;
MonitoredTool absorbs `Task.step` semantics (STOP_ACTION,
obs_postprocess, validate_per_step, finished()); LLM auto-emits.

#### Agent-private tools

Both paths work the same. The framework has no hook for "agent-owned
tools" — agents have full Python. Hold tools as instance fields and
either call them directly inside `step` / `run`, or wrap them in a
local `Toolbox` if you want unified dispatch:

```python
class AgentWithMemory(Agent):
    def __init__(self, config, llm):
        super().__init__(config)
        self.llm = llm
        self.memory = MemoryTool()  # private; never reaches Episode

    def attach_recorder(self, recorder):
        super().attach_recorder(recorder)
        self.llm.attach_recorder(recorder)
        # self.memory is intentionally NOT attached — agent-private
        # tools don't appear in the trajectory.

    async def run(self, initial_obs, env_tool):
        combined = Toolbox([env_tool, as_async(self.memory)])
        # ...same loop as above; `remember` and `bash` dispatch through
        # the same call site. Only env_tool's dispatches emit
        # ToolCallEvents; memory calls don't.
```

### `EventStreamer`

The recorder is no longer an agent-facing API. It is a sink:
event producers (LLM, MonitoredTool) emit through it; the recorder
forwards to storage + summary (and, in Phase 2, to OTel / RL HTTP
sinks via `EventStreamerConfig`).

```python
class EventStreamer:
    """The trajectory's event sink. Built by Episode; attached to event
    producers via their `attach_recorder` methods."""

    def __init__(
        self,
        trajectory_id: str,
        storage: Storage | None,
        budget: Budget | None,
    ): ...

    # --- Single fan-out entry point. ---
    def emit(self, te: TrajectoryEvent) -> str:
        """Fold per-episode stats counters (under a lock) + forward
        to each sink. Sole event-flow path: every producer (LLM,
        MonitoredTool, the boundary helpers below) funnels through
        this method."""

    # --- Producer-facing hook (called by LLM.call() auto-emit). ---
    def on_llm_call(
        self,
        call: LLMCall,
        profiling: dict[str, tuple[float, float]] | None = None,
        error: StepError | None = None,
    ) -> str:
        """Emit one LLMCallEvent, bump Budget (LLM usage cost+tokens),
        enforce caps. Returns the event id (stashed as the active
        parent_event_id so subsequent ToolCallEvents inherit it via
        `parent_event_id_getter`)."""

    # --- Agent-loop hook (called once per Agent.run iteration). ---
    def on_step(self) -> None:
        """Bump `budget.turns` + enforce. Turn-counting is per agent
        step, NOT per LLM call (one step may make 0..N LLM calls)."""

    # --- Episode-only boundary helpers (not called by agents). ---
    def record_reset(self, initial: EnvironmentOutput) -> None: ...
    def record_failure(self, exc: BaseException) -> None: ...     # → AgentErrorEvent
    def record_evaluation(self, reward: float, info: dict | None = None, *,
                          is_terminal: bool = True) -> None: ...
    def current_parent_event_id(self) -> str: ...                          # for MonitoredTool

    # --- Final stats (queried by Episode at finalize). ---
    def summary_stats(self, *, duration: float | None,
                      final_reward: float) -> dict:
        """Returns the per-episode stats dict written to
        TrajectoryMetadata.summary_stats. Replaces the dropped
        SummaryProcessor; no separate episode_summary.jsonl."""

    @property
    def budget(self) -> Budget: ...                                # for agent introspection
```

The legacy `record(AgentOutput)` / `begin_turn() / Turn / add_*`
surface was dropped in favor of producer auto-emit. Less ceremony,
streaming-friendly (no batched turn boundaries to flush), and the
RL HTTP sink can subscribe to a clean stream of events.

`EventStreamerConfig` is a forward seam on `EpisodeConfig` for sink
configuration (OTel, RL HTTP, custom). Phase 1 ships an empty
`EventStreamerConfig` — `FileStorage` is the sole sink (per-episode
stats are folded directly inside the streamer; no separate sink for
them). Phase 2 adds fields like `enable_otel: bool` and
`rl_http_endpoint: str | None`.

Cross-episode state (sinks, budget) lives on `Episode` and is bound
into the `EventStreamer` at construction. Agents never read or write
that state directly.

### `TaskDone` — end-of-episode signal

cube-standard's gym contract carried `done` inside `EnvironmentOutput`.
With `Agent.run` taking a `Toolbox` (not a `Task`), the `done` flag
disappears from the agent's surface. The replacement:

```python
class TaskDone(BaseException):
    """Raised by MonitoredTool when the task indicates the episode is
    over. Fires on two paths:

    - Agent emitted STOP_ACTION and `task.accept_agent_stop=True`.
    - `task.finished(obs)` returned True after the inner tool call.

    Like BudgetExceeded, subclasses BaseException so agent code's
    `try / except Exception` doesn't swallow it. Episode catches it in
    its outer `except TaskDone:` and finalizes normally."""
```

MonitoredTool internally calls `task.finished(obs)` after every
successful tool dispatch; on True, raises `TaskDone`. Same for
`STOP_ACTION` — checked BEFORE dispatch, raised before any inner
call. From the agent's POV, control flow simply unwinds out of
`toolbox.execute_action` when the task decides it's done.

### `MonitoredTool`

`MonitoredTool` subclasses `cube.tool.AbstractTool` and exposes a **dual
API**: `execute_action` (sync; sync inner only) and `async_execute_action`
(async; works for both sync and async inners). It is a transparent
decorator: agents and `task.step` call it identically to any other tool.
The previously-proposed split into `MonitoredTool(AbstractTool)` +
`AsyncMonitoredTool(AbstractAsyncTool)` collapsed into this single class.

```python
class MonitoredTool(AbstractTool):
    def __init__(
        self,
        inner: AbstractTool | AbstractAsyncTool,
        emit: Callable[[TrajectoryEvent], str],   # streamer.emit
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        task: Any | None = None,
    ): ...

    @property
    def action_set(self) -> list[ActionSchema]:
        return self.inner.action_set                          # transparent

    def execute_action(self, action: Action) -> Observation | StepError:
        """Sync path. Requires inner to be a sync AbstractTool."""
        ...

    async def async_execute_action(self, action: Action) -> Observation | StepError:
        if self.budget.exhausted:
            raise BudgetExceeded(action=action)
        # Await directly when inner is AsyncTool;
        # wrap with asyncio.to_thread when inner is sync Tool.
        result = await self._invoke_inner(action)
        self._record_tool_call_event(action, result)          # storage + summary + trajectory
        self.budget.tool_calls += 1
        return result                                          # unchanged
```

When a multi-tool container is needed, use `cube.tool.AsyncToolbox`
directly — cube-standard's companion PR (#152) relaxed it to accept mixed
sync + async leaves, so monitored and unmonitored tools coexist in one
toolbox without a harness-side wrapper class. Since `AsyncToolbox` is-a
tool, the composition is recursive: a toolbox may contain monitored
tools, unmonitored tools, and nested toolboxes side-by-side. The agent
calls `toolbox.async_execute_action(action)`; dispatch by action name
routes to the right member, monitored or not.

Three things to note about this shape:

1. **`MonitoredTool` returns `Observation | StepError`** — same as any
   `Tool`. It does not return `EnvironmentOutput` and does not detect
   `done`. Those remain the responsibility of `Task.step` (cube-standard's
   gym wrapper), which calls into the toolbox and constructs the
   `EnvironmentOutput` from `obs + finished(obs) + evaluate(obs) + …`.
2. **Done detection is unchanged.** `Task.step` returns
   `EnvironmentOutput.done = True` when `self.finished(obs)` is True. The
   default `Agent.run` inspects that and terminates. Tool-level agents that
   bypass `task.step` need their own done logic — typically a "submit"
   tool whose obs triggers `task.finished()` for the next gym caller, or
   the agent returning when its own success criterion is met.
   (Open question resolved: an async `task.astep` is deferred — agents that
   want async LLM dispatch override `_arun` directly.)
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
| `ToolCallEvent` persistence (env-step save) | `MonitoredTool` | Records to trajectory + storage on every call. |
| Per-call summary update (env-step counter, reward) | `MonitoredTool` | `summary.on_event(tool_call_event)`. |
| Budget enforcement (`max_steps`, etc.) | `MonitoredTool` | Counts calls; raises `BudgetExceeded(BaseException)`. |
| Heartbeat / `status.json` | `MonitoredTool` | Updates `last_heartbeat_at` per call. Cheap. |
| Tool-side error handling (`StepError` from inner) | `MonitoredTool` | Records the `StepError` into the `ToolCallEvent`, returns it (does not raise). |
| `AgentEvent` persistence (agent-step save) | `EventStreamer` | `record()` or `Turn.__exit__` flushes one AgentEvent. |
| Per-turn summary update (LLM calls, tokens, cost) | `EventStreamer` | Updates `summary` from `AgentEvent.llm_calls`. |
| Agent output logging | `EventStreamer` | Logs alongside the event flush. |
| Per-turn OTel span (`tracer.step("turn_N")`) | **Dropped** | Today's loop-level span goes away; agent-owns-loop has no central per-turn point to wrap. Per-turn data lives in `AgentEvent` (richer than a span name). The episode-level span (below) is preserved. |
| Agent-side error capture (`agent.step` raised) | `EventStreamer.record_failure` | Episode's `except` calls it after `agent.run` raises. |
| `done` detection | `Task.step` (cube-standard, unchanged) | Returns `EnvironmentOutput.done`. Default agent reads it. |
| Per-step `reward` | `Task.step` (cube-standard, unchanged) | Comes through `EnvironmentOutput.reward`. |
| Step-wise `evaluate` (`validate_per_step`) | `Task.step` (cube-standard, unchanged) | Built into `task.step` at [task.py:346](../../../src/cube/task.py#L346). |
| `obs_postprocess` | `Task.step` (cube-standard, unchanged) | |
| Action validation (empty → break) | Default `Agent.run` | Convention; not enforced. |
| `task.reset` | `Episode` (before `agent.run`) | Initial obs handed to the agent. |
| Terminal `task.evaluate` | `Episode` (in `finally`) | `recorder.record_evaluation(reward, info)`. |
| `task.close` | `Episode` (in `finally`) | |
| Storage finalize | `Episode` (in `finally`) | |
| Episode-level OTel span | `Episode` | Wraps the whole `try / except / finally` (preserved from today). |

### Connector taxonomy (forward-looking, Phase 2)

`Agent.run(initial_obs, task, recorder)` is the seam. Each external agent
framework is plugged in via an `Agent` subclass that bridges to it. Three
buckets cover the landscape:

```
┌────────────────────────────────────────────────────────────────┐
│ Bucket 1 — In-process Python frameworks                         │
│   Agent.run() instantiates the framework, wires tools,          │
│   drives its loop, captures events → recorder.                  │
│   LangGraph, Pydantic AI, Inspect AI, OpenAI Agents SDK,        │
│   Claude Agent SDK, smolagents.                                 │
│   Connector size: ~30 LOC tool-shim + ~150 LOC agent class.     │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ Bucket 2 — CLI / subprocess agents (run inside cube sandbox)    │
│   Agent.run() launches the binary, points it at cube.server's   │
│   MCP URL for tools, parses its JSON event stream → recorder.   │
│   Codex CLI, Goose, Pi.                                         │
│   Connector size: ~150 LOC subprocess + JSONL parser.           │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ Bucket 3 — HTTP / A2A agents                                    │
│   Agent.run() acts as A2A client — POSTs Message with task      │
│   instruction, streams Task state, captures Message exchanges   │
│   → recorder. Tools bridged via cube.server JSON-RPC.           │
│   AgentBeats agents, any A2A-compliant agent.                   │
│   Connector size: ~200 LOC (A2A client + tool bridge).          │
└────────────────────────────────────────────────────────────────┘
```

Module layout (Phase 2):

```
cube_harness/connectors/
├── pydantic_ai/    langgraph/    openai_agents/    inspect_ai/
├── claude_agent_sdk/    smolagents/
├── codex_cli/    goose/    pi_cli/
└── a2a/
```

Each module ships `{framework}_agent.py` (an `Agent` subclass) and
`tools.py` (a ~30-line shim mapping `cube.tool.Tool` → framework-native
tool form). No core changes per connector; each is self-contained.

#### Trade-offs by bucket

What the recorder can capture depends on what the framework lets us observe:

| Bucket | Tool calls | LLM calls | Tokens/cost | Per-step trajectory |
|---|---|---|---|---|
| 1 — In-process Python | ✅ full | ✅ usually | ✅ | ✅ |
| 2 — CLI subprocess | ✅ via `cube.server` | ❌ | ✅ from CLI's JSON output | Partial (per-turn) |
| 3 — A2A / HTTP | Messages only | ❌ | Depends on AgentCard | Message-level |

The trade-off is acceptable: connectors trade fine-grained traces for
**evaluation parity** — a benchmark score from an off-the-shelf agent in
~minutes of integration work, vs. days of re-implementation.

#### Excluded from connector planning

- **ATA** (claimed AgentBeats protocol) — does not exist as a distinct
  spec; AgentBeats agents speak A2A + MCP. Covered by the A2A connector.
- **AGNTCY / SLIM** (Cisco enterprise stack) — heavy gRPC/mTLS layer;
  every SLIM agent today also speaks A2A. Defer.
- **AG2 / AutoGen** — conversation-flavored, high impedance mismatch.
  Skip unless explicitly requested.
- **Mastra** (TypeScript framework) — not a binary, requires a TS project
  scaffold. Mis-shaped for CLI-style adapter.

### `Episode.run` (lifecycle owner)

```python
async def run(self) -> TrajectoryView:
    task = self.task_config.make(...)
    meta = TrajectoryMetadata(id=self.id, metadata={...}, start_time=now(), end_time=None)
    self.storage.save_metadata(meta)              # WRITE-AT-START: crashed runs are loadable
    budget = Budget(max_agent_steps=self.config.max_agent_steps, ...)  # was max_turns; counts agent-loop iterations, not LLM turns

    # Wrap each member of task.tool with MonitoredTool, baking in the
    # task ref so wrappers absorb cube-standard Task.step semantics
    # (STOP_ACTION, obs_postprocess, validate_per_step, finished).
    # Events stream to disk via storage.save_event; nothing in memory.
    install_monitoring(task, trajectory_id=self.id, budget=..., storage=..., summary=...)

    # Compose the toolbox the agent sees: monitored task tools + the
    # agent's own (non-monitored) tools from agent_config.own_tool_configs.
    own_tools = [cfg.make() for cfg in self.config.agent_config.own_tool_configs]
    toolbox = Toolbox([*task.tool.tools, *own_tools]) if own_tools else task.tool

    recorder = EventStreamer(
        trajectory_id=self.id,
        storage=self.storage,
        summary=self.summary,
        budget=budget,
    )
    try:
        initial = task.reset()
        recorder.record_reset(initial)            # Episode-only helper
        # Wire LLMs etc. to the recorder so LLM.call() auto-emits
        # LLMCallEvent. Sub-component-aware agents (Genny, React,
        # GenericAgent) override attach_recorder to propagate.
        self.agent.attach_recorder(recorder)
        env_tool = as_async(task.tool or task.toolbox)
        await self.agent.run(initial.obs, env_tool)
    except BudgetExceeded as e:
        recorder.record_failure(e)
    except TaskDone:
        pass                                       # clean end — task said done
    except Exception as e:
        recorder.record_failure(e)
        raise
    finally:
        try:
            reward, info = task.evaluate()        # terminal eval, obs optional
            recorder.record_evaluation(reward, info, is_terminal=True)
        except Exception as e:
            recorder.record_failure(e)
            raise
        meta = meta.model_copy(update={
            "end_time": now(),
            "reward_info": recorder.reward_info,
            "summary_stats": self.summary.snapshot(),
        })
        self.storage.finalize_episode(meta)       # UPDATE-AT-END: same file, full data
        self.summary.on_episode_complete(meta, self.storage)
        task.close()
    return self.storage.load_episode(self.id)     # lazy view onto what we just wrote
```

Episode- and benchmark-level OTel spans (today's `tracer.episode(...)` around
the `try` block, `tracer.benchmark(...)` higher up in `exp_runner`) and
`tracer.shutdown()` in `finally` are preserved from today's `Episode.run` —
elided from the pseudo-code above for clarity. This RFC does not add new
OTel surface (no per-tool-call, no per-turn spans). The trajectory event
stream is the harness's structured per-call/per-turn observability.

The agent cannot prevent finalization. The agent receives only
`env_tool` — the `task` reference never leaks, the recorder is
attached separately. The monitoring wrappers are installed onto
`task.tool`'s leaves once, baking the task ref in for `Task.step`-
equivalent semantics (STOP, postprocess, done detection, step-eval).
`record_reset` / `record_failure` / `record_evaluation` are
Episode-only helpers on `EventStreamer` (not actively hidden from
agents, but conventionally Episode's). `on_llm_call` is the only
producer-facing entrypoint — called automatically by `LLM.call(...)`
when a recorder is attached.

Note: `task.evaluate()` is called with no obs in `finally`. Tasks that need
the final obs to evaluate must track it internally (cube-standard's `Task`
already does for the gym path). This avoids the harness having to chase the
"last good obs" through the agent's loop logic.

### Event-stream trajectory

```python
class LLMCallEvent(TypedBaseModel):
    id: str                            # parent_event_id for child ToolCallEvents
    call: LLMCall | None               # full prompt/response/usage (None on legacy decode only)
    profiling: dict[str, tuple[float, float]]
    error: StepError | None

class ToolCallEvent(TypedBaseModel):
    id: str                            # for step-wise EvaluationEvent.parent_event_id back-ref
    parent_event_id: str               # the originating LLMCallEvent.id (or RESET sentinel);
                                       # parallel-sibling tool calls share this id by construction —
                                       # it is the sole grouping primitive (no separate turn_id field)
    action_id: str | None              # echoes Action.id
    action: Action | None              # full action payload — self-contained trajectory
    obs: Observation                   # what came back to the agent (empty when error)
    error: StepError | None            # set when execute_action returned a StepError

class EvaluationEvent(TypedBaseModel):
    reward: float
    info: dict
    is_terminal: bool                  # True iff this is Episode's final evaluate
    parent_event_id: str | None        # for step-wise: the ToolCallEvent.id; None for terminal

class AgentErrorEvent(TypedBaseModel):
    id: str
    error: StepError                   # Episode-level failure not tied to a specific call

class TrajectoryEvent(TypedBaseModel):
    output: LLMCallEvent | ToolCallEvent | EvaluationEvent | AgentErrorEvent
    start_time: float
    end_time: float
```

`Trajectory` (a Pydantic class holding `events` + metadata) is **deleted**.
Two replacement abstractions take its place — see *Storage & Loaders* below.

Why this shape:

- **One `LLMCallEvent` per LLM API call**, not per "turn" — collapses the
  prior batched `AgentEvent`. Streaming-friendly: each event lands as soon
  as the LLM call completes; no batched flush at turn boundaries. An agent
  that makes 3 LLM calls per step (Genny: compact + summarize + act) emits
  3 LLMCallEvents; XRay groups them by `parent_event_id`.
- **`ToolCallEvent` carries only `obs` + `error`** — the agent's view of
  what came back. `reward` / `done` / `info` are NOT here:
  - `done` is signalled by the `TaskDone(BaseException)` exception
    raised by MonitoredTool. No `done` field in the trajectory.
  - Step-wise `reward` / `info` (when `task.validate_per_step=True`)
    live on a sibling `EvaluationEvent` with `is_terminal=False` and
    `parent_event_id` referencing the `ToolCallEvent.id`.
  - The terminal `reward` / `info` is an `EvaluationEvent` with
    `is_terminal=True` and `parent_event_id=None`, emitted by Episode.
- **`ToolCallEvent.action_id` references back to the parent
  `AgentEvent.actions[i].id`**, so the event stream is a flat list but the
  parent-child structure is recoverable.
- **`parent_event_id` groups parallel calls** so XRay can render
  sibling `ToolCallEvent`s sharing the same originating `LLMCallEvent.id`
  as horizontal lanes within a turn. A separate `turn_id` field was
  considered and dropped — `parent_event_id` is already the grouping
  primitive by construction.
- **One `EvaluationEvent` type for both step-wise and terminal**
  evaluations — discriminated by `is_terminal` and the presence of
  `parent_event_id`. One type, two flavors; no discriminated union in
  the trajectory event stream.

Storage filenames evolve from `000_obs.msgpack.zst` / `001_act.msgpack.zst` to
`000_agent.msgpack.zst` / `001_tool_call.msgpack.zst` / `002_eval.msgpack.zst`.
Old V2 layouts (steps/) remain loadable via the legacy-upgrade view.

### Storage & Loaders — `TrajectoryMetadata` + `TrajectoryView`

The `Trajectory` pydantic class is deleted. Two abstractions replace it.

```python
class TrajectoryMetadata(TypedBaseModel):
    """Persisted at episode.metadata.json. Written at episode start with
    end_time=None and stub summary fields; updated at episode end with
    the final summary. Tiny — never holds events."""
    id: str
    metadata: dict                       # task_id, agent_config dict, infra, …
    start_time: float | None
    end_time: float | None               # None until episode finalizes
    summary_stats: dict | None           # filled at end
    reward_info: dict                    # filled at end (mirrors final EvaluationEvent)
```

```python
class TrajectoryView:
    """Lazy reader bound to (Storage, trajectory_id).

    Holds `.metadata` eagerly (one JSON read). Events are decoded from
    disk on demand — never held in a list. Random access via view[i]
    populates an internal dict cache scoped to the view's lifetime.
    """
    storage: Storage
    id: str
    metadata: TrajectoryMetadata            # eager
    _cache: dict[int, TrajectoryEvent]   # populated on access; cleared with the view

    def __len__(self) -> int             # from directory listing — cheap
    def __getitem__(self, i: int) -> TrajectoryEvent
    def __iter__(self) -> Iterator[TrajectoryEvent]    # decodes one at a time
    def iter_events(self) -> Iterator[TrajectoryEvent] # alias for __iter__
    # Convenience shortcuts to .metadata fields:
    @property
    def summary_stats(self) -> dict | None
    @property
    def reward_info(self) -> dict
    @property
    def is_complete(self) -> bool  # metadata.end_time is not None
```

Modeled on AgentLab's `Result`/`Episode` loader pattern:

- **Studies of 500 episodes never touch bulk data** — `list_episodes()`
  returns `list[TrajectoryMetadata]` (one cheap JSON read each). DataFrame
  aggregation, Atlas indexing, and EpisodeRecord generation use metadata
  only.
- **Only episodes you actually inspect pay event I/O** — XRay's data
  layer iterates `view`, decoding one event per render.
- **Crash-safe loading.** The metadata file is written upfront with
  `end_time=None`; on crash, `view.is_complete == False` and XRay
  renders what's on disk plus the `status.json` failure summary.

Storage protocol changes:

```python
class Storage(Protocol):
    def save_metadata(self, meta: TrajectoryMetadata) -> None
    def save_event(self, event: TrajectoryEvent, trajectory_id: str, n: int) -> None
    def load_episode(self, trajectory_id: str) -> TrajectoryView   # replaces load_trajectory
    def finalize_episode(self, meta: TrajectoryMetadata) -> None   # writes final metadata + summary
    def list_episodes(self) -> list[TrajectoryMetadata]            # cheap study scan
```

Legacy V1 (steps/-layout) episodes load through `TrajectoryView` with a
synthesis layer: `AgentOutput` steps map to `AgentEvent`s,
`EnvironmentOutput` steps map to `ToolCallEvent`s parented to the most
recent agent event. Synthesized events live only in the iterator — never
rewritten to disk.

### Internal cache policy

`TrajectoryView._cache` is a plain `dict[int, TrajectoryEvent]`. AgentLab's
pattern: no LRU, no eviction. RAM is bounded by ONE episode (~thousands of
events tops at SWE-bench scale). When XRay opens a new episode it
constructs a new `TrajectoryView`; the old one is GC'd along with its cache.

### XRay rewrite

The viewer is rebuilt around the event stream and consumes the lazy
`TrajectoryView`. The `_events_to_legacy_steps` shim is **deleted** — there
is no remaining caller after this PR.

- **Data layer:** `view = storage.load_episode(id)`. The viewer iterates
  once to build a lightweight index `event_index -> kind` (~bytes per
  event, no payloads). Each card lazy-loads its full event body when
  rendered via `view[i]`. The view's internal cache keeps re-renders fast
  without holding the whole trajectory; switching episodes constructs a
  new view and the old cache is GC'd.
- **Timeline:** one card per `TrajectoryEvent`. Colours by kind: `agent`
  (LLM / thoughts / response text), `tool_call` (one per action),
  `evaluation` (final). Parallel `tool_call` siblings share a
  `parent_event_id` (the originating `LLMCallEvent.id`) and render in
  horizontal lanes within a turn group.
- **Selection:** clicking a card sets `selected_event_index`. The viewer
  derives `last_agent_event_index` and `last_observation_event_index` by
  walking back from the selection (cached in the view).
- **Tabs:**
  - **Reasoning / Chat** — renders the AgentEvent at
    `last_agent_event_index` (thoughts, response_text, intended actions,
    LLM messages).
  - **Observation** — renders the obs from `selected_event` if it's a
    `tool_call`, else from `last_observation_event`. Screenshots are
    content inside the observation, not a separate tab.
  - **Turn observations** — all `tool_call` events sharing the selected
    event's `parent_event_id`. Empty for non-tool events.
  - **Profiling** — per-event timing breakdown (already supported via
    `AgentEvent.profiling`).
  - Header strip always shows: `Event X / N — kind, turn=…, t=…s`.
- **Crashed/in-flight episodes** (`view.is_complete == False`): timeline
  renders what's on disk; banner shows the `status.json` failure summary.
- Drop the standalone screenshot tab; it's redundant with Observation.
- Drop the legacy steps/ pairing logic (env+agent paired into a single
  navigation unit) — navigation is per-event.

### RPC layer

cube-standard already has the canonical RPC surface: `cube.server` exposes
`tools/list`, `tools/call`, `cube/step`, etc. as JSON-RPC 2.0 (MCP-compatible).
The Phase 1 PR does **not** change `cube.server`. The companion cube-standard
change (`cube-standard/openspec/changes/agent-owns-loop/`) only clarifies that
`MonitoredTool` lives in the harness (it captures harness-side trajectory
state, composed into `cube.tool.AsyncToolbox`) and that future external-agent
connectivity will use the existing `cube.server` endpoint with a per-session
monitoring context attached on the harness side. The harness's duplicate `cube_harness/mcp/server.py` is left
alone in Phase 1 and slated for retirement in a follow-up.

### Async-first

The new `Agent.run` is `async def`. Sync `step()` is wrapped in
`asyncio.to_thread` by the default `run()`. The `Agent` base ships dual
`_run` / `_arun` paths, selected by `AgentConfig.parallel_actions` — the
parallel-dispatch mode lives on `Genny` as a config flag rather than a
separate parallel-agent class. Monitored tools compose into
`cube.tool.AsyncToolbox` (cube-standard's companion PR relaxed it to
accept mixed sync + async leaves). LLM calls become awaitable through
`cube_harness.llm` — out of scope for this RFC if LiteLLM async is
already available (likely yes, follow-up if not).

---

## Migration

- **`Trajectory` is deleted.** No deprecation alias. Callers that imported
  it move to `TrajectoryMetadata` (for fields) or `TrajectoryView` (for iteration).
  In-tree callers are migrated in this PR (storage, episode, experiment,
  EpisodeRecord, XRay, investigator, ~30 tests).
- **Legacy on-disk episodes remain loadable.** `TrajectoryView` detects
  `events/` vs `steps/` at open time. For steps/-layout episodes, the
  iterator synthesizes `AgentEvent` / `ToolCallEvent` from
  `AgentOutput` / `EnvironmentOutput` step files on the fly. No
  on-disk rewrite — synthesis is read-time only.
- **`trajectory.json` (V1 metadata file)** is still read by the legacy
  loader. New writes always use `episode.metadata.json`. The V1 reader
  path stays for backward compat with archived runs.
- Existing agents (`ReactAgent`, `Genny`) keep their `step()` / `run()` —
  no agent-side changes for backward compat. The previously-proposed
  parallel-agent subclass folds into `Genny` with
  `GennyConfig(..., parallel_actions=True)`; selection of `_run` vs
  `_arun` follows `AgentConfig.parallel_actions`.
- `EpisodeRecord.from_trajectory(traj)` → `EpisodeRecord.from_view(view)`.
  Body reads `view.metadata` only — no event I/O.
- Tests that hand-built `Trajectory(id=..., steps=[...])` migrate to
  `make_fake_episode(events=[...]) -> TrajectoryView`, a new test helper
  that writes events to a `TmpStorage` and returns a real lazy view.
  Tests then exercise the actual load path, not a mock.

---

## Risks

- **XRay rewrite is the largest single piece.** Mitigated by: (a) the
  legacy V1 reader path stays, so old experiment dirs still load; (b)
  smoke `xray_loads_event_trajectory.py` is updated to assert the full
  event-card UI renders content from both a fresh events/ trajectory and
  a legacy steps/ trajectory; (c) manual `make xray` check against both.
- **Behaviour drift in default `Agent.run` vs. today's loop.** Mitigated by
  routing today's loop through the same hook helpers — diff is structural,
  not behavioural.
- **Subclass agents that override `step()` but expect a specific call
  cadence.** No such agents exist in-tree; out-of-tree agents may need a
  trivial update. Documented in the agent spec.
- **Storage format change** — addressed by reading both formats and writing
  only the new one. One release window for tooling to catch up.
- **`Trajectory` deletion breaks out-of-tree code that imported it.** No
  deprecation alias is shipped: the type lived in `cube_harness.core` for
  ~2 years but in-tree callers were always thin (mostly fixture
  construction in tests). Out-of-tree callers move to `TrajectoryView` for
  iteration or `TrajectoryMetadata` for fields. Documented in PR notes.

---

## Spec changes

See `deltas.md` (cube-harness) and `cube-standard/openspec/changes/agent-owns-loop/deltas.md`.
