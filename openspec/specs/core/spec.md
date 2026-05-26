# Core Types

**Module:** `cube_harness.core`

## Purpose

Data structures that layer on top of cube-standard's core types. `AgentOutput` captures
what one agent step produced; `Trajectory` accumulates the alternating sequence of
agent and environment steps over an episode.

## Public API

### `AgentOutput`
```python
class AgentOutput(TypedBaseModel):
    actions: list[Action] = []           # actions produced by this agent step
    llm_calls: list[LLMCall] = []        # every LLM call made this step; tag each via LLMCall.tag
    error: StepError | None = None       # populated if agent.step() raised
    profiling: dict[str, tuple[float, float]] = {}  # label → (start, end) unix timestamps
    thoughts: str | None = None          # chain-of-thought / extended thinking text
```

`__str__` excludes `llm_calls` for a compact debug line (they're large).

### `TrajectoryStep`
```python
class TrajectoryStep(TypedBaseModel):
    output: EnvironmentOutput | AgentOutput
    start_time: float | None = None
    end_time: float | None = None
```

A step is either the environment's output (obs/reward/done/info) or the agent's output
(actions/LLM calls/error). They alternate in a trajectory: env₀ → agent₀ → env₁ → agent₁ → …

### `Trajectory`
```python
class Trajectory(TypedBaseModel):
    id: str                                # usually f"{task_id}_ep{episode_id}"
    steps: list[TrajectoryStep] = []       # lazy/on-disk view; EMPTY on a runner-produced traj
    metadata: dict = {}                    # task_id, agent_name, info dump
    start_time: float | None = None
    end_time: float | None = None
    reward_info: dict = {}                 # final reward + info dict
    summary_stats: dict | None = None      # streamed by SummaryProcessor (the aggregate source)

    def last_env_step(self) -> EnvironmentOutput  # raises if none present (needs steps loaded)
    @property n_agent_steps: int
    @property n_env_steps: int
```

A `Trajectory` returned by `Episode.run()` / the runners carries metadata + `summary_stats`
+ `reward_info` but **no steps** (they stream to disk during the run). Use `summary_stats` /
`reward_info` for aggregates; call `FileStorage.load_trajectory(id)` to materialize step
content (xray/investigator do this on demand).

### `ActionSpace`
```python
class ActionSpace(frozenset[Callable]):
    """frozenset of action callables. Supports &, -, | for composition."""
    def __new__(cls, *actions: Callable) -> "ActionSpace"
    @property names: frozenset[str]
```

## Invariants

1. Trajectory steps alternate: `EnvironmentOutput` and `AgentOutput` in sequence (env first).
2. `last_env_step()` walks backwards and raises `ValueError` if none exists — callers
   must have at least reached the reset.
3. `LLMCall.tag` is how multiple LLM calls per step (e.g., "act", "summary") are
   distinguished — agents should tag consistently.
4. `AgentOutput.error=None` means the step succeeded; a non-None error means the agent
   itself failed (not the environment). `EnvironmentOutput.error` covers the other case.

## Gotchas

- `profiling` uses absolute Unix timestamps, not durations. The XRay viewer computes
  the delta.
- `ActionSpace` inherits from `frozenset` — use set operators, not list ops.
