# RFC: Multi-Episode Rollouts for Cross-Episode Memory Agents

**Status:** DRAFT
**Author:** Oleksiy Ostapenko
**Date:** 2026-05-25
**Base branch:** `dev` (per CLAUDE.md)

---

## RFC scope vs. additive changes

Per the constitution's RFC process (post-`068a718f`): "Additive, backward-compatible changes (a new method, a new optional field) skip this — just keep the living spec accurate and describe the change in the PR."

That makes only **one** part of this proposal genuinely RFC-bound:

| Part | RFC-bound? | Why |
|---|---|---|
| New `Rollout` layer (`src/cube_harness/rollout.py`, new spec) | **Yes** | Net-new architectural layer, not an additive extension to an existing one |
| `Agent.reflect()` default no-op | No (additive) | New method with default impl; existing agents unaffected |
| Storage `rollout_id` / `episode_index_in_rollout` fields | No (additive) | New optional fields, default `None`; pre-existing trajectories remain readable |

This document still describes all three because they are designed together — but only the Rollout layer requires async review. The other two can be applied directly to their specs as part of the same PR.

---

## Problem

cube-harness today runs one episode at a time: each `Episode` constructs a fresh `Agent`
via `AgentConfig.make()`, plays it against a `Task`, persists the trajectory, and exits.
This shape is correct for evaluating a fixed policy on a benchmark, but it does not
support a class of meta-RL methods in which the agent is expected to *adapt across
multiple episodes against the same task* by carrying memory between attempts.

The motivating example is **LaMer** ([arxiv 2512.16848](https://arxiv.org/abs/2512.16848)),
a meta-RL framework that improves over RL baselines on Sokoban (+11pp), MineSweeper
(+14pp), and Webshop (+19pp). LaMer's two innovations both depend on a cross-episode
unit of execution:

1. **Cross-episode training framework** — the training signal is a discounted sum of
   per-episode rewards across N attempts, not a single-episode reward. This incentivises
   solving the task as early as possible while preserving exploration value.
2. **In-context policy adaptation via reflection** — between episodes, the agent
   reflects on the just-finished trajectory and updates its in-context memory; no
   gradient updates happen between episodes within a rollout.

Neither is expressible in the current `Episode` / `Experiment` API. There is no place
for an agent instance to survive across episodes, no place for an inter-episode
`reflect()` hook, and no place to aggregate reward across episodes with a discount.

This is also the natural foundation for any future meta-RL approach in cube-harness —
LaMer is the first concrete consumer, not the only one. Scoping the new abstraction to
a generic "rollout of N episodes with persistent agent" rather than "LaMer-specific
machinery" keeps it useful beyond the initial paper.

## Scope

### In scope (V1)

- **New layer**: `Rollout` orchestrator (`cube_harness/rollout.py`) that runs N episodes
  against the same task with a long-lived `Agent` instance.
- **Agent API extension**: one additive optional hook on `Agent` — `reflect(trajectory,
  final_reward) -> None`, default no-op. Existing agents (ReAct, Genny) keep working
  unchanged.
- **Reward aggregation**: `RolloutResult` carries per-episode rewards and a discounted
  aggregate `Σ γ^k · r_k`.
- **Storage extension**: trajectories produced within a rollout carry `rollout_id` and
  `episode_index_in_rollout` metadata so they can be linked back to their rollout
  post-hoc.
- **Validation cube**: MiniWob via cube-browser-tool's `SyncPlaywrightTool` (the
  only browser-tool path post-PR removing `cube_harness/tools/browsergym.py`).
  `MiniWobTask.reset()` (`cubes/miniwob/src/miniwob_cube/task.py:37`) satisfies
  the cube-standard contract the orchestrator depends on. No cube changes needed.
  Recipe will follow the current `recipes/hello_miniwob.py` pattern
  (`MINIWOB_CONFIGS["default"]` via the new `ConfigRegistry`).
- **POC agent**: a `LaMerAgent` subclassing `ReactAgent` with a memory list, a
  reflection prompt, and prompt-injection of the memory near the top of the context.
  Ships as a recipe under `recipes/`, not as a permanent agent.

### Out of scope (deferred to V2+)

- **Ray parallelism over rollouts.** V1 runs rollouts sequentially. Once the contract
  is stable, parallelizing is the standard `run_with_ray` extension — each Ray task
  owns one whole rollout (preserves the "agent doesn't cross process boundaries"
  invariant from CLAUDE.md).
- **Cross-rollout / global memory persistence.** LaMer explicitly scopes memory to
  within a single rollout; persistent memory across rollouts requires an external
  service (MCP / vector DB) and is a separate proposal.
- **Reward-model integration / meta-RL training loop.** The training backend
  (PipelineRL / NeMo-Aligner / TRL / custom) consumes `RolloutResult.discounted_reward`
  exactly as it currently consumes per-episode rewards. No harness-side training
  loop is added.
- **New cube.** MiniWob suffices for V1 validation; cube selection is an experiment
  concern, not a spec concern.

## Design

### Layer placement

```
Experiment           ← unchanged: collection of tasks
  └─ Rollout         ← NEW: N episodes against one task with persistent agent
       └─ Episode    ← unchanged: one task attempt
            └─ Agent.step() loop
```

`Rollout` sits between `Experiment` and `Episode`. Crucially, it **uses** `Episode`
internally — `Episode` is not extended to be the rollout, because that would conflate
two units (one task attempt vs. N attempts with shared memory) and break the existing
episode spec.

### `Rollout` class

```python
class RolloutConfig(TypedBaseModel):
    n_episodes: int
    gamma: float = 1.0          # 1.0 = no discount; <1.0 = LaMer-style discounting
    # task_config and agent_config inherited from the surrounding Experiment context

class RolloutResult(TypedBaseModel):
    rollout_id: str             # uuid linking per-episode trajectories
    trajectories: list[Trajectory]
    per_episode_rewards: list[float]
    discounted_reward: float    # Σ γ^k · r_k

class Rollout:
    def __init__(
        self,
        task: Task,
        agent: Agent,
        config: RolloutConfig,
    ) -> None: ...

    def run(self) -> RolloutResult: ...
```

The invariant: **the same `agent` instance handles all N episodes**. That's the entire
architectural change. Internally:

```python
def run(self) -> RolloutResult:
    trajectories, rewards = [], []
    for k in range(self.config.n_episodes):
        self.task.reset()                          # cube-standard contract
        trajectory, reward = self._run_one_episode()
        trajectories.append(trajectory)
        rewards.append(reward)
        self.agent.reflect(trajectory, reward)     # optional hook (default no-op)
    discounted = sum(self.config.gamma**k * r for k, r in enumerate(rewards))
    return RolloutResult(rollout_id=..., trajectories=trajectories,
                         per_episode_rewards=rewards, discounted_reward=discounted)
```

`_run_one_episode()` reuses the existing `Episode` machinery — it constructs an
`Episode` with the pre-existing `agent` instance instead of calling `agent_config.make()`.

### `Agent.reflect()` hook

```python
class Agent(ABC):
    @abstractmethod
    def step(self, obs: Observation) -> AgentOutput: ...

    def reflect(self, trajectory: Trajectory, final_reward: float) -> None:
        """Called between episodes within a Rollout. Default: no-op.

        Implementations may update internal state read by subsequent step() calls.
        Single-Episode runs never call this — preserves existing behavior.
        """
```

Three properties matter:

- **Default no-op** — ReAct, Genny, and any external agent keep working unchanged. They
  behave as N independent episodes inside a rollout (no memory carryover).
- **Called between episodes, not after the last one** — there is no policy use for the
  reflection after the rollout ends.
- **Sees one trajectory at a time** — implementations that want to look at all prior
  trajectories accumulate them in instance state. Keeps the contract minimal.

### Cube-standard alignment

The orchestrator depends on `Task.reset()` returning the task to its initial state
without recreating its underlying `Resource`. This is already part of cube-standard's
`Task` contract — `MiniWobTask.reset()` (`cubes/miniwob/src/miniwob_cube/task.py:37`)
and `SolveArithmeticTask.reset()` (`cubes/arithmetic-cube/src/arithmetic_cube/task.py:24`)
both implement it. No cube-standard changes required.

Environment note: the validation path requires `cube-standard>=0.1.0rc9` (provides
`cube.core.ConfigRegistry` used by the new `MINIWOB_CONFIGS` registry) and
`cube-browser-tool>=0.3.0` (currently from cube-standard dev branch
`cube-tools/cube-browser-tool` subdirectory; not yet on PyPI). The root pyproject
and `cubes/miniwob/pyproject.toml` handle these as git sources during this transition.

For cubes whose `reset()` is expensive (Docker re-spin, browser relaunch), the cost is
paid N times per rollout. V1 only validates against MiniWob; cubes that need a cheaper
reset path are a per-cube concern.

### Storage and provenance

A trajectory's record currently identifies its episode but not its rollout. We add:

- `rollout_id: str | None` — set by `Rollout`; `None` for trajectories produced by a
  standalone `Episode`.
- `episode_index_in_rollout: int | None` — 0-indexed position; `None` outside a rollout.

Both are optional and additive — existing trajectory files remain readable. XRay and
the trajectory-judge tooling work unchanged on standalone trajectories; rollout-aware
grouping is a follow-up.

### LaMer agent (illustrative, not part of the spec)

The POC agent is small (~100 lines on top of `ReactAgent`):

```python
class LaMerAgent(ReactAgent):
    def __init__(self, config: LaMerAgentConfig, tools: list[ActionSchema]):
        super().__init__(config, tools)
        self.memory: list[str] = []

    def reflect(self, trajectory: Trajectory, final_reward: float) -> None:
        prompt = self._build_reflection_prompt(trajectory, final_reward)
        response = self.llm(prompt)
        self.memory.append(response.message.content)

    def choose_steps_to_render(self, history):
        base = super().choose_steps_to_render(history)
        if self.memory:
            base.insert(1, {"role": "user", "content": self._format_memory()})
        return base
```

This agent is shipped as a recipe (`recipes/lamer_miniwob.py`), not as a permanent
agent under `src/cube_harness/agents/`. Once the approach is validated, a production
variant on top of Genny's prompt skeleton may follow.

## Alternatives considered

**Extend `Episode` to internally loop.** Rejected — conflates "one task attempt"
with "N attempts." Existing code, specs, and downstream consumers (XRay, eval log,
trajectory judge) all assume `Episode = one attempt`. Breaking that invariant has
wide blast radius for marginal benefit.

**Inject memory via an external MCP service from day one.** Rejected for V1 —
adds operational complexity (run a service, handle eventual consistency under
parallelism) for no V1 value since LaMer's memory is scoped within a rollout. The
external-memory path is the right design for *cross-rollout* persistence (V2), where
the in-process approach doesn't work.

**Carry memory state inside `EpisodeConfig` between episodes.** Rejected — the
fresh-agent-per-episode invariant is intentional and currently relied upon by the
serialization boundary (workers receive `EpisodeConfig`, not live `Agent` instances).
Threading mutable agent state through pickled configs invites bugs. The Rollout
approach instead keeps the agent alive *inside one worker* for the rollout's lifetime,
which preserves the serialization boundary at the rollout level.

**Make `reflect()` part of a separate `MetaAgent` interface.** Rejected as
over-engineering. A default no-op on the base `Agent` ABC adds one line, requires no
new abstract base class, and lets any existing agent opt-in by overriding a single
method.

## Open questions

1. **Reflection input granularity.** Should `reflect()` receive `(trajectory,
   final_reward)` for just the latest episode, or `(list_of_trajectories,
   list_of_rewards)` for all prior episodes in the rollout? V1 takes the latest only;
   the agent accumulates prior context in `self.memory`. If a stronger contract is
   needed later, expanding the signature is forward-compatible.

2. **Reward-aggregation location.** Discounted reward computation lives in `Rollout`
   for V1, returned as `RolloutResult.discounted_reward`. The training backend may
   prefer to receive raw `per_episode_rewards` and compute its own aggregate (with
   value baselines, GAE-like smoothing, etc.). Both paths remain available — backends
   that want raw rewards read `per_episode_rewards`.

3. **Episode budget across the rollout.** Should `max_actions` be per-episode (existing
   semantics) or per-rollout (a meta-budget the agent allocates)? V1: per-episode,
   matching today's behavior. Per-rollout budget is a config-level extension.

4. **XRay visualization.** Rollouts are currently displayed as N independent
   trajectories. A rollout-aware view (timeline across episodes, memory diff per
   step) is desirable but out of scope for V1.

## References

- LaMer paper: [arxiv 2512.16848](https://arxiv.org/abs/2512.16848)
- Cube-standard `Task.reset()` contract: cube-standard `openspec/specs/task/spec.md`
- Existing rollout-adjacent code: `src/cube_harness/episode.py`, `src/cube_harness/experiment.py`
- Recipe template to mirror: `recipes/hello_miniwob.py` (current MiniWoB reference recipe)
- Validation cube: `cubes/miniwob/src/miniwob_cube/task.py`
- Browser tool the cube uses: `cube_browser_tool.SyncPlaywrightTool` (installed via
  `cubes/miniwob/pyproject.toml`'s git source until 0.3.0 ships to PyPI)
