# Agent Layer

**Module:** `cube_harness.agent`

## Purpose

`Agent` is the harness-level interface an LLM-driven decision-maker must implement.
It consumes an `Observation` and produces an `AgentOutput` containing the actions to
execute and the LLM calls made to arrive at them. `AgentConfig` is the serializable
factory.

## Public API

### `AgentConfig` (abstract, serializable)
```python
class AgentConfig(TypedBaseModel, ABC):
    description_overrides: dict[str, str] = {}   # action_name -> replacement description

    @abstractmethod
    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> Agent
```

Workers receive the config, deserialize, and call `.make(action_set)` with the task's
filtered action set.

`description_overrides` is an experiment-time knob for testing better action wording
without editing the tool: an agent applies `apply_description_overrides(...)` to its
encoded tool schemas, replacing the docstring-derived description for each named action
before the schema reaches the LLM. Keys must match an action in the current action set
(it raises otherwise — catching typos and stale keys after a rename). A proven override
graduates into the tool's docstring at the source via a PR.

### `Agent` (abstract)
```python
class Agent(ABC):
    name: str                          # identifier (react_agent, genny, etc.)
    description: str                   # one-line description
    input_content_types: list[str]     # e.g. ["image/png", "text/plain"]
    output_content_types: list[str]    # usually ["application/json"]

    def __init__(self, config: AgentConfig)

    @abstractmethod
    def step(self, obs: Observation) -> AgentOutput
```

`step()` semantics:
- Called by `Episode` once per turn.
- Returns an `AgentOutput` with `actions` to execute next. Empty `actions` (and no
  error) tells the episode loop to stop gracefully.
- Must attach every LLM call to `output.llm_calls` with a tag so traces are readable.
- Can surface thinking/reasoning via `output.thoughts`.
- Exceptions propagate: the episode loop wraps them in `StepError` and re-raises
  after saving the trajectory step.

## Concrete implementations (reference)

| Agent | File | Purpose |
|-------|------|---------|
| `ReactAgent` | `agents/react.py` | ReAct loop with tool calls, history compaction |
| `Genny` (genny) | `agents/genny.py` | Context-aware agent with rolling summaries, think/act pattern |
| `GenericAgent` (legacy) | `agents/legacy_generic_agent.py` | **Deprecated** XML-tag-based agent |

## Invariants

1. `AgentConfig.make(action_set)` must return an `Agent` subclass.
2. `Agent.step()` must not block indefinitely — the episode loop has a `max_steps`
   cap, but exceptions should surface promptly.
3. Every LLM call inside `step()` must be captured in `AgentOutput.llm_calls` with a
   tag. The XRay viewer and ADP export depend on this.

## Contracts for implementers

- If your agent holds state across steps (history, memory), initialize it in
  `__init__`. The episode keeps the same `Agent` instance for all turns.
- Inject `STOP_ACTION` into your tool list if the agent should be able to
  self-terminate — do not rely on the task to detect completion.
- Tag LLM calls: `LLMCall(tag="act", ...)` and `LLMCall(tag="summary", ...)` make
  traces readable.
- Prefer emitting actions as `Action(id=..., name=..., arguments=...)` with a
  stable `id` so logs correlate across env/agent steps.

## Gotchas

- `action_set` may be `None` (passed through from `AgentConfig.make()`). Agents that
  need actions for tool-call formatting must handle this case or declare it required.
- `parallel_tool_calls=False` in `LLMConfig` is the default — the LLM returns one
  tool call at a time. If your agent expects multiple actions per step, set it True
  and handle the list in your tool-call parser.
- The `legacy_generic_agent` is slated for removal. New agents should not depend on
  its prompt-building utilities (see `DEPRECATED.md`).
