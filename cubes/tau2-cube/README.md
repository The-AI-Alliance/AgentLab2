# tau2-cube

Wraps [τ²-bench](https://github.com/sierra-research/tau2-bench) (Sierra's conversational
tool-use benchmark) as a CUBE benchmark. One catalog spans all four domains —
**mock**, **airline**, **retail**, **telecom** (288 tasks) — selectable via per-domain
subsets.

A tau-2 task is a customer-service conversation: the agent talks to a simulated
**user** (an LLM) and calls **domain tools** (operating on an in-memory database) to
satisfy the customer's request, following the domain policy.

## Architecture (what wraps what)

| tau2 piece | cube layer |
|---|---|
| `Environment` (DB + domain tools) | `CubeTool._env` — the shared seam |
| domain tools (`@is_tool`) | `CubeTool.action_set` — built dynamically from `env.get_tools()`, dispatched via `env.get_response` |
| `UserSimulator` | the synthetic `send_message_to_user` action |
| `evaluate_simulation` | `CubeTask.evaluate()` — scores the recorded conversation |
| `Task` (scenario, initial_state, criteria) | `CubeTaskMetadata.tau2_task` |
| `registry` | `CubeTool.reset()` + `benchmark.py` task loading |

We deliberately do **not** use tau2's own Agent or Orchestrator — cube-harness's
`Episode` is the driver. `CubeTool.execute_action` records every turn into
`self._messages` (tau2 `Message` format) so `evaluate()` can hand a `SimulationRun`
to tau2's real evaluator.

## Layout

```
src/tau2_cube/
├── tool.py        CubeTool / CubeToolConfig — domain-agnostic action surface + user-sim
├── task.py        CubeTask — reset (env + user session), evaluate (evaluate_simulation)
├── benchmark.py   CubeBenchmarkConfig — all domains, ids namespaced <domain>/<id>, named_subsets
└── debug.py       deterministic gold-action replay (+ STOP) for `cube test`
debug_cycle.py     dev scratch: walk one task through the full cycle (any domain), optional vLLM agent
```

## Running

```bash
uv pip install -e cubes/tau2-cube           # editable install (pulls tau2 from git)
cube test tau2-cube                         # debug compliance suite (mock, LLM-free)
```

Per-domain subset for training/eval:
```python
from tau2_cube.benchmark import CubeBenchmarkConfig
CubeBenchmarkConfig().named_subset("airline")   # or retail / telecom / mock
CubeBenchmarkConfig()                            # all 288 tasks
```

Agent + user-sim both route through LiteLLM. Point them at a local vLLM:
```python
LLMConfig(model_name="hosted_vllm/qwen-32b", tool_choice="required")   # agent — see note below
CubeToolConfig(user_llm="hosted_vllm/qwen-32b")                        # user simulator
# env: HOSTED_VLLM_API_BASE=http://localhost:8000/v1  HOSTED_VLLM_API_KEY=local-key
```

**`tool_choice="required"` is important:** tau-2 agents must *act* every turn — talk via
`send_message_to_user`, act via domain tools, end via `final_step`. With `"auto"` the
model can reply in plain text, which the harness drops (no tool call → episode stops).

## Scoring

`evaluate()` delegates to tau2's `evaluate_simulation` over the recorded conversation.
Reward = product of the checkers in the task's `reward_basis`:

| reward_basis component | checks |
|---|---|
| `DB` | final DB matches gold (gold = replay reference `actions`; `actions=[]` ⇒ expect *no* change) |
| `COMMUNICATE` | required substrings appear in the agent's messages |
| `NL_ASSERTION` | LLM-judge over the conversation |
| `ENV_ASSERTION` | environment-state assertions (telecom) |
| `ACTION` | specific tool calls were made |

## Known limitations / TODO

- **NL judge = `gpt-4.1`** (tau2's `DEFAULT_LLM_NL_ASSERTIONS`, hardcoded). Fires on tasks
  with `NL_ASSERTION` in `reward_basis` (≈ most retail). It is **not** the local vLLM —
  override `tau2.config.DEFAULT_LLM_NL_ASSERTIONS` for fully-local runs.
- **Dual-control** (telecom user-side tools): the user simulator is created with
  `tools=None`, so tasks needing user actions can't reach reward 1.0. See
  `CubeTool.start_user_session`.
- **`termination_reason` is always `AGENT_STOP`** in `evaluate()`. A run that maxes out
  without the agent stopping is still scored (vs tau2's strict 0 for premature end).
- **Degenerate tasks**: a few tasks have `actions=None` and empty `communicate_info`
  (trivially 1.0). Consider a "scorable" subset filter before RL training.
- **Initial message history**: tasks that start mid-conversation aren't pre-seeded into
  `self._messages` (rare; one-line fix in `reset()`).
