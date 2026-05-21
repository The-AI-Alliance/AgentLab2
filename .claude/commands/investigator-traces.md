---
trigger: /investigator-traces
description: Post-hoc LLM investigator for cube-harness episode trajectories. Reads trajectory steps from an experiment or episode directory, produces structured Findings (blame attribution, hypothesis, evidence). Based on RFC #358.
---

# Trajectory Investigator

You are a post-hoc investigator for agent episodes. Your job is to read a trajectory, understand what the agent did, and produce a structured failure analysis.

## Input

The user will give you one of:
- **An experiment directory** — investigator all COMPLETED episodes (or failures only if `--failures-only` is specified)
- **One or more episode directories** — investigator those specific episodes
- **A `--n N` flag** — limit to N episodes (default: all)

Episode directories are named like `<task_id>_ep<N>/` and live under `<experiment_dir>/episodes/`.

## Step 1 — Decode the trajectory

Each episode directory contains a `steps/` folder with `NNN_obs.msgpack.zst` and `NNN_act.msgpack.zst` files. Use this script to extract a readable transcript:

```python
import zstandard, msgpack, json
from pathlib import Path

def decode_step(p: Path) -> dict:
    with open(p, 'rb') as f:
        data = zstandard.ZstdDecompressor().decompress(f.read())
    return msgpack.unpackb(data, raw=False)

def extract_transcript(episode_dir: Path) -> str:
    steps_dir = episode_dir / "steps"
    lines = []
    task_desc = None

    for step_file in sorted(steps_dir.iterdir()):
        obj = decode_step(step_file)
        output = obj.get("output", obj)  # handle both wrapped and unwrapped
        step_idx = int(step_file.name[:3])

        if "_obs" in step_file.name:
            obs = output.get("obs", output)
            contents = obs.get("contents", []) if isinstance(obs, dict) else []
            for c in contents:
                data = c.get("data", "") if isinstance(c, dict) else str(c)
                tool_call_id = c.get("tool_call_id") if isinstance(c, dict) else None
                if tool_call_id is None and task_desc is None:
                    task_desc = data  # first obs with no tool_call_id = task description
                    lines.append(f"[Step {step_idx}] TASK:\n{data[:2000]}")
                else:
                    lines.append(f"[Step {step_idx}] OBS (tool_call_id={tool_call_id}):\n{str(data)[:1500]}")

        elif "_act" in step_file.name:
            actions = output.get("actions", [])
            llm_calls = output.get("llm_calls", [])
            for llm_call in llm_calls:
                thinking = llm_call.get("thinking", "")
                if thinking:
                    lines.append(f"[Step {step_idx}] THINKING:\n{thinking[:800]}")
            for action in actions:
                name = action.get("name", "?")
                args = action.get("arguments", {})
                if name == "bash":
                    lines.append(f"[Step {step_idx}] ACTION bash:\n{args.get('command', '')[:800]}")
                elif name == "final_step":
                    lines.append(f"[Step {step_idx}] ACTION final_step (DONE)")
                else:
                    lines.append(f"[Step {step_idx}] ACTION {name}:\n{json.dumps(args, default=str)[:600]}")

    return "\n\n".join(lines)
```

Run with the project venv python (e.g. `.venv/bin/python3`). The `zstandard` and `msgpack` packages are installed as part of `cube-harness`.

> **For batch / non-interactive investigating**, prefer the Python module: `pip install 'cube-harness[investigator]'` then `ch-investigate <experiment_dir> [--sample 0.1 | --ids ...] [--summary]`. It calls Claude Code via the `claude-agent-sdk`, persists `Findings` into `episode_record.json`, and writes an aggregate `experiment_investigation_summary.json`. This slash command is for interactive ad-hoc deep-dives.

## Step 2 — Read episode metadata

```bash
python3 -c "
import json
m = json.load(open('<episode_dir>/episode.metadata.json'))
print('task_id:', m.get('metadata', {}).get('task_id') or m.get('task_id'))
print('reward:', m.get('reward_info', {}).get('reward'))
print('steps:', m.get('n_agent_steps'))
print('cost_usd:', m.get('cost_usd'))
"
```

## Step 3 — Investigator each episode

For each episode, read the transcript and produce a `Findings`. Think through the evidence carefully before committing to structured fields.

### Output schema

```json
{
  "task_id": "repo__repo-NNNN",
  "reward": 0.0,
  "analysis": "<multi-paragraph scratchpad — reason through what happened before filling fields below>",
  "outcome": "<success|success_lucky|almost|failure|should_have_been_rewarded>",
  "summary": "<1-3 sentences>",
  "primary_blame": "<see taxonomy>",
  "primary_blame_confidence": 0,
  "other_blames": [],
  "evidence": [
    {"step": 7, "quote": "exact excerpt from transcript"}
  ],
  "hypothesis": "<1-2 sentences: what change would most likely fix this class of failure>",
  "hypothesis_confidence": 0
}
```

### Outcome taxonomy

| Value | Meaning |
|---|---|
| `success` | Agent solved the task correctly (reward=1.0, clean approach) |
| `success_lucky` | reward=1.0 but agent reached it by accident or wrong approach |
| `almost` | Agent understood the task and had the right strategy; failed on a minor detail |
| `failure` | Task not solved |
| `should_have_been_rewarded` | Agent did the right thing but eval function was too brittle to accept it |

### Blame taxonomy

| Category | Use when |
|---|---|
| `task_unclear` | Task description is ambiguous, contradictory, or missing context the agent needed |
| `model_capability` | Agent understood the task but lacked reasoning ability or domain knowledge to solve it |
| `tool_failure` | Tool raised an exception or returned unexpected output — bug in the tool wrapper |
| `env_failure` | Container crash, network timeout, VM restart — outside agent/tool control |
| `agent_scaffolding` | System prompt design, budget limits, context management, or submission protocol caused the failure |
| `action_space_limited` | Agent couldn't complete the task because a required action doesn't exist in its tool set |
| `insufficient_observation` | Observation was missing crucial info (truncated output, pruned context) |
| `eval_brittle` | Agent produced a correct/acceptable solution but evaluator rejected it |
| `submission_format` | Agent reached correct solution but never called `final_step` or submitted wrong way |
| `none` | Clean success, or too ambiguous to attribute without speculation |

### Confidence scale (0–5)

| Score | Meaning |
|---|---|
| 5 | Certain — evidence is unambiguous |
| 4 | High — strong evidence, one minor alternative |
| 3 | Medium — plausible but another reading is credible |
| 2 | Low — best guess, thin evidence |
| 1 | Very low — mostly speculation |
| 0 | No basis |

### Hallucination rules

- `evidence` must contain actual verbatim quotes from the transcript when `primary_blame != "none"`
- `primary_blame` must be from the taxonomy — do not invent categories
- Write `analysis` first as a scratchpad; your structured fields must be consistent with it
- If the transcript is genuinely ambiguous, use `confidence=2` or lower and say so in `analysis`

## Step 4 — Output

Write results to stdout as a JSON array (one object per episode). Also write each individual result to `<episode_dir>/findings.json` for persistence.

Print a summary table at the end:

```
task_id                           outcome    blame              conf  hypothesis_conf
--------------------------------  ---------  -----------------  ----  ---------------
stanfordnlp__dspy-1609            failure    model_capability   4     3
wireservice__csvkit-1274          failure    agent_scaffolding  3     4
...

Blame distribution: model_capability=5, agent_scaffolding=3, eval_brittle=1
```

## SWE-bench specific context

For SWE-bench episodes:
- The agent is given a GitHub issue description as the task
- Tools available: `bash`, `read_file`, `write_file`, `final_step` (or bash-only variant)
- The sandbox is at `/testbed` with the repo already cloned
- Agent must modify source files (not tests) to fix the issue
- Eval applies the agent's patch and runs the test suite
- The fail-to-pass tests are **removed from the repo** before the agent runs — the agent cannot run them directly; green on existing tests does not mean the f2p tests will pass
- Common failure patterns: agent applies fix from issue description without probing the actual failure first; agent uses `--maxfail=1` or runs the full suite and hits an unrelated pre-existing failure before the target test; agent's fix is logically correct but uses the wrong API name (issue description vs. test expectation)
