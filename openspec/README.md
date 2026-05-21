# OpenSpec — cube-harness

Living specifications for the cube-harness runtime layers. Read these before modifying
code in the corresponding layer — specs define the contracts, invariants, and gotchas
that aren't obvious from reading the source alone.

For the full OpenSpec workflow (how to sync specs, write delta proposals, and manage
breaking changes), see [cube-standard's openspec/README.md](https://github.com/The-AI-Alliance/cube-standard/blob/main/openspec/README.md).

## Layer index

| Layer | Source | Spec |
|-------|--------|------|
| Core types (Trajectory, AgentOutput) | `src/cube_harness/core.py` | [core/spec.md](specs/core/spec.md) |
| Agent | `src/cube_harness/agent.py` | [agent/spec.md](specs/agent/spec.md) |
| LLM | `src/cube_harness/llm.py` | [llm/spec.md](specs/llm/spec.md) |
| Episode | `src/cube_harness/episode.py` | [episode/spec.md](specs/episode/spec.md) |
| Experiment + runners | `src/cube_harness/experiment.py`, `exp_runner.py` | [experiment/spec.md](specs/experiment/spec.md) |
| Storage | `src/cube_harness/storage.py`, `summary.py` | [storage/spec.md](specs/storage/spec.md) |
| Metrics / OTel | `src/cube_harness/metrics/` | [metrics/spec.md](specs/metrics/spec.md) |
| XRay / Analyze | `src/cube_harness/analyze/` | [analyze/spec.md](specs/analyze/spec.md) |
| MCP server | `src/cube_harness/mcp/` | [mcp/spec.md](specs/mcp/spec.md) |
| EvalLog | `src/cube_harness/eval_log.py` | [eval_log/spec.md](specs/eval_log/spec.md) |

Cross-repo: cube-harness consumes cube-standard's contracts (`Task`, `Benchmark`, `Tool`,
`Resource`). When your change touches the base protocol, check cube-standard's spec first.

## Changes

Active proposals live in [`changes/`](changes/).  
Completed ones are archived in [`changes/archive/`](changes/archive/).
