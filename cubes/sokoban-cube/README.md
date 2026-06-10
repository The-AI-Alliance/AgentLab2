# sokoban-cube — CUBE benchmark template

This directory is the canonical starting point for a new CUBE benchmark package.
Copy it, rename things, and follow the TODOs in each file.

> **New to CUBE?** Read the [Authoring a CUBE guide](https://the-ai-alliance.github.io/cube-standard/authoring-a-cube) for the end-to-end walkthrough — interviewer skill, implementation order, validation, and publishing.

**If you copy-paste manually** (rather than using `cube init`), find every placeholder
that needs renaming:

```bash
grep -r "sokoban_cube\|sokoban-cube\|CubeTask\|CubeBenchmark\|CubeTool\|CubeEnv" src/ pyproject.toml
```

Replace all occurrences with names that match your benchmark.

## Quick start

```bash
# 1. Scaffold from the template
cube init my-bench        # copies _template/sokoban_cube and renames placeholders
cd my-bench

# 2. Install in editable mode
uv sync

# 3. Run the debug compliance suite
cube test my-bench        # resolves via cube.benchmarks entry-point
# or: cube test my_bench.debug
```

## File map

```
sokoban_cube/
├── pyproject.toml              ← package metadata & cube.benchmarks entry point
└── src/sokoban_cube/
    ├── __init__.py
    ├── benchmark.py            ← CubeBenchmark (registry, metadata, task list)
    ├── benchmark_metadata.json ← Option B: load benchmark metadata from JSON
    ├── task.py                 ← CubeTask + CubeTaskConfig (episode loop)
    ├── task_metadata.json      ← Option B: load task metadata from JSON
    ├── tool.py                 ← CubeTool + CubeToolConfig + @tool_action methods
    └── debug.py                ← deterministic agent for `cube test`
```

## The five layers

Work through the files in this order — each layer depends on the one above it:

| # | File | What to implement |
|---|------|-------------------|
| 1 | `tool.py` | Subclass `Tool`; add `@tool_action` methods; expose config via `CubeToolConfig` |
| 2 | `task.py` | `reset()` (opening observation) and `evaluate()` (reward); `finished()` is optional |
| 3 | `benchmark.py` | Fill `BenchmarkMetadata` and `task_metadata` (inline or via CSV/JSON); `_setup()` and `close()` to spinup / close resources |
| 4 | `debug.py` | One deterministic action sequence per task; must reach `reward == 1.0` |
| 5 | `pyproject.toml` | Update `name`, `description`, and the `cube.benchmarks` entry-point key |

See `examples/counter-cube/` in the cube-standard repo for a complete reference implementation covering all five layers.

## Checklist

- [ ] `tool.py` — add `@tool_action` methods; delete `example_action` placeholder
- [ ] `task.py` — implement `reset()` and `evaluate()`; optionally `finished()`
- [ ] `benchmark.py` — fill in `BenchmarkMetadata` and `task_metadata` (or switch to JSON/CSV files); implement `_setup()` and `close()`; optionally `install()` and `uninstall()`
- [ ] `debug.py` — add one entry to `_TASK_ACTIONS` per task
- [ ] `pyproject.toml` — update `name`, `description`, and the `cube.benchmarks` entry-point key
- [ ] Run `cube test <your-benchmark-name>` — all tasks must pass
- [ ] Run `/review-cube ./` in Claude Code for a pre-submission self-audit
- [ ] Submit to the registry with `cube registry add --submit`

## Key invariants

- Every `@tool_action` must return something `Content.from_data()` can wrap (str, dict, PIL Image, …).
- `evaluate()` must return `(reward: float, info: dict)` — `reward == 1.0` means solved.
- `TaskConfig` must be lightweight (no reference to TaskMetadata) and JSON-serializable: it travels over the network to workers
- `debug.py` action sequences must be deterministic and reach `reward == 1.0` — `cube test` enforces this.

## How `cube test` works

`cube test` accepts either the registered benchmark name (`my-bench`) or the
dotted module path (`my_bench.debug`).  When given a name it resolves it via
the `cube.benchmarks` entry-point group and automatically loads
`<package_root>.debug`.

The debug module must expose two callables:

| symbol | signature |
|---|---|
| `get_debug_benchmark()` | `() → Benchmark` |
| `make_debug_agent(task_id)` | `(str) → agent callable` |

See `debug.py` for the full template.
