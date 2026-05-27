# swebench-live-cube

[SWE-bench Live](https://swe-bench-live.github.io/) ported to the [CUBE](../../) protocol — **1,895** continuously-updated, contamination-resistant GitHub issue resolution tasks across many open-source repos.

## Overview

Each task is a real GitHub issue paired with its merged fix. The agent gets the problem statement plus a git checkout at the base commit and must produce a patch that makes the upstream `fail_to_pass` tests pass without breaking `pass_to_pass`. Unlike SWE-bench Verified (a fixed 500-task snapshot of pre-2024 issues), SWE-bench Live keeps refreshing the task pool, which makes it useful for testing contamination resistance.

`SWEBenchLiveTask` uses `TerminalTool` from cube-standard for `bash` / `read_file` / `write_file` access into the per-task Docker container; evaluation runs the upstream `test_cmds` and reports resolution if **at least one** `fail_to_pass` test passes (Linux-only convention).

## Prerequisites

- Docker daemon reachable from the harness (any backend works — local Docker, Modal, Daytona, EAI Toolkit).
- Network access to HuggingFace for the one-time dataset download (cached under `~/.cube/swebench-live-cube/`).
- For agent runs: an LLM provider key (anything LiteLLM speaks).

## Installation

```bash
uv pip install swebench-live-cube
cube install swebench-live-cube      # one-time: download splits, populate per-task execution cache
```

Re-running `install` is idempotent — it skips when the cache is already populated.

## Usage

### Via recipe (full evaluation run)

The unified SWE recipe handles both Verified and Live. See [`recipes/swe_agent_recipe.py`](../../recipes/swe_agent_recipe.py).

```bash
# 2 oracle debug tasks, sequential, no LLM:
.venv/bin/python recipes/swe_agent_recipe.py --benchmark live --debug

# SWE-bench Live golden 30 on EAI Toolkit:
.venv/bin/python recipes/swe_agent_recipe.py --benchmark live --subset live-golden-30 \
    --toolkit --eai-path ~/bin/eai --n-parallel 30

# Full 'lite' subset (300 tasks):
.venv/bin/python recipes/swe_agent_recipe.py --benchmark live --subset lite \
    --toolkit --eai-path ~/bin/eai --n-parallel 50
```

Named subsets: `solvable-lite` (223 gold-confirmed snapshot — see `src/swebench_live_cube/lite_solvable_<date>.json`), `live-golden-30` (30 confirmed-solvable), `lite` (300), `verified` (499 Linux-runnable), `full` (1,895), `test`.

### Programmatic

```python
from cube.tools.terminal import TerminalToolConfig
from swebench_live_cube import SWEBenchLiveBenchmarkConfig

cfg = SWEBenchLiveBenchmarkConfig(
    tool_config=TerminalToolConfig(working_dir="/testbed", enable_file_actions=True),
    oracle_mode=False,
).named_subset("lite")

cfg.install()                     # one-time
bench = cfg.make(infra=...)       # see cube-harness recipes for infra wiring
for task_cfg in cfg.get_task_configs():
    task = bench.spawn(task_cfg)
    obs, info = task.reset()
    # ... agent loop ...
    reward, eval_info = task.evaluate()
    task.close()
bench.close()
```

## Gold-patch baseline

[`swebench_live_cube.gold_patch`](src/swebench_live_cube/gold_patch/) provides an oracle baseline that applies the gold patch from `task.execution_info` and calls `final_step`. Useful for sanity-checking the evaluation pipeline and identifying which tasks the upstream environment can actually resolve. SWE-bench-Live images assume `USER root`, so the non-root EAI Toolkit scores 0 on tasks needing root; **Daytona (root) resolves them** (275/300 on the lite subset vs 223/300 on Toolkit). Requires `cube-harness` on the path.

[`recipe.py`](src/swebench_live_cube/gold_patch/recipe.py) is a standard declarative recipe — `run()` ships the generic CLI (`--experiment` picks infra, `--ray`/`--limit` control execution). Because Live needs root, **`--experiment` defaults to `daytona`**; `local` (local Docker) and `toolkit` stay selectable for smokes and the root-vs-non-root comparison. Each infra needs a `~/.cube/infra.py` entry. For a different task subset, edit `bench = ...` in the file.

```bash
# lite subset (300 tasks) on Daytona (root) — the default, 20 Ray workers:
.venv/bin/python -m swebench_live_cube.gold_patch.recipe --ray 20

# On EAI Toolkit with 50 workers (non-root — misses root-needing tasks):
.venv/bin/python -m swebench_live_cube.gold_patch.recipe --experiment toolkit --ray 50

# Quick smoke — first 3 tasks, in-process, local Docker:
.venv/bin/python -m swebench_live_cube.gold_patch.recipe --experiment local --limit 3
```

(SWE-bench-Live images are large; if Daytona launches time out, raise `launch_timeout_seconds` on the `DaytonaInfraConfig` in `~/.cube/infra.py`.)

List which tasks resolved after a run (reward == 1.0):

```python
from swebench_live_cube.gold_patch import extract_solvable, intersect_solvable

extract_solvable(run_dir)              # resolved task IDs from one run
intersect_solvable([dir1, dir2, dir3]) # (stable, flaky) across repeated runs
```

## Solvable snapshots

A solvable snapshot is a JSON file shipped inside the package that records which task IDs resolved (reward == 1.0) under the gold-patch oracle on a given source subset, on a given date. Filename convention: `<subset>_solvable_<YYYY-MM-DD>.json`. Schema:

```json
{
  "date": "2026-05-12",
  "source_set": "swe-bench-live/lite",
  "description": "...",
  "n_tasks": 223,
  "task_ids": ["..."]
}
```

Snapshots are recipe-level pinned subsets — they freeze a deterministic task list so two agents can be compared on the same ground truth without re-running the gold-patch oracle. To regenerate, run the gold-patch recipe (see [Gold-patch baseline](#gold-patch-baseline)) one or more times, then extract the resolved IDs post-hoc:

```python
from swebench_live_cube.gold_patch import extract_solvable, intersect_solvable

task_ids = extract_solvable(run_dir)                  # one run
task_ids, _flaky = intersect_solvable([d1, d2, d3])   # stable across N runs
```

Then wrap the resulting list in the schema above and rename to encode the date (and add the new filename to `[tool.uv-build] include` in `pyproject.toml` so it ships in the wheel).

## Evaluation

`SWEBenchLiveTask.evaluate()`:

1. Captures the baseline `test_cmds` run (some `pass_to_pass` tests may already fail in the base image — they're excluded so the agent isn't penalised for upstream flakes).
2. Applies the upstream `test_patch` (the patch that gates resolution).
3. Re-runs `test_cmds`, parses output with the task's declared `log_parser`.
4. Returns `1.0` if **at least one** `fail_to_pass` test now passes and no previously-passing `pass_to_pass` test regresses; `0.0` otherwise. The info dict carries `fail_to_pass_passed`, `pass_to_pass_passed`, and trimmed raw output.

The Linux-only "at-least-one" criterion matches the upstream SWE-bench Live convention and is more permissive than SWE-bench Verified's "all fail_to_pass must pass".

## Debug suite

Two oracle tasks exercise the full pipeline end-to-end via `cube test swebench-live-cube`:

- `cyclotruc__gitingest-94`
- `dynaconf__dynaconf-1241`

Both apply the gold patch and assert `reward == 1.0`.

## Regenerating `task_metadata.json`

The shipped `task_metadata.json` is generated by [`scripts/create_task_metadata.py`](scripts/create_task_metadata.py). It pulls every split from HuggingFace, normalises log-parser names, and writes ~1 KB/task of public metadata. Heavy per-task data (problem statements, patches, test patches) lives in the execution cache populated by `BenchmarkConfig.install()` and never lands in the shipped wheel.

## References

- Upstream: [github.com/SWE-bench-Live/SWE-bench-Live](https://github.com/SWE-bench-Live/SWE-bench-Live)
- Project page: <https://swe-bench-live.github.io/>
