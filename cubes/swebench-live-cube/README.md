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

[`swebench_live_cube.gold_patch`](src/swebench_live_cube/gold_patch/) provides an oracle baseline that applies the gold patch from `task.execution_info` and calls `final_step`. Useful for sanity-checking the evaluation pipeline and identifying which tasks the upstream environment can actually resolve. Requires `cube-harness` on the path.

```bash
# Single gold pass over the lite subset:
.venv/bin/python -m swebench_live_cube.gold_patch.recipe --subset lite \
    --toolkit --eai-path ~/bin/eai --n-parallel 50

# Identify stable solvable subset across 3 runs:
.venv/bin/python -m swebench_live_cube.gold_patch.recipe --subset lite \
    --n-runs 3 --dump-solvable solvable_lite_stable.json \
    --toolkit --eai-path ~/bin/eai --n-parallel 50

# Post-hoc intersect already-completed run dirs (no fresh runs):
.venv/bin/python -m swebench_live_cube.gold_patch.recipe \
    --from-runs dir1 dir2 dir3 --dump-solvable solvable_lite_stable.json
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

Snapshots are recipe-level pinned subsets — they freeze a deterministic task list so two agents can be compared on the same ground truth without re-running the gold-patch oracle. Regenerate with:

```bash
.venv/bin/python -m swebench_live_cube.gold_patch.recipe --subset lite \
    --n-runs 3 --dump-solvable <new-snapshot-path>.json \
    --toolkit --eai-path ~/bin/eai --n-parallel 50
```

Then wrap the resulting bare list in the schema above and rename to encode the date.

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
