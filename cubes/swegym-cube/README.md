# swegym-cube

[SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) ported to the [CUBE](../../) protocol — the full **2438**-task SWE-Gym training set, with [SWE-Gym-Lite](https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Lite) (**230** tasks) as the official, score-comparable named subset.

## Overview

[SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) (ICML 2025) is a training environment of 2438 real-world Python software-engineering tasks collected with the SWE-bench methodology. The official 230-task **lite** subset (the upstream SWE-Gym-Lite split) is the score-comparable evaluation set; select it with `SWEGymBenchmarkConfig().named_subset("lite")`.

Each task gives the agent a GitHub issue inside a pre-built, executable Docker container (the published `xingyaoww/sweb.eval.x86_64.*` eval images). `SWEGymTask` uses `ContainerTerminalTool` from cube-standard for `bash` / `read_file` / `write_file` access into that container. Resolution requires **all** `fail_to_pass` tests to pass after the agent's patch, with `pass_to_pass` tests remaining green — the strict SWE-bench criterion.

> A small fraction (~3%) of full-set tasks have no published eval image; those fail loudly at docker-pull (never a silent score-0). The 230-task lite subset is fully imaged.

> Unlike SWE-bench Verified, SWE-Gym is a *training* split: it ships no human `difficulty` annotation, and every repo uses pytest (no Django/sympy bespoke runners).

## Prerequisites

- Docker daemon reachable from the harness (any backend — local Docker, Modal, Daytona, EAI Toolkit). The eval images are `linux/amd64`; on Apple Silicon run them under a Rosetta-enabled VM.
- Network access to HuggingFace for the one-time dataset download and Docker Hub for the images.
- For agent runs: an LLM provider key.

## Installation

```bash
uv pip install swegym-cube
cube install swegym-cube      # one-time: download dataset + populate execution cache
```

`install()` is idempotent.

## Usage

### Debug suite

Two oracle tasks exercise the full pipeline end-to-end via `cube test swegym-cube`:

- `getmoto__moto-5699`
- `iterative__dvc-5822`

Both run in `oracle_mode` (the gold patch is written to `/tmp/gold_patch.diff` in `reset()`), apply it, and assert `reward == 1.0`.

### Programmatic

```python
from swegym_cube import SWEGymBenchmarkConfig

cfg = SWEGymBenchmarkConfig(
    oracle_mode=False,      # if True, the gold patch is written to /tmp/gold_patch.diff in reset()
)

cfg.install()
bench = cfg.make(infra=...)          # defaults to LocalInfraConfig() (local Docker)
for task_cfg in cfg.get_task_configs():
    task = task_cfg.make(bench.runtime_context)
    obs, info = task.reset()
    # ... agent loop ...
    reward, eval_info = task.evaluate()
bench.close()
```

Use the official lite subset for score-comparable evaluation:

```python
cfg = SWEGymBenchmarkConfig().named_subset("lite")   # the official 230-task subset
```

For any other task subset, chain the standard `BenchmarkConfig` helpers, e.g.
`SWEGymBenchmarkConfig().subset_from_list(["getmoto__moto-5699", ...])`.

## Task-level features

- **`append_submission_instructions`** (default `True`) — appends submission instructions to the problem statement reminding the agent how to submit (`final_step` after `git diff > patch.txt && cat patch.txt`). Disable for raw-benchmark comparisons where the prompt must match upstream verbatim.
- **`oracle_mode`** — writes the gold patch to `/tmp/gold_patch.diff` so an oracle agent can apply it directly. Used by the debug suite.
- **non-root writability guard** — `reset()` fails loud with `IncompatibleInfraError` (never a silent reward-0) if the gold patch's target files aren't writable on a non-root infra; root-capable infras pay nothing.

## Evaluation

`SWEGymTask.evaluate()`:

1. Runs `pass_to_pass` on the unpatched tree to record a baseline (so a pre-existing environmental flake isn't charged to the agent).
2. Applies the upstream `test_patch`.
3. Runs `fail_to_pass` — **all** must pass (strict).
4. Runs `pass_to_pass` — must remain green (relaxed: exit-4 "no tests collected" tolerated for truncated test IDs; a post-patch failure that already failed on the baseline is not counted as a regression).
5. Returns `1.0` only if both checks pass; the info dict carries `fail_to_pass_passed`, `pass_to_pass_passed`, `pass_to_pass_baseline_passed`, and trimmed output (last 200 lines).

Every SWE-Gym repo uses pytest, so test directives run via a single `python -m pytest` invocation (no per-repo special-casing).

## Gold-patch baseline

[`swegym_cube.gold_patch`](src/swegym_cube/gold_patch/) provides an oracle baseline that applies the gold patch (written to `/tmp/gold_patch.diff` by `reset()` under `oracle_mode`) and calls `final_step`. Unlike the 2-task debug suite, it runs **all 2438 tasks** (or any subset, e.g. `.named_subset("lite")`) to sanity-check the evaluation pipeline and identify which tasks the environment can actually resolve. Requires `cube-harness` on the path.

[`recipe.py`](src/swegym_cube/gold_patch/recipe.py) is a standard declarative recipe — `run()` ships the generic CLI (`--experiment` picks infra, `--ray`/`--limit` control execution). Pick infra with `--experiment` (`default` = local Docker; `toolkit`/`daytona` if configured in `~/.cube/infra.py`). For a task subset, edit `bench = ...subset_from_list([...])` in the file.

```bash
# All 2438 tasks on local Docker (default), 8 Ray workers:
.venv/bin/python -m swegym_cube.gold_patch.recipe

# On EAI Toolkit with 50 workers:
.venv/bin/python -m swegym_cube.gold_patch.recipe --experiment toolkit --ray 50

# Quick smoke — first 3 tasks, in-process:
.venv/bin/python -m swegym_cube.gold_patch.recipe --experiment toolkit --limit 3
```

List which tasks resolved after a run (reward == 1.0):

```python
from swegym_cube.gold_patch import extract_solvable, intersect_solvable

extract_solvable(run_dir)              # resolved task IDs from one run
intersect_solvable([dir1, dir2, dir3]) # (stable, flaky) across repeated runs
```

## Regenerating task metadata

`task_metadata.json` (the shipped lightweight per-task metadata, including the resolved Docker image reference) is generated from the HuggingFace dataset:

```bash
.venv/bin/python scripts/create_task_metadata.py --force
```

## References

- Paper: [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139) (ICML 2025)
- Code: [github.com/SWE-Gym/SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)
- Dataset: [huggingface.co/datasets/SWE-Gym/SWE-Gym-Lite](https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Lite)
