# terminalbench2-cube

[Terminal-Bench 2](https://github.com/harbor-framework/terminal-bench-2) (Laude Institute / Harbor Framework) ported to the [CUBE](../../) protocol — **89** real-world terminal tasks (compile, debug, deploy, query, modernize) with pytest-based validation.

## Overview

Each task hands the agent a Linux shell pre-loaded with a project, asks for a concrete deliverable (a fixed bug, a passing test, a compiled binary, an inferred answer), and verifies the result by running an upstream `pytest` test suite the agent never sees. Tasks span 16 categories — data-processing, debugging, file-operations, scientific-computing, security, software-engineering, system-administration, and more — with `difficulty` ∈ {`easy`, `medium`, `hard`}.

`TerminalBench2Task` uses [`TerminalTool`](https://github.com/The-AI-Alliance/cube-standard/tree/main/src/cube/tools/terminal) from cube-standard for `bash` / `read_file` / `write_file` access into the per-task Docker container — no cube-specific tool subclass.

## Prerequisites

- Docker daemon reachable from the harness (any backend — local Docker, Modal, Daytona, EAI Toolkit).
- Network access to `github.com` for the one-time upstream clone at `install()` time.
- For agent runs: an LLM provider key (per [cube-harness](../../) recipe).

## Installation

```bash
uv pip install terminalbench2-cube
cube install terminalbench2-cube      # one-time: clone terminal-bench-2 and populate execution cache
```

`install()` is idempotent. The shipped `task_metadata.json` carries only lightweight public fields (id, difficulty, category, tags); task instructions and archives are downloaded to a per-task execution cache the first time.

## Usage

### Via recipe

```bash
# 2 oracle debug tasks (no LLM, sequential, local Docker):
.venv/bin/python recipes/swe_agent_recipe.py --benchmark tbench --debug

# Full 89-task run on a named Toolkit profile (declared in ~/.cube/infra.json):
.venv/bin/python recipes/swe_agent_recipe.py --benchmark tbench --infra yul101 --n-parallel 20

# Stable 40-task medium subset for iteration (proportional across categories):
.venv/bin/python recipes/swe_agent_recipe.py --benchmark tbench --subset tbench-iter-40 --infra yul101
```

Infra choice (Toolkit / Daytona / local) is driven by named profiles in
`~/.cube/infra.json` — see `cube_harness.infra_profile` for the schema. The
sidecar + uv bundle that used to require `--sidecar-data` / `--assets-data`
flags is now auto-provisioned by `ToolkitInfraConfig` (`cube_data="auto"`,
mounted at `/opt/cube/` on first launch).

### Programmatic

```python
from cube.tools.terminal import TerminalToolConfig
from terminalbench2_cube import TerminalBench2BenchmarkConfig

cfg = TerminalBench2BenchmarkConfig(
    tool_config=TerminalToolConfig(working_dir="/app", max_timeout=900, enable_file_actions=True),
    oracle_mode=False,      # if True, the gold solution is uploaded to /tmp/solution in reset()
)

cfg.install()
bench = cfg.make(infra=...)
for task_cfg in cfg.get_task_configs():
    task = bench.spawn(task_cfg)
    obs, info = task.reset()
    # ... agent loop ...
    reward, eval_info = task.evaluate()
    task.close()
bench.close()
```

## Subsets

`TerminalBench2BenchmarkConfig` ships named subsets keyed by `difficulty` and `category`. Filter via `subset_from_glob` (single key) or compose multiple subsets via `subset_from_list` with explicit task IDs. The iter recipe ships a fixed 40-task medium subset chosen proportionally across all 14 categories for reproducible iteration.

## Task-level features

- **`oracle_mode`** — uploads the gold solution to `/tmp/solution` so a scripted debug agent (or a "give-up" recipe) can apply it directly. Used by the debug suite.
- **`relocate_if_readonly`** — when `/app` is a read-only mount (some non-root backends like EAI Toolkit), the working directory relocates to `/tmp/app` and the agent's instruction text is rewritten in-place so prompts match the path the evaluator checks.
- **`_ensure_uv_preinstalled()`** — bootstraps a local `uv` binary inside the container so test.sh's `curl https://astral.sh/uv/.../install.sh | sh` line works on minimal images. Three fall-back paths: (1) **fast path** copying from `/opt/cube/` when `ToolkitInfraConfig` auto-mounts the cube_data bundle; (2) root `apt-get install python3 + pip install uv`; (3) non-root `apt-get download + dpkg-deb --extract` for images without root.

## Evaluation

`TerminalBench2Task.evaluate()` uploads the upstream pytest test harness into the container, runs `test.sh` (which writes a `reward.txt`), and returns the upstream-defined reward (`1.0` = all tests passed; `0.0` otherwise). Output is parsed via [`PytestParser`](src/terminalbench2_cube/pytest_parser.py) with regex fall-back for non-standard pytest envelopes.

`evaluate()` deliberately mutates container state (uploads test files, installs `uv`) — this is acceptable here because `validate_per_step=False` makes it a single terminal call.

## Debug suite

Two oracle tasks exercise the pipeline end-to-end via `cube test terminalbench2-cube`:

- `fix-git`
- `overfull-hbox`

Both apply the gold solution and assert `reward == 1.0`.

## Legal

- **Wrapper code** (this cube): Apache-2.0, inheriting from the [cube-harness root LICENSE](../../LICENSE.Apache-2.0).
- **Upstream benchmark**: Terminal-Bench 2 is licensed Apache-2.0. License URL: <https://github.com/harbor-framework/terminal-bench-2/blob/main/LICENSE>.
- **Task content**: instructions, archives, and test scripts are downloaded from the upstream repo at `install()` time — not redistributed in the cube wheel. The shipped `task_metadata.json` carries only structural metadata (task IDs, difficulty, category, tags).
- **No third-party software registration**, no live-website cloning, no proprietary data dependencies.

When submitting this cube to [cube-registry](https://github.com/The-AI-Alliance/cube-registry), the corresponding `legal:` block is:

```yaml
legal:
  wrapper_license: Apache-2.0
  benchmark_license:
    reported: Apache-2.0
    source_url: "https://github.com/harbor-framework/terminal-bench-2/blob/main/LICENSE"
    verified_by_original_authors: false
```

## References

- Upstream: <https://github.com/harbor-framework/terminal-bench-2> (formerly `laude-institute/terminal-bench-2` — GitHub redirects)
- Paper / project page: <https://www.tbench.ai/>
