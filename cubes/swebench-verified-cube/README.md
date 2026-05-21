# swebench-verified-cube

[SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) ported to the [CUBE](../../) protocol — 500 human-validated GitHub issues with test-based resolution criteria.

## Overview

SWE-bench Verified is Princeton + OpenAI's curated 500-task subset of [SWE-bench](https://github.com/princeton-nlp/SWE-bench) where every task was manually checked for:

- A clear, unambiguous problem statement.
- A reference test that actually validates the fix.
- A reproducible execution environment.

`SWEBenchVerifiedTask` uses `TerminalTool` from cube-standard for `bash` / `read_file` / `write_file` access into the per-task Docker container. Resolution requires **all** `fail_to_pass` tests to pass after the patch, with `pass_to_pass` tests remaining green — the strict SWE-bench criterion.

## Prerequisites

- Docker daemon reachable from the harness (any backend — local Docker, Modal, Daytona, EAI Toolkit).
- Network access to HuggingFace for the one-time dataset download.
- For agent runs: an LLM provider key.

## Installation

```bash
uv pip install swebench-verified-cube
cube install swebench-verified-cube      # one-time: download dataset + populate execution cache
```

`install()` is idempotent.

## Usage

### Via recipe

```bash
# 2 oracle debug tasks (no LLM, sequential):
.venv/bin/python recipes/swe_agent_recipe.py --debug

# Full 500-task run on EAI Toolkit:
.venv/bin/python recipes/swe_agent_recipe.py --toolkit --eai-path ~/bin/eai --n-parallel 20

# Princeton HAL-50 subset (25 Django + 25 Sphinx):
.venv/bin/python recipes/swe_agent_recipe.py --subset hal_mini \
    --toolkit --eai-path ~/bin/eai --n-parallel 20
```

### Programmatic

```python
from cube.tools.terminal import TerminalToolConfig
from swebench_verified_cube import SWEBenchVerifiedBenchmarkConfig

cfg = SWEBenchVerifiedBenchmarkConfig(
    tool_config=TerminalToolConfig(working_dir="/testbed", enable_file_actions=True),
    include_hints=False,    # if True, hints_text is appended to the problem statement
    oracle_mode=False,      # if True, the gold patch is written to /tmp/gold_patch.diff in reset()
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

## Task-level features

- **`append_submission_instructions`** (default `True`) — appends `_TASK_INSTRUCTIONS` to the problem statement reminding the agent how to submit (`final_step` after `git diff > patch.txt && cat patch.txt`). Disable for raw-benchmark comparisons where the prompt must match upstream verbatim.
- **`include_hints`** — surface the curated hints text shipped with each task (when present).
- **`oracle_mode`** — writes the gold patch to `/tmp/gold_patch.diff` so an oracle agent can apply it directly. Used by the debug suite.
- **`filter_actions()`** — patches `STOP_ACTION`'s empty parameters schema to `{"type": "object", "properties": {}}` so Anthropic models don't reject it.
- **`_build_tool()`** — pre-flight `git config --global --add safe.directory` plus a writable-file `cp + mv` pass so non-root containers can edit the testbed.

## Evaluation

`SWEBenchVerifiedTask.evaluate()`:

1. Applies the upstream `test_patch`.
2. Runs `fail_to_pass` — **all** must pass (strict).
3. Runs `pass_to_pass` — must remain green (relaxed: exit-4 "no tests collected" tolerated for truncated test IDs; non-zero exit with zero failures tolerated for sympy import-deprecation noise).
4. Returns `1.0` only if both checks pass; info dict carries `fail_to_pass_passed`, `pass_to_pass_passed`, and trimmed output (last 200 lines).

The Django runtests.py path uses [`_normalize_django_directive`](src/swebench_verified_cube/task.py) to convert SWE-bench's unittest verbose format (`test_foo (module.ClassName)`) into Django's expected dotted form (`module.ClassName.test_foo`), with a None-on-malformed contract that downstream filtering relies on.

## Debug suite

Two oracle tasks exercise the full pipeline end-to-end via `cube test swebench-verified-cube`:

- `django__django-11099`
- `astropy__astropy-12907`

Both apply the gold patch and assert `reward == 1.0`.

## References

- Upstream: [github.com/princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench)
- Verified subset: <https://openai.com/index/introducing-swe-bench-verified/>
