# Forensic note: the 25 unresolved lite tasks

The 25 lite tasks NOT in [`lite_solvable_daytona_2026-05-25.json`](./lite_solvable_daytona_2026-05-25.json) did not score `reward == 1.0` under the unscoped gold-patch oracle on Daytona. Investigated in [PR #473](https://github.com/The-AI-Alliance/cube-harness/pull/473) (closed unmerged); this is the forensic summary so future investigators don't re-derive the same lessons.

## Root cause

The dataset's `test_cmds` typically runs the **whole upstream test suite** (e.g. `pytest keras -rA` → ~30 k tests), even though `evaluate()` only cares about a known list of `fail_to_pass` / `pass_to_pass` node IDs. For heavy repos that doesn't fit in `eval_timeout = 1800 s` — the post-fix `pytest` is SIGKILL'd before the `fail_to_pass` tests even run, and the gold patch silently scores 0 despite being correct.

## What was tried

A prototype `SWEBenchLiveBenchmarkConfig(scoped_eval=True)` rewrote each pytest `test_cmd` via `xargs` reading from a file of the specific f2p/p2p node IDs, with:

- a parser preserving env-var prefixes (placed BEFORE `xargs` so xargs inherits them), wrappers (`poetry run`, `uv run`, `python -m`), and every flag;
- an `_is_truncated_id` filter dropping dataset-truncated parametrized IDs (commas inside brackets, unbalanced quotes/parens — pytest's strict matching aborts the whole run on these);
- a chunked base64 file writer (Daytona's container `exec` arg cap is much tighter than ARG_MAX — ~30 KB silently fails, 16 KB chunks land) with per-chunk `[exit_code: N]` verification.

## Empirical recovery on the 25

### ✓ Resolved (6) — including 4 heavy timeout-bound tasks

| task | f2p | p2p_failed |
|---|---|---|
| `python-sdk-167` | 2/2 | 0/156 |
| `feast-5036` | 1/1 | 0/487 |
| **`conan-17514`** (heavy) | 4/4 | **0/4154** |
| **`keras-20389`** (heavy) | 1/1 | **0/7686** |
| **`keras-20443`** (heavy) | 1/1 | **0/7795** |
| **`keras-20534`** (heavy) | 1/1 | **0/7913** |

### ⚠ f2p PASSED but p2p flake-blocked (4) — near-recoveries

| task | f2p | p2p_failed |
|---|---|---|
| `kedro-4406` | 1/1 | 49/1591 |
| `pybamm-4816` | 8/8 | 27/1599 |
| `dspy-1801` | 5/5 | 1/197 |
| `helm-3467` | 1/1 | 1/290 |

In all four, the scoped post-fix run PASSED every f2p test — the gold patch is correct and the eval reached it. They miss only because the smaller (scoped) baseline didn't catch some pre-existing p2p flakes.

### ✗ f2p NOT passed (4) — orthogonal issues, not scoped_eval-fixable

| task | reason |
|---|---|
| `beets-5457` | real eval — network-dependent lyrics tests |
| `haystack-8489` | `hatch run test:unit` — non-pytest, falls back to unscoped path |
| `pybamm-4865` | `pytest -m unit` interaction with explicit node IDs |
| `wemake-3114` | gold patch's f2p genuinely doesn't pass in this env |

### Ø Daytona infra cut-off (11) — not given a fair eval

Mid-run a Daytona session-token / proxy outage hit `ContainerExecError: unauthorized: Bearer toke…` / `HTTPSConnectionPool(host='proxy.app.daytona.io')` for 7 tasks; 4 more were stuck `RUNNING` for ~2 h with no log activity and were killed. xarray×4, sympy, wemake×2, fast-f1, dvc, pyt-bot, pybamm-4644 all went down this way. Worth re-running on a stable Daytona window to see how many of these would have joined the resolved or near-miss buckets.

## Verdict

**~450 LOC + ongoing maintenance burden judged not worth the ~6 outright recoveries** out of ~2 k full / 300 lite. PR #473 closed.

## Most promising follow-up

The 4 f2p-passing near-misses are the strongest forward signal. A **much smaller** PR could land them by widening the pre-existing-p2p detection in `evaluate()` — e.g., for each task, also run the f2p test FILES in the baseline (not just the explicit p2p IDs), or accept a small flake tolerance (≤ 1 net p2p failure when no fresh `test_patch`-introduced regressions). This avoids all of the xargs / chunked-write / truncation machinery and would lift recovery to ~10/25 without the complexity.
