# Cluster results — CUBE NeurIPS 2026 rebuttal

**Cluster:** Mila (SLURM, `login-4.server.mila.quebec`). **Run by:** coding agent, 2026-07-26.
**Repo:** `/network/scratch/o/omar.younis/cube-harness` @ `paper/neurips26-rebuttal` (3260209).
**All artifacts:** `/network/scratch/o/omar.younis/cube_rebuttal/`

> **Status: IN PROGRESS.** This file is updated as jobs land. Sections marked
> *pending* have not produced numbers yet.

---

## 0. Environment — what had to change before any job could run

This is the single biggest caveat on everything below, so it comes first.

**Mila has no Docker.** Not on the login node, not on compute nodes, and there is
no Docker daemon or socket anywhere. J1, J2 and J5 all hard-require a container
runtime. The workaround was to make **rootless podman** (5.2.2, present on compute
nodes with `crun`) impersonate Docker. Three distinct problems had to be solved:

| Problem | Symptom | Fix |
|---|---|---|
| podman's default graphroot is `$HOME` (Lustre) | `configure storage: a network file system with user namespaces is not supported` | pin `--root`/`--runroot` to node-local `/tmp` (ext3, ~7 TB free) |
| account has **no `subuid`/`subgid` ranges** → rootless single-UID mapping | every real image fails to unpack: `potentially insufficient UIDs or GIDs available in user namespace (requested 0:42 for /etc/gshadow)` | `--storage-opt overlay.ignore_chown_errors=true` |
| cube-standard drives Docker **two ways** — the docker-py SDK *and* shell-outs to a `docker` binary (`pull`/`ps`/`stop`/`rm`; `shutil.which("docker")` also gates the `docker` + `container:root` capabilities) | `docker` not found → zero capabilities → benchmark refuses to run | a `docker`→`podman` shim on `PATH`, pinned to the same store as `podman system service`, with `DOCKER_HOST` pointing at that socket |

Setup script: `/network/scratch/o/omar.younis/cube_rebuttal/scratch/podman_env.sh`.

After this, `LocalInfraConfig.capabilities()` reports
`['container:cgroupns-host', 'container:privileged', 'container:root', 'docker']`
— i.e. **containers are root-capable**, which per the runbook is the regime that
produced 275/300 rather than 223/300 on SWE-bench Live.

**Credit where due:** cube-standard's `_make_docker_client` already normalises
Podman's `http+unix://` socket, so the SDK half needed no changes. Only the
CLI-shell-out half needed the shim.

### A real bug in `--set` (fixed; `src/cube_harness/recipe.py`)

**Both J1 and J2 died on their first launch** with a bare
`TypeError: unsupported operand type(s) for /: 'str' and 'str'`, thrown deep in
`exp_runner._experiment_lifecycle` at `exp_dir / EXPERIMENT_STATUS_FILENAME` —
*after* the benchmark had been built and all 500 episode configs written.

Cause: `--set output_dir=/some/path` stored a **`str`** in a field annotated
`Path | None`. `_apply_override` did a bare `setattr`, and its docstring's claim
that "`ValidatedConfig` validates the type" is **false for `Experiment`**, which
subclasses `TypedBaseModel` — `validate_assignment` is not enabled
(`Experiment.model_config.get("validate_assignment")` is `None`). So every
`--set` value was stored uncoerced, and any type error surfaced far from its
origin, or not at all.

Fix: `_apply_override` now coerces each value through
`TypeAdapter(field.annotation).validate_python(...)` and raises
`typer.BadParameter` at parse time for a bad value or an unknown field.
Full test suite passes (1071 passed, 26 skipped).

**Design note for the maintainers (not fixed here):** the deeper issue is that
`Experiment` is mutated after construction — recipes do exactly this
(`agent.llm_config = ...`) — yet subclasses `TypedBaseModel` rather than
`ValidatedConfig`, whose docstring says *"Subclass this instead of
`TypedBaseModel` for any config a user mutates after construction."* Switching
it would catch this class of bug for every mutation path, not just `--set`. That
is a core change that ripples into every cube and recipe, so it belongs in a
change proposal rather than a rebuttal-week patch.

### A near-miss that would have silently shrunk J1's denominator

Worth recording because it is precisely the failure the runbook warns about, and
because it produced a *green-looking* log.

The pre-puller wrote its progress to `prepull_state.json` on **shared scratch**,
while the podman image store is **node-local `/tmp`**. The first successful pull
of all 500 images happened on `cn-h002`. When the job was later resubmitted and
landed on `cn-h001`, the puller read that shared state, concluded every image was
already fetched, and reported:

```
[prepull] 0 already local, 0 to pull
[prepull] COMPLETE ok=500 failed=0 elapsed=0m
```

The node in fact had **zero** images (`docker images` → 1 line, the header). The
oracle then began 500 tasks against an empty cache, so every task would have
pulled on demand through cube-standard's `_docker_pull` — 6 attempts, ~15 min of
backoff — across 24 concurrent Ray workers. Registry stalls would have been
recorded as *task* failures, and the resolved-rate would have been computed over
a silently reduced denominator.

Caught by noticing `elapsed=0m` on a node that could not have had the images, and
the run was cancelled ~2 minutes in. Fixes: the state file now lives at
`/tmp/cube-podman/prepull_state.json` (with the store it describes); the pull
list is derived from the **actual local store** rather than from bookkeeping; and
the puller ends by re-reading the store and exiting non-zero with an explicit
`MISSING <image>` line per absent image. No J1 numbers were produced from the
affected run.

### Two runbook corrections

1. `--experiment local` is wrong. The gold-patch recipe maps the local infra to
   the key `default` (`{"default" if name == "local" else name: ...}`), so the
   command is `--experiment default`. `--experiment local` exits with
   `Invalid value: --experiment 'local' not in ['default']`.
2. **A required install step is missing.** Without
   `SWEBenchVerifiedBenchmarkConfig.install()` (equivalently
   `cube install swebench-verified-cube`) *every* episode dies with
   `SWE-bench Verified per-task execution cache is empty at
   ~/.cube/swebench-verified-cube/tasks_execution_info`. It writes 500 JSON
   files and must run before any oracle run.

### Cluster limits worth knowing for future runs

- Per-user QOS caps make the runbook's `--ray 32` impossible on the obvious
  partitions: `main-cpu` allows **cpu=8, mem=64G per user**, `main` allows
  **cpu=8, gpu=2, mem=48G**. The `long`/`long-cpu` partitions have **no
  per-user TRES cap** (7-day limit) and are what everything below uses.
- `long`/`long-cpu` are **preemptible** (`PreemptMode=REQUEUE`). Long runs are
  therefore launched with a pinned `output_dir` (8-hex suffix, so
  `_ensure_unique_output_dir` leaves it intact) plus `resume=true`, so a requeue
  resumes instead of restarting 500 tasks.
- Docker Hub rate-limits anonymous pulls to **100/hour** on Mila's shared NAT IP
  (`64.15.78.148`). Observed sustained pull rate was nonetheless ~1000/h, so the
  limit did not bind in practice; images are pre-pulled in a separate phase
  regardless, so that a rate-limit stall can never be miscounted as a task failure.
  **All 500 SWE-bench Verified images pulled successfully (ok=500, failed=0) in
  109 min**, occupying ~120 GB on node-local disk after overlay layer sharing.
- **Do not put a Python venv used by multi-process jobs on `/network/scratch`
  (BeeGFS).** J3's first launch died with
  `FileNotFoundError: .../transformers/models/bert_generation/tokenization_bert_generation.py`
  — for a file that exists and imports fine. `transformers` 5.x walks its entire
  `models/` tree at import (2275 `.py` files via
  `create_import_structure_from_path`), and with `tensor-parallel-size=4` that
  walk runs in four worker processes at once; BeeGFS metadata does not stay
  consistent under it. Fix: `scratch/stage_vllm.sh` copies the 7.5 GB venv to
  node-local `/tmp` per job. Model weights stay on scratch (few large sequential
  reads, which BeeGFS handles fine). The venv must then be invoked as
  `$VLLM_PY -m vllm.entrypoints.openai.api_server`, **not** via the `vllm`
  console script, whose shebang hard-codes the original scratch interpreter and
  would silently re-run the networked copy.

---

## J1 — SWE-bench Verified gold-patch oracle

Status: **pass 1 complete** (pass 2 queued for the stable-vs-flaky split)

Run dirs: `/network/scratch/o/omar.younis/cube_rebuttal/results/j1_oracle_pass1_a1b2c3d4`
(pass 2, queued: `.../results/j1_oracle_pass2_b2c3d4e5`, pinned to `cn-h001` to reuse
that node's image store)
Infra: `local` (rootless podman as above), Mila `long-cpu`, 32 CPU / 128 GB, 24 Ray workers,
**root-capable containers: yes**
Model: none (oracle applies the gold patch; no LLM)
Wall clock: 194 min image pre-pull (501 images, 258 GB) + 4 h 27 min evaluation

### Smoke (pre-flight)

3/3 tasks resolved, avg final reward **1.0000**, 0 failed
(`results/20260726_070930_..._f9071d6f`). Confirms the podman bridge, the gold-patch
apply path, test execution and reward computation all work end to end.

### Numbers

| | |
|---|---|
| Attempted | **500** |
| Completed | **500** (no episode failed to run) |
| Resolved | **491** |
| Oracle parity | **98.2 %** |

LaTeX row (`scripts/paper/oracle_parity.py`): `SWE-bench Verified & 500 & 491 & 98.2\% \\`
Solvable-id list: `cube_rebuttal/solvable_verified.json`

**The score field was verified, not trusted.** `oracle_parity.py` reads
`EpisodeRecord.score`, a different code path from the `reward_info` that J2's export
uses — and that field turned out to be a trap there (see J2). All 500 episodes were
therefore cross-checked against the terminal evaluation event's `resolved` flag:
**0 mismatches**.

### What broke — all 9 unresolved instances

Every one has a non-empty `model_patch`, so the gold patch applied in all cases;
these are test-outcome failures, not infrastructure crashes.

| Instance | Cause | Class |
|---|---|---|
| `pylint-dev__pylint-4661` | `ModuleNotFoundError: No module named 'appdirs'` — conftest import fails | image env |
| `astropy__astropy-8707` | pytest raises `PytestRemovedIn8Warning` as an error (nose-style `setup(self)`) | image env |
| `astropy__astropy-8872` | collection error: `distutils Version` `DeprecationWarning` raised as error | image env |
| `django__django-13821` | **PASS_TO_PASS suite timed out** (FAIL_TO_PASS passed) | infra / timeout |
| `django__django-11532` | `AttributeError: _fqdn` at `delattr(DNS_NAME, '_fqdn')` — cached value never populated when only the F2P test runs | test isolation |
| `astropy__astropy-13398` | `TypeError: unsupported operand type(s) for -: 'Time' and 'float'` at `astropy/utils/iers/iers.py:271`; also the only instance with pre-existing P2P failures (`pass_to_pass_baseline_passed=False`) | test failure |
| `sphinx-doc__sphinx-9229` | `test_class_alias_having_doccomment`: `assert [] == ['', '.. py:a...']`; autodoc fails to import `classes.OtherAlias` | test failure |
| `sympy__sympy-12481` | `RecursionError: maximum recursion depth exceeded`; 42 exceptions in `test_args.py` | test failure |
| `sympy__sympy-13877` | `test_refine`: `refine(det(A), Q.orthogonal(A)) == 1` fails | test failure |

Six of nine (image env, timeout, test isolation) are properties of the upstream
task images and of running the specified test subset — not of CUBE's evaluator.

### Caveats

- **Single pass.** `django__django-13821` failed on a test-suite *timeout*, which is
  by definition non-deterministic; a task that resolves in one pass and not another is
  an infrastructure signal, not an evaluator signal. Pass 2 is queued to separate the
  stable set from the flaky one. **Until it lands, 98.2 % is a one-pass number and the
  9 failures are not all established as stable.**
- The four "test failure" rows are reported as observed. They have not been traced to
  a root cause in CUBE's evaluator, and no attempt was made to make them pass —
  per the runbook, a disagreement is the result.

---

## J2 — Cross-harness regrade

Status: **blocked on credentials**

No LLM API key exists on this cluster (no `.env`; only `WANDB_API_KEY` is set), and
J2 requires a *fresh* agent run because `model_patch` capture is new in this branch.
Everything else is staged and verified:

- upstream harness installed and importable: `swebench 4.1.0` in
  `/network/scratch/o/omar.younis/cube_rebuttal/swebench-venv`
- agent recipe for a deterministic 100-task subset:
  `recipes/paper/j2_swe_agent_subset.py` (round-robin across all 12 repos, ~10 each;
  ids recorded to `cube_rebuttal/j2_subset_ids.json`)
- job script ready: `cube_rebuttal/scratch/j2_run.sh` (agent run → export → upstream
  `run_evaluation` → `compare`)

**To unblock:** put `OPENAI_API_KEY=...` in `/network/scratch/o/omar.younis/cube-harness/.env`.

### Caveats (already known)

The 100-task subset equalises repos rather than reproducing SWE-bench Verified's
django-heavy mix. This maximises coverage of distinct test harnesses (where the
three documented relaxations bite) but means **the subset's resolve rate is not
comparable to a published SWE-bench Verified score.**

---

## J3 — Open-weight RL rollout

Status: **queued** (start ~09:05, waiting on 4×L40S)

Infra: Mila `long`, 4×L40S (48 GB each), 16 CPU / 128 GB
Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`, revision **`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`**
Server: vLLM **0.26.0**, torch 2.11.0+cu130, TP=4, `--max-model-len 32768`

### Numbers

*pending*

---

## J4 — Open-weight MiniWoB matrix cells

Status: **not started** (runs after J3; own vLLM server)

---

## J5 — Terminal-Bench budget lift

Status: **not attempted** (optional; gated behind J1–J4, and also needs an LLM key)

---

## Files changed in the working tree

- `recipes/paper/j2_swe_agent_subset.py` — **new**, J2's 100-task subset recipe.
- `paper/neurips26/cluster/RESULTS.md` — this file.

No commits, no pushes. Everything else lives outside the repo in
`/network/scratch/o/omar.younis/cube_rebuttal/`.
