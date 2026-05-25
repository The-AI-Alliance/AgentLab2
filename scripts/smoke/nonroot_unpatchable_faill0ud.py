#!/usr/bin/env python3
"""Smoke: non-root-unpatchable SWE-bench tasks fail loud, not silent-0 (auto-fix #446).

Regression guard for the auto-fix(446) band-aid. On a non-root infra (EAI Toolkit,
uid 13011) some SWE-bench images ship root-owned package subdirs (e.g. psf/requests'
``/testbed/requests/``) that the writability normalisation in
``SWEBenchVerifiedTask._build_tool`` cannot reparent without root. Any patch — gold
or agent — to a file there dies with ``git apply: Permission denied``, and the task
would silently score a *correct* fix 0. The band-aid raises ``IncompatibleInfraError``
(terminal & non-retriable, episode.py) instead, scoped to the **gold patch's target
files** so unrelated root-owned vendored dirs (e.g. astropy's ``astropy/_erfa``) do
not false-positive.

This smoke asserts, on toolkit:
  1. ``psf__requests-1142`` (gold target ``requests/models.py`` in a root-owned dir)
     ends ``INVALID_CONFIG`` with ``error_type == "IncompatibleInfraError"`` — NOT a
     reward-0 completion, and NOT retried.
  2. ``astropy__astropy-12907`` (gold target ``astropy/modeling/`` is writable, even
     though ``astropy/_erfa`` is root-owned) resolves normally (reward 1.0) — no
     false-positive.

SKIP if ``eai`` is absent / not authed (safe in any CI).

Run from the cube-harness repo root with the venv:
    EAI_PROFILE=yul101 .venv/bin/python scripts/smoke/nonroot_unpatchable_faill0ud.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

_NAME = "nonroot_unpatchable_faill0ud"
_BLOCKED = "psf__requests-1142"  # gold target requests/models.py — root-owned dir
_CONTROL = "astropy__astropy-12907"  # gold target astropy/modeling/ — writable


def banner(status: str, reason: str = "") -> int:
    print(f"\nSMOKE {status}: {_NAME}" + (f" — {reason}" if reason else ""))
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def main() -> int:
    if shutil.which("eai") is None:
        return banner("SKIP", "eai not on PATH")

    # Imports after the SKIP gate so a CI box without the deps still skips cleanly.
    from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmarkConfig
    from swebench_verified_cube.gold_patch.agent import GoldPatchAgentConfig

    from cube_harness.exp_runner import run_with_ray
    from cube_harness.experiment import Experiment
    from cube_harness.infra import INFRA_CONFIGS
    from cube_harness.storage import FileStorage

    if "toolkit" not in INFRA_CONFIGS:
        return banner("SKIP", "no 'toolkit' infra configured in ~/.cube/infra.py")

    tmp = Path(tempfile.mkdtemp(prefix="smoke_faill0ud_"))
    bench = SWEBenchVerifiedBenchmarkConfig(oracle_mode=True).subset_from_list([_BLOCKED, _CONTROL])
    exp = Experiment(
        name=f"{_NAME}",
        agent_config=GoldPatchAgentConfig(),
        benchmark_config=bench,
        infra=INFRA_CONFIGS["toolkit"],
        max_steps=5,
        output_dir=tmp / "run",
    )
    try:
        run_with_ray(exp, n_cpus=2)
    except Exception as exc:  # a per-episode raise must not crash the batch
        return banner("FAIL", f"run_with_ray raised at batch level: {type(exc).__name__}: {exc}")

    store = FileStorage(exp.output_dir)
    statuses = {s.task_id: s for s in store.list_episode_statuses().values()}
    blocked = statuses.get(_BLOCKED)
    control = statuses.get(_CONTROL)
    if blocked is None or control is None:
        return banner("FAIL", f"missing episode status (blocked={blocked!r}, control={control!r})")

    # 1. blocked task must fail loud (IncompatibleInfraError), terminal & non-retriable.
    if blocked.error_type != "IncompatibleInfraError":
        return banner(
            "FAIL",
            f"{_BLOCKED}: expected error_type=IncompatibleInfraError (fail loud), got "
            f"status={blocked.status!r} error_type={blocked.error_type!r} reward={blocked.reward!r} "
            f"— a silent reward-0 here is the bug this guards against",
        )
    if blocked.status != "INVALID_CONFIG":
        return banner(
            "FAIL", f"{_BLOCKED}: IncompatibleInfraError should be terminal INVALID_CONFIG, got {blocked.status!r}"
        )
    if blocked.retry_count != 0:
        return banner("FAIL", f"{_BLOCKED}: should be non-retriable, but retry_count={blocked.retry_count}")

    # 2. control task must NOT false-positive — resolves normally.
    if not (control.status == "COMPLETED" and (control.reward or 0) >= 1.0):
        return banner(
            "FAIL",
            f"{_CONTROL}: false-positive — a writable task regressed to "
            f"status={control.status!r} reward={control.reward!r} (its gold target dir IS writable)",
        )

    shutil.rmtree(tmp, ignore_errors=True)
    return banner("OK", f"{_BLOCKED} → fail-loud INVALID_CONFIG/IncompatibleInfraError; {_CONTROL} → resolved 1.0")


if __name__ == "__main__":
    sys.exit(main())
