#!/usr/bin/env python3
"""Smoke: the SWE-bench-Live gold-patch recipe runs end-to-end on local Docker.

Mirrors the recipe's local experiment (`gold_patch.recipe._exp("local")`) but pins it
to one known gold-solvable task, then runs it with no LLM (GoldPatchAgent applies the
gold patch + calls final_step) and asserts the episode resolved (reward >= 1.0). This
exercises the real path a downstream user hits: Experiment + GoldPatchAgentConfig +
local Docker infra + SWEBenchLiveTask.evaluate.

Prerequisites:
  - Docker daemon reachable (honors DOCKER_HOST — e.g. `colima start`).
  - QEMU, which `LocalInfraConfig.install()` provisions on first use.
  - One-time `cube install swebench-live-cube` (downloads the task execution cache).
SKIPs (exit 2) if Docker or QEMU is unavailable, so it is safe to run anywhere.

Run from the cube-harness repo root with the venv:
    .venv/bin/python cubes/swebench-live-cube/scripts/smoke/gold_patch_recipe.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_NAME = "gold_patch_recipe"
# gitingest is a small repo (fast container) and is in the gold-solvable debug suite.
_TASK = "cyclotruc__gitingest-94"


def banner(status: str, reason: str = "") -> int:
    print(f"\nSMOKE {status}: {_NAME}" + (f" — {reason}" if reason else ""))
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def main() -> int:
    if shutil.which("docker") is None:
        return banner("SKIP", "docker not on PATH")
    try:
        info = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
    except Exception as exc:  # daemon hung / socket missing
        return banner("SKIP", f"`docker info` failed: {type(exc).__name__}: {exc}")
    if info.returncode != 0:
        return banner("SKIP", "docker daemon unreachable (set DOCKER_HOST? e.g. colima)")
    # LocalInfraConfig runs inside a QEMU microVM; without qemu its install() would try
    # to build it (and can fail), so skip rather than fail when it is absent.
    if shutil.which("qemu-system-x86_64") is None:
        return banner("SKIP", "qemu-system-x86_64 absent (LocalInfraConfig can't provision)")

    # Imports after the SKIP gate so a box without the deps still skips cleanly.
    from swebench_live_cube.benchmark import SWEBenchLiveBenchmarkConfig
    from swebench_live_cube.gold_patch.agent import GoldPatchAgentConfig

    from cube_harness.exp_runner import run_with_ray
    from cube_harness.experiment import Experiment
    from cube_harness.infra import INFRA_CONFIGS
    from cube_harness.storage import FileStorage

    tmp = Path(tempfile.mkdtemp(prefix="smoke_gold_live_"))
    # Mirrors gold_patch.recipe._exp("local"), narrowed to one task.
    exp = Experiment(
        name=_NAME,
        agent_config=GoldPatchAgentConfig(),
        benchmark_config=SWEBenchLiveBenchmarkConfig(oracle_mode=True).subset_from_list([_TASK]),
        infra=INFRA_CONFIGS["local"],
        max_steps=5,
        output_dir=tmp / "run",
    )
    try:
        run_with_ray(exp, n_cpus=1)
    except Exception as exc:  # a per-episode raise must not crash the batch
        return banner("FAIL", f"run_with_ray raised at batch level: {type(exc).__name__}: {exc}")

    store = FileStorage(exp.output_dir)
    statuses = {s.task_id: s for s in store.list_episode_statuses().values()}
    st = statuses.get(_TASK)
    if st is None:
        return banner("FAIL", f"no episode status for {_TASK} (cube install run?)")
    if not (st.status == "COMPLETED" and (st.reward or 0) >= 1.0):
        return banner(
            "FAIL",
            f"{_TASK}: expected COMPLETED reward>=1.0, got status={st.status!r} "
            f"reward={st.reward!r} error_type={st.error_type!r}",
        )

    shutil.rmtree(tmp, ignore_errors=True)
    return banner("OK", f"{_TASK} → gold patch applied, resolved reward {st.reward}")


if __name__ == "__main__":
    sys.exit(main())
