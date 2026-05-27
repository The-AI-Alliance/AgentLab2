#!/usr/bin/env python3
"""Smoke: the startup GC reaps a dead run's orphaned toolkit jobs by run_id (auto-fix #206).

End-to-end across repos (toolkit `CUBE_RUN_ID` tagging + harness `_reap_dead_runs`):

  1. Set `CUBE_RUN_ID=<dead>` (what `_experiment_lifecycle` does) and launch a real eai
     job → the toolkit tags it `cube_run_id=<dead>`.
  2. Simulate a crashed/slept client: drop the handle (no close) and write a stale
     `RUNNING` experiment_status.json carrying `<dead>` — i.e. an orphaned job whose
     owner's heartbeat is long gone.
  3. A *new* run's `_reap_dead_runs(...)` must reap that orphaned job — and leave a
     concurrently-launched LIVE run's job (fresh heartbeat) alone.

SKIP if `eai` is absent / not authed.

Run from the cube-harness repo root with the venv (cube-standard must carry the
CUBE_RUN_ID change — cube-standard#209):
    EAI_PROFILE=yul101 .venv/bin/python scripts/smoke/experiment_reap_on_death.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from cube.resource import DockerServiceConfig
from cube_infra_toolkit.toolkit import ToolkitInfraConfig

from cube_harness.exp_runner import _DEAD_RUN_STALE_AFTER_S, _reap_dead_runs
from cube_harness.experiment_status import ExperimentStatus

_NAME = "experiment_reap_on_death"
_ACTIVE = {"queued", "queuing", "running"}


def banner(status: str, reason: str = "") -> int:
    print(f"\nSMOKE {status}: {_NAME}" + (f" — {reason}" if reason else ""))
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def _state(eai: str, profile: str, job_id: str) -> str:
    r = subprocess.run(
        [eai, "--profile", profile, "job", "get", job_id, "--fields", "state", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    for ln in r.stdout.splitlines():
        if ln.strip():
            try:
                return str(json.loads(ln).get("state", "")).lower()
            except json.JSONDecodeError:
                continue
    return ""


def _released(eai: str, profile: str, job_id: str, *, deadline_s: int = 30) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if (st := _state(eai, profile, job_id)) and st not in _ACTIVE:
            return True
        time.sleep(2)
    return False


def _write_status(exp_dir: Path, run_id: str, heartbeat_age_s: float) -> None:
    now = time.time()
    ExperimentStatus(
        status="RUNNING",
        mode="ray",
        pid=999999,  # not us
        host="dead-host",
        started_at=now - heartbeat_age_s - 1,
        last_heartbeat_at=now - heartbeat_age_s,
        total_episodes=1,
        run_id=run_id,
    ).write(exp_dir / "experiment_status.json")


def main() -> int:
    eai = "eai" if shutil.which("eai") else os.path.expanduser("~/bin/eai")
    if shutil.which("eai") is None and not os.path.exists(eai):
        return banner("SKIP", "eai CLI not found")
    profile = os.environ.get("EAI_PROFILE", "yul101")

    root = Path(tempfile.mkdtemp(prefix="smoke_206_death_"))
    dead_run, live_run = f"smoke206-{uuid.uuid4().hex[:8]}", f"smoke206-{uuid.uuid4().hex[:8]}"
    infra = ToolkitInfraConfig(profile=profile, eai_path=eai, cube_data=None, default_ttl_seconds=600)
    resource = DockerServiceConfig(name="cube-smoke-206d", scope="task", docker_images=["python:3.12-slim"])
    infra.provision(resource)
    prev_env = os.environ.get("CUBE_RUN_ID")

    dead_job = live_job = None
    try:
        os.environ["CUBE_RUN_ID"] = dead_run  # what _experiment_lifecycle exports
        dead_h = infra.launch(resource)  # tagged cube_run_id=dead_run
        dead_job = dead_h.id
        os.environ["CUBE_RUN_ID"] = live_run
        live_h = infra.launch(resource)  # a concurrent, healthy run
        live_job = live_h.id

        # Simulate the crash: dead run's handle is lost (no close), its heartbeat is stale;
        # the live run is heartbeating now.
        dead_h = None  # noqa: F841 — drop handle, do NOT close (orphans the job)
        _write_status(root / "dead", dead_run, heartbeat_age_s=_DEAD_RUN_STALE_AFTER_S + 120)
        _write_status(root / "live", live_run, heartbeat_age_s=0)

        # A new run starts → its startup GC runs.
        _reap_dead_runs(infra, root / "current", current_run_id=f"smoke206-{uuid.uuid4().hex[:8]}")

        if not _released(eai, profile, dead_job):
            return banner("FAIL", f"startup GC did not reap the dead run's orphaned job {dead_job[:8]}")
        if _state(eai, profile, live_job) not in _ACTIVE:
            return banner("FAIL", f"startup GC wrongly killed the LIVE run's job {live_job[:8]}")
        return banner("OK", f"GC reaped dead-run job {dead_job[:8]}; live-run job {live_job[:8]} untouched")
    except Exception as exc:  # noqa: BLE001
        return banner("FAIL", f"{type(exc).__name__}: {exc}")
    finally:
        os.environ["CUBE_RUN_ID"] = prev_env if prev_env is not None else ""
        if prev_env is None:
            os.environ.pop("CUBE_RUN_ID", None)
        for rid in (dead_run, live_run):
            try:
                infra.cleanup(rid)
            except Exception:  # noqa: BLE001, S110
                pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
