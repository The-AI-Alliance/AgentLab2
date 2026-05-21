"""Per-task-container cube × infra integration matrix.

Runs every (cube, task_id, infra) combination end-to-end through the cube's
debug suite (oracle-mode agent replays gold actions, benchmark-defined
evaluator must return reward == 1.0).

Cubes:  terminalbench2, swebench-verified, swebench-live
Infras: local (Docker), daytona, toolkit, modal

Each parametrised case is independently skippable based on prerequisites
(Docker daemon for local, ``DAYTONA_API_KEY`` for Daytona, ``eai`` CLI +
``EAI_PROFILE`` for Toolkit, ``MODAL_TOKEN_ID`` or ``~/.modal.toml`` for
Modal), so the matrix degrades gracefully in CI.

Each (cube, task_id, infra) triple is a separate pytest item — you get
per-task pass/fail visibility and individual reruns.

Run examples
------------
    # All available infras (skip unavailable ones automatically)
    cd integration-tests
    ./run_matrix.sh

    # Specific cube + infra
    ./run_matrix.sh terminalbench2 toolkit

    # Specific task
    ./run_matrix.sh terminalbench2 toolkit overfull-hbox

    # Direct pytest (same thing, more flags)
    uv run --group toolkit pytest test_debug_matrix.py -v
        --log-cli-level=INFO -k "terminalbench2 and toolkit"
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from types import ModuleType

import pytest
import swebench_live_cube.debug as _swebench_live
import swebench_verified_cube.debug as _swebench_verified
import terminalbench2_cube.debug as _terminalbench2
from cube.resource import InfraConfig
from cube_integration_tests.debug_harness import run_debug_task

# ── cubes ────────────────────────────────────────────────────────────────────

_CUBES: list[tuple[str, ModuleType]] = [
    ("terminalbench2", _terminalbench2),
    ("swebench_verified", _swebench_verified),
    ("swebench_live", _swebench_live),
]


# ── infras (lazy factories: imports only when the factory is called) ─────────


def _local_infra() -> InfraConfig:
    from cube.infra_local import LocalInfraConfig

    return LocalInfraConfig()


def _daytona_infra() -> InfraConfig:
    from cube_infra_daytona import DaytonaInfraConfig

    return DaytonaInfraConfig()


_EAI_CANDIDATE_PATHS: list[str] = [
    os.path.expanduser("~/bin/eai"),
    os.path.expanduser("~/.local/bin/eai"),
    "/usr/local/bin/eai",
]


def _toolkit_infra() -> InfraConfig:
    from cube_infra_toolkit import ToolkitInfraConfig

    eai_path = shutil.which("eai") or next((p for p in _EAI_CANDIDATE_PATHS if os.path.isfile(p)), "eai")
    return ToolkitInfraConfig(eai_path=eai_path)


def _modal_infra() -> InfraConfig:
    from cube_infra_modal import ModalInfraConfig

    return ModalInfraConfig()


# ── prerequisite checks (run at collection time to decide skip/run) ──────────


def _has_docker() -> bool:
    try:
        subprocess.run(["docker", "ps", "-q"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def _has_daytona() -> bool:
    return bool(os.environ.get("DAYTONA_API_KEY"))


def _has_toolkit() -> bool:
    if not bool(os.environ.get("EAI_PROFILE")):
        return False
    if shutil.which("eai") is not None:
        return True
    return any(os.path.isfile(p) for p in _EAI_CANDIDATE_PATHS)


def _has_modal() -> bool:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    return os.path.exists(os.path.expanduser("~/.modal.toml"))


_INFRAS: list[tuple[str, Callable[[], InfraConfig], Callable[[], bool], str]] = [
    ("local", _local_infra, _has_docker, "Docker daemon not reachable"),
    ("daytona", _daytona_infra, _has_daytona, "DAYTONA_API_KEY not set"),
    ("toolkit", _toolkit_infra, _has_toolkit, "eai CLI or EAI_PROFILE not set"),
    ("modal", _modal_infra, _has_modal, "Modal credentials not configured"),
]


# (cube, task_id, infra) combinations that are expected to fail for known
# reasons outside this migration's scope.  Kept in the matrix so the gap
# stays visible in CI output.
_KNOWN_XFAIL: dict[tuple[str, str, str], str] = {}


def _build_matrix() -> list[pytest.param]:
    """Return one pytest.param per (cube, task_id, infra) triple."""
    params = []
    for cube_name, cube_mod in _CUBES:
        task_ids = list(cube_mod._TASK_ACTIONS.keys())
        for infra_name, factory, ready_check, skip_reason in _INFRAS:
            for task_id in task_ids:
                test_id = f"{cube_name}--{task_id}--{infra_name}"
                marks = []
                if not ready_check():
                    marks.append(pytest.mark.skip(reason=skip_reason))
                elif (cube_name, task_id, infra_name) in _KNOWN_XFAIL:
                    reason = _KNOWN_XFAIL[(cube_name, task_id, infra_name)]
                    marks.append(pytest.mark.xfail(reason=reason, strict=False))
                params.append(pytest.param(cube_mod, task_id, factory, id=test_id, marks=marks))
    return params


# ── the matrix ───────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("cube_module,task_id,infra_factory", _build_matrix())
def test_debug_suite(cube_module: ModuleType, task_id: str, infra_factory: Callable[[], InfraConfig]) -> None:
    result = run_debug_task(cube_module, task_id, infra_factory())
    assert not result["error"], f"Task {task_id!r} errored: {result['error']}"
    assert result["done"], f"Task {task_id!r} did not complete (reward={result['reward']})"
    assert result["reward"] == 1.0, f"Task {task_id!r} reward={result['reward']} (expected 1.0)"
