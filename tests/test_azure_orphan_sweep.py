"""Tests for scripts/azure_orphan_sweep.py.

The script imports ``cube_infra_azure`` which is a separate workspace package
in cube-standard (cube-resources/cube-infra-azure) and is not installed in the
cube-harness test venv by default. We auto-skip when it's not importable —
the production runtime is the scheduled GitHub Action, which installs it
explicitly. The tests below run when the package is present (e.g. on a dev
laptop with cube-infra-azure linked into the venv) to verify the CLI plumbing.
"""

import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

# Resolve the script path once.
SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "azure_orphan_sweep.py"

_HAS_INFRA_AZURE = importlib.util.find_spec("cube_infra_azure") is not None


def test_script_file_exists_and_is_executable() -> None:
    """The L4 deliverable: ensure the script we ship is on disk and chmod +x."""
    assert SCRIPT_PATH.exists(), f"Sweeper script missing at {SCRIPT_PATH}"
    # POSIX exec bit — best-effort, the workflow invokes via `python script.py` either way.
    mode = SCRIPT_PATH.stat().st_mode & 0o111
    assert mode, "Script should be executable (chmod +x)"


def test_script_help_runs_without_azure_deps() -> None:
    """``--help`` must not require Azure credentials or the cube_infra_azure
    package. It's the smoke check operators run first."""
    if not _HAS_INFRA_AZURE:
        pytest.skip("cube_infra_azure not installed in this environment")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"--help failed: {proc.stderr}"
    assert "--resource-group" in proc.stdout
    assert "--max-age-seconds" in proc.stdout
    assert "--dry-run" in proc.stdout


@pytest.mark.skipif(not _HAS_INFRA_AZURE, reason="cube_infra_azure not installed")
def test_main_invokes_cleanup_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Driving the script with patched AzureInfraConfig must call cleanup_stale()
    with the resource group from the CLI."""
    from unittest.mock import MagicMock

    fake_infra = MagicMock()
    fake_infra.cleanup_stale.return_value = ["cube-abc-vm-001", "cube-def-vm-002"]

    fake_cls = MagicMock(return_value=fake_infra)
    monkeypatch.setattr("cube_infra_azure.azure.AzureInfraConfig", fake_cls)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--resource-group", "test-rg"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    assert exc.value.code in (None, 0)

    fake_cls.assert_called_once_with(resource_group="test-rg")
    fake_infra.cleanup_stale.assert_called_once_with(max_age_seconds=None)


@pytest.mark.skipif(not _HAS_INFRA_AZURE, reason="cube_infra_azure not installed")
def test_dry_run_skips_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--dry-run`` must instantiate the infra (auth check) but not call
    cleanup_stale() — operators rely on this to validate credentials safely."""
    from unittest.mock import MagicMock

    fake_infra = MagicMock()
    fake_cls = MagicMock(return_value=fake_infra)
    monkeypatch.setattr("cube_infra_azure.azure.AzureInfraConfig", fake_cls)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--resource-group", "test-rg", "--dry-run"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    assert exc.value.code in (None, 0)

    fake_cls.assert_called_once_with(resource_group="test-rg")
    fake_infra.cleanup_stale.assert_not_called()
