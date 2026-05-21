"""Tests for cube_harness.exp_runner._experiment_lifecycle.

Focused on the L2 robustness layer: SIGTERM handler installation and the
``cleanup_stale()`` sweep that runs on lifecycle exit. Compounds with L3
(cleanup at benchmark setup) and L4 (external scheduled sweeper) to limit
the window in which orphaned cloud VMs can accumulate.
"""

import signal
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cube.resource import InfraConfig

from cube_harness.exp_runner import _experiment_lifecycle


def _drain_lifecycle(exp_dir: Path, infra: InfraConfig | None) -> None:
    """Drive the lifecycle context to normal completion."""
    with _experiment_lifecycle(exp_dir, mode="sequential", infra=infra):
        pass


def test_cleanup_stale_called_on_normal_exit(tmp_path: Path) -> None:
    """On a clean exit, ``cleanup_stale()`` must still run as a defensive sweep
    of expired resources from concurrent or prior runs."""
    infra = MagicMock(spec=InfraConfig)
    infra.cleanup_stale.return_value = []

    _drain_lifecycle(tmp_path, infra)

    infra.cleanup_stale.assert_called_once_with()


def test_cleanup_stale_called_on_exception(tmp_path: Path) -> None:
    """When the body of the lifecycle raises, ``cleanup_stale()`` must still
    fire — the whole point of L2 is to catch the abnormal-exit path."""
    infra = MagicMock(spec=InfraConfig)
    infra.cleanup_stale.return_value = []

    with pytest.raises(RuntimeError, match="boom"):
        with _experiment_lifecycle(tmp_path, mode="sequential", infra=infra):
            raise RuntimeError("boom")

    infra.cleanup_stale.assert_called_once_with()


def test_cleanup_stale_not_called_when_infra_is_none(tmp_path: Path) -> None:
    """No infra → nothing to sweep. Don't call APIs that don't exist."""
    # Pure smoke: just ensure entering and exiting works without raising.
    _drain_lifecycle(tmp_path, infra=None)


def test_cleanup_stale_runs_on_keyboard_interrupt(tmp_path: Path) -> None:
    """Ctrl+C still triggers ``cleanup_stale()`` — so Ray workers get their
    teardown window and stale infra resources get reclaimed. Bounded by the
    grace timeout (see test_cleanup_stale_timeout_does_not_block_exit) and
    interruptible by a second Ctrl+C."""
    infra = MagicMock(spec=InfraConfig)
    infra.cleanup_stale.return_value = []

    with pytest.raises(KeyboardInterrupt):
        with _experiment_lifecycle(tmp_path, mode="sequential", infra=infra):
            raise KeyboardInterrupt

    infra.cleanup_stale.assert_called_once_with()


def test_cleanup_stale_timeout_does_not_block_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``cleanup_stale`` hangs (network blip, throttle), the lifecycle must
    exit within the grace window, not wait forever. The daemon thread keeps
    running but dies when the main process exits — next launch sweeps residue."""
    import cube_harness.exp_runner as exp_runner_mod

    monkeypatch.setattr(exp_runner_mod, "_CLEANUP_GRACE_TIMEOUT_S", 0.5)

    infra = MagicMock(spec=InfraConfig)
    infra.cleanup_stale.side_effect = lambda: time.sleep(10)

    start = time.time()
    with _experiment_lifecycle(tmp_path, mode="sequential", infra=infra):
        pass
    elapsed = time.time() - start

    assert elapsed < 3.0, f"Lifecycle exit blocked for {elapsed:.1f}s waiting on hung cleanup"
    infra.cleanup_stale.assert_called_once_with()


def test_cleanup_stale_called_on_systemexit(tmp_path: Path) -> None:
    """SystemExit (which is what our SIGTERM handler raises) is NOT a
    KeyboardInterrupt — orchestrator-driven shutdowns get the full cleanup
    because the orchestrator gives us a grace period to do it."""
    infra = MagicMock(spec=InfraConfig)
    infra.cleanup_stale.return_value = []

    with pytest.raises(SystemExit):
        with _experiment_lifecycle(tmp_path, mode="sequential", infra=infra):
            raise SystemExit(143)

    infra.cleanup_stale.assert_called_once_with()


def test_cleanup_stale_failure_does_not_propagate(tmp_path: Path) -> None:
    """If the sweep itself fails (transient cloud error, throttling, etc.) we
    must not mask the original exception or crash on a clean exit. Failures are
    logged at WARNING."""
    infra = MagicMock(spec=InfraConfig)
    infra.cleanup_stale.side_effect = RuntimeError("transient Azure 503")

    # Should not raise despite the side_effect.
    _drain_lifecycle(tmp_path, infra)

    infra.cleanup_stale.assert_called_once_with()


def test_sigterm_handler_installed_and_restored(tmp_path: Path) -> None:
    """The lifecycle must install a SIGTERM handler on entry (so ``finally``
    blocks run on orchestrator-driven shutdown) and restore the prior handler
    on exit so it doesn't leak across runs."""
    infra = MagicMock(spec=InfraConfig)
    infra.cleanup_stale.return_value = []

    sentinel_handler = signal.getsignal(signal.SIGTERM)

    with _experiment_lifecycle(tmp_path, mode="sequential", infra=infra):
        in_lifecycle = signal.getsignal(signal.SIGTERM)
        assert in_lifecycle is not sentinel_handler, "Expected lifecycle to install its own SIGTERM handler"

    after = signal.getsignal(signal.SIGTERM)
    assert after is sentinel_handler, "Lifecycle must restore the prior SIGTERM handler on exit"


def test_no_sigterm_handler_when_infra_is_none(tmp_path: Path) -> None:
    """No infra → no need to intercept SIGTERM."""
    sentinel_handler = signal.getsignal(signal.SIGTERM)

    with _experiment_lifecycle(tmp_path, mode="sequential", infra=None):
        in_lifecycle = signal.getsignal(signal.SIGTERM)
        assert in_lifecycle is sentinel_handler, (
            "Lifecycle should not install a SIGTERM handler when no infra is configured"
        )
