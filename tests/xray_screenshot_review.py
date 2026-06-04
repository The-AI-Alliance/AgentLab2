"""Standalone visual review script for XRay — event-stream edition.

Builds synthetic experiments, launches XRay, puppets through key UI states,
and saves labeled screenshots to /tmp/xray_screenshots/ for agent analysis.

Modes
-----
  tables   — Agents → Trajectories tabs: status symbols, counts, retry badge
  dashboard — Dashboard tab: experiment stats + progress bar
  episode  — Event card rail + detail tabs (Chat/Observation/AXTree/Evaluation)
  all      — Runs all three modes in sequence (default)

Usage
-----
  .venv/bin/python tests/xray_screenshot_review.py --mode episode
  .venv/bin/python tests/xray_screenshot_review.py --mode all
  .venv/bin/python tests/xray_screenshot_review.py --mode all --headed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator

from playwright.sync_api import Browser, Page, sync_playwright

# Allow running from repo root or tests/
_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from tests.xray_fixture import build_demo_experiment  # noqa: E402
from tests.xray_test_helpers import EXTENDED_SCENARIOS, build_experiment, free_port, wait_for_server  # noqa: E402

OUT_DIR = Path("/tmp/xray_screenshots")
# Use the worktree venv directly — never `uv run` (it re-syncs and clobbers the
# editable cube-standard install). Requires `make install` to have been run.
_VENV_PYTHON = _REPO / ".venv" / "bin" / "python"
if not _VENV_PYTHON.exists():
    sys.exit(f"ERROR: venv not found at {_VENV_PYTHON}. Run `make install` first.")
_PYTHON = str(_VENV_PYTHON)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


@contextmanager
def xray_server(results_dir: Path) -> Generator[str, None, None]:
    port = free_port()
    proc = subprocess.Popen(
        [_PYTHON, "-m", "cube_harness.analyze.xray", "--results-dir", str(results_dir), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(url, timeout=60.0)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


def settle(page: Page, ms: int = 700) -> None:
    page.wait_for_timeout(ms)


def shot(page: Page, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    print(f"  📸  {path}")
    return path


def open_xray(page: Page, url: str, settle_ms: int = 3500) -> None:
    """Navigate to XRay; the auto-loader selects the first experiment."""
    page.goto(url, wait_until="domcontentloaded")
    settle(page, settle_ms)


def click_tab(page: Page, label: str, wait_ms: int = 600) -> None:
    tab = page.get_by_role("tab", name=label, exact=False)
    if tab.count():
        tab.first.click()
        settle(page, wait_ms)


def click_card(page: Page, index: int, wait_ms: int = 700) -> bool:
    """Click the Nth event card in the vertical rail. Returns False if no such card."""
    cards = page.locator(".xray-event-card")
    if cards.count() > index:
        cards.nth(index).click()
        settle(page, wait_ms)
        return True
    return False


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_tables(page: Page, url: str) -> None:
    """Agent + Trajectory tables — all status symbols, counts, retry badge."""
    print("\n[tables] Agents/Trajectories table views…")
    open_xray(page, url)
    shot(page, "tables_01_landing")

    click_tab(page, "Agents")
    shot(page, "tables_02_agents")

    click_tab(page, "Trajectories")
    shot(page, "tables_03_trajectories")

    click_tab(page, "Dashboard")
    shot(page, "tables_04_dashboard")


def run_dashboard(page: Page, url: str) -> None:
    """Dashboard tab — experiment-level stats and progress bar."""
    print("\n[dashboard] Dashboard…")
    open_xray(page, url)
    click_tab(page, "Dashboard")
    shot(page, "dashboard_01_stats")


def run_episode(page: Page, url: str) -> None:
    """Event card rail + every detail tab — the event-stream UI core."""
    print("\n[episode] Event card rail + detail tabs…")
    open_xray(page, url)
    shot(page, "episode_01_landing_rail")

    # Walk through the first 6 cards:
    # 0=reset obs, 1=llm act, 2=click, 3=llm act(parallel), 4=type, 5=read_page
    for idx in range(6):
        if click_card(page, idx, wait_ms=400):
            shot(page, f"episode_02_card_{idx}")

    # Select the parallel turn (card 3 — LLM that dispatches 2 tool calls).
    click_card(page, 3)
    for label in ("Chat", "Observation", "AXTree", "Evaluation", "Error", "Debug"):
        click_tab(page, label, wait_ms=500)
        shot(page, f"episode_03_tab_{label.lower()}")

    # Error card (card 6 — LLM error event).
    if click_card(page, 6):
        click_tab(page, "Error")
        shot(page, "episode_04_error_card")

    # Terminal evaluation (last card).
    n = page.locator(".xray-event-card").count()
    if n > 0 and click_card(page, n - 1):
        click_tab(page, "Evaluation")
        shot(page, "episode_05_terminal_eval")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_MODES: dict[str, tuple[str, Callable[[Page, str], None]]] = {
    "tables": ("tables", run_tables),
    "dashboard": ("tables", run_dashboard),
    "episode": ("episode", run_episode),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="XRay visual review — event-stream edition.")
    parser.add_argument("--mode", choices=[*_MODES, "all"], default="all")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed (visible) mode.")
    args = parser.parse_args()

    modes_to_run = list(_MODES.keys()) if args.mode == "all" else [args.mode]

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp)

        # Metadata-only fixture: drives Agents/Trajectories tables and Dashboard.
        tables_exp = results_dir / "exp_20260101_status"
        build_experiment(tables_exp, EXTENDED_SCENARIOS)

        # Event-format fixture: drives the card rail and detail panes.
        episode_exp = results_dir / "exp_20260101_episode"
        build_demo_experiment(episode_exp)

        with xray_server(results_dir) as url:
            with sync_playwright() as pw:
                browser: Browser = pw.chromium.launch(headless=not args.headed)
                for mode_key in modes_to_run:
                    _, fn = _MODES[mode_key]
                    page: Page = browser.new_page(viewport={"width": 1500, "height": 900})
                    fn(page, url)
                    page.close()
                browser.close()

    print(f"\nScreenshots saved to {OUT_DIR}/")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
