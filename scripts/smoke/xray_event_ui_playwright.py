#!/usr/bin/env python3
"""SMOKE: launch the real XRay viewer on a synthetic event-format experiment
and drive it with Playwright, capturing screenshots of the event-card UI.

This is the visual feedback loop for the agent-owns-loop XRay rewrite. It:
  1. synthesizes a rich event trajectory on disk (tests/xray_fixture.py),
  2. launches `python -m cube_harness.analyze.xray` as a subprocess,
  3. opens it in headless Chromium, navigates agent -> trajectory,
  4. screenshots the landing page and the trajectory detail view + a few tabs.

Screenshots land in --out (default /tmp/xray_shots). Run:
    .venv/bin/python scripts/smoke/xray_event_ui_playwright.py
Prints SMOKE OK|FAIL: xray_event_ui_playwright  (exit 0|1).
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# tests/ is not a package; add it to the path for the fixture import.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests"))
from xray_fixture import build_demo_experiment  # noqa: E402

NAME = "xray_event_ui_playwright"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, proc: subprocess.Popen, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _fail(msg: str) -> int:
    print(f"  ✗ {msg}")
    print(f"SMOKE FAIL: {NAME}")
    return 1


def main() -> int:
    # Optional: point at a real results dir to debug a specific experiment, e.g.
    #   xray_event_ui_playwright.py --results-dir ~/cube_harness_results
    # With no arg, synthesizes the deterministic demo fixture (CI-safe).
    real_dir: Path | None = None
    if "--results-dir" in sys.argv:
        real_dir = Path(sys.argv[sys.argv.index("--results-dir") + 1]).expanduser()

    out_dir = Path("/tmp/xray_shots")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tmp: Path | None = None
    if real_dir is not None:
        exp_root = real_dir
        print(f"  • pointing at real results dir: {exp_root}")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="xray-ui-"))
        exp_root = tmp
        build_demo_experiment(exp_root / "demo_experiment")

    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    proc = subprocess.Popen(
        [sys.executable, "-m", "cube_harness.analyze.xray", "--results-dir", str(exp_root), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_server(url, proc):
            return _fail("xray server did not come up")

        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:  # pragma: no cover
            print(f"  ⚠ playwright not importable ({e})")
            print(f"SMOKE SKIP: {NAME}")
            return 2

        shots: list[str] = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(3500)  # auto-load selects + renders the first episode
            page.screenshot(path=str(out_dir / "01_landing.png"), full_page=True)
            shots.append("01_landing.png")

            def _click(locator, ms: int = 700) -> bool:
                try:
                    locator.first.click(timeout=3000)
                    page.wait_for_timeout(ms)
                    return True
                except Exception:
                    return False

            # The first episode auto-loads, so the detail view is already shown.
            # Walk the event cards (rail) and the detail tabs, screenshotting each.
            cards = page.locator(".xray-event-card")
            n_cards = cards.count()
            for idx in range(min(n_cards, 6)):
                if _click(cards.nth(idx)):
                    page.screenshot(path=str(out_dir / f"02_card_{idx}.png"), full_page=True)
                    shots.append(f"02_card_{idx}.png")

            for i, label in enumerate(("Chat", "Observation", "Screenshot", "AXTree", "Debug", "Step Details")):
                tab = page.get_by_role("tab", name=label, exact=False)
                if tab.count() and _click(tab):
                    fname = f"03_tab_{i}_{label.split()[0].lower()}.png"
                    page.screenshot(path=str(out_dir / fname), full_page=True)
                    shots.append(fname)
            browser.close()

        print(f"  ✓ captured {len(shots)} screenshots in {out_dir}: {', '.join(shots)}")
        print(f"SMOKE OK: {NAME}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if tmp is not None:  # only remove the synthesized fixture, never a real dir
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
