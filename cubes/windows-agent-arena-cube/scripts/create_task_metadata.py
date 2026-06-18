#!/usr/bin/env python3
"""Generate src/waa_cube/task_metadata.json from the WindowsAgentArena repo.

This is a developer tool. Run it after cloning (or updating) the WAA repo
to regenerate the shipped package resource. The output file is committed to
the repository — end users never need to run this script.

Usage:
    python scripts/create_task_metadata.py [--force]

Options:
    --force      Overwrite task_metadata.json even if it already exists.

Requires WAA_EVAL_EXAMPLES_DIR to be set (or pass --eval-dir).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make the package importable from the cube root without venv activation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cube.task import TaskMetadata

logger = logging.getLogger(__name__)

WAA_EVAL_EXAMPLES_ENV = "WAA_EVAL_EXAMPLES_DIR"
_DEFAULT_OUTPUT = Path(__file__).parent.parent / "src" / "waa_cube" / "task_metadata.json"
_TEST_SETS = ["test_all", "test_small", "test_custom"]

# Upstream WAA tasks ship Linux launch commands (e.g. ['google-chrome', ...]).
# Translate the ones we know are Linux-isms to their Windows equivalents so
# they actually run on the Windows VM. Add new entries here as we hit them.
_LINUX_TO_WINDOWS_BIN = {
    "google-chrome": ["start", "chrome"],
}
# PowerShell cmdlets aren't binaries — they must be invoked through powershell.
_PS_CMDLETS = {"Stop-Process", "Start-Process", "Get-Process", "Set-ItemProperty"}
# Translate Linux home paths → Windows. Upstream WAA's Linux container uses
# user "user" (HOME=/home/user); our Windows VM uses user "Docker".
_LINUX_HOME = "/home/user"
_WINDOWS_HOME = "C:\\Users\\Docker"


def _windows_ize_path(s: str) -> str:
    """Rewrite Linux-style WAA paths (and any embedded forward slashes) for Windows."""
    if not isinstance(s, str) or _LINUX_HOME not in s:
        return s
    rest = s.split(_LINUX_HOME, 1)[1]
    return _WINDOWS_HOME + rest.replace("/", "\\")


def _windows_ize_command(cmd: list) -> list:
    """Rewrite a single launch command list for Windows."""
    if not cmd:
        return cmd
    head = cmd[0]
    if head in _LINUX_TO_WINDOWS_BIN:
        return _LINUX_TO_WINDOWS_BIN[head] + list(cmd[1:])
    if head in _PS_CMDLETS:
        return ["powershell", "-Command", *cmd]
    return cmd


def _windows_ize_value(v):
    """Recursively translate /home/user paths in nested dict/list/str values."""
    if isinstance(v, str):
        return _windows_ize_path(v)
    if isinstance(v, list):
        return [_windows_ize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _windows_ize_value(x) for k, x in v.items()}
    return v


# Chrome on our Windows image silently refuses to bind --remote-debugging-port
# unless launched with a fresh --user-data-dir. Probe v3 confirmed: the original
# `start chrome ...` shell-handler path doesn't actually pass flags to a new
# chrome process; the working invocation is chrome.exe directly with the data
# dir. We also drop the ['socat', ...] step because socat isn't installed —
# Caddy is already running as a system service that proxies 9222 → 1337.
_CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
_CDP_USER_DATA_DIR = r"C:\Temp\cdp-profile"


def _is_chrome_launch_step(step: dict) -> bool:
    if step.get("type") != "launch":
        return False
    cmd = (step.get("parameters") or {}).get("command")
    if not isinstance(cmd, list) or not cmd:
        return False
    head = cmd[0]
    # Either "start chrome ..." (already-windows-ized) or upstream "google-chrome ..."
    return head == "google-chrome" or (head == "start" and len(cmd) > 1 and cmd[1] == "chrome")


def _is_socat_step(step: dict) -> bool:
    if step.get("type") != "launch":
        return False
    cmd = (step.get("parameters") or {}).get("command")
    return isinstance(cmd, list) and bool(cmd) and cmd[0] == "socat"


def _chrome_replacement_steps(orig_cmd: list) -> list[dict]:
    """Build the kill-then-launch-with-fresh-profile sequence for chrome."""
    # Preserve any flags after the original binary (e.g. --force-renderer-accessibility)
    # except --remote-debugging-port, which we always set explicitly.
    extras = [
        a for a in orig_cmd[(2 if orig_cmd[0] == "start" else 1) :] if not a.startswith("--remote-debugging-port")
    ]
    return [
        {"type": "execute", "parameters": {"command": "taskkill /IM chrome.exe /F", "shell": "true"}},
        {"type": "sleep", "parameters": {"seconds": 2}},
        {
            "type": "launch",
            "parameters": {
                "command": [
                    "cmd",
                    "/c",
                    "start",
                    "",
                    "/b",
                    _CHROME_EXE,
                    "--remote-debugging-port=1337",
                    f"--user-data-dir={_CDP_USER_DATA_DIR}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    *extras,
                ],
                "shell": False,
            },
        },
    ]


def _windows_ize_config(config: list[dict]) -> list[dict]:
    """Rewrite Linux-only launch commands AND /home/user paths AND chrome launches
    AND drop redundant socat steps in upstream task configs for Windows."""
    fixed: list[dict] = []
    for step in config:
        # Drop socat steps — Caddy already proxies 9222 → 1337.
        if _is_socat_step(step):
            continue

        # Special-case chrome: replace with kill + chrome.exe direct + fresh user-data-dir.
        if _is_chrome_launch_step(step):
            orig_cmd = (step.get("parameters") or {}).get("command") or []
            for replacement in _chrome_replacement_steps(orig_cmd):
                fixed.append(_windows_ize_value(replacement))
            continue

        new_step = dict(step)
        params = dict(step.get("parameters") or {})
        cmd = params.get("command")
        if step.get("type") == "launch" and isinstance(cmd, list) and cmd:
            new_cmd = _windows_ize_command(cmd)
            if new_cmd is not cmd:
                params["command"] = new_cmd
        # Translate any /home/user path inside parameters (e.g. open path, download dest).
        new_step["parameters"] = _windows_ize_value(params)
        fixed.append(new_step)
    return fixed


def _windows_ize_evaluator(evaluator: dict) -> dict:
    """Same translation, applied to evaluator.postconfig (run before scoring)
    and to any embedded /home/user paths in result/expected configs."""
    if not isinstance(evaluator, dict):
        return evaluator
    fixed = _windows_ize_value(evaluator)
    if isinstance(fixed.get("postconfig"), list):
        fixed["postconfig"] = _windows_ize_config(fixed["postconfig"])
    return fixed


def generate_task_metadata(
    eval_examples_dir: Path,
    output_path: Path = _DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> int:
    """Parse the WAA evaluation_examples_windows/ dir and write task_metadata.json.

    Returns the number of tasks written (0 if skipped).
    """
    if output_path.exists() and not force:
        logger.info("task_metadata.json already exists at %s — skipping. Pass --force to regenerate.", output_path)
        return 0

    if not eval_examples_dir.exists():
        raise RuntimeError(f"evaluation_examples_windows not found at {eval_examples_dir}")

    # Collect which test sets each task belongs to
    task_sets: dict[str, list[str]] = {}
    task_domains: dict[str, str] = {}

    for set_name in _TEST_SETS:
        set_file = eval_examples_dir / f"{set_name}.json"
        if not set_file.exists():
            logger.warning("Test set file not found: %s", set_file)
            continue
        with open(set_file) as f:
            tasks_by_domain: dict[str, list[str]] = json.load(f)
        for domain_name, task_ids in tasks_by_domain.items():
            for task_id in task_ids:
                task_sets.setdefault(task_id, []).append(set_name)
                task_domains.setdefault(task_id, domain_name)

    # Load each task JSON and build metadata
    all_task_ids = set()
    for set_name in _TEST_SETS:
        set_file = eval_examples_dir / f"{set_name}.json"
        if not set_file.exists():
            continue
        with open(set_file) as f:
            for task_ids in json.load(f).values():
                all_task_ids.update(task_ids)

    metadata: list[dict] = []
    for task_id in sorted(all_task_ids):
        domain = task_domains.get(task_id, "unknown")
        task_file = eval_examples_dir / "examples" / domain / f"{task_id}.json"
        if not task_file.exists():
            logger.warning("Task file not found: %s", task_file)
            continue
        try:
            with open(task_file) as f:
                td = json.load(f)
        except Exception as exc:
            logger.error("Failed to load task %s: %s", task_id, exc)
            continue

        tm = TaskMetadata(
            # NOTE: `td["id"]` (the task JSON's internal id) is NOT unique — two
            # WAA task pairs share an internal id while their filenames (`task_id`,
            # which carry a -2/-3 suffix) differ. Using td["id"] here collides
            # those into one dict key and silently drops 2 tasks (154 -> 152). To
            # recover all 154, switch this to the unique `task_id` (filename) and
            # regenerate task_execution_info.json with the same key. See the note
            # in benchmark.py. Left as-is to avoid renaming every shipped task id.
            id=td.get("id", task_id),
            abstract_description=td.get("instruction", ""),
            extra_info={
                "domain": domain,
                "snapshot": td.get("snapshot", "init_state"),
                "config": _windows_ize_config(td.get("config", [])),
                "evaluator": _windows_ize_evaluator(td.get("evaluator", {})),
                "related_apps": td.get("related_apps", []),
                "test_sets": task_sets.get(task_id, []),
            },
        )
        metadata.append(tm.model_dump())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Saved %d tasks to %s", len(metadata), output_path)
    return len(metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    parser.add_argument("--eval-dir", type=str, default=None, help="Path to evaluation_examples_windows/")
    args = parser.parse_args()

    eval_dir = args.eval_dir or os.environ.get(WAA_EVAL_EXAMPLES_ENV)
    if not eval_dir:
        print(f"Error: set {WAA_EVAL_EXAMPLES_ENV} or pass --eval-dir", file=sys.stderr)
        sys.exit(1)

    n = generate_task_metadata(Path(eval_dir), force=args.force)
    if n:
        print(f"Generated task_metadata.json with {n} tasks")
    else:
        print("Skipped (already exists). Use --force to regenerate.")
