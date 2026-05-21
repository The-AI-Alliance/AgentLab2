#!/usr/bin/env python3
"""Generate src/terminalbench2_cube/task_metadata.json from the terminal-bench-2 repo.

This is a developer tool. Run it when the terminal-bench-2 task list changes to
regenerate the shipped package resource. The output file is committed to the
repository — end users never need to run this script.

Only lightweight public fields are written (difficulty, category, tags,
max_agent_timeout_sec, container_config). Heavy execution data (instruction,
archive) is written by TerminalBench2BenchmarkConfig.install() into the per-task
execution cache and is never committed.

Usage:
    python scripts/create_task_metadata.py [--force] [--repo-dir DIR]

Options:
    --force          Overwrite task_metadata.json even if it already exists.
    --repo-dir DIR   Use an already-cloned terminal-bench-2 repo instead of
                     cloning a fresh copy (useful for development / offline use).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

# Make the package importable when executed from the cube root without venv activation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cube.container import ContainerConfig

from terminalbench2_cube.benchmark import REPO_URL, TerminalBench2BenchmarkConfig
from terminalbench2_cube.task import TerminalBench2TaskMetadata

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent.parent / "src" / "terminalbench2_cube" / "task_metadata.json"


def _parse_gb(value: str | int | float) -> float:
    """Parse a memory/storage string like '4G' to a float in GB."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).upper().strip()
    if s.endswith("GB"):
        return float(s[:-2])
    if s.endswith("G"):
        return float(s[:-1])
    if s.endswith("M"):
        return float(s[:-1]) / 1024
    return 4.0


def _build_task_metadata(tasks: list[dict]) -> dict[str, TerminalBench2TaskMetadata]:
    """Build lightweight TaskMetadata from raw task dicts.

    Only extracts public fields (difficulty, category, tags, max_agent_timeout_sec,
    container_config). Heavy execution fields (instruction, archive) live in the
    per-task execution cache written by install().
    """
    metadata: dict[str, TerminalBench2TaskMetadata] = {}
    for t in tasks:
        tid = t["task_id"]
        # abstract_description = first line of the instruction, capped at 200 chars
        first_line = t["base_description"].split("\n", 1)[0]
        metadata[tid] = TerminalBench2TaskMetadata(
            id=tid,
            abstract_description=first_line[:200],
            recommended_max_steps=None,
            container_config=ContainerConfig(
                image=t.get("docker_image", "python:3.13"),
                cpu_cores=float(t.get("cpus", 1)),
                ram_gb=_parse_gb(t.get("memory", "4G")),
                disk_gb=_parse_gb(t.get("storage", "10G")),
            ),
            difficulty=t.get("difficulty", "unknown"),
            category=t.get("category", ""),
            tags=t.get("tags", []),
            max_agent_timeout_sec=int(t.get("max_agent_timeout_sec", 900)),
        )
    return metadata


def generate_task_metadata(
    output_path: Path = _DEFAULT_OUTPUT,
    repo_dir: Path | None = None,
    *,
    force: bool = False,
) -> int:
    """Clone (or reuse) the terminal-bench-2 repo and write the shipped task_metadata.json.

    Args:
        output_path: Destination path. Defaults to src/terminalbench2_cube/task_metadata.json.
        repo_dir:    Path to an already-cloned terminal-bench-2 repo. If None, a fresh
                     shallow clone is made into a temporary directory.
        force:       Overwrite even if output_path already exists.

    Returns:
        Number of tasks written (0 if skipped due to idempotency).
    """
    if output_path.exists() and not force:
        logger.info(
            "task_metadata.json already exists at %s — skipping. Pass --force to regenerate.",
            output_path,
        )
        return 0

    if repo_dir is not None:
        tasks = _load_tasks_from_repo(repo_dir)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            clone_dir = Path(tmpdir) / "terminal-bench-2"
            logger.info("Cloning %s ...", REPO_URL)
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)],
                check=True,
                timeout=300,
            )
            tasks = _load_tasks_from_repo(clone_dir)

    logger.info("  %d tasks loaded", len(tasks))
    metadata = _build_task_metadata(tasks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([tm.model_dump() for tm in metadata.values()], indent=2))
    logger.info("Saved %d tasks to %s", len(metadata), output_path)
    return len(metadata)


def _load_tasks_from_repo(repo_dir: Path) -> list[dict]:
    """Load all tasks from a terminal-bench-2 repo directory."""
    tasks = []
    for item in sorted(repo_dir.iterdir()):
        if item.is_dir() and (item / "task.toml").exists():
            task = TerminalBench2BenchmarkConfig._load_task_from_repo(item)
            if task:
                tasks.append(task)
                logger.info("  Loaded: %s (%s)", task["task_id"], task["difficulty"])
    return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    parser.add_argument(
        "--repo-dir",
        metavar="DIR",
        default=None,
        help="Path to an already-cloned terminal-bench-2 repo (skips cloning)",
    )
    args = parser.parse_args()

    generate_task_metadata(
        force=args.force,
        repo_dir=Path(args.repo_dir) if args.repo_dir else None,
    )
