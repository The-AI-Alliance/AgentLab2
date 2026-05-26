#!/usr/bin/env python3
"""Generate src/swebench_verified_cube/task_metadata.json from HuggingFace.

This is a developer tool.  Run it when the SWE-bench Verified dataset is updated
to regenerate the shipped package resource.  The output file is committed to
the repository — end users never need to run this script.

Only lightweight public fields are written (repo, base_commit, difficulty, version).
Heavy execution data (problem_statement, patch, test_patch, etc.) is written by
SWEBenchVerifiedBenchmarkConfig.install() into the per-task execution cache and is never committed.

Usage:
    python scripts/create_task_metadata.py [--force] [--hf-cache DIR]

Options:
    --force          Overwrite task_metadata.json even if it already exists.
    --hf-cache DIR   Where to store the downloaded HF dataset.
                     Defaults to ~/.cube/swebench-verified-cube/huggingface_cache
                     (same as SWEBenchVerifiedBenchmarkConfig.install()).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Make the package importable when executed from the cube root without venv activation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cube.resource import ContainerConfig
from datasets import load_dataset

from swebench_verified_cube.benchmark import TASKS_REQUIRING_ROOT, SWEBenchVerifiedBenchmarkConfig, _DATASET_NAME
from swebench_verified_cube.task import SWEBenchVerifiedTaskMetadata

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent.parent / "src" / "swebench_verified_cube" / "task_metadata.json"
_DEFAULT_HF_CACHE = SWEBenchVerifiedBenchmarkConfig.cache_dir() / "huggingface_cache"

_DOCKER_NAMESPACE = "swebench"
_IMAGE_TAG = "latest"


def _normalize_instance_id(instance_id: str) -> str:
    """Normalize instance_id for Docker image naming: replace __ with _1776_ and lowercase."""
    return instance_id.replace("__", "_1776_").lower()


def _get_docker_image(instance_id: str) -> str:
    normalized = _normalize_instance_id(instance_id)
    return f"{_DOCKER_NAMESPACE}/sweb.eval.x86_64.{normalized}:{_IMAGE_TAG}"


def _build_task_metadata(rows: list[dict[str, Any]]) -> dict[str, SWEBenchVerifiedTaskMetadata]:
    """Build lightweight TaskMetadata from HF dataset rows.

    Only extracts public fields (repo, base_commit, difficulty, version).
    Heavy execution fields (problem_statement, patch, test_patch, etc.) live
    in the per-task execution cache written by install().
    """
    metadata: dict[str, SWEBenchVerifiedTaskMetadata] = {}
    for row in rows:
        iid = row["instance_id"]
        # abstract_description = first line of problem_statement, capped at 200 chars
        first_line = row["problem_statement"].split("\n", 1)[0]
        metadata[iid] = SWEBenchVerifiedTaskMetadata(
            id=iid,
            abstract_description=first_line[:200],
            recommended_max_steps=100,
            container_config=ContainerConfig(
                image=_get_docker_image(iid),
                cpu_cores=2.0,
                ram_gb=4.0,
                disk_gb=10.0,
                # A few images ship a root-owned package subdir a non-root infra can't patch
                # (silent score=0); declaring container:root makes such infras incompatible
                # at make(). See TASKS_REQUIRING_ROOT / cube-harness#446.
                requires={"container:root"} if iid in TASKS_REQUIRING_ROOT else set(),
            ),
            repo=row["repo"],
            base_commit=row["base_commit"],
            difficulty=row.get("difficulty", "unknown"),
            version=row.get("version", ""),
        )
    return metadata


def generate_task_metadata(
    output_path: Path = _DEFAULT_OUTPUT,
    hf_cache: Path = _DEFAULT_HF_CACHE,
    *,
    force: bool = False,
) -> int:
    """Download the HF dataset and write the shipped task_metadata.json.

    Args:
        output_path:  Destination path. Defaults to src/swebench_verified_cube/task_metadata.json.
        hf_cache:     HuggingFace cache directory.
                      Defaults to ~/.cube/swebench-verified-cube/huggingface_cache.
        force:        Overwrite even if output_path already exists.

    Returns:
        Number of tasks written (0 if skipped due to idempotency).
    """
    if output_path.exists() and not force:
        logger.info(
            "task_metadata.json already exists at %s — skipping. Pass --force to regenerate.",
            output_path,
        )
        return 0

    logger.info("Downloading %s from HuggingFace...", _DATASET_NAME)
    ds = load_dataset(_DATASET_NAME, split="test", cache_dir=str(hf_cache))
    rows = list(ds)  # type: ignore[arg-type]
    logger.info("  %d tasks downloaded", len(rows))

    metadata = _build_task_metadata(rows)  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([tm.model_dump() for tm in metadata.values()], indent=2))
    logger.info("Saved %d tasks to %s", len(metadata), output_path)
    return len(metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    parser.add_argument(
        "--hf-cache",
        metavar="DIR",
        default=None,
        help=f"HuggingFace cache directory (default: {_DEFAULT_HF_CACHE})",
    )
    args = parser.parse_args()

    generate_task_metadata(
        force=args.force,
        hf_cache=Path(args.hf_cache) if args.hf_cache else _DEFAULT_HF_CACHE,
    )
