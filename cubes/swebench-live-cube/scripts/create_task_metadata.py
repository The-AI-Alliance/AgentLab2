#!/usr/bin/env python3
"""Generate src/swebench_live_cube/task_metadata.json from HuggingFace.

This is a developer tool.  Run it when the SWE-bench Live dataset is updated
to regenerate the shipped package resource.  The output file is committed to
the repository — end users never need to run this script.

Only lightweight public fields are written (repo, base_commit, splits,
log_parser).  Heavy execution data (problem_statement, patch, test_patch, etc.)
is written by SWEBenchLiveBenchmarkConfig.install() into the per-task execution
cache and is never committed.

Usage:
    python scripts/create_task_metadata.py [--force] [--hf-cache DIR]

Options:
    --force          Overwrite task_metadata.json even if it already exists.
    --hf-cache DIR   Where to store the downloaded HF dataset.
                     Defaults to ~/.cube/swebench-live-cube/huggingface_cache
                     (same as SWEBenchLiveBenchmarkConfig.install()).
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

from swebench_live_cube.benchmark import (
    SWEBenchLiveBenchmarkConfig,
    _DATASET_NAME,
    _SPLIT_PRIORITY,
    _merge_rows_by_split,
)
from swebench_live_cube.gold_subset import tag_lite_gold
from swebench_live_cube.task import SWEBenchLiveTaskMetadata

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent.parent / "src" / "swebench_live_cube" / "task_metadata.json"
_DEFAULT_HF_CACHE = SWEBenchLiveBenchmarkConfig.cache_dir() / "huggingface_cache"

_DOCKER_NAMESPACE = "starryzhang"
_IMAGE_TAG = "latest"


def _normalize_instance_id(instance_id: str) -> str:
    """Normalize instance_id for Docker image naming: replace __ with _1776_ and lowercase."""
    return instance_id.replace("__", "_1776_").lower()


def _get_docker_image(instance_id: str) -> str:
    normalized = _normalize_instance_id(instance_id)
    return f"{_DOCKER_NAMESPACE}/sweb.eval.x86_64.{normalized}:{_IMAGE_TAG}"


def _build_task_metadata(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, SWEBenchLiveTaskMetadata]:
    """Build lightweight TaskMetadata from HF dataset rows.

    Only extracts public fields (repo, base_commit, splits, log_parser).
    Heavy execution fields (problem_statement, patch, test_patch, etc.) live
    in the per-task execution cache written by install().
    """
    merged = _merge_rows_by_split(rows_by_split)
    metadata: dict[str, SWEBenchLiveTaskMetadata] = {}
    for iid, (row, splits_present) in merged.items():
        # abstract_description = first line of problem_statement, capped at 200 chars
        first_line = row["problem_statement"].split("\n", 1)[0]
        metadata[iid] = SWEBenchLiveTaskMetadata(
            id=iid,
            abstract_description=first_line[:200],
            recommended_max_steps=100,
            container_config=ContainerConfig(
                image=_get_docker_image(iid),
                cpu_cores=2.0,
                ram_gb=4.0,
                disk_gb=10.0,
            ),
            repo=row["repo"],
            base_commit=row["base_commit"],
            splits=splits_present,
            log_parser=row.get("log_parser", "pytest"),
        )
    return metadata


def generate_task_metadata(
    output_path: Path = _DEFAULT_OUTPUT,
    hf_cache: Path = _DEFAULT_HF_CACHE,
    *,
    force: bool = False,
) -> int:
    """Download HF splits and write the shipped task_metadata.json.

    Args:
        output_path:  Destination path. Defaults to src/swebench_live_cube/task_metadata.json.
        hf_cache:     HuggingFace cache directory.
                      Defaults to ~/.cube/swebench-live-cube/huggingface_cache.
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

    # Download all 4 splits from HuggingFace
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in _SPLIT_PRIORITY:
        logger.info("Downloading %s split=%r from HuggingFace...", _DATASET_NAME, split)
        ds = load_dataset(_DATASET_NAME, split=split, cache_dir=str(hf_cache))
        rows_by_split[split] = list(ds)  # type: ignore[arg-type]
        logger.info("  %d tasks in split=%r", len(rows_by_split[split]), split)

    metadata = _build_task_metadata(rows_by_split)

    rows = [tm.model_dump() for tm in metadata.values()]
    # Stamp the official lite-gold marker from the committed gold-solvable id list so
    # named_subset("lite-gold") survives regeneration.
    tagged = tag_lite_gold(rows)
    logger.info("Tagged %d tasks with the 'lite-gold' split", tagged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n")
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
