#!/usr/bin/env python3
"""Generate src/swegym_cube/task_metadata.json from HuggingFace.

This is a developer tool.  Run it when the SWE-Gym dataset is updated
to regenerate the shipped package resource.  The output file is committed to
the repository — end users never need to run this script.

Only lightweight public fields are written (repo, base_commit, version).
Heavy execution data (problem_statement, patch, test_patch, etc.) is written by
SWEGymBenchmarkConfig.install() into the per-task execution cache and is never committed.

Usage:
    python scripts/create_task_metadata.py [--force] [--hf-cache DIR]

Options:
    --force          Overwrite task_metadata.json even if it already exists.
    --hf-cache DIR   Where to store the downloaded HF dataset.
                     Defaults to ~/.cube/swegym-cube/huggingface_cache
                     (same as SWEGymBenchmarkConfig.install()).
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

from swegym_cube.benchmark import (
    TASKS_REQUIRING_ROOT,
    SWEGymBenchmarkConfig,
    _DATASET_NAME,
    _DATASET_SPLIT,
)
from swegym_cube.task import SWEGymTaskMetadata

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent.parent / "src" / "swegym_cube" / "task_metadata.json"
_DEFAULT_HF_CACHE = SWEGymBenchmarkConfig.cache_dir() / "huggingface_cache"

# The official 230-task lite subset is a separate HuggingFace dataset whose
# instance_ids are a subset of the full 2438-task SWE-Gym/SWE-Gym split. We load
# it once to stamp ``subset="lite"`` on the matching rows of the full set.
_LITE_DATASET_NAME = "SWE-Gym/SWE-Gym-Lite"

# SWE-Gym ships pre-built per-instance eval images under the `xingyaoww` Docker Hub
# namespace (same `sweb.eval.x86_64.<normalized_id>` recipe SWE-bench uses).
_DOCKER_NAMESPACE = "xingyaoww"
_IMAGE_TAG = "latest"


def _normalize_instance_id(instance_id: str) -> str:
    """Normalize instance_id for Docker image naming.

    SWE-Gym's published ``xingyaoww`` eval images use the older SWE-bench
    normalization that replaces ``__`` with ``_s_`` (not the newer ``_1776_``
    used by ``swebench/*`` Verified images), then lowercases. So
    ``getmoto__moto-5699`` → ``getmoto_s_moto-5699``.
    """
    return instance_id.replace("__", "_s_").lower()


def _get_docker_image(instance_id: str) -> str:
    normalized = _normalize_instance_id(instance_id)
    return f"{_DOCKER_NAMESPACE}/sweb.eval.x86_64.{normalized}:{_IMAGE_TAG}"


def _load_lite_instance_ids(hf_cache: Path) -> set[str]:
    """Return the set of instance_ids in the official 230-task lite subset.

    Loaded from the separate ``SWE-Gym/SWE-Gym-Lite`` dataset; its instance_ids
    are a subset of the full ``SWE-Gym/SWE-Gym`` split, so we use them to stamp
    ``subset="lite"`` on the matching full-set rows.
    """
    logger.info("Downloading %s from HuggingFace (for lite subset ids)...", _LITE_DATASET_NAME)
    ds = load_dataset(_LITE_DATASET_NAME, split=_DATASET_SPLIT, cache_dir=str(hf_cache))
    ids = {row["instance_id"] for row in ds}  # type: ignore[union-attr]
    logger.info("  %d lite instance_ids loaded", len(ids))
    return ids


def _build_task_metadata(rows: list[dict[str, Any]], lite_ids: set[str]) -> dict[str, SWEGymTaskMetadata]:
    """Build lightweight TaskMetadata from HF dataset rows.

    Only extracts public fields (repo, base_commit, version, subset).
    ``subset`` is ``"lite"`` if the instance is in the official lite subset, else
    ``"full"``. Heavy execution fields (problem_statement, patch, test_patch, etc.)
    live in the per-task execution cache written by install().
    """
    metadata: dict[str, SWEGymTaskMetadata] = {}
    for row in rows:
        iid = row["instance_id"]
        # abstract_description = first line of problem_statement, capped at 200 chars
        first_line = row["problem_statement"].split("\n", 1)[0]
        metadata[iid] = SWEGymTaskMetadata(
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
            version=row.get("version", ""),
            subset="lite" if iid in lite_ids else "full",
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
        output_path:  Destination path. Defaults to src/swegym_cube/task_metadata.json.
        hf_cache:     HuggingFace cache directory.
                      Defaults to ~/.cube/swegym-cube/huggingface_cache.
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
    ds = load_dataset(_DATASET_NAME, split=_DATASET_SPLIT, cache_dir=str(hf_cache))
    rows = list(ds)  # type: ignore[arg-type]
    logger.info("  %d tasks downloaded", len(rows))

    lite_ids = _load_lite_instance_ids(hf_cache)

    metadata = _build_task_metadata(rows, lite_ids)  # type: ignore
    n_lite = sum(1 for tm in metadata.values() if tm.subset == "lite")
    logger.info("  %d/%d tasks stamped subset='lite'", n_lite, len(metadata))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # mode="json" so set-valued fields (ContainerConfig.requires) serialize as lists.
    output_path.write_text(json.dumps([tm.model_dump(mode="json") for tm in metadata.values()], indent=2))
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
