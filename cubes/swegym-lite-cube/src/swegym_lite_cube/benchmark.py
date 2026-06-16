"""Benchmark for swegym-lite-cube — SWE-Gym Lite with test-based validation."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Generator
from typing import Any, ClassVar, cast

from cube import LocalInfraConfig
from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.resource import InfraConfig
from cube.task import TaskConfig

from swegym_lite_cube.task import SWEGymLiteTaskConfig, SWEGymLiteTaskMetadata

logger = logging.getLogger(__name__)

_DATASET_NAME = "SWE-Gym/SWE-Gym-Lite"
_DATASET_SPLIT = "train"


def _build_execution_info(row: dict[str, Any]) -> dict[str, Any]:
    """Extract execution-only fields from a HuggingFace dataset row.

    These fields are only needed when a task runs; they are never loaded at
    import time. Stored in the per-task execution cache by install().
    """
    return {
        "problem_statement": row["problem_statement"],
        "patch": row["patch"],
        "test_patch": row["test_patch"],
        "fail_to_pass": json.loads(row["FAIL_TO_PASS"])
        if isinstance(row["FAIL_TO_PASS"], str)
        else row["FAIL_TO_PASS"],
        "pass_to_pass": json.loads(row["PASS_TO_PASS"])
        if isinstance(row["PASS_TO_PASS"], str)
        else row["PASS_TO_PASS"],
        "eval_timeout": 1800,
    }


# ---------------------------------------------------------------------------
# SWEGymLiteBenchmark (runtime pair)
# ---------------------------------------------------------------------------


class SWEGymLiteBenchmark(Benchmark["SWEGymLiteBenchmarkConfig"]):
    """Runtime pair — publishes ``self._infra`` (stashed by the base
    ``Benchmark.__init__``) into ``runtime_context["infra"]`` so per-task
    container launches flow through ``Task.runtime_context``.
    """

    def _setup(self) -> None:
        """Publish the shared InfraConfig to runtime_context; containers are launched per-task."""
        if self._infra is not None:
            self._runtime_context["infra"] = self._infra
        logger.info(
            "SWEGymLiteBenchmark ready with %d tasks (infra=%s)",
            self.config.num_tasks,
            self._infra.fingerprint() if self._infra is not None else "<none>",
        )

    def close(self) -> None:
        logger.info("SWE-Gym Lite benchmark closed")


# ---------------------------------------------------------------------------
# SWEGymLiteBenchmarkConfig
# ---------------------------------------------------------------------------


class SWEGymLiteBenchmarkConfig(BenchmarkConfig[SWEGymLiteTaskMetadata]):
    """SWE-Gym Lite — 230 real-world GitHub issues with test-based validation."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="swegym-lite-cube",
        version="0.1.0",
        description="SWE-Gym Lite — 230 executable real-world GitHub issues with test-based validation",
        num_tasks=230,
        tags=["swe", "github", "docker", "training"],
    )
    task_config_class: ClassVar[type[TaskConfig]] = SWEGymLiteTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = SWEGymLiteBenchmark

    # User-configurable fields
    oracle_mode: bool = False

    # ------------------------------------------------------------------
    # Data lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def install(cls) -> None:
        """Populate the per-task execution cache from HuggingFace.

        Downloads heavy fields (problem_statement, patch, test_patch, etc.) and writes
        one JSON file per task into ``task_config_class.task_execution_cache_dir()``.
        Idempotent: skips if the cache directory already exists and is non-empty. If
        the HuggingFace data has not been downloaded yet, it is fetched into
        ``cache_dir()/huggingface_cache/``.
        """
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists() and any(exec_cache_dir.iterdir()):
            logger.info("Execution cache already populated, skipping installation")
            return
        exec_cache_dir.mkdir(parents=True, exist_ok=True)

        # Download into our own cache folder (not the default ~/.cache/huggingface).
        # load_dataset is idempotent: if the data is already cached there, no download occurs.
        from datasets import load_dataset

        hf_cache = cls.cache_dir() / "huggingface_cache"
        logger.info(f"Downloading {_DATASET_NAME} from HuggingFace (cache: {hf_cache})...")
        ds = load_dataset(_DATASET_NAME, split=_DATASET_SPLIT, cache_dir=str(hf_cache))
        logger.info(f"  {len(ds)} tasks loaded")  # type: ignore[arg-type]

        n = 0
        for row in ds:
            iid = row["instance_id"]  # type: ignore
            (exec_cache_dir / f"{iid}.json").write_text(json.dumps(_build_execution_info(row)))  # type: ignore
            n += 1

        logger.info(f"Saved {n} execution cache files to {exec_cache_dir}")

    @classmethod
    def uninstall(cls) -> None:
        """Remove the per-task execution cache and the HuggingFace dataset cache.

        The shipped task_metadata.json is not removed.
        """
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists():
            shutil.rmtree(exec_cache_dir)
            logger.info(f"Removed execution cache at {exec_cache_dir}")

        hf_cache = cls.cache_dir() / "huggingface_cache"
        if hf_cache.exists():
            shutil.rmtree(hf_cache)
            logger.info(f"Removed HuggingFace dataset cache at {hf_cache}")

    # ------------------------------------------------------------------
    # Factory / task generation
    # ------------------------------------------------------------------

    def make(self, infra: InfraConfig | None = None) -> SWEGymLiteBenchmark:
        """Resolve a default infra of ``LocalInfraConfig`` if none provided, then
        delegate to the base ``BenchmarkConfig.make`` for provisioning + setup.
        """
        return cast(SWEGymLiteBenchmark, super().make(infra=infra or LocalInfraConfig()))

    def get_task_configs(self) -> Generator[SWEGymLiteTaskConfig, None, None]:
        """Yield TaskConfigs with oracle_mode forwarded from benchmark settings."""
        for tm in self.tasks().values():
            yield SWEGymLiteTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                oracle_mode=self.oracle_mode,
            )


# ---------------------------------------------------------------------------
# Tasks that require a root container (cube-harness#446)
# ---------------------------------------------------------------------------

# SWE-Gym images (xingyaoww/sweb.eval.x86_64.*) follow the same SWE-bench build recipe
# and run ``USER root``. Most tasks run fine on a non-root infra, but some ship a
# root-owned, non-writable package subdir even when ``/testbed`` itself is world-writable.
# A non-root runtime user (the EAI toolkit pins uid 13011) cannot make those files writable
# by any non-root means (can't chmod/chown — not owner; can't rename or rm the dir — needs
# write on the dir), so ``git apply`` of the gold patch — or any agent edit — dies with
# "Permission denied" and a *correct* fix silently scores 0. Declaring ``container:root``
# makes a non-root infra report these tasks incompatible at ``BenchmarkConfig.make()``
# (raises ``IncompatibleInfraError`` — no spend, no silent 0); root-capable infras
# (daytona/local/aws/azure) run them normally. See cube-harness#446.
#
# Unlike swebench-verified-cube, this set starts EMPTY: the offending instances must be
# identified empirically for SWE-Gym Lite via the gold-patch differential (toolkit-fail +
# daytona-root-pass). Until then, the fail-loud probe in ``SWEGymLiteTask.reset()`` (#452)
# protects every (untagged) task at runtime — a non-root infra that hits an unpatchable
# root-owned dir raises ``IncompatibleInfraError`` rather than silently scoring 0. Populate
# this set once the differential surfaces concrete instance_ids.
#
# ``scripts/create_task_metadata.py`` imports it and stamps ``requires={"container:root"}``
# into the ``container_config`` of these tasks in the generated ``task_metadata.json``. The
# capability gate reads ``container_config.requirements()`` at ``make()`` — no runtime patching.
TASKS_REQUIRING_ROOT: frozenset[str] = frozenset()
