"""Benchmark for swebench-live-cube — SWE-bench Live with test-based validation."""

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
from datasets import load_dataset

from swebench_live_cube.task import SWEBenchLiveTaskConfig, SWEBenchLiveTaskMetadata

logger = logging.getLogger(__name__)

_DATASET_NAME = "SWE-bench-Live/SWE-bench-Live"
# Priority order for conflict resolution: earlier = higher priority.
# When the same instance_id appears in multiple splits with different data,
# the split with the highest priority wins.
_SPLIT_PRIORITY = ["verified", "full", "test", "lite"]


def _merge_rows_by_split(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, tuple[dict[str, Any], list[str]]]:
    """Merge rows across splits by instance_id.

    Returns {iid: (winning_row, splits_present)} using the priority order from
    _SPLIT_PRIORITY. Logs a warning for each instance_id whose data differs
    across splits so cube developers can spot upstream inconsistencies.
    """
    # First pass: collect all rows per instance_id across splits
    all_rows: dict[str, dict[str, dict[str, Any]]] = {}  # iid -> {split: row}
    for split in _SPLIT_PRIORITY:
        for row in rows_by_split.get(split, []):
            iid = row["instance_id"]
            all_rows.setdefault(iid, {})[split] = row

    # Second pass: pick the highest-priority row and report conflicts
    n_conflicts = 0
    result: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for iid, split_rows in all_rows.items():
        splits_present = [s for s in _SPLIT_PRIORITY if s in split_rows]
        winning_split = splits_present[0]  # highest priority
        winning_row = split_rows[winning_split]

        if len(splits_present) > 1:
            differing = [s for s in splits_present[1:] if split_rows[s] != winning_row]
            if differing:
                n_conflicts += 1
                diff_lines = []
                for other_split in differing:
                    other_row = split_rows[other_split]
                    changed_fields = {k for k in winning_row if winning_row[k] != other_row.get(k)}
                    for field in sorted(changed_fields):
                        diff_lines.append(
                            f"  field={field!r}:\n"
                            f"    {winning_split}: {repr(winning_row[field])[:200]}\n"
                            f"    {other_split}: {repr(other_row[field])[:200]}"
                        )
                logger.warning(
                    f"Conflict for {iid!r}: data differs between {differing} and {winning_split!r}:\n"
                    + "\n".join(diff_lines)
                    + f"\n  -> Using {winning_split!r}."
                )

        result[iid] = (winning_row, splits_present)

    if n_conflicts:
        logger.warning(
            f"{n_conflicts} task(s) had conflicting data across splits. Split priority used: {_SPLIT_PRIORITY}."
        )

    return result


def _build_execution_info(row: dict[str, Any]) -> dict[str, Any]:
    """Extract execution-only fields from a HF dataset row.

    These fields are only needed when a task runs; they are never loaded
    at import time. Stored in the per-task execution cache by install().
    """
    fail_to_pass = row["FAIL_TO_PASS"] if isinstance(row["FAIL_TO_PASS"], list) else []
    pass_to_pass = row["PASS_TO_PASS"] if isinstance(row["PASS_TO_PASS"], list) else []
    return {
        "problem_statement": row["problem_statement"],
        "hints_text": row.get("hints_text", ""),
        "patch": row["patch"],
        "test_patch": row["test_patch"],
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "test_cmds": row.get("test_cmds", []),
        "eval_timeout": 1800,
    }


# ---------------------------------------------------------------------------
# SWEBenchLiveBenchmark (runtime pair)
# ---------------------------------------------------------------------------


class SWEBenchLiveBenchmark(Benchmark["SWEBenchLiveBenchmarkConfig"]):
    """Runtime pair — publishes ``self._infra`` (stashed by the base
    ``Benchmark.__init__``) into ``runtime_context["infra"]`` so per-task
    container launches flow through ``Task.runtime_context``.
    """

    def _setup(self) -> None:
        """Publish the shared InfraConfig to runtime_context; containers are launched per-task."""
        if self._infra is not None:
            self._runtime_context["infra"] = self._infra
        logger.info(
            "SWEBenchLiveBenchmark ready with %d tasks (infra=%s)",
            self.config.num_tasks,
            self._infra.fingerprint() if self._infra is not None else "<none>",
        )

    def close(self) -> None:
        logger.info("SWE-bench Live benchmark closed")


# ---------------------------------------------------------------------------
# SWEBenchLiveBenchmarkConfig
# ---------------------------------------------------------------------------


class SWEBenchLiveBenchmarkConfig(BenchmarkConfig[SWEBenchLiveTaskMetadata]):
    """SWE-bench Live — continuously updated GitHub issue resolution benchmark."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="swebench-live-cube",
        version="0.1.0",
        description=(
            "SWE-bench Live — continuously updated, contamination-resistant GitHub issue resolution. "
            "By default the benchmark contains all tasks across all splits. "
            "Use cfg.named_subset('lite'), cfg.named_subset('verified'), etc. to get a specific split.\n"
            "\n"
            "CUBE DEVELOPER NOTES:\n"
            "---------------------\n"
            "task_metadata.json is a shipped package resource containing lightweight public fields. "
            "Heavy execution data (problem_statement, patch, test_patch, etc.) is stored in the "
            "per-task execution cache populated by install(). "
            "All tasks from all splits (test, lite, verified, full) are included, deduplicated by "
            "instance_id with split priority: verified > full > test > lite. "
            "Each task stores which splits it belongs to in the typed 'splits' field."
        ),
        tags=["swe", "github", "docker", "live"],
        num_tasks=1895,  # total unique tasks across all splits as of 2026-04-02
        # Splits overlap heavily: full(1887) ⊇ verified(499) ⊇ lite(300); test(1000) adds 8 unique tasks not in full.
        named_subsets={
            "test": ("splits", "*'test'*"),
            "lite": ("splits", "*'lite'*"),
            "verified": ("splits", "*'verified'*"),
            "full": ("splits", "*'full'*"),
            # Official gold-solvable subset: the 275 lite tasks whose gold patch resolves
            # under the root (Daytona) oracle. Tag stamped by create_task_metadata.py from
            # gold_subset.LITE_GOLD_SOLVABLE_JSON. NB: "*'lite'*" won't match "'lite-gold'".
            "lite-gold": ("splits", "*'lite-gold'*"),
        },
    )
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchLiveTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = SWEBenchLiveBenchmark

    # User-configurable fields
    include_hints: bool = False
    oracle_mode: bool = False

    # ------------------------------------------------------------------
    # Data lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def install(cls) -> None:
        """Download all SWE-bench Live splits and populate the per-task execution cache.

        The shipped task_metadata.json is a package resource and is not modified here.
        Downloads from HuggingFace (cached under cache_dir()/huggingface_cache/) and
        writes one JSON file per task (problem_statement, patch, test_patch, etc.)
        to ``task_config_class.task_execution_cache_dir()``.

        Safe to call multiple times: skips if the execution cache is already populated.
        """
        task_execution_info_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if task_execution_info_cache_dir.exists() and any(task_execution_info_cache_dir.iterdir()):
            logger.info("Execution cache already populated, skipping installation")
            return
        task_execution_info_cache_dir.mkdir(parents=True, exist_ok=True)

        # Download from HuggingFace into our own cache folder (not the default ~/.cache/huggingface)
        # load_dataset is idempotent: if the data is already cached there, no download occurs.
        hf_cache = str(cls.cache_dir() / "huggingface_cache")
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for split in _SPLIT_PRIORITY:
            logger.info(f"Downloading {_DATASET_NAME} split={split!r} from HuggingFace...")
            ds = load_dataset(_DATASET_NAME, split=split, cache_dir=hf_cache)
            rows_by_split[split] = list(ds)  # type: ignore[arg-type]
            logger.info(f"  {len(rows_by_split[split])} tasks in split={split!r}")

        merged = _merge_rows_by_split(rows_by_split)

        # Write one execution-cache file per task
        n = 0
        for iid, (row, _) in merged.items():
            exec_info = _build_execution_info(row)
            (task_execution_info_cache_dir / f"{iid}.json").write_text(json.dumps(exec_info))
            n += 1

        logger.info(f"Saved {n} execution cache files to {task_execution_info_cache_dir}")

    @classmethod
    def uninstall(cls) -> None:
        """Remove the per-task execution cache and the HuggingFace dataset cache.

        The shipped task_metadata.json is not removed.
        """
        task_execution_info_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if task_execution_info_cache_dir.exists():
            shutil.rmtree(task_execution_info_cache_dir)
            logger.info(f"Removed execution cache at {task_execution_info_cache_dir}")

        hf_cache = cls.cache_dir() / "huggingface_cache"
        if hf_cache.exists():
            shutil.rmtree(hf_cache)
            logger.info(f"Removed HuggingFace dataset cache at {hf_cache}")

    # ------------------------------------------------------------------
    # Factory / task generation
    # ------------------------------------------------------------------

    def make(self, infra: InfraConfig | None = None) -> SWEBenchLiveBenchmark:
        """Resolve a default infra of ``LocalInfraConfig`` if none provided, then
        delegate to the base ``BenchmarkConfig.make`` for provisioning + setup.
        """
        return cast(SWEBenchLiveBenchmark, super().make(infra=infra or LocalInfraConfig()))

    def get_task_configs(self) -> Generator[SWEBenchLiveTaskConfig, None, None]:
        """Yield TaskConfigs with include_hints and oracle_mode forwarded from benchmark settings."""
        for tm in self.tasks().values():
            yield SWEBenchLiveTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                include_hints=self.include_hints,
                oracle_mode=self.oracle_mode,
            )
