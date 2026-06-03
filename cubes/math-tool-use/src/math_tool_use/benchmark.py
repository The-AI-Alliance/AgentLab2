import json
import logging
import os
from typing import ClassVar, Iterator

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from datasets import load_dataset

from math_tool_use.task import MathToolUseTaskConfig, MathToolUseTaskMetadata

logger = logging.getLogger(__name__)

_DATASET_NAMES = ["open_reasoner_zero_57k", "open_reasoner_zero_extended_72k", "aime_2025"]
_DATASET_URLS = {
    "open_reasoner_zero_57k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json",
    "open_reasoner_zero_extended_72k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_72k_collection_extended.json",
}


def _max_tasks_per_dataset() -> int | None:
    """Dev/plumbing cap: MATH_TOOL_MAX_TASKS_PER_DATASET limits tasks per dataset (None = all)."""
    raw = os.environ.get("MATH_TOOL_MAX_TASKS_PER_DATASET")
    return int(raw) if raw else None


def _load_aime_rows(year: int, upsample_factor: int = 0) -> list[dict]:
    if year == 2025:
        aime = load_dataset("MathArena/aime_2025", split="train")
    else:
        aime = load_dataset("AI-MO/aimo-validation-aime", split="train").filter(lambda x: str(year) in x["url"])
    name = f"aime_{year}" + ("" if upsample_factor > 0 else "_original")
    rows = [{"dataset": name, "task": it["problem"], "answer": "\\boxed{" + str(it["answer"]) + "}"} for it in aime]
    return rows * upsample_factor if upsample_factor > 0 else rows


def _load_dataset_rows(dataset_name: str) -> list[dict]:
    """Return [{dataset, task, answer}] for one dataset (HF-cached after first download)."""
    if "open_reasoner" in dataset_name:
        ds = load_dataset("json", data_files=_DATASET_URLS[dataset_name], split="train")
        # ORZ items sometimes carry a preamble in the value; we use it as-is.
        return [
            {
                "dataset": dataset_name,
                "task": it["0"]["value"],
                "answer": "\\boxed{" + it["1"]["ground_truth"]["value"] + "}",
            }
            for it in ds
        ]
    if "aime" in dataset_name:
        year = int(dataset_name.split("_")[1])
        return _load_aime_rows(year, upsample_factor=0 if dataset_name.endswith("_original") else 16)
    return []


def _iter_tasks() -> Iterator[dict]:
    """Yield {id, dataset, question, expected} for every task (respecting the dev cap).

    Shared by ``load_task_metadata`` (lightweight registry) and ``install`` (heavy cache) so the
    download happens once per process and the two stay in sync.
    """
    cap = _max_tasks_per_dataset()
    for dataset_name in _DATASET_NAMES:
        rows = _load_dataset_rows(dataset_name)
        if not rows:
            logger.warning("math-tool-use: no rows for dataset %s", dataset_name)
            continue
        for i, t in enumerate(rows):
            if cap is not None and i >= cap:
                break
            yield {"id": f"{dataset_name}_{i}", "dataset": t["dataset"], "question": t["task"], "expected": t["answer"]}


def load_task_metadata() -> dict[str, TaskMetadata]:
    """Build the LIGHTWEIGHT eager registry — id/split/dataset only (no question/expected)."""
    md: dict[str, TaskMetadata] = {}
    for t in _iter_tasks():
        md[t["id"]] = MathToolUseTaskMetadata(
            id=t["id"],
            split="train",
            dataset=t["dataset"],
            abstract_description="Solve a math problem with Python tool use; submit a LaTeX \\boxed{} answer.",
            recommended_max_steps=3,
        )
    logger.info("math-tool-use: %d task metadata entries (lightweight)", len(md))
    return md


_TASK_METADATA: dict[str, TaskMetadata] = load_task_metadata()


class MathToolUseBenchmark(Benchmark["MathToolUseBenchmarkConfig"]):
    """Runtime pair — math tasks need no shared infrastructure (the sandbox is an external service)."""

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class MathToolUseBenchmarkConfig(BenchmarkConfig[MathToolUseTaskMetadata]):
    """Math-via-tool-use benchmark: solve a problem with Python, submit a LaTeX answer.

    Heavy per-task data (question/expected) is written to the per-task execution cache by
    ``install()`` and lazy-loaded in ``MathToolUseTaskConfig.make()``.
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="math-tool-use",
        version="0.1.0",
        description="Solve math by planning, using deterministic Python calls, then submitting a LaTeX answer",
        num_tasks=len(_TASK_METADATA),
        tags=["math", "tool-use", "python"],
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = _TASK_METADATA
    task_config_class: ClassVar[type[TaskConfig]] = MathToolUseTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = MathToolUseBenchmark

    @classmethod
    def install(cls) -> None:
        """Write each task's heavy {question, expected} to the per-task execution cache.

        Idempotent. Read back lazily by ``MathToolUseTaskConfig.make()`` via
        ``load_task_execution_info()``. Must NOT mutate ``task_metadata``.
        """
        super().install()
        cache_dir = cls.task_config_class.task_execution_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        written = skipped = 0
        for t in _iter_tasks():
            dest = cache_dir / f"{t['id']}.json"
            if dest.exists():  # idempotent; another worker already wrote it
                skipped += 1
                continue
            # Atomic write (temp + os.replace) so concurrent install() across Ray workers
            # never exposes a partially-written file to a reader in make().
            tmp = cache_dir / f".{t['id']}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps({"question": t["question"], "expected": t["expected"]}))
            os.replace(tmp, dest)
            written += 1
        logger.info("math-tool-use: execution cache at %s (wrote %d, skipped %d existing)", cache_dir, written, skipped)

    # cube_rl drives the instantiated config object directly through
    # install() + setup() + get_task_configs() + close(); math needs no runtime infra.
    def setup(self) -> None:
        pass

    def close(self) -> None:
        pass
