"""BrowseCompBenchmark: 1,266 web information-retrieval tasks."""

import csv
import io
import json
import logging
import shutil
import urllib.request
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig

from browsercomp_cube.task import BrowseCompTaskConfig, BrowseCompTaskMetadata

logger = logging.getLogger(__name__)

_DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
_CSV_FILENAME = "browse_comp_test_set.csv"


class BrowseCompBenchmark(Benchmark["BrowseCompBenchmarkConfig"]):
    """BrowseComp benchmark: 1,266 hard web information-retrieval tasks."""

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class BrowseCompBenchmarkConfig(BenchmarkConfig[BrowseCompTaskMetadata]):
    """Serializable BrowseComp benchmark configuration."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="browsercomp-cube",
        version="0.1.0",
        description="BrowseComp benchmark — hard web information retrieval requiring multi-step browsing",
        num_tasks=1266,
        tags=["web", "browser", "reasoning", "nlp"],
        named_subsets={
            "art": ("topic", "Art"),
            "geography": ("topic", "Geography"),
            "history": ("topic", "History"),
            "music": ("topic", "Music"),
            "other": ("topic", "Other"),
            "politics": ("topic", "Politics"),
            "science-and-technology": ("topic", "Science & technology"),
            "sports": ("topic", "Sports"),
            "tv-shows-and-movies": ("topic", "TV shows & movies"),
            "video-games": ("topic", "Video games"),
        },
    )
    task_config_class: ClassVar[type[TaskConfig]] = BrowseCompTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = BrowseCompBenchmark

    # LLM judge used to grade answers. Has a default so the config is
    # instantiable for introspection (registry compliance, `cube list`);
    # override per-experiment. The deterministic debug benchmark doesn't use it.
    scorer_model: str = "openai/gpt-4o"

    @classmethod
    def install(cls) -> None:
        """Download the encrypted dataset and split it into the per-task execution cache.

        Downloads ``browse_comp_test_set.csv`` from the OpenAI public blob into
        ``cache_dir()`` (idempotent), then writes one JSON file per task into
        ``task_execution_cache_dir()`` containing the still-encrypted
        ``{problem, answer, canary}`` triple. Decryption happens at task make()
        time so cleartext never lands on disk.

        The shipped task_metadata.json is a package resource and is not modified.
        To regenerate task_metadata.json (developer use only), run
        ``scripts/generate_task_metadata.py``.
        """
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists() and any(exec_cache_dir.iterdir()):
            logger.info("Execution cache already populated, skipping installation")
            return
        exec_cache_dir.mkdir(parents=True, exist_ok=True)

        csv_path = cls._download_dataset()
        text = csv_path.read_text(encoding="utf-8")
        n = 0
        for idx, row in enumerate(csv.DictReader(io.StringIO(text))):
            task_id = f"browsecomp-{idx:04d}"
            (exec_cache_dir / f"{task_id}.json").write_text(
                json.dumps({"problem": row["problem"], "answer": row["answer"], "canary": row["canary"]})
            )
            n += 1
        logger.info("Wrote %d encrypted records to %s", n, exec_cache_dir)

    @classmethod
    def uninstall(cls) -> None:
        """Remove the per-task execution cache and the cached source CSV."""
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists():
            shutil.rmtree(exec_cache_dir)
            logger.info("Removed execution cache at %s", exec_cache_dir)
        csv_path = cls.cache_dir() / _CSV_FILENAME
        if csv_path.exists():
            csv_path.unlink()
            logger.info("Removed cached dataset at %s", csv_path)

    @classmethod
    def _download_dataset(cls) -> Path:
        """Download the source CSV into ``cache_dir()`` if not already present."""
        csv_path = cls.cache_dir() / _CSV_FILENAME
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s -> %s", _DATASET_URL, csv_path)
            urllib.request.urlretrieve(_DATASET_URL, csv_path)
        return csv_path

    def get_task_configs(self) -> Generator[BrowseCompTaskConfig, None, None]:
        for tm in self.tasks().values():
            yield BrowseCompTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                scorer_model=self.scorer_model,
            )
