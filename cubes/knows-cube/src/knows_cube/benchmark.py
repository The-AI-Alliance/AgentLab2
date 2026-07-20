"""KNOWS benchmark — targets Google Workspace, so there is no infrastructure to provision."""

import logging
from collections.abc import Generator
from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig
from cube.tool import ToolConfig
from pydantic import SerializeAsAny

from knows_cube.evaluator import missing_credentials
from knows_cube.task import KnowsTaskConfig, KnowsTaskMetadata, default_tool_config

logger = logging.getLogger(__name__)


class KnowsBenchmark(Benchmark["KnowsBenchmarkConfig"]):
    """Runtime pair.

    KNOWS targets Google Workspace — a remote SaaS — so there is no Docker stack,
    no declared resource, and nothing to launch or tear down.
    """

    def _setup(self) -> None:
        """Warn loudly if credentials are missing, since the failure mode is silent zeros."""
        missing = missing_credentials()
        if missing:
            logger.warning(
                "KNOWS credentials incomplete — every task will grade 0.0 with "
                "info['evaluation_error'] set. Missing: %s",
                "; ".join(missing),
            )

    def close(self) -> None:
        """Nothing to release. Created Drive files are intentionally left in place."""


class KnowsBenchmarkConfig(BenchmarkConfig[KnowsTaskMetadata]):
    """KNOWS — 110 Google Workspace authoring tasks (22 families x 5 instances).

    Requires live Google + Gemini credentials; without them every task scores 0.0
    with ``info["evaluation_error"]`` set. See the cube README.

    Filtering is user-land::

        config.subset_from_glob("workspace_kind", "slides")
        config.subset_from_glob("task_family_folder", "docs_*")
        config.subset_from_list(["knows.docs_1_formal_letter.1"])

    Regenerate ``task_metadata.json`` with ``scripts/generate_task_metadata.py``.
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="knows-cube",
        version="0.1.0",
        description="KNOWS — 110 Google Docs/Sheets/Slides authoring tasks graded by upstream evaluators",
        num_tasks=110,
        tags=["browser", "web", "google-workspace", "docs", "sheets", "slides", "knows"],
        named_subsets={
            "docs": ("workspace_kind", "docs"),
            "sheets": ("workspace_kind", "sheets"),
            "slides": ("workspace_kind", "slides"),
            "gradeable": ("is_gradeable", "True"),
        },
    )
    task_config_class: ClassVar[type[TaskConfig]] = KnowsTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = KnowsBenchmark

    tool_config: SerializeAsAny[ToolConfig] = default_tool_config()
    """SerializeAsAny is load-bearing: without it a BgymToolConfig round-trips
    through Ray as a bare ToolConfig and the tool silently comes back wrong."""

    agent_name: str = "cube-harness"
    settle_seconds: float = 8.0

    def get_task_configs(self) -> Generator[KnowsTaskConfig, None, None]:
        """Forward the cube-level knobs onto each task config.

        Dispatches through ``self.task_config_class`` rather than naming
        ``KnowsTaskConfig`` directly, so subclasses (the debug benchmark) get
        their own config type. Deliberately replicates the base class's
        seed-generator expansion — overriding this without it silently drops
        seed handling.
        """
        config_class = self.task_config_class
        for tm in self.tasks().values():
            seeds = list(self.seed_generator(tm)) if self.seed_generator is not None else [None]
            for seed in seeds:
                yield config_class(
                    metadata=tm,
                    tool_config=self.tool_config,
                    seed=seed,
                    agent_name=self.agent_name,
                    settle_seconds=self.settle_seconds,
                )
