"""
OSWorldBenchmarkConfig / OSWorldBenchmark — CUBE benchmark for the OSWorld
desktop-automation suite.

Entry point::

    config = OSWorldBenchmarkConfig(tool_config=ComputerConfig())
    benchmark = config.make(infra=AWSInfraConfig())  # provisions VM image, publishes infra into runtime_context
    for tc in config.get_task_configs():
        task = tc.make(runtime_context=benchmark._runtime_context)
        obs, info = task.reset()
        ...
        task.close()
    benchmark.close()

Filter by domain or other metadata field::

    chrome_config = config.subset_from_glob("domain", "chrome")
"""

from __future__ import annotations

import enum
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import ClassVar, cast

from dotenv import load_dotenv

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.infra_local import LocalInfraConfig
from cube.resource import InfraConfig, ResourceConfig
from cube.task import RuntimeContext, TaskConfig, TaskMetadata

from osworld_cube._paths import OSWORLD_BASE_DIR, OSWORLD_REPO_DIR, OSWORLD_VM_DIR
from osworld_cube.computer import ComputerConfig
from osworld_cube.task import (
    OSWORLD_UBUNTU_RESOURCE,
    OSWorldExecutionInfo,
    OSWorldTask,
    OSWorldTaskMetadata,
)

logger = logging.getLogger(__name__)


# Pinned OSWorld commit for reproducibility
OSWORLD_COMMIT = "cb834f7"


# ---------------------------------------------------------------------------
# helper functions for install()
# ---------------------------------------------------------------------------


def ensure_proxy_config_in_env(env_path: Path = Path(".env")) -> None:
    """Append PROXY_CONFIG_FILE to .env if it is not already defined there.

    The value mirrors the default set in computer.py so that desktop_env
    resolves the path correctly regardless of the current working directory.
    """
    key = "PROXY_CONFIG_FILE"
    value = str(OSWORLD_REPO_DIR / "evaluation_examples" / "settings" / "proxy" / "dataimpulse.json")

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                logger.debug(f"{key} already present in {env_path}, skipping.")
                return

    with env_path.open("a") as f:
        f.write(f"\n{key}={value}\n")
    logger.info(f"Appended {key} to {env_path}")


def _fix_config_paths(config: list[dict]) -> list[dict]:
    """
    Prepend OSWorld repo path to settings_file paths in config items.

    Keeps relative paths working regardless of CWD.
    """
    result = deepcopy(config)
    for config_item in result:
        params = config_item.get("parameters", {})
        if "settings_file" in params:
            params["settings_file"] = str(OSWORLD_REPO_DIR / params["settings_file"])
    return result


def _build_task_execution_info_from_repo() -> dict[str, dict]:
    """
    Build heavy per-task execution info from the OSWorld repo.
    """
    assert OSWORLD_REPO_DIR.exists(), (
        f"OSWorld repo not found at {OSWORLD_REPO_DIR}. Run OSWorldBenchmarkConfig.install() to clone it first."
    )
    eval_examples_dir = OSWORLD_REPO_DIR / "evaluation_examples"
    exec_info_by_id: dict[str, dict] = {}

    for test_set_file in eval_examples_dir.glob("test_*.json"):
        with open(test_set_file) as f:
            tasks_by_domain: dict[str, list[str]] = json.load(f)
        for domain_name, task_ids in tasks_by_domain.items():
            for task_id in task_ids:
                task_file = eval_examples_dir / "examples" / domain_name / f"{task_id}.json"
                if not task_file.exists():
                    logger.warning("Task file not found: %s", task_file)
                    continue
                try:
                    with open(task_file) as f:
                        td = json.load(f)
                except Exception as e:
                    logger.error("Failed to load task %s: %s", task_id, e)
                    continue

                raw = {"config": td.get("config", []), "evaluator": td.get("evaluator", {})}
                if task_id in exec_info_by_id:
                    assert raw == exec_info_by_id[task_id], (
                        f"Task {task_id!r} appears in domain {domain_name!r} with content "
                        f"that conflicts with a previously loaded copy"
                    )
                    continue
                exec_info_by_id[task_id] = raw

    # Convert relative paths in config to absolute paths pointing to the repo.
    exec_info_by_id_abs = {
        task_id: {
            "config": _fix_config_paths(raw["config"]),
            "evaluator": raw["evaluator"],
        }
        for task_id, raw in exec_info_by_id.items()
    }
    logger.info("Built %d task execution info entries from OSWorld repo", len(exec_info_by_id_abs))
    return exec_info_by_id_abs


def _clone_osworld_repo() -> None:
    """Clone and pin the OSWorld repository to OSWORLD_COMMIT."""
    OSWORLD_BASE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/xlang-ai/OSWorld", str(OSWORLD_REPO_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", OSWORLD_COMMIT],
        cwd=str(OSWORLD_REPO_DIR),
        check=True,
    )
    OSWORLD_VM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# OSWorldTestSet
# ---------------------------------------------------------------------------


class OSWorldTestSet(str, enum.Enum):
    """Valid test-set index files shipped with the OSWorld repo."""

    TEST_ALL = "test_all.json"
    TEST_INFEASIBLE = "test_infeasible.json"
    TEST_NOGDRIVE = "test_nogdrive.json"
    TEST_SMALL = "test_small.json"


# ---------------------------------------------------------------------------
# OSWorldTaskConfig
# ---------------------------------------------------------------------------


class OSWorldTaskConfig(TaskConfig[OSWorldTaskMetadata]):
    """
    Serialisable config for a single OSWorld task.

    Heavy execution data (setup config, evaluator) is loaded lazily on the
    worker by ``make()`` from the per-task execution cache populated by
    ``OSWorldBenchmarkConfig.install()``.
    """

    use_som: bool = False
    """Set-of-Marks observation post-processing toggle, propagated from the benchmark config."""

    def verify_installed(self) -> None:
        """Fail fast if the per-task cache or the OSWorld repo are missing."""
        cache_dir = type(self).task_execution_cache_dir()
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise RuntimeError(
                f"OSWorld per-task execution cache is empty at {cache_dir}. "
                f"Run `cube install osworld-cube` (or `OSWorldBenchmarkConfig.install()`) "
                f"on this worker first."
            )
        if not OSWORLD_REPO_DIR.exists():
            raise RuntimeError(
                f"OSWorld repo not found at {OSWORLD_REPO_DIR}. "
                f"Run `cube install osworld-cube` (or `OSWorldBenchmarkConfig.install()`) "
                f"on this worker first."
            )

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> OSWorldTask:
        """Instantiate OSWorldTask from this config.

        Loads typed execution info from the per-task cache populated by
        ``OSWorldBenchmarkConfig.install()`` and surfaces it via
        ``OSWorldTask.execution_info``.
        """
        self.verify_installed()
        raw = self.load_task_execution_info()
        execution_info = OSWorldExecutionInfo.model_validate(raw)
        return OSWorldTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config or ComputerConfig(),
            runtime_context=runtime_context,
            use_som=self.use_som,
        )


# ---------------------------------------------------------------------------
# OSWorldBenchmark (runtime pair)
# ---------------------------------------------------------------------------


class OSWorldBenchmark(Benchmark["OSWorldBenchmarkConfig"]):
    """Runtime pair — publishes ``self._infra`` (stashed by the base
    ``Benchmark.__init__``) into ``runtime_context["infra"]`` so per-task VM
    launches flow naturally through ``Task.runtime_context``.
    """

    def _setup(self) -> None:
        provider = type(self._infra).__name__ if self._infra is not None else "<none>"
        logger.info(f"Setting up OSWorldBenchmark (provider={provider})...")

        self._runtime_context["osworld"] = True
        if self._infra is not None:
            self._infra.cleanup_stale()
            self._runtime_context["infra"] = self._infra

        logger.info("OSWorldBenchmark ready with %d tasks", self.config.num_tasks)

    def close(self) -> None:
        """No global VM resources to release here — VM lifecycle is per-task."""
        logger.info("Closing OSWorldBenchmark — no global resources to release")


# ---------------------------------------------------------------------------
# OSWorldBenchmarkConfig
# ---------------------------------------------------------------------------


class OSWorldBenchmarkConfig(BenchmarkConfig[OSWorldTaskMetadata]):
    """
    CUBE BenchmarkConfig wrapping the OSWorld desktop-automation evaluation suite.

    Reference: https://github.com/xlang-ai/OSWorld

    Class-level attributes (required by cube.benchmark.BenchmarkConfig):
        benchmark_metadata:  ClassVar[BenchmarkMetadata]
        task_metadata:       ClassVar[dict[str, OSWorldTaskMetadata]]  (auto-loaded from task_metadata.json)
        task_config_class:   type[TaskConfig] = OSWorldTaskConfig
        benchmark_class:     type[Benchmark]  = OSWorldBenchmark

    Instance fields:
        tool_config:  ComputerConfig (action_space selects variant)
        use_som:      bool — Set-of-Marks mode for all tasks
        resources:    by default declares OSWORLD_UBUNTU_RESOURCE so make(infra)
                      provisions the VM image idempotently.

    Filter by any metadata field::

        cfg = OSWorldBenchmarkConfig().subset_from_glob("domain", "chrome")
    """

    # ------------------------------------------------------------------
    # Required class variables
    # ------------------------------------------------------------------

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="osworld-cube",
        version="1.0.0",
        description=("OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"),
        authors=["Tianbao Xie et al."],
        license="CC-BY-4.0",
        requirements={
            "vm": "Ubuntu 22.04 (docker or vmware)",
            "ram_gb": 8,
            "disk_gb": 40,
        },
        num_tasks=368,
        tags=["desktop", "gui", "multimodal"],
        named_subsets={
            "test_all": ("test_sets", "*'test_all'*"),
            "test_small": ("test_sets", "*'test_small'*"),
            "test_nogdrive": ("test_sets", "*'test_nogdrive'*"),
            "test_infeasible": ("test_sets", "*'test_infeasible'*"),
        },
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]]
    """Auto-loaded from task_metadata.json shipped next to this module. Values are
    ``OSWorldTaskMetadata`` instances by way of the ``_type`` discriminator;
    ``self.tasks()`` narrows the read view to ``Mapping[str, OSWorldTaskMetadata]``."""

    task_config_class: ClassVar[type[TaskConfig]] = OSWorldTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = OSWorldBenchmark

    # ------------------------------------------------------------------
    # Instance fields
    # ------------------------------------------------------------------

    tool_config: ComputerConfig = ComputerConfig()  # type: ignore[assignment]
    """Default computer-tool config; overridden per-construction."""

    use_som: bool = False
    """Enable Set-of-Marks annotation for all tasks in this benchmark run."""

    resources: list[ResourceConfig] = [OSWORLD_UBUNTU_RESOURCE]
    """VM image required to run OSWorld tasks (declared for the harness resource lifecycle)."""

    # ------------------------------------------------------------------
    # overrides
    # ------------------------------------------------------------------

    @classmethod
    def cache_dir(cls) -> Path:
        """OSWORLD_BASE_DIR — the OSWorld repo clone lives here. The per-task
        execution cache is reachable via
        ``OSWorldTaskConfig.task_execution_cache_dir()`` which returns
        ``OSWORLD_BASE_DIR / 'tasks_execution_info'`` (also under this tree).
        """
        return OSWORLD_BASE_DIR

    def make(self, infra: InfraConfig | None = None) -> OSWorldBenchmark:
        """Resolve a default infra of ``LocalInfraConfig`` if none provided, then
        delegate to the base ``BenchmarkConfig.make`` for provisioning + setup.
        """
        return cast(OSWorldBenchmark, super().make(infra=infra or LocalInfraConfig()))

    def get_task_configs(self) -> Generator[OSWorldTaskConfig, None, None]:
        """Yield OSWorldTaskConfig objects, propagating use_som from the benchmark."""
        for tm in self.tasks().values():
            yield OSWorldTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                use_som=self.use_som,
            )

    # ------------------------------------------------------------------
    # install() / uninstall()
    # ------------------------------------------------------------------

    @classmethod
    def install(cls) -> None:
        """Clone the OSWorld repo (if missing) and populate the per-task execution cache.

        task_metadata.json is a shipped package resource and is NOT generated here.
        Run scripts/create_task_metadata.py to regenerate it from the repo.
        """
        if not cls.task_metadata:
            raise RuntimeError(
                "task_metadata is empty — task_metadata.json is missing or was not loaded. "
                "Run scripts/create_task_metadata.py to generate it."
            )

        logger.info("Installing OSWorld benchmark...")
        # Repo is needed at task execution time; always ensure it is present.
        if not OSWORLD_REPO_DIR.exists():
            _clone_osworld_repo()
            logger.info(f"OSWorld repo cloned to {OSWORLD_REPO_DIR}")
        else:
            logger.info(f"OSWorld repo already present at {OSWORLD_REPO_DIR}")
        ensure_proxy_config_in_env()
        load_dotenv()  # Load the .env file
        logger.info(f"Set PROXY_CONFIG_FILE={os.environ.get('PROXY_CONFIG_FILE', 'not set')}")

        exec_info_by_id_abs = _build_task_execution_info_from_repo()

        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        exec_cache_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for task_id, exec_info in exec_info_by_id_abs.items():
            cache_file = exec_cache_dir / f"{task_id}.json"
            new_content = json.dumps(exec_info, indent=2)
            if cache_file.exists():
                if cache_file.read_text() == new_content:
                    continue
                else:
                    logger.warning(
                        f"Execution cache for task {task_id} already exists but content differs from repo; overwriting"
                    )
            cache_file.write_text(new_content)
            written += 1

        logger.info(f"Wrote {written} execution cache files to {exec_cache_dir}")
        logger.info("OSWorldBenchmarkConfig.install() done")

    @classmethod
    def uninstall(cls) -> None:
        """Remove the execution cache and the cloned OSWorld repo."""
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists():
            shutil.rmtree(exec_cache_dir)
            logger.info(f"Removed execution cache at {exec_cache_dir}")
        if OSWORLD_REPO_DIR.exists():
            shutil.rmtree(OSWORLD_REPO_DIR)
            logger.info(f"Removed OSWorld repo at {OSWORLD_REPO_DIR}")
