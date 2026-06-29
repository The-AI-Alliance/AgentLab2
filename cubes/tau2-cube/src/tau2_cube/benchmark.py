"""Benchmark for tau2_cube — all tau2 domains in one catalog.

BenchmarkConfig is the serialisable registry; CubeBenchmark is the runtime pair
(no shared infra — tau2 envs are built per-task in CubeTaskConfig.make()).

Tasks from every domain in DOMAINS are loaded at import and namespaced as
``<domain>/<tau2_id>`` to avoid cross-domain id collisions. Each carries its
``domain`` field, so:

    CubeBenchmarkConfig().named_subset("airline")   # train on one domain
    CubeBenchmarkConfig()                            # all domains mixed

named_subsets glob on the ``domain`` field (see CubeTaskMetadata).
"""

from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from tau2.registry import registry

from tau2_cube.task import CubeTaskConfig, CubeTaskMetadata

DOMAINS = ["mock", "airline", "retail", "telecom"]


def _load_all_task_metadata(domains: list[str]) -> dict[str, TaskMetadata]:
    """Wrap every tau2 task across `domains` as a CubeTaskMetadata, keyed `<domain>/<id>`."""
    catalog: dict[str, TaskMetadata] = {}
    for domain in domains:
        # Call with NO args so each domain's own default split applies. Do NOT pass
        # None: telecom defaults to its curated "base" split (114 tasks), but None
        # means the full unsplit set (2285). The Callable[[Optional[str]], ...] type
        # hides the default, hence the type: ignore.
        for t in registry.get_tasks_loader(domain)():  # type: ignore[call-arg]
            task_id = f"{domain}/{t.id}"
            catalog[task_id] = CubeTaskMetadata(
                id=task_id,
                abstract_description=f"tau2 {domain} task: {t.id}",
                recommended_max_steps=30,
                domain=domain,
                tau2_task=t,
            )
    return catalog


_TASK_METADATA = _load_all_task_metadata(DOMAINS)  # built once at import


class CubeBenchmark(Benchmark):
    """Runtime pair — tau2 envs are per-task, so nothing shared to set up."""

    def _setup(self) -> None: ...

    def close(self) -> None: ...


class CubeBenchmarkConfig(BenchmarkConfig):
    """Registry of tau2 tasks across all domains."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="tau2-cube",
        version="0.1.0",
        description="tau2-bench conversational tool-use tasks (mock/airline/retail/telecom) as a cube.",
        num_tasks=len(_TASK_METADATA),
        tags=["tool-use", "conversational", "tau2"],
        named_subsets={dom: ("domain", dom) for dom in DOMAINS},
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = _TASK_METADATA
    task_config_class: ClassVar[type[TaskConfig]] = CubeTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = CubeBenchmark


if __name__ == "__main__":
    cfg = CubeBenchmarkConfig()
    print(f"{cfg.benchmark_metadata.num_tasks} tasks across {DOMAINS}")
    print("named subsets:", CubeBenchmarkConfig.named_subsets())
    for dom in DOMAINS:
        print(f"  {dom}: {len(cfg.named_subset(dom).tasks())} tasks")
