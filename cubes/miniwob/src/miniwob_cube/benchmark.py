from importlib.resources import files
from typing import ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig

from miniwob_cube.task import MiniWobTaskConfig, MiniWobTaskMetadata


class MiniWobBenchmark(Benchmark["MiniWobBenchmarkConfig"]):
    """Runtime pair for MiniWoB task enumeration.

    Each live MiniWob task owns its own HTTP server so concurrent trajectories do
    not share browser HTML state or a server process.
    """

    def _setup(self) -> None:
        pass
    
    def close(self) -> None:
	pass

class MiniWobBenchmarkConfig(BenchmarkConfig[MiniWobTaskMetadata]):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="miniwob-cube",
        version="1.0.0",
        description="MiniWob++ browser automation benchmark tasks",
        num_tasks=125,
        tags=["browser", "web", "ui"],
    )
    task_config_class: ClassVar[type[TaskConfig]] = MiniWobTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = MiniWobBenchmark

    html_path: str = files("miniwob").joinpath("html").as_posix()  # type: ignore
    port: int | None = None
    remove_human_display: bool = True
    episode_max_time: int = 1000000
    server_start_timeout: float = 10.0
    server_start_poll_interval: float = 0.1

    def get_task_configs(self) -> Generator[MiniWobTaskConfig, None, None]:
        for tm in self.tasks().values():
            yield MiniWobTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                html_path=self.html_path,
                port=self.port,
                remove_human_display=self.remove_human_display,
                episode_max_time=self.episode_max_time,
                server_start_timeout=self.server_start_timeout,
                server_start_poll_interval=self.server_start_poll_interval,
            )
