from io import TextIOWrapper
import logging
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.resource import InfraConfig
from cube.task import TaskConfig

from miniwob_cube.task import MiniWobTaskConfig, MiniWobTaskMetadata

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Ask the OS for a free TCP port (so parallel benchmark instances don't collide on 8000)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


class MiniWobBenchmark(Benchmark["MiniWobBenchmarkConfig"]):
    """Runtime pair — owns the local HTTP server process serving MiniWob HTML."""

    def __init__(self, config: "MiniWobBenchmarkConfig", infra: InfraConfig | None = None) -> None:
        super().__init__(config, infra=infra)
        self._server_process: subprocess.Popen | None = None
        self._stdout_file: TextIOWrapper | None = None
        self._stderr_file: TextIOWrapper | None = None

    def _setup(self) -> None:
        cfg = self.config
        if cfg.port == 0:  # auto-assign a free port — each parallel Ray worker gets its own server
            cfg.port = _find_free_port()
        tmp_dir = Path(tempfile.gettempdir())
        self._stdout_file = open(tmp_dir / f"miniwob_server_{cfg.port}_stdout.log", "w")
        self._stderr_file = open(tmp_dir / f"miniwob_server_{cfg.port}_stderr.log", "w")
        logger.info(f"Starting MiniWob server at port {cfg.port} serving from {cfg.html_path}...")
        self._server_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(cfg.port)],
            cwd=cfg.html_path,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
        )
        startup_deadline = time.monotonic() + cfg.server_start_timeout
        last_response_error: Exception | None = None

        while time.monotonic() < startup_deadline:
            if self._server_process.poll() is not None:
                self._stderr_file.flush()
                stderr_path = Path(tempfile.gettempdir()) / f"miniwob_server_{cfg.port}_stderr.log"
                stderr_content = stderr_path.read_text() if stderr_path.exists() else "No stderr available"
                returncode = self._server_process.returncode
                self.close()
                raise RuntimeError(f"MiniWob server failed to start (exit code {returncode}): {stderr_content}")

            try:
                urllib.request.urlopen(cfg.base_url, timeout=1)
                logger.info(f"MiniWob server responding at {cfg.base_url}")
                break
            except Exception as e:
                last_response_error = e
                time.sleep(cfg.server_start_poll_interval)
        else:
            self.close()
            raise RuntimeError(
                f"MiniWob server failed to respond at {cfg.base_url} within {cfg.server_start_timeout:.1f}s"
            ) from last_response_error

        self._runtime_context["base_url"] = cfg.base_url

    def close(self) -> None:
        if self._server_process is not None:
            logger.info("Shutting down MiniWob server...")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, killing...")
                self._server_process.kill()
            self._server_process = None

        if self._stdout_file is not None:
            self._stdout_file.close()
            self._stdout_file = None

        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None


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
    port: int = 8000  # set 0 to auto-assign a free port (needed for many parallel benchmark instances)
    remove_human_display: bool = True
    episode_max_time: int = 1000000
    server_start_timeout: float = 10.0
    server_start_poll_interval: float = 0.1

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/miniwob"

    def get_task_configs(self) -> Generator[MiniWobTaskConfig, None, None]:
        for tm in self.tasks().values():
            yield MiniWobTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                base_url=self.base_url,
                remove_human_display=self.remove_human_display,
                episode_max_time=self.episode_max_time,
            )
