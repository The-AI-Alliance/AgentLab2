import json
import logging
import os
import random
import subprocess
import tempfile
import time
import urllib.request
from random import shuffle
from typing import Any, TextIO

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.miniwob.task import MiniWobTask

logger = logging.getLogger(__name__)


class MiniWobBenchmark(Benchmark):
    metadata: dict = {
        "name": "miniwob",
        "description": "MiniWob benchmark for web-based tasks",
        "task_cls": MiniWobTask.__name__,
    }
    dataset_dir: str = "./data/miniwob-plusplus"
    port: int = 8000
    remove_human_display: bool = True
    episode_max_time: int = 1000000
    shuffle: bool = True
    shuffle_seed: int = 42

    # Runtime state (not serialized)
    _server_process: subprocess.Popen | None = None
    _stdout_file: TextIO | None = None
    _stderr_file: TextIO | None = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.tasks = [
            MiniWobTask(
                id=task["subdomain"],
                desc=task["desc"],
                subdomain=task["subdomain"],
                base_url=self.base_url,
                remove_human_display=self.remove_human_display,
                episode_max_time=self.episode_max_time,
            )
            for task in self.load_task_infos()
        ]
        if self.shuffle:
            random.seed(self.shuffle_seed)
            shuffle(self.tasks)

    def load_task_infos(self) -> list[dict]:
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        _tasks_file = os.path.join(_module_dir, "miniwob_tasks.json")
        with open(_tasks_file) as f:
            task_infos = json.load(f)
        return task_infos

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/miniwob"

    def setup(self):
        html_path = os.path.join(self.dataset_dir, "miniwob", "html")
        tmp_dir = tempfile.gettempdir()
        self._stdout_file = open(os.path.join(tmp_dir, "miniwob_server_stdout.log"), "w")
        self._stderr_file = open(os.path.join(tmp_dir, "miniwob_server_stderr.log"), "w")
        self._server_process = subprocess.Popen(
            ["python", "-m", "http.server", str(self.port)],
            cwd=html_path,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
        )
        time.sleep(1)
        # Check if the server is running by attempting to connect
        try:
            urllib.request.urlopen(f"{self.base_url}/", timeout=5)
            logger.info(f"MiniWob server responding at {self.base_url}")
        except Exception as e:
            self.close()
            raise RuntimeError(f"MiniWob server failed to respond: {e}")

    def close(self):
        """Shutdown the MiniWob server and close file handles."""
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
