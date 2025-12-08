import os
import subprocess
import tempfile
import time
import urllib.request
from typing import Any

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.miniwob.all_tasks import ALL_MINIWOB_TASKS
from agentlab2.benchmarks.miniwob.task import MiniWobTask


class MiniWobBenchmark(Benchmark):
    metadata: dict = {
        "name": "miniwob",
        "description": "MiniWob benchmark for web-based tasks",
        "task_cls": MiniWobTask.__name__,
    }
    dataset_dir: str = "./data/miniwob-plusplus"
    base_url: str = "http://localhost:8000/miniwob"
    remove_human_display: bool = True
    episode_max_time: int = 1000000

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.tasks = [
            MiniWobTask(
                id=task.subdomain,
                desc=task.desc,
                subdomain=task.subdomain,
                base_url=self.base_url,
                remove_human_display=self.remove_human_display,
                episode_max_time=self.episode_max_time,
            )
            for task in ALL_MINIWOB_TASKS
        ]

    def prepare(self):
        html_path = os.path.join(self.dataset_dir, "miniwob", "html")
        tmp_dir = tempfile.gettempdir()
        stdout_file = open(os.path.join(tmp_dir, "miniwob_server_stdout.log"), "w")
        stderr_file = open(os.path.join(tmp_dir, "miniwob_server_stderr.log"), "w")
        subprocess.Popen(
            ["python", "-m", "http.server", "8000"],
            cwd=html_path,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        time.sleep(1)
        # Check if the server is running by attempting to connect

        try:
            urllib.request.urlopen(f"{self.base_url}/", timeout=5)
        except Exception as e:
            raise RuntimeError(f"MiniWob server failed to start: {e}")
