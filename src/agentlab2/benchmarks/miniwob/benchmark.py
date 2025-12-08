from typing import Any

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.miniwob.all_tasks import ALL_MINIWOB_TASKS
from agentlab2.benchmarks.miniwob.task import MiniWobTask


class MiniWobBenchmark(Benchmark):
    metadata = {
        "name": "miniwob",
        "description": "MiniWob benchmark for web-based tasks",
        "task_cls": MiniWobTask.__name__,
    }
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
