from typing import ClassVar

import logging
from datasets import load_dataset

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from math_tool_use.task import MathToolUseTaskConfig

logger = logging.getLogger(__name__)

def process_aime_and_amc(dataset, dataset_name):
    for item in dataset:
        task = item["problem"]
        answer = "\\boxed{" + str(item["answer"]) + "}"
        yield {
            "dataset": dataset_name,
            "task": task,
            "answer": answer,
        }

def _load_aime_dataset(year: int, upsample_factor: int = 0) -> list[dict]:
    if year == 2025:
        aime_dataset = load_dataset("MathArena/aime_2025", split="train")
    else:
        aime_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
        aime_dataset = aime_dataset.filter(lambda x: str(year) in x["url"])

    dataset_name = f"aime_{year}" + ("" if upsample_factor > 0 else "_original")
    tasks_data = [s for s in process_aime_and_amc(aime_dataset, dataset_name) if s is not None]

    if upsample_factor > 0:
        tasks_data *= upsample_factor

    return tasks_data


def process_open_reasoner(dataset, dataset_name):
    for item in dataset:
        # Note: Open Reasoner tasks sometimes have preamble, e.g.
        # - Example 31 (2004 College Entrance Examination Hunan Paper)
        # - 8.
        # - 4. (7 points)
        # We are currently ignoring the preamble
        task = item["0"]["value"]
        answer = "\\boxed{" + item["1"]["ground_truth"]["value"] + "}"
        yield {"dataset": dataset_name, "task": task, "answer": answer}

def load_task_metadata() -> dict[str, TaskMetadata]:
    """Load task metadata from the Open Reasoner Zero dataset."""

    dataset_names = ["open_reasoner_zero_57k", "open_reasoner_zero_extended_72k", "aime_2025"]
    dataset_urls = {
        "open_reasoner_zero_57k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json",
        "open_reasoner_zero_extended_72k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_72k_collection_extended.json"
    }

    metadata: dict[str, TaskMetadata] = {}

    for dataset_name in dataset_names:
        tasks_data = []
        if "open_reasoner" in dataset_name:
            dataset = load_dataset(
                "json",
                data_files=dataset_urls.get(dataset_name),
                split="train",
                trust_remote_code=True,
            )
            tasks_data = [s for s in process_open_reasoner(dataset, dataset_name) if s is not None]
        
        if "aime" in dataset_name:
            year = int(dataset_name.split("_")[1])
            if dataset_name.endswith("_original"):
                tasks_data = _load_aime_dataset(year, upsample_factor=0)
            else:               
                tasks_data = _load_aime_dataset(year, upsample_factor=16)

        if len(tasks_data) == 0:
            logger.warning(f"No tasks loaded for dataset {dataset_name}")
            continue

        for i, t in enumerate(tasks_data):
            instance_id = f"{dataset_name}_{i}"
            metadata[instance_id] = TaskMetadata(
                id=instance_id,
                abstract_description="Solve math by tool-using with Python and submit final LaTeX answer via MathAnswer",
                recommended_max_steps=3,
                split="train",
                extra_info={
                    "dataset": t["dataset"],
                    "question": t["task"],
                    "expected": t["answer"],
                    "dataset": t["dataset"],
                    "rewards": {
                        "correct_answer_finished": 1.0,
                        "correct_answer_not_finished": 0,
                        "wrong_answer_finished": 0,
                        "wrong_answer_not_finished": 0,
                        "no_answer_finished": 0,
                        "no_answer_not_finished": 0,
                        "unparsable_finished": 0,
                        "unparsable_not_finished": 0}
                },
            )

        logger.info(f"Loading {dataset_name}: {len(tasks_data)} tasks")
        
    logger.info(f"Loading total of {len(metadata)} tasks")
    return metadata

class MathToolUseBenchmark(Benchmark):
    """Arithmetic tasks requiring deterministic Python tool use before final LaTeX answer submission."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="math-tool-use",
        version="0.1.0",
        description="Solve arithmetic by planning, using one deterministic Python call, then submitting a LaTeX answer",
        num_tasks=4,
        tags=["example", "arithmetic", "tool-use", "python"],
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = load_task_metadata()
    task_config_class: ClassVar[type[TaskConfig]] = MathToolUseTaskConfig

    @classmethod
    def install(cls) -> None:
        cls.benchmark_metadata = cls.benchmark_metadata.model_copy(update={"num_tasks": len(cls.task_metadata)})

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass
