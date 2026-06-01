from typing import Tuple

import math_verify
from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskExecutionInfo, TaskMetadata
from pydantic import BaseModel

from math_tool_use.tool import MathToolUseTool, MathToolUseToolConfig


class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float

    def get_reward_range(self) -> tuple[float, float]:
        values = [
            self.wrong_answer_not_finished,
            self.wrong_answer_finished,
            self.no_answer_not_finished,
            self.no_answer_finished,
            self.unparsable_not_finished,
            self.unparsable_finished,
            self.correct_answer_not_finished,
            self.correct_answer_finished,
        ]
        return min(values), max(values)


# The companion cube stored an identical reward map on every task's extra_info. It is not
# per-task data, so it lives here as a single default rather than in TaskMetadata.
DEFAULT_REWARD_TABLE = RewardTable(
    correct_answer_finished=1.0,
    correct_answer_not_finished=0.0,
    wrong_answer_finished=0.0,
    wrong_answer_not_finished=0.0,
    no_answer_finished=0.0,
    no_answer_not_finished=0.0,
    unparsable_finished=0.0,
    unparsable_not_finished=0.0,
)


def get_reward(answer_status: str, finished: bool, reward_table: RewardTable) -> float:
    match (answer_status, finished):
        case ("wrong", False):
            return reward_table.wrong_answer_not_finished
        case ("wrong", True):
            return reward_table.wrong_answer_finished
        case ("no_answer", False):
            return reward_table.no_answer_not_finished
        case ("no_answer", True):
            return reward_table.no_answer_finished
        case ("unparsable", False):
            return reward_table.unparsable_not_finished
        case ("unparsable", True):
            return reward_table.unparsable_finished
        case ("correct", False):
            return reward_table.correct_answer_not_finished
        case ("correct", True):
            return reward_table.correct_answer_finished
        case _:
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{finished}")


def _verify_answer_status(
    prediction: str,
    gold: str,
    *,
    strict: bool = True,
    max_prediction_length: int = 1000,
) -> str:
    boxed_start = prediction.rfind("\\boxed{")
    if boxed_start < 0:
        return "no_answer"

    boxed_prediction = prediction[boxed_start:]
    if "\\boxed{}" in boxed_prediction:
        return "unparsable"
    if len(boxed_prediction) > max_prediction_length:
        return "unparsable"

    try:
        gold_parsed = math_verify.parse(gold)
        boxed_prediction_parsed = math_verify.parse(boxed_prediction)
        if not boxed_prediction_parsed:
            return "unparsable"
        equivalent = math_verify.verify(gold_parsed, boxed_prediction_parsed, strict=strict, timeout_seconds=1)
        return "correct" if equivalent else "wrong"
    except Exception:
        return "unparsable"


class MathToolUseTaskMetadata(TaskMetadata):
    """Lightweight, eager per-task metadata — held for every task in the registry.

    Only the cheap ``dataset`` label lives here (id/split/etc. come from the base). The
    heavy problem text + gold answer live in ``MathToolUseTaskExecutionInfo`` (lazy).
    """

    dataset: str

    @property
    def extra_info(self) -> dict:
        # Back-compat shim: cube_rl's registry filters tasks via
        # ``task_metadata.extra_info["dataset"]``. Keep that working without re-introducing a
        # generic extra_info field on cube-standard's TaskMetadata.
        return {"dataset": self.dataset}


class MathToolUseTaskExecutionInfo(TaskExecutionInfo):
    """Heavy, lazy per-task execution data — loaded in ``make()`` from the cache that
    ``MathToolUseBenchmarkConfig.install()`` wrote, so the 130k problem texts are never held
    eagerly in the metadata registry."""

    question: str
    expected: str


class MathToolUseTask(Task[MathToolUseTaskMetadata]):
    """Solve a math problem via a Python tool, then submit a final LaTeX answer."""

    accept_agent_stop: bool = False  # the agent must submit an answer, not stop early
    validate_per_step: bool = False  # score only at the end, from the submitted answer

    @property
    def _exec(self) -> MathToolUseTaskExecutionInfo:
        assert isinstance(self.execution_info, MathToolUseTaskExecutionInfo), (
            "MathToolUseTask requires execution_info; populate it in MathToolUseTaskConfig.make()"
        )
        return self.execution_info

    def reset(self) -> tuple[Observation, dict]:
        self.tool.reset()
        question = self._exec.question
        return Observation.from_text(question), {"question": question, "expected": self._exec.expected}

    def evaluate(self, obs: Observation | None = None) -> Tuple[float, dict]:
        assert isinstance(self.tool, MathToolUseTool)
        if self.tool.final_answer is not None:
            submitted = self.tool.final_answer
        else:
            submitted = self.tool._last_python_output or ""

        answer_status = _verify_answer_status(submitted, self._exec.expected, strict=True)
        reward = get_reward(answer_status, self.finished(obs), DEFAULT_REWARD_TABLE)

        return reward, {
            "success": answer_status == "correct",
            "no_answer": answer_status == "no_answer",
            "no_error": answer_status != "unparsable",
            "num_python_calls": self.tool.python_call_count,
            "overflow": not self.tool.submitted_final_answer,
        }

    def finished(self, obs: Observation | None = None) -> bool:
        assert isinstance(self.tool, MathToolUseTool)
        return self.tool.submitted_final_answer


class MathToolUseTaskConfig(TaskConfig[MathToolUseTaskMetadata]):
    """Serializable config that produces a MathToolUseTask.

    Lazy-loads the heavy ``MathToolUseTaskExecutionInfo`` from the per-task cache written by
    ``MathToolUseBenchmarkConfig.install()`` (raises an actionable error if install hasn't run).
    """

    def make(self, runtime_context: RuntimeContext | None = None) -> MathToolUseTask:
        execution_info = MathToolUseTaskExecutionInfo.model_validate(self.load_task_execution_info())
        tool_cfg = self.tool_config or MathToolUseToolConfig()
        return MathToolUseTask(
            metadata=self.metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            execution_info=execution_info,
            accept_agent_stop=False,
        )
