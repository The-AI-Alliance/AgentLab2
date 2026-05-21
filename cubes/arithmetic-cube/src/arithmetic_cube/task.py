from typing import Any, Literal

from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from arithmetic_cube.tool import ArithmeticTool, ArithmeticToolConfig


class ArithmeticTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for arithmetic tasks — typed per-task fields
    that previously lived in ``extra_info``.
    """

    a: int
    b: int
    op: Literal["+", "-", "*"]
    expected: int


class SolveArithmeticTask(Task[ArithmeticTaskMetadata]):
    """Task: solve a math problem by calling submit_answer() once with the correct integer."""

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        m = self.metadata
        question = f"What is {m.a} {m.op} {m.b}?"
        return Observation.from_text(question), {"question": question, "expected": m.expected}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, ArithmeticTool)
        answer = self.tool.last_answer
        expected = self.metadata.expected
        correct = answer == expected
        return (1.0 if correct else 0.0), {"answer": answer, "expected": expected, "correct": correct}

    def finished(self, obs: Observation | None = None) -> bool:
        assert isinstance(self.tool, ArithmeticTool)
        return self.tool.last_answer is not None


class ArithmeticTaskConfig(TaskConfig[ArithmeticTaskMetadata]):
    """Serializable configuration that produces a SolveArithmeticTask."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> SolveArithmeticTask:
        tool_cfg = self.tool_config or ArithmeticToolConfig()
        return SolveArithmeticTask(
            metadata=self.metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
        )
