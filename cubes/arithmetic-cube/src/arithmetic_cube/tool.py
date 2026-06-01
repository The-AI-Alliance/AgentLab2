from typing import Any

from cube.container import Container
from cube.tool import Tool, ToolConfig, tool_action


class ArithmeticToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> "ArithmeticTool":
        return ArithmeticTool()


class ArithmeticTool(Tool):
    def __init__(self) -> None:
        self._submitted: int | None = None

    def reset(self) -> None:
        self._submitted = None

    @tool_action
    def submit_answer(self, answer: int) -> str:
        """Submit your integer answer to the arithmetic question.

        Args:
            answer: Your integer answer.
        """
        self._submitted = answer
        return f"Answer {answer} submitted."

    @tool_action
    def _unknown_tool(self, name: str, arguments: dict[str, Any]) -> str:
        # TirAgent maps hallucinated/unknown tool calls to this sentinel; return feedback
        # instead of letting cube.tool raise, so RL rollouts stay alive. Mirrors math-tool-use.
        return f"Unknown tool '{name}' with arguments {arguments}. Use submit_answer(answer: int)."

    @property
    def last_answer(self) -> int | None:
        return self._submitted
