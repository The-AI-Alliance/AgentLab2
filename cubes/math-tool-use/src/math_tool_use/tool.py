import re
from typing import Any

from cube.container import Container
from cube.tool import Tool, ToolConfig, tool_action
from sandbox_fusion import RunCodeRequest, run_code

_BLOCKED_PATTERNS = [
    re.compile(r"\bsys\.exit\b"),
    re.compile(r"\bos\._exit\b"),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\.popen\b"),
    re.compile(r"\bos\.exec\w*\b"),
    re.compile(r"\bos\.spawn\w*\b"),
    re.compile(r"\bos\.kill\b"),
    re.compile(r"\bshutil\.rmtree\b"),
    re.compile(r"\bos\.remove\b"),
    re.compile(r"\bos\.unlink\b"),
]


class MathToolUseToolConfig(ToolConfig):
    sandbox_endpoint: str | None = None
    sandbox_language: str = "python"
    sandbox_compile_timeout: float = 10.0
    sandbox_run_timeout: float = 10.0

    def make(self, container: Container | None = None) -> "MathToolUseTool":
        return MathToolUseTool(config=self)


class MathToolUseTool(Tool):
    def __init__(self, config: MathToolUseToolConfig) -> None:
        self.config = config
        self._python_call_count: int = 0
        self._final_answer: str | None = None
        self._submitted_final_answer: bool = False
        self._last_python_output: str | None = None

    def reset(self) -> None:
        self._python_call_count = 0
        self._final_answer = None
        self._submitted_final_answer = False
        self._last_python_output = None

    @tool_action
    def _unknown_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return f"Unknown tool '{name}' with arguments {arguments}"

    @tool_action
    def run_python_code(self, code: str) -> str:
        """Execute Python code. Print only the final result.

        Args:
            code: Python code to execute.
        """
        self._python_call_count += 1
        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(code):
                self._last_python_output = f"Blocked: code contains forbidden pattern '{pattern.pattern}'"
                return self._last_python_output

        try:
            request = RunCodeRequest(
                code=code,
                language=self.config.sandbox_language,
                compile_timeout=self.config.sandbox_compile_timeout,
                run_timeout=self.config.sandbox_run_timeout,
            )
            kwargs: dict[str, Any] = {}
            if self.config.sandbox_endpoint:
                kwargs["endpoint"] = self.config.sandbox_endpoint

            response = run_code(request, **kwargs)

            stdout = ""
            stderr = ""
            if response.run_result:
                stdout = response.run_result.stdout or ""
                stderr = response.run_result.stderr or ""

            status = response.status.value if hasattr(response.status, "value") else str(response.status)
            is_timeout = "timeout" in status.lower() or "timeout" in (response.message or "").lower()

            parts = []
            if stdout:
                parts.append(stdout.rstrip())
            if stderr:
                parts.append(f"[stderr]\n{stderr.rstrip()}")
            if is_timeout:
                parts.append("[execution timed out]")
            if not parts:
                parts.append("[no output]")

            self._last_python_output = "\n".join(parts)
            return self._last_python_output
        except Exception as exc:  # pragma: no cover - defensive for runtime tool use
            self._last_python_output = f"[execution error: {exc}]"
            return self._last_python_output

    @tool_action
    def MathAnswer(self, answer: str) -> str:  # noqa: N802
        """Submit the final answer in LaTeX \\boxed{} format (for example: ``\\boxed{42}``).

        Args:
            answer: The final answer.
        """
        self._final_answer = answer.strip()
        self._submitted_final_answer = True
        return f"Answer submitted: {self._final_answer}"

    @property
    def python_call_count(self) -> int:
        return self._python_call_count

    @property
    def final_answer(self) -> str | None:
        return self._final_answer
    
    @property
    def submitted_final_answer(self) -> bool:
        return self._submitted_final_answer

    @property
    def last_python_output(self) -> str | None:
        return self._last_python_output
