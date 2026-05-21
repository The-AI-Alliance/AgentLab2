"""BrowseCompTask, BrowseCompTaskMetadata, and BrowseCompTaskConfig for BrowseComp benchmark."""

import re
from typing import Any

import litellm

from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskExecutionInfo, TaskMetadata
from cube.tool import Toolbox, ToolboxConfig

from browsercomp_cube.crypto import decrypt
from browsercomp_cube.tool import SubmitAnswerTool, SubmitAnswerToolConfig
from cube_web_tool import BraveWebSearchToolConfig, WebFetchToolConfig

_GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous \
[correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' \
if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only \
on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on \
any background to the problem, do not attempt to solve the problem, do not argue for any answer different than \
[correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small \
margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, \
ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no \
confidence score available."""

_FORMAT_INSTRUCTIONS = (
    "\n\nPlease structure your final answer as:\n"
    "Explanation: <your reasoning>\n"
    "Exact Answer: <the precise answer>\n"
    "Confidence: <0-100>"
)


class BrowseCompTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for BrowseComp tasks.

    Lightweight public fields shipped in task_metadata.json. The encrypted
    problem/answer payload lives in the per-task execution cache populated by
    BrowseCompBenchmarkConfig.install() and is decrypted at make() time.
    """

    topic: str = ""
    """Coarse subject area (e.g. 'Art', 'Sports'). Mirrored into abstract_description."""


class BrowseCompExecutionInfo(TaskExecutionInfo):
    """Heavy decrypted BrowseComp payload, populated in TaskConfig.make()."""

    problem: str
    answer: str


class BrowseCompTask(Task[BrowseCompTaskMetadata]):
    """A single BrowseComp information-retrieval task."""

    validate_per_step: bool = False
    accept_agent_stop: bool = True
    grader_retries: int = 3
    scorer_model: str

    @property
    def _exec(self) -> BrowseCompExecutionInfo:
        """Typed view on execution_info — fails fast if it was not populated."""
        if not isinstance(self.execution_info, BrowseCompExecutionInfo):
            raise RuntimeError(
                f"BrowseCompTask {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected BrowseCompExecutionInfo. "
                f"Construct via BrowseCompTaskConfig.make() so it is populated."
            )
        return self.execution_info

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        prompt = self._exec.problem + _FORMAT_INSTRUCTIONS
        return Observation.from_text(prompt), {"problem": self._exec.problem}

    def _call_grader(self, prompt: str, scorer_model: str) -> tuple[bool, str]:
        completion = litellm.completion(
            model=scorer_model,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content or ""
        match = re.search(r"correct:\s*(yes|no)", response, re.IGNORECASE)
        if not match:
            raise ValueError(f"Grader response missing 'correct: yes/no':\n{response}")
        return match.group(1).lower() == "yes", response

    def _submit_tool(self) -> SubmitAnswerTool:
        assert isinstance(self.tool, Toolbox)
        tool = self.tool.find_tool(SubmitAnswerTool)
        assert isinstance(tool, SubmitAnswerTool)
        return tool

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        submitted = self._submit_tool().last_answer
        if submitted is None:
            return 0.0, {"correct": False, "submitted": None, "reason": "No answer submitted"}

        prompt = _GRADER_TEMPLATE.format(
            question=self._exec.problem,
            response=submitted,
            correct_answer=self._exec.answer,
        )

        last_error: Exception | None = None
        for _ in range(self.grader_retries):
            try:
                is_correct, grader_response = self._call_grader(prompt, self.scorer_model)
                return (1.0 if is_correct else 0.0), {
                    "correct": is_correct,
                    "submitted": submitted,
                    "grader_response": grader_response,
                }
            except Exception as e:
                last_error = e

        return 0.0, {"correct": False, "submitted": submitted, "grader_error": str(last_error)}

    def finished(self, obs: Observation | None = None) -> bool:
        return self._submit_tool().last_answer is not None


class BrowseCompTaskConfig(TaskConfig[BrowseCompTaskMetadata]):
    """Serializable configuration that produces a BrowseCompTask.

    The encrypted record (problem, answer, canary) for ``task_id`` is loaded
    from the per-task execution cache and decrypted in ``make()`` so the heavy
    payload never crosses process boundaries inside the config object.
    """

    scorer_model: str

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> BrowseCompTask:
        self.verify_installed()
        encrypted = self.load_task_execution_info()
        canary = encrypted["canary"]
        execution_info = BrowseCompExecutionInfo(
            problem=decrypt(encrypted["problem"], canary),
            answer=decrypt(encrypted["answer"], canary),
        )

        tool_cfg = self.tool_config or ToolboxConfig(
            tool_configs=[BraveWebSearchToolConfig(), WebFetchToolConfig(), SubmitAnswerToolConfig()]
        )
        return BrowseCompTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=tool_cfg,
            scorer_model=self.scorer_model,
            runtime_context=runtime_context,
        )
