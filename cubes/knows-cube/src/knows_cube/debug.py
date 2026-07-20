"""Hermetic debug benchmark for knows-cube — no Google, no Gemini, no browser.

WHAT THIS PROVES
----------------
Cube plumbing only: goal construction, submit/infeasible detection, idempotent
terminal evaluation (including the truncation path), KNOWS ``Result`` -> reward
normalisation, and the flat ``eval.cp{i}_*`` info schema.

WHAT THIS DOES NOT PROVE
------------------------
Anything about the real KNOWS evaluators. All 22 upstream families route at
least one scoring step through a Gemini vision model, and the cheapest task
(docs_1, 10 points) has a deterministic ceiling of 8/10 because
``verify_image_in_region`` has no non-VLM tier. **Reward 1.0 on a real KNOWS task
is not reachable deterministically with any credentials**, which is why the
debug suite is synthetic rather than a real task with a scripted cheat.

Real verification lives in ``scripts/smoke/knows_gold_eval.py``, which requires
full credentials and asserts a threshold rather than equality.

Public API (cube.testing protocol)
----------------------------------
get_debug_benchmark()           -> DebugKnowsBenchmarkConfig
make_debug_agent(task_id: str)  -> DebugAgent

Usage::

    uv run python -m knows_cube.debug
"""

import logging
import sys
from typing import Any, ClassVar

from browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result
from browsergym.knows.task import KnowsWorkspaceTask
from cube.benchmark import Benchmark, RuntimeContext
from cube.core import Action, ActionSchema, Observation
from cube.task import TaskConfig, TaskMetadata
from cube.testing import run_debug_suite
from cube.tool import ToolboxConfig, ToolConfig
from pydantic import SerializeAsAny

from knows_cube.benchmark import KnowsBenchmark, KnowsBenchmarkConfig
from knows_cube.task import KnowsTask, KnowsTaskConfig, KnowsTaskMetadata
from knows_cube.tool import SubmitWorkToolConfig

logger = logging.getLogger(__name__)

_WINNING_PHRASE = "debug-complete"

_DEBUG_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "knows-debug.docs.1",
        "task_family_folder": "docs_1_formal_letter",
        "task_id_prefix": "knows-debug.docs",
        "workspace_kind": "docs",
        "goal": "Debug task: write a formal letter. No real Google Doc is involved.",
    },
    {
        "id": "knows-debug.sheets.1",
        "task_family_folder": "sheets_7_running_analysis",
        "task_id_prefix": "knows-debug.sheets",
        "workspace_kind": "sheets",
        "goal": "Debug task: analyse running data. No real Google Sheet is involved.",
    },
)

_GOALS: dict[str, str] = {rec["id"]: rec["goal"] for rec in _DEBUG_TASKS}


def _debug_task_metadata() -> dict[str, TaskMetadata]:
    """Synthetic metadata — deliberately not read from task_metadata.json."""
    return {
        rec["id"]: KnowsTaskMetadata(
            id=rec["id"],
            split="test",
            recommended_max_steps=3,
            abstract_description=rec["goal"],
            task_family_folder=rec["task_family_folder"],
            task_id_prefix=rec["task_id_prefix"],
            instance_number=1,
            workspace_kind=rec["workspace_kind"],
            n_checkpoints=2,
            max_points=10,
            is_gradeable=True,
        )
        for rec in _DEBUG_TASKS
    }


class DebugKnowsTask(KnowsTask):
    """KnowsTask with both credentialed seams replaced by deterministic stubs.

    Everything between ``reset()`` and ``evaluate()`` — submit detection,
    browsing history, idempotent caching, ``Result`` normalisation — is the real
    code path, so the debug suite exercises the cube's own logic rather than a
    parallel implementation of it.
    """

    def _reset_workspace(self) -> tuple[str, str]:
        doc_id = f"debug-{self.metadata.id}"
        return doc_id, self._workspace_url(doc_id)

    def _navigate_to_workspace(self, doc_url: str) -> None:
        """No browser attached."""
        del doc_url

    def _page_observation(self) -> Observation:
        return Observation.from_text("[debug] no browser attached")

    def _build_goal(self, doc_url: str) -> str:
        return f"{_GOALS[self.metadata.id]}\n\nFile URL: {doc_url}\nCall `submit_work` when finished."

    def _record_current_url(self) -> None:
        """No browser attached — replay the workspace URL so history is non-empty."""
        self._visited_urls.append(self._doc_url or "")

    def _grade(self, doc_id: str) -> tuple[float, dict[str, Any]]:
        """Deterministic grader.

        Builds a real KNOWS ``Result`` and runs it through upstream's
        ``_summarize_result``, so the reward maths and the ``eval.*`` key schema
        are genuinely exercised rather than faked.
        """
        summary = self._submit_tool.summary or ""
        submitted_ok = _WINNING_PHRASE in summary
        result = Result(
            checkpoints=[
                Checkpoint(name="submitted", total=5, result=5 if submitted_ok else 0),
                Checkpoint(name="doc_id_present", total=5, result=5 if doc_id else 0),
            ],
            total_execution_time=None,
        )
        info: dict[str, Any] = {
            "doc_id": doc_id,
            "task_family": self.metadata.task_family_folder,
            "workspace_kind": self.metadata.workspace_kind,
            "visited_urls": list(self._visited_urls),
            "debug_grader": True,
        }
        reward, breakdown = KnowsWorkspaceTask._summarize_result(result, info)
        info.update(breakdown)
        return reward, info


class DebugKnowsTaskConfig(KnowsTaskConfig):
    """Task config that builds a :class:`DebugKnowsTask` with a browser-free toolbox."""

    def make(self, runtime_context: RuntimeContext | None = None) -> DebugKnowsTask:
        return DebugKnowsTask(
            metadata=self.metadata,
            tool_config=self.tool_config or ToolboxConfig(tool_configs=[SubmitWorkToolConfig()]),
            runtime_context=runtime_context,
            settle_seconds=0.0,
        )


class DebugKnowsBenchmark(KnowsBenchmark):
    """Debug runtime pair — skips the credential warning, which does not apply."""

    def _setup(self) -> None:
        """The debug benchmark needs no credentials."""


class DebugKnowsBenchmarkConfig(KnowsBenchmarkConfig):
    """Hermetic two-task benchmark used by ``cube test knows-cube``."""

    task_metadata: ClassVar[dict[str, TaskMetadata]] = _debug_task_metadata()
    task_config_class: ClassVar[type[TaskConfig]] = DebugKnowsTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = DebugKnowsBenchmark

    # Must mirror KnowsBenchmarkConfig's annotation exactly: a bare `Any` here drops
    # both validation and the polymorphic serializer, so the config round-trips
    # through Ray/storage as a plain dict. See benchmark.py:70-72.
    tool_config: SerializeAsAny[ToolConfig] = ToolboxConfig(tool_configs=[SubmitWorkToolConfig()])
    # get_task_configs is inherited: KnowsBenchmarkConfig's override already
    # dispatches through task_config_class, which is DebugKnowsTaskConfig here.


class DebugAgent:
    """Deterministic agent that submits the phrase the debug grader rewards."""

    def get_action(self, obs: Observation) -> Action:
        """Return the single scripted action."""
        del obs
        return Action(name="submit_work", arguments={"summary": _WINNING_PHRASE})

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        """Callable shorthand for task-loop compatibility."""
        del action_set
        return self.get_action(obs)


def make_debug_agent(task_id: str) -> DebugAgent:
    """Return a fresh deterministic debug agent. Every debug task uses the same one."""
    del task_id
    return DebugAgent()


def get_debug_benchmark() -> DebugKnowsBenchmarkConfig:
    """Return the hermetic two-task debug benchmark. Requires no credentials."""
    return DebugKnowsBenchmarkConfig()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    results = run_debug_suite("knows-cube", sys.modules[__name__])
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] != 1.0]
    sys.exit(1 if failed else 0)
