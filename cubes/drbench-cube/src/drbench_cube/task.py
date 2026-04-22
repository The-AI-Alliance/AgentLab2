"""CUBE Task implementation for DRBench.

DrBenchTask implements the gym-like reset/step/evaluate loop.
DrBenchTaskConfig is the serializable config that creates a DrBenchTask.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata

from drbench_cube.container import DrBenchContainerBackend
from drbench_cube.tool import DrBenchToolConfig
from drbench.score_report import score_report
from drbench.task_loader import get_task_from_id

logger = logging.getLogger(__name__)

_RESET_PROMPT_TEMPLATE = """\
You are {persona_name}, {persona_role} at {company_name} ({company_industry}).

Your credentials:
  username: {username}
  password: {password}

You have access to the following systems:
  - Nextcloud (company file storage): {nextcloud_url}
  - Mattermost (team communication): {mattermost_url}
  - FileBrowser (shared file storage): {filebrowser_url}
  - Email (IMAP): available via search_emails / get_email / list_email_folders actions

Research question:
{dr_question}

When you have gathered sufficient information, call submit_report(report_text=...) \
with your complete analysis. Your report should address the research question \
comprehensively, citing all relevant sources you discovered.
"""


class DrBenchTask(Task):
    """
    CUBE Task for a single DRBench episode.

    Lifecycle:
      reset()     → returns initial Observation (persona + question prompt)
      step(...)   → executes tool actions (handled by CUBE base class)
      evaluate()  → scores submitted report with score_report()
      finished()  → True once submit_report() has been called
    """

    # Set by DrBenchTaskConfig.make() — not Pydantic fields
    _eval_model: str = "gpt-4o"
    _eval_embedding_model: str = "text-embedding-3-large"
    _eval_metrics: List[str] = ["insights_recall", "factuality"]

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self.tool.reset()

        drbench_task = get_task_from_id(self.metadata.id)
        task_cfg = drbench_task.get_task_config()
        persona = task_cfg.get("persona", {})
        company = task_cfg.get("company_info", {})

        # Store for evaluate()
        self._task_cfg = task_cfg
        self._eval_cfg = drbench_task.get_eval_config()

        prompt = _RESET_PROMPT_TEMPLATE.format(
            persona_name=persona.get("name", "Unknown"),
            persona_role=persona.get("department", "Employee"),
            company_name=company.get("name", "Company"),
            company_industry=company.get("industry", ""),
            username=persona.get("username", ""),
            password=persona.get("password", ""),
            dr_question=task_cfg.get("dr_question", ""),
            nextcloud_url=self.tool._container.get_url(8081),
            mattermost_url=self.tool._container.get_url(8082),
            filebrowser_url=self.tool._container.get_url(8090),
        )
        return Observation.from_text(prompt), {"task_id": self.metadata.id}

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        report_text = self.tool._submitted_report
        if not report_text:
            return 0.0, {"error": "no_report_submitted"}

        try:
            scores = score_report(
                predicted_report_text=report_text,
                task_config=self._task_cfg,
                eval_config=self._eval_cfg,
                metrics=self._eval_metrics,
                verbose=False,
                model=self._eval_model,
                embedding_model=self._eval_embedding_model,
            )
            # Primary reward: harmonic mean of insights_recall and factuality,
            # matching the composite score reported in the DRBench paper.
            recall = float(scores.get("insights_recall", 0.0))
            factuality = float(scores.get("factuality", 0.0))
            if recall + factuality > 0:
                primary = 2 * recall * factuality / (recall + factuality)
            else:
                primary = 0.0
            scores["reward"] = primary
            return primary, scores
        except Exception as e:
            logger.error(f"evaluate() failed for task {self.metadata.id}: {e}")
            return 0.0, {"error": str(e)}

    def finished(self, obs: Observation) -> bool:
        return self.tool._submitted_report is not None

    def close(self) -> None:
        """Stop and remove the container, then clean up the tool."""
        try:
            if self._container is not None:
                self._container.stop()
                self._container = None
        except Exception as e:
            logger.warning(f"Error stopping container during close(): {e}")
        super().close()


class DrBenchTaskConfig(TaskConfig):
    """
    Serializable config for a single DRBench task.

    Stores task_id and evaluation model config. Persona credentials and full
    task configs are loaded from DRBENCH_DATA_DIR at runtime so this model
    stays JSON-serializable and Ray-picklable.

    Args:
        task_id: DRBench task identifier (e.g. "DR0001")
        eval_model: LLM used by score_report() to judge submitted reports.
            Changing this produces scores incomparable with official DRBench results.
        eval_embedding_model: Embedding model for factuality's semantic retrieval.
            Defaults to text-embedding-3-large, matching the paper's methodology.
        eval_metrics: Metrics to compute. Defaults to ["insights_recall", "factuality"].
            Reward is the harmonic mean of all specified metrics.
    """

    task_id: str
    eval_model: str = "gpt-4o"
    eval_embedding_model: str = "text-embedding-3-large"
    eval_metrics: List[str] = ["insights_recall", "factuality"]

    def make(
        self,
        runtime_context=None,
        container_backend: ContainerBackend | None = None,
    ) -> DrBenchTask:
        from drbench_cube.benchmark import DrBenchBenchmark

        metadata: TaskMetadata = DrBenchBenchmark.task_metadata[self.task_id]

        # Load persona credentials to build the ToolConfig
        drbench_task = get_task_from_id(self.task_id)
        persona = drbench_task.get_task_config().get("persona", {})
        tool_config = DrBenchToolConfig(
            persona_username=persona.get("username", ""),
            persona_password=persona.get("password", ""),
        )

        task = DrBenchTask(
            metadata=metadata,
            tool_config=tool_config,
            container_backend=container_backend or DrBenchContainerBackend(),
        )
        task._eval_model = self.eval_model
        task._eval_embedding_model = self.eval_embedding_model
        task._eval_metrics = self.eval_metrics
        return task


# RuntimeContext is a TYPE_CHECKING-only import in cube.task — resolve it now.
DrBenchTask.model_rebuild()
DrBenchTaskConfig.model_rebuild()
