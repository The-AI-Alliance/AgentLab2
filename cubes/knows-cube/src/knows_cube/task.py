"""KNOWS task — Google Workspace authoring against the real docs.google.com."""

import logging
import time
from typing import Any, Literal

from browsergym.knows.doc_setup import create_task_workspace
from browsergym.knows.task import _WORKSPACE_ID_RE, _WORKSPACE_URL_SEGMENT, KnowsWorkspaceTask
from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Toolbox, ToolboxConfig
from cube.tools.browser import BrowserTool
from cube_browser_tool.bgym_tool import BgymToolConfig
from pydantic import PrivateAttr, SerializeAsAny

from knows_cube import evaluator as knows_evaluator
from knows_cube.tool import KnowsBrowserTool, SubmitWorkTool, SubmitWorkToolConfig

logger = logging.getLogger(__name__)

WorkspaceKind = Literal["docs", "sheets", "slides"]

_WORKSPACE_LABEL: dict[str, str] = {
    "docs": "Google Doc",
    "sheets": "Google Sheet",
    "slides": "Google Slides presentation",
}

_GOAL_SUFFIX = (
    "\n\n---\n"
    "A {label} has already been created for you. You MUST complete this task "
    "inside that file; do NOT create a new file and do NOT rename it.\n"
    "File URL: {url}\n"
    "When you are finished, call the `submit_work` action with a short summary "
    "of what you changed. If the task is genuinely impossible, call "
    "`report_infeasible` instead."
)


class KnowsTaskMetadata(TaskMetadata):
    """Lightweight, JSON-serialisable KNOWS task identity and routing data.

    Generated offline by ``scripts/generate_task_metadata.py`` so that neither
    ``browsergym.knows`` nor the KNOWS filesystem tree is touched when metadata
    is loaded.
    """

    task_family_folder: str
    """Upstream folder name, verbatim including casing. Load-bearing: it is the
    on-disk evaluator path segment and the dynamic evaluator module name."""

    task_id_prefix: str
    """Upstream gym-id prefix, e.g. ``knows.sheets_6_stock_tracker``. NOT derivable
    from ``task_family_folder`` — 9 of 22 families diverge, and 6 of those are
    semantic renames no string transform recovers. Carried explicitly."""

    instance_number: int
    """1-5. Never inferred from ``knows_task_id`` — docs_1 instance_1 is '1b'."""

    workspace_kind: WorkspaceKind
    """Drives the create URL, the URL segment, and the doc-id regex filter."""

    knows_task_id: str | None = None
    """Upstream short code ('1a', '6b'). Display only — 5 of 110 are missing."""

    evaluator_accepted_kwargs: list[str] = []
    """Sorted kwargs ``grade_checkpoints`` accepts. Diagnostic and offline
    validation only; runtime dispatch re-introspects, so it cannot drift."""

    n_checkpoints: int = 0

    max_points: int | None = None
    """Static point sum, or None when a checkpoint computes ``total`` at runtime.
    Diagnostic ONLY — reward is already normalised upstream. Never re-normalise."""

    has_dynamic_points: bool = False

    is_gradeable: bool = True
    """False where the upstream evaluator cannot run at all (see known_issues)."""

    known_issues: list[str] = []


class KnowsTask(Task):
    """One KNOWS instance: create the workspace file, then grade it.

    Terminal evaluation is idempotent and runs on every episode exit path,
    including truncation — ``cube_harness.episode`` calls ``task.evaluate()`` in
    the terminal block regardless of how the agent loop ended. That subsumes
    upstream's separate ``finalize()`` hook.
    """

    metadata: KnowsTaskMetadata  # type: ignore[assignment]

    agent_name: str = "cube-harness"
    """Prefix for the created file's name. Cosmetic — aids Drive triage."""

    existing_doc_id: str | None = None
    """Point the task at a pre-existing (e.g. gold-instance) file instead of
    creating one. Used by the credentialed smoke script."""

    settle_seconds: float = 8.0
    """Wait after navigating to the workspace file — the Docs/Sheets/Slides
    editors take several seconds to become interactive."""

    validate_per_step: bool = False
    """Pinned off, and unsafe to enable. Grading exports the doc, OCRs it and
    calls a paid Gemini model, so per-step evaluation would cost one full grade
    per action; ``evaluate()`` is terminal-only and caches its verdict, so a
    mid-episode call would also freeze the result the episode is scored on."""

    _doc_id: str | None = PrivateAttr(default=None)
    _doc_url: str | None = PrivateAttr(default=None)
    _visited_urls: list[str] = PrivateAttr(default_factory=list)
    _evaluator_module: Any = PrivateAttr(default=None)
    _eval_cache: tuple[float, dict[str, Any]] | None = PrivateAttr(default=None)

    # ---- tool resolution -------------------------------------------------

    @property
    def _browser_tool(self) -> KnowsBrowserTool:
        """Resolve the browser tool whether it is bare or inside a Toolbox."""
        if isinstance(self.tool, Toolbox):
            found = self.tool.find_tool(BrowserTool)
            if found is None:
                raise RuntimeError("No browser tool found in Toolbox")
            tool: Any = found
        else:
            tool = self.tool
        if not isinstance(tool, KnowsBrowserTool):
            raise RuntimeError(
                "The browser tool must satisfy the KnowsBrowserTool protocol "
                f"(e.g. BgymTool or SyncPlaywrightTool), got {type(tool).__name__}"
            )
        return tool

    @property
    def _submit_tool(self) -> SubmitWorkTool:
        """Resolve the SubmitWorkTool, which always lives inside the Toolbox."""
        if not isinstance(self.tool, Toolbox):
            raise TypeError(f"KNOWS requires a Toolbox containing SubmitWorkTool, got {type(self.tool).__name__}")
        tool = self.tool.find_tool(SubmitWorkTool)
        if not isinstance(tool, SubmitWorkTool):
            raise RuntimeError("SubmitWorkTool not found in Toolbox")
        return tool

    # ---- seams overridden by the debug subclass --------------------------

    def _reset_workspace(self) -> tuple[str, str]:
        """Create (or adopt) the workspace file; return ``(doc_id, doc_url)``.

        Delegates to upstream's ``create_task_workspace``, which drives the real
        Docs UI with our Playwright page and then shares the file with the
        evaluator's service account (falling back to the share UI).

        Unlike upstream — which swallows creation errors and lets the agent run a
        full episode against nothing — this raises. A missing workspace voids the
        episode, and a silent void is worse than a loud failure.
        """
        if self.existing_doc_id:
            return self.existing_doc_id, self._workspace_url(self.existing_doc_id)
        page = self._browser_tool.page
        doc_id, doc_url = create_task_workspace(
            page,
            agent_name=self.agent_name,
            task_name=self.metadata.task_id_prefix,
            instance_id=self.metadata.instance_number,
            kind=self.metadata.workspace_kind,
        )
        return doc_id, doc_url

    def _navigate_to_workspace(self, doc_url: str) -> None:
        """Open the workspace file and wait for the editor to settle.

        ``noop()`` is mandatory: we mutate the page outside the tool, so without
        it ``page_obs()`` can return the pre-navigation DOM.
        """
        browser = self._browser_tool
        browser.page.goto(doc_url)
        browser.noop()
        if self.settle_seconds > 0:
            time.sleep(self.settle_seconds)

    def _page_observation(self) -> Observation:
        """Observation of the current browser page."""
        return self._browser_tool.page_obs()

    def _grade(self, doc_id: str) -> tuple[float, dict[str, Any]]:
        """Run the upstream evaluator and normalise its ``Result`` to ``(reward, info)``.

        Never raises: every failure becomes ``info["evaluation_error"]`` with
        reward 0.0, matching upstream. On that path none of the other ``eval.*``
        keys exist, so a genuine 0.0 is distinguishable from a crash *only* via
        ``evaluation_error``. Any aggregate that averages reward without checking
        that key conflates infrastructure failure with agent failure.
        """
        info: dict[str, Any] = {
            "doc_id": doc_id,
            "doc_url": self._doc_url,
            "instance_number": self.metadata.instance_number,
            "task_family": self.metadata.task_family_folder,
            "workspace_kind": self.metadata.workspace_kind,
            "visited_urls": list(self._visited_urls),
        }
        if not self.metadata.is_gradeable:
            info["evaluation_error"] = f"task marked is_gradeable=False: {'; '.join(self.metadata.known_issues)}"
            info["eval.score_fraction"] = 0.0
            return 0.0, info
        try:
            if self._evaluator_module is None:
                self._evaluator_module = knows_evaluator.load_evaluator(
                    self.metadata.task_family_folder, self.metadata.instance_number
                )
            result = knows_evaluator.grade(self._evaluator_module, doc_id, list(self._visited_urls))
            # Upstream staticmethod: reward = sum(cp.result) / sum(cp.total). Reused
            # verbatim so the eval.* key schema stays identical to upstream's.
            reward, score_breakdown = KnowsWorkspaceTask._summarize_result(result, info)
            info.update(score_breakdown)
            return reward, info
        except Exception as exc:  # noqa: BLE001 — parity with upstream: grading never fails the episode
            logger.exception("KNOWS evaluation failed for %s", self.metadata.id)
            info["evaluation_error"] = str(exc)
            info["eval.score_fraction"] = 0.0
            return 0.0, info

    # ---- helpers ---------------------------------------------------------

    def _workspace_url(self, doc_id: str) -> str:
        """Editor URL for a doc id, per workspace kind."""
        segment = _WORKSPACE_URL_SEGMENT[self.metadata.workspace_kind]
        return f"https://docs.google.com/{segment}/d/{doc_id}/edit"

    def _extract_doc_id(self, text: str) -> str | None:
        """Pull a doc id out of text, preferring the URL segment for our kind."""
        expected = _WORKSPACE_URL_SEGMENT.get(self.metadata.workspace_kind)
        for match in _WORKSPACE_ID_RE.finditer(text or ""):
            if match.group(1) == expected:
                return match.group(2)
        match = _WORKSPACE_ID_RE.search(text or "")
        return match.group(2) if match else None

    def _build_goal(self, doc_url: str) -> str:
        """``task.md`` plus the workspace instruction block."""
        body = knows_evaluator.read_task_markdown(
            self.metadata.task_family_folder, self.metadata.instance_number
        ).rstrip()
        label = _WORKSPACE_LABEL[self.metadata.workspace_kind]
        return body + _GOAL_SUFFIX.format(label=label, url=doc_url)

    # ---- cube-standard lifecycle ----------------------------------------

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Create the workspace file, navigate to it, and return goal + page state."""
        self._doc_id = None
        self._doc_url = None
        self._visited_urls = []
        self._eval_cache = None
        self.tool.reset()

        doc_id, doc_url = self._reset_workspace()
        self._doc_id, self._doc_url = doc_id, doc_url
        self._visited_urls.append(doc_url)
        self._navigate_to_workspace(doc_url)

        goal = self._build_goal(doc_url)
        obs = Observation.from_text(goal) + self._page_observation()
        info = {
            "task_id": self.metadata.id,
            "doc_id": doc_id,
            "doc_url": doc_url,
            "workspace_kind": self.metadata.workspace_kind,
        }
        return obs, info

    def _record_current_url(self) -> None:
        """Append the browser's current URL to the browsing history, de-duplicated.

        94 of the 110 upstream evaluators accept ``browsing_history`` and grade
        against it, so this is scored data, not telemetry — but a browser that
        has gone away must not break grading, hence the broad catch.
        """
        try:
            url = self._browser_tool.page.url
        except Exception:  # noqa: BLE001 — a dead browser must not abort grading
            return
        if url and (not self._visited_urls or self._visited_urls[-1] != url):
            self._visited_urls.append(url)

    def _post_action(self, obs: Observation, role: str | None = None) -> None:
        """Record the page URL after every action.

        ``_post_action`` is cube-standard's designated home for per-action side
        effects: it fires once per ACTION on both views (gym ``Task.step`` and
        ``AgentView.execute_action``) and, on both, *before* ``evaluate()``.

        An earlier draft used ``obs_postprocess`` because ``_post_action`` did not
        exist on ``Task`` at the time. It does as of cube-standard 0.1.0rc11, whose
        docstring explicitly rules ``obs_postprocess`` out for this — that hook
        transforms the observation, fires once per STEP rather than per action, and
        runs after ``evaluate()``. Recording there under-counted multi-action steps
        in the ``browsing_history`` that 94 of the 110 evaluators grade against.
        """
        del obs, role
        self._record_current_url()

    def finished(self, obs: Observation | None = None) -> bool:
        """True once the agent has submitted work or reported infeasible. No I/O."""
        del obs
        return self._submit_tool.submitted

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        """Terminal evaluation. Idempotent, and correct on the truncation path.

        ``cube_harness.episode`` calls this on every exit path — clean submit,
        agent stop, and budget/max-steps truncation — so upstream's separate
        ``finalize()`` hook (which silently left truncated episodes ungraded) is
        unnecessary here.
        """
        del obs
        if self._eval_cache is not None:
            reward, cached = self._eval_cache
            return reward, dict(cached)

        # _post_action already fires before evaluate() on both views, but the
        # AgentStop path breaks out of step()'s action loop *before* it, so the
        # final URL would otherwise miss the grade. _record_current_url()
        # de-duplicates consecutive URLs, so the overlap is free.
        self._record_current_url()

        submit = self._submit_tool
        if submit.infeasible_reason is not None:
            end_state = "reported_infeasible"
        elif submit.summary is not None:
            end_state = "submitted"
        else:
            end_state = "truncated"

        doc_id = self._doc_id or self._extract_doc_id(submit.summary or "")
        if doc_id is None:
            self._eval_cache = (
                0.0,
                {"episode_end": end_state, "evaluation_error": "no_doc_id", "eval.score_fraction": 0.0},
            )
            return 0.0, dict(self._eval_cache[1])

        reward, info = self._grade(doc_id)
        info["episode_end"] = end_state
        if submit.summary is not None:
            info["agent_summary"] = submit.summary
        if submit.infeasible_reason is not None:
            info["infeasible_reason"] = submit.infeasible_reason
        # Cache a copy and hand out copies: cube.task.Task.step writes
        # info["profiling"] into whatever evaluate() returns, which would otherwise
        # mutate the cached terminal record that gets persisted for the episode.
        self._eval_cache = (reward, dict(info))
        return reward, dict(info)


class KnowsTaskConfig(TaskConfig[KnowsTaskMetadata]):
    """Serialisable KNOWS task config — this is what crosses the Ray boundary."""

    agent_name: str = "cube-harness"
    existing_doc_id: str | None = None
    settle_seconds: float = 8.0

    def make(self, runtime_context: RuntimeContext | None = None) -> KnowsTask:
        """Instantiate the live task on the worker."""
        return KnowsTask(
            metadata=self.metadata,
            tool_config=self.tool_config or default_tool_config(),
            runtime_context=runtime_context,
            agent_name=self.agent_name,
            existing_doc_id=self.existing_doc_id,
            settle_seconds=self.settle_seconds,
        )


def default_tool_config() -> SerializeAsAny[ToolboxConfig]:
    """Browser plus the submit terminator — the fallback when none is supplied.

    A factory, not a module constant: each caller gets its own config instance so
    no mutable state is shared between tasks (EX-002).
    """
    return ToolboxConfig(
        tool_configs=[
            BgymToolConfig(use_html=False, use_axtree=True, use_screenshot=True),
            SubmitWorkToolConfig(),
        ]
    )
