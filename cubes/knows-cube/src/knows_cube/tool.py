"""Tools and protocols for KNOWS tasks."""

from typing import Any, Protocol, runtime_checkable

from cube.core import Observation
from cube.tool import Tool, ToolConfig, tool_action
from playwright.sync_api import Page


@runtime_checkable
class KnowsBrowserTool(Protocol):
    """Browser tools usable by KnowsTask — must expose a raw Playwright ``page``.

    KNOWS creates its workspace file by driving the real Google Docs UI, so the
    task needs the live Page object, not just an observation. Satisfied by
    ``BgymTool`` and ``SyncPlaywrightTool``.

    NOTE: ``@runtime_checkable`` Protocols check method NAMES only, never
    signatures — this is a smoke check, not a type guarantee.
    """

    @property
    def page(self) -> Page: ...

    def noop(self) -> Any: ...

    def page_obs(self) -> Observation: ...


class SubmitWorkTool(Tool):
    """Terminator tool — replaces upstream's "reply DONE" substring detection.

    Upstream KNOWS decides an episode is over by scanning the last chat message
    for the literal substring ``"DONE"``. That is unreliable: any prose
    containing the word fires it. An explicit action makes completion a
    first-class, observable event and keeps ``finished()`` free of I/O — which
    matters because ``finished()`` runs after every single action.
    """

    def __init__(self) -> None:
        self._summary: str | None = None
        self._infeasible_reason: str | None = None

    def reset(self) -> None:
        self._summary = None
        self._infeasible_reason = None

    def close(self) -> None:
        """No resources held."""

    @property
    def submitted(self) -> bool:
        """True once the agent has either submitted work or reported infeasible."""
        return self._summary is not None or self._infeasible_reason is not None

    @property
    def summary(self) -> str | None:
        """The agent's submitted summary, or None."""
        return self._summary

    @property
    def infeasible_reason(self) -> str | None:
        """The agent's infeasibility explanation, or None."""
        return self._infeasible_reason

    @tool_action
    def submit_work(self, summary: str) -> str:
        """Submit your finished work and end the task.

        Args:
            summary: One or two sentences describing what you changed in the file.
        """
        self._summary = summary
        return f"Work submitted: {summary}"

    @tool_action
    def report_infeasible(self, explanation: str) -> str:
        """Report that this task cannot be completed. Use only when it is objectively impossible.

        Args:
            explanation: Brief explanation of why the task cannot be completed.
        """
        self._infeasible_reason = explanation
        return "Reported task as infeasible."


class SubmitWorkToolConfig(ToolConfig):
    """Configuration for :class:`SubmitWorkTool`."""

    def make(self, container: Any = None) -> SubmitWorkTool:
        """Instantiate the tool. ``container`` is unused — this tool is local."""
        del container
        return SubmitWorkTool()
