"""Render one episode with the real XRay viewer, for the annotation portal.

The point of the portal is evidence parity: the judge read the whole episode, so
the human must too. That means the annotator needs screenshots, the event rail,
per-step LLM detail — the XRay view the team already uses — not a truncated text
transcript.

Nothing is reimplemented here. `xray_utils` already holds every renderer as a
pure function of `(EpisodeEvents, EventGroup)`, so this module only loads the
episode through the production path (`FileStorage.load_episode` →
`EpisodeEvents.from_view`, exactly what `XRayState._reload_events` does) and
assembles the same fragment `xray.py` lays out.

**Blinding is structural.** `xray.py` and `xray_utils.py` contain no reference to
findings, blame, or the investigator, so no renderer here has a code path that
could surface the judge's verdict. The header shows the blind `uid`, never the
experiment name or the results-directory path.
"""

from __future__ import annotations

import html as html_lib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from cube_harness.analyze import xray_utils
from cube_harness.analyze.annotate.store import Episode
from cube_harness.analyze.xray_events import EpisodeEvents
from cube_harness.storage import FileStorage

#: How many decoded episodes to keep in memory. Every event-rail click
#: re-renders, so re-decoding the episode each time would make navigation
#: crawl; a handful of entries covers the "one annotator, one episode, many
#: clicks" access pattern with a bounded footprint (screenshots are PIL images).
CACHE_SIZE = 4


@dataclass
class ViewerRender:
    """Everything the portal's viewer column shows for one selected event."""

    header: str = ""
    stats: str = ""
    goal: str = ""
    rail: str = ""
    reasoning: str = ""
    action: str = ""
    images: list[Image.Image] = field(default_factory=list)
    observation: str = ""
    chat: str = ""
    evaluation: str = ""

    def all_text(self) -> str:
        """Every string this render puts on screen, joined.

        Exists for the blinding audit: the study's claim is that no rendered
        surface carries the judge's verdict, and that is only checkable against
        *all* of them at once. `tests/test_annotate_panel.py` and
        `scripts/smoke/annotate_portal.py` both assert against this.
        """
        surfaces = (
            self.header,
            self.stats,
            self.goal,
            self.rail,
            self.reasoning,
            self.action,
            self.observation,
            self.chat,
            self.evaluation,
        )
        return "\n".join(surfaces)


def _panel(title: str, body: str, *, bg: str, border: str, title_bg: str, title_fg: str) -> str:
    """One `.info-panel` block — the same markup xray.py's panels use."""
    return (
        f'<div class="info-panel" style="background:{bg}; border-color:{border};">'
        f'<div class="info-panel-title" style="background:{title_bg}; color:{title_fg};">{title}</div>'
        f'<div class="info-panel-body">{body}</div>'
        "</div>"
    )


def render_goal_panel(text: str) -> str:
    """Task goal as a styled panel (mirrors `xray._render_goal_panel`)."""
    safe = html_lib.escape(text).replace("\n", "<br>")
    return _panel("📋 Goal", safe, bg="#f0f4ff", border="#c7d2fe", title_bg="#e0e7ff", title_fg="#4338ca")


def _stats_html(stats: dict[str, Any]) -> str:
    """Compact duration / token / cost line for the episode header."""
    bits: list[str] = []
    duration = stats.get("duration")
    if duration:
        bits.append(f"⏱️ **{xray_utils.format_duration(float(duration))}**")
    steps = stats.get("n_env_steps")
    if steps:
        bits.append(f"👣 **{int(steps)}** env steps")
    calls = stats.get("total_llm_calls")
    if calls:
        bits.append(f"🧠 **{int(calls)}** LLM calls")
    return " &nbsp;&nbsp; ".join(bits)


class EpisodeViewer:
    """Loads and renders study episodes. One instance per portal process."""

    def __init__(self) -> None:
        self._cache: OrderedDict[str, EpisodeEvents | None] = OrderedDict()
        self._stats: dict[str, dict[str, Any]] = {}

    # --- loading ----------------------------------------------------------

    def events(self, episode: Episode) -> EpisodeEvents | None:
        """The episode's event stream, or None if it cannot be decoded.

        A single unreadable episode must not take the portal down mid-study, so
        failures degrade to an empty view (the same choice
        `XRayState._reload_events` makes) and the annotator can skip it.
        """
        if episode.uid in self._cache:
            self._cache.move_to_end(episode.uid)
            return self._cache[episode.uid]

        events: EpisodeEvents | None = None
        try:
            episode_dir = Path(episode.episode_dir)
            # sample.json records `<results>/<experiment>/episodes/<traj_id>`;
            # FileStorage is rooted at the experiment directory.
            storage = FileStorage(episode_dir.parent.parent)
            view = storage.load_episode(episode.trajectory_id)
            events = EpisodeEvents.from_view(view)
            # `compute_trajectory_stats` falls through to `traj.steps`, which a
            # lazy TrajectoryView does not have; read summary_stats directly.
            self._stats[episode.uid] = dict(view.summary_stats or {})
        except Exception:  # noqa: BLE001 — one corrupt episode must not sink the study
            self._stats[episode.uid] = {}

        self._cache[episode.uid] = events
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return events

    def n_events(self, episode: Episode) -> int:
        events = self.events(episode)
        return len(events) if events is not None else 0

    def initial_index(self, episode: Episode) -> int:
        """First event worth showing — the first LLM call, not the reset obs."""
        return 1 if self.n_events(episode) > 1 else 0

    def step(self, episode: Episode, selected: int, where: str) -> int:
        """Move the selection: 'first' | 'prev' | 'next' | 'last'."""
        events = self.events(episode)
        if events is None or len(events) == 0:
            return 0
        selected = max(0, min(selected, len(events) - 1))
        if where == "first":
            return events.first_group_root()
        if where == "last":
            return events.last_group_root()
        if where == "prev":
            return events.prev_group_root(selected)
        if where == "next":
            return events.next_group_root(selected)
        return selected

    # --- rendering --------------------------------------------------------

    def render(self, episode: Episode, selected: int) -> ViewerRender:
        """Assemble the full viewer fragment for `(episode, selected)`."""
        events = self.events(episode)
        stats = self._stats.get(episode.uid, {})
        header = (
            f"**{episode.uid}** │ {episode.modality} │ task `{html_lib.escape(episode.task_id)}` "
            f"│ final score {episode.score:g}"
        )
        if events is None or len(events) == 0:
            return ViewerRender(
                header=header,
                goal=render_goal_panel("(this episode could not be loaded — please Skip it)"),
                rail="<div style='padding:10px;color:#666;'>No events to display</div>",
                reasoning=_reasoning_panel("<em>Episode unavailable.</em>"),
                action=_action_panel("<em>Episode unavailable.</em>"),
                observation="<em>Episode unavailable.</em>",
                chat="<em>Episode unavailable.</em>",
                evaluation="*Episode unavailable.*",
            )

        selected = max(0, min(selected, len(events) - 1))
        group = events.group_for(selected)
        header += f" │ Event {selected + 1}/{len(events)}"
        images, observation = xray_utils.render_group_observation_html(events, group)
        return ViewerRender(
            header=header,
            stats=_stats_html(stats),
            goal=render_goal_panel(xray_utils.goal_from_events(events)),
            rail=xray_utils.render_event_rail_html(events, selected),
            reasoning=_reasoning_panel(xray_utils.render_group_reasoning_html(events, group)),
            action=_action_panel(xray_utils.render_group_action_html(events, group)),
            images=images,
            observation=observation,
            chat=xray_utils.render_group_chat_html(events, group),
            evaluation=xray_utils.render_group_evaluation_md(events, group),
        )


def _reasoning_panel(body: str) -> str:
    return _panel("🧠 Reasoning", body, bg="#eff6ff", border="#bfdbfe", title_bg="#dbeafe", title_fg="#1d4ed8")


def _action_panel(body: str) -> str:
    return _panel("🤖 Action", body, bg="#f0fdf4", border="#bbf7d0", title_bg="#dcfce7", title_fg="#15803d")
