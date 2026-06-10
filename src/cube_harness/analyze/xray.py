"""cube-harness XRay Viewer.

A Gradio-based experiment viewer with agent/task hierarchy, lazy tab loading, and
rich per-event inspection.

Event model: an episode is a flat, ordered stream of events (LLM call / tool call
/ evaluation / error), loaded via `storage.load_episode` → `EpisodeEvents`. The
vertical card rail lists every event coloured by kind; selecting one surfaces its
whole logical group (the LLM call + the observation(s) it produced + their
evaluations + any error) in the detail tabs. Navigation moves between events.
"""

import argparse
import html as html_lib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import pandas as pd

from cube_harness import EXP_DIR
from cube_harness.analyze import inspect_results, xray_utils
from cube_harness.analyze.xray_events import EpisodeEvents
from cube_harness.core import Trajectory
from cube_harness.experiment_status import EXPERIMENT_STATUS_FILENAME, ExperimentStatus
from cube_harness.reproducibility import submissions
from cube_harness.storage import FileStorage

# ---------------------------------------------------------------------------
# State identifiers
# ---------------------------------------------------------------------------


@dataclass
class StepId:
    """Identifies the selected event (flat-stream index) in the loaded episode.

    Used purely as a Gradio gr.State trigger value — changing it re-renders the
    detail tabs and the card rail. `step` is the event index, not an env-step.
    """

    step: int = 0


# ---------------------------------------------------------------------------
# XRayState — all mutable viewer state, captured by closures
# ---------------------------------------------------------------------------


@dataclass
class XRayState:
    """All mutable state for the XRay viewer, captured by handler closures."""

    results_dir: Path
    trajectories: list[Trajectory] = field(default_factory=list)
    selected_agent_key: str | None = None
    # Metadata stub for the open episode (tables/header/stats/logs read it).
    current_trajectory: Trajectory | None = None
    # Flat event stream of the open episode (the detail panes consume this).
    current_events: EpisodeEvents | None = None
    # Index of the selected event in current_events (the active card).
    selected: int = 0

    # One FileStorage per loaded experiment
    _storages: list[FileStorage] = field(default_factory=list, repr=False)
    # Parallel to self.trajectories: _traj_storages[i] is the storage that owns trajectories[i].
    # Index-based (not traj_id-based) so experiments with overlapping task IDs don't collide.
    _traj_storages: list[FileStorage] = field(default_factory=list, repr=False)
    # Names of currently selected experiments (for change-detection in the UI)
    _selected_exp_names: list[str] = field(default_factory=list, repr=False)
    # Per-storage timestamp tag (parsed from exp dir name); keyed by id(storage).
    # Always appended to agent_name so each trajectory is unambiguously identified.
    _exp_tags: dict[int, str] = field(default_factory=dict, repr=False)
    # Per-storage backfill names for backwards-compat (agent class short name from config); keyed by id(storage).
    _backfill_names: dict[int, str | None] = field(default_factory=dict, repr=False)
    # Per-storage config JSON strings for the Config tabs; keyed by id(storage).
    # Value: (agent_config_json, exp_config_json), both may be None if unavailable.
    _storage_configs: dict[int, tuple[str | None, str | None]] = field(default_factory=dict, repr=False)
    # Live polling: tracks which trajectories are done (skip on future ticks) and their mtimes
    _completed_ids: set[str] = field(default_factory=set, repr=False)
    _traj_mtimes: dict[str, float] = field(default_factory=dict, repr=False)
    # Timestamp of last detected file change — used for stale experiment detection
    _last_change_time: float = field(default=0.0, repr=False)
    # Ordered traj_ids matching the last rendered trajectory table rows; used for row→traj_id lookup
    _traj_row_ids: list[str] = field(default_factory=list, repr=False)

    def load_experiments(self, exp_dirs: list[Path]) -> bool:
        """Load trajectory metadata stubs from one or more experiment directories.

        Each directory gets its own FileStorage instance. Trajectories from all
        directories are merged into self.trajectories. The parallel _traj_storages
        list maps each trajectory to its owning storage by index, avoiding collisions
        when multiple experiments share identical task/episode IDs.

        Returns True if at least one trajectory was loaded.

        Stats (steps/tokens/cost/duration) come from each trajectory's persisted
        ``summary_stats`` on the metadata stub, so the tables render without loading any
        steps. Full steps are loaded lazily — by ``select_trajectory`` when you open a
        trajectory, and by ``refresh_experiment`` for in-flight ones — never eagerly.
        """
        self._storages = [FileStorage(d) for d in exp_dirs]
        self._selected_exp_names = [d.name for d in exp_dirs]
        self.trajectories = []
        self._traj_storages = []
        self._exp_tags = {}
        self._backfill_names = {}
        self._storage_configs = {}
        for exp_dir, storage in zip(exp_dirs, self._storages):
            trajs = storage.load_all_trajectory_metadata()
            stubs = storage.load_missing_trajectory_stubs()
            self._load_experiment_config(exp_dir, storage)
            self._exp_tags[id(storage)] = xray_utils._parse_exp_date(exp_dir)
            for traj in trajs + stubs:
                self._apply_agent_name(traj, storage)
                self._apply_exp_tag(traj, storage)
            self.trajectories.extend(trajs + stubs)
            self._traj_storages.extend([storage] * (len(trajs) + len(stubs)))
        self.selected_agent_key = None
        self.current_trajectory = None
        self.current_events = None
        self.selected = 0
        self._completed_ids = {t.id for t in self.trajectories if t.end_time is not None}
        self._traj_mtimes = {}
        for storage in self._storages:
            self._traj_mtimes.update(storage.list_trajectory_ids_with_mtime())
        self._last_change_time = time.time()
        return len(self.trajectories) > 0

    def should_poll(self) -> bool:
        """Whether the live-refresh timer should run: an experiment is loaded and not yet
        complete. (Historical/complete experiments need no polling.)"""
        return bool(self.trajectories) and not self.is_experiment_complete()

    def load_experiment(self, exp_dir: Path) -> bool:
        """Convenience wrapper: load a single experiment directory."""
        return self.load_experiments([exp_dir])

    def _load_experiment_config(self, exp_dir: Path, storage: FileStorage) -> None:
        """Read experiment_config.json and store per-storage config JSON strings.

        Populates _storage_configs[id(storage)] with (agent_config_json, exp_config_json)
        and _backfill_names[id(storage)] with the resolved AgentConfig.agent_name.

        Falls back to the first episode_configs/*.json when experiment_config.json
        is absent (e.g. experiments run before save_config() was added).
        """
        agent_cfg: dict = {}
        exp_cfg_display: dict = {}
        config_path = exp_dir / "experiment_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    exp_cfg = json.load(f)
                agent_cfg = exp_cfg.get("agent_config", {})
                exp_cfg_display = {**exp_cfg, "agent_config": "(see Agent Config tab)"}
            except Exception:
                pass
        if not agent_cfg:
            # Fallback: extract agent_config from the first available episode config
            episode_cfgs = (
                sorted((exp_dir / "episode_configs").glob("*.json")) if (exp_dir / "episode_configs").exists() else []
            )
            for ep_path in episode_cfgs:
                try:
                    with open(ep_path) as f:
                        ep_cfg = json.load(f)
                    agent_cfg = ep_cfg.get("agent_config", {})
                    if agent_cfg:
                        break
                except Exception:
                    continue
        derived_name = xray_utils.agent_name_from_config(agent_cfg) or None
        self._backfill_names[id(storage)] = derived_name
        self._storage_configs[id(storage)] = (
            json.dumps(agent_cfg, indent=2) if agent_cfg else None,
            json.dumps(exp_cfg_display, indent=2) if exp_cfg_display else None,
        )

    def _apply_agent_name(self, traj: Trajectory, storage: FileStorage) -> None:
        """Override trajectory agent_name with the config-derived name.

        The config is the source of truth — this corrects stale class-name strings
        (e.g. "GennyConfig") written by older versions of episode.py.
        Only applied when a derived name is available (i.e. a config file exists).
        """
        name = self._backfill_names.get(id(storage))
        if name:
            traj.metadata["agent_name"] = name

    def _apply_exp_tag(self, traj: Trajectory, storage: FileStorage) -> None:
        """Append this storage's timestamp tag to traj's agent_name."""
        tag = self._exp_tags.get(id(storage), "")
        if tag:
            traj.metadata["agent_name"] = traj.metadata.get("agent_name", "unknown") + f" [{tag}]"

    def get_config_jsons(self) -> tuple[str, str]:
        """Return (agent_config_json, exp_config_json) for the currently selected agent.

        Looks up which storage owns the selected agent's trajectories, then returns
        that storage's config strings. Returns ("", "") when nothing is selected or found.
        """
        if self.selected_agent_key is None:
            return "", ""
        for i, traj in enumerate(self.trajectories):
            if traj.metadata.get("agent_name") == self.selected_agent_key:
                storage = self._traj_storages[i]
                agent_cfg, exp_cfg = self._storage_configs.get(id(storage), (None, None))
                return agent_cfg or "", exp_cfg or ""
        return "", ""

    def refresh_experiment(self) -> bool:
        """Incrementally reload new or changed trajectories from disk. Returns True if anything changed.

        Uses mtime-based change detection: only trajectories whose files have changed since the
        last check are reloaded. Completed trajectories (end_time set) are skipped entirely.
        Called on each bg_timer tick while the experiment is still running.
        """
        if not self._storages:
            return False
        changed = False
        known_ids = {t.id for t in self.trajectories}

        for storage in self._storages:
            id_mtimes = storage.list_trajectory_ids_with_mtime()
            for traj_id, mtime in id_mtimes.items():
                if traj_id in self._completed_ids:
                    continue
                prev_mtime = self._traj_mtimes.get(traj_id, 0.0)
                if mtime <= prev_mtime and traj_id in known_ids:
                    continue
                # Record the mtime up front: a metadata-less stub fails load_trajectory
                # every tick otherwise, re-reading it forever.
                self._traj_mtimes[traj_id] = mtime
                try:
                    full = storage.load_trajectory(traj_id)
                except Exception:
                    # No trajectory file yet (e.g. a QUEUED stub whose status.json flipped
                    # to STALE on driver death) or an unreadable file: fall back to a cheap
                    # status-only refresh so the terminal flip still surfaces live.
                    if self._reinject_episode_status(traj_id, storage):
                        changed = True
                    continue
                self._apply_agent_name(full, storage)
                self._apply_exp_tag(full, storage)
                # Refresh only updates table stats, never displayed content, so steps are
                # never read here — drop them to bound RAM on long live runs. The detail
                # panes render from the event stream loaded by select_trajectory.
                if full.summary_stats is None:
                    full.summary_stats = xray_utils.compute_trajectory_stats(full)
                full.steps = []
                is_current = self.current_trajectory is not None and self.current_trajectory.id == traj_id
                changed = True
                # Find the existing slot owned by this storage (avoids ID collision)
                idx = next(
                    (
                        i
                        for i, t in enumerate(self.trajectories)
                        if t.id == traj_id and self._traj_storages[i] is storage
                    ),
                    None,
                )
                if idx is not None:
                    self.trajectories[idx] = full
                    self._traj_storages[idx] = storage
                    if is_current:
                        self.current_trajectory = full
                        self._reload_events(storage, traj_id)
                else:
                    self.trajectories.append(full)
                    self._traj_storages.append(storage)
                    known_ids.add(traj_id)
                if full.end_time is not None:
                    self._completed_ids.add(traj_id)
        if changed:
            self._last_change_time = time.time()
        return changed

    def _reinject_episode_status(self, traj_id: str, storage: FileStorage) -> bool:
        """Re-inject ``_episode_status`` (+ retry/error fields) from status.json onto an
        already-loaded trajectory/stub that has no (new) trajectory file to load.

        The cheap counterpart to a full reload: one small JSON read, no step decode. Lets
        a status-only transition (e.g. QUEUED→STALE) update the display in place. Returns
        True if the displayed status actually changed.
        """
        status = storage.read_episode_status(traj_id)
        if status is None:
            return False
        idx = next(
            (i for i, t in enumerate(self.trajectories) if t.id == traj_id and self._traj_storages[i] is storage),
            None,
        )
        if idx is None:
            return False
        meta = self.trajectories[idx].metadata
        changed = meta.get("_episode_status") != status.status
        meta["_episode_status"] = status.status
        meta["_retry_count"] = status.retry_count
        meta["_error_type"] = status.error_type
        meta["_error_message"] = status.error_message
        return changed

    def ray_dashboard_links(self) -> list[tuple[str, str]]:
        """Return ``[(exp_name, ray_dashboard_url)]`` for selected experiments whose
        experiment_status.json records a Ray dashboard URL.

        Populated only in Ray mode while the driver is up (the URL points at the live Ray
        cluster); empty for sequential runs and usually dead once the run completes. One
        small file read per selected experiment — not per-episode.
        """
        links: list[tuple[str, str]] = []
        for storage in self._storages:
            status = ExperimentStatus.read(storage.output_dir / EXPERIMENT_STATUS_FILENAME)
            if status is not None and status.ray_dashboard_url:
                links.append((storage.output_dir.name, status.ray_dashboard_url))
        return links

    def is_experiment_complete(self) -> bool:
        """Return True when every known trajectory has reached a terminal status."""
        if not self.trajectories:
            return False
        return all(xray_utils.trajectory_status(t) in xray_utils.TERMINAL_STATUSES for t in self.trajectories)

    def is_experiment_stale(self, timeout_s: float = 1200.0) -> bool:
        """Return True if no file changes have been detected for timeout_s seconds.

        Used to stop the live-polling timer when an experiment appears to have stalled
        (e.g., the runner crashed without setting end_time on every trajectory).
        Default timeout is 20 minutes.
        """
        return time.time() - self._last_change_time > timeout_s

    def select_agent(self, agent_key: str) -> None:
        """Select an agent; resets the open trajectory + its event stream."""
        self.selected_agent_key = agent_key
        self.current_trajectory = None
        self.current_events = None
        self.selected = 0

    def select_trajectory(self, traj_id: str) -> None:
        """Open a trajectory by ID and load its event stream.

        The metadata stub stays as `current_trajectory` (tables/header/stats/logs
        read it); the detail panes consume `current_events`, loaded fresh via
        `storage.load_episode` → `EpisodeEvents.from_view`. When multiple
        experiments share task/episode IDs, prefers the slot whose agent_name
        matches the current selection, falling back to the first match.
        """
        # Prefer the slot whose agent matches the current selection (multi-experiment safety)
        idx = next(
            (
                i
                for i, t in enumerate(self.trajectories)
                if t.id == traj_id and t.metadata.get("agent_name") == self.selected_agent_key
            ),
            None,
        )
        if idx is None:
            idx = next((i for i, t in enumerate(self.trajectories) if t.id == traj_id), None)
        if idx is None:
            self.current_trajectory = None
            self.current_events = None
            self.selected = 0
            return
        self.current_trajectory = self.trajectories[idx]
        self.selected = 0
        storage = self._traj_storages[idx]
        if self.current_trajectory.metadata.get("_missing"):
            self.current_events = None  # no episode dir on disk to load
        else:
            self._reload_events(storage, traj_id)
            # Default to the first event AFTER the initial observation (the first
            # LLM call / action) rather than the reset observation itself.
            if self.n_events() > 1:
                self.selected = 1

    def _reload_events(self, storage: FileStorage, traj_id: str) -> None:
        """(Re)load the open episode's event stream into `current_events`."""
        try:
            self.current_events = EpisodeEvents.from_view(storage.load_episode(traj_id))
        except Exception:
            self.current_events = None  # renders degrade to an empty state

    # --- event/group accessors (consumed by the detail panes) -------------

    def n_events(self) -> int:
        return len(self.current_events) if self.current_events is not None else 0

    def selected_group(self):  # -> EventGroup | None
        """The resolved logical group of the selected event, or None."""
        if self.current_events is None or self.n_events() == 0:
            return None
        sel = max(0, min(self.selected, self.n_events() - 1))
        return self.current_events.group_for(sel)

    def current_storage(self) -> FileStorage | None:
        """Return the FileStorage that owns the currently selected trajectory."""
        if self.current_trajectory is None:
            return None
        idx = next((i for i, t in enumerate(self.trajectories) if t is self.current_trajectory), None)
        if idx is None:
            return None
        return self._traj_storages[idx]


# ---------------------------------------------------------------------------
# Lazy tab loading decorator
# ---------------------------------------------------------------------------


def if_active(tab_name: str, n_out: int = 1) -> Callable:
    """Decorator factory that makes a handler a no-op when the given tab is not active.

    The wrapped function receives `active_tab` as its first positional argument (a str).
    When active_tab != tab_name: returns gr.skip() (or a tuple of n_out gr.skip()).
    When active_tab == tab_name: calls the original function (render functions read
    state via closure and take no extra arguments).

    Usage:
        step_id.change(
            fn=if_active("AXTree")(render_axtree),
            inputs=[active_tab, step_id],
            outputs=axtree_code,
        )
    """

    def decorator(fn: Callable) -> Callable:
        def wrapper(active_tab: str, *_args: Any, **_kwargs: Any) -> Any:
            if active_tab != tab_name:
                if n_out == 1:
                    return gr.skip()
                return tuple(gr.skip() for _ in range(n_out))
            # Render functions read state via closure — no args to forward.
            return fn()

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# CSS and keyboard shortcuts JS
# ---------------------------------------------------------------------------


_CSS = """
html {
    color-scheme: light only;
}
/* Stable scroll container for the event rail: overflow lives here (not on the
   re-rendered inner HTML), so clicking a card keeps the scroll position. */
#xray_rail {
    max-height: 72vh;
    overflow-y: auto;
    padding: 4px;
    background: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}
/* Tiny, tight nav buttons (jump-to-first ⤒ · prev ◀ · next ▶ · jump-to-last ⤓),
   centred and hugging the rail. */
#xray_first_btn, #xray_prev_btn, #xray_next_btn, #xray_last_btn {
    min-width: 28px !important;
    max-width: 34px;
    padding: 2px 6px !important;
    flex: 0 0 auto;
}
.xray-nav-row {
    justify-content: center !important;
    gap: 6px !important;
    margin-bottom: 2px !important;
    min-height: 0 !important;
}
/* Experiments toolbar: one row that never wraps. Dir controls on the left; the
   growing dir label pushes the two action split-buttons (Archive 🤖✓ · Submit 🤖✓)
   to the right. Each (action, auto-select) pair renders as one split-button. */
.xray-exp-toolbar {
    gap: 0 !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    width: 100%;
}
/* The dir label grows to fill, right-aligning the actions; ellipsis if long. */
#exp_dir_label {
    flex: 1 1 auto !important;
    min-width: 30px;
    margin: 0 10px !important;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
#exp_refresh_btn {
    min-width: 34px !important;
    max-width: 40px;
    padding: 2px 8px !important;
    flex: 0 0 auto;
    margin-left: 6px !important;
}
/* auto-select buttons: stretch to the action button's height (so the split-
   button halves match) and fit "🤖✓" on ONE line (no vertical wrap). */
#exp_pick_archivable_btn, #exp_pick_submittable_btn {
    min-width: 0 !important;
    flex: 0 0 auto;
    align-self: stretch !important;
}
#exp_pick_archivable_btn button, #exp_pick_submittable_btn button {
    height: 100% !important;
    padding: 2px 10px !important;
    white-space: nowrap !important;
}
/* gap between the Archive split-button and the Submit cluster */
#exp_submit_registry_btn {
    margin-left: 16px !important;
}
/* Submit cluster = [Registry | EEE | 🤖✓] joined into one unit. Left segment
   (Registry) keeps its left radius; the middle (EEE) is square; the 🤖✓ keeps
   the right radius (handled by the shared pick-button rule below). */
#exp_archive_btn button, #exp_submit_registry_btn button {
    border-top-right-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
}
#exp_submit_eee_btn button {
    border-radius: 0 !important;
    border-left: 1px solid rgba(0, 0, 0, 0.18) !important;
}
#exp_pick_archivable_btn button, #exp_pick_submittable_btn button {
    border-top-left-radius: 0 !important;
    border-bottom-left-radius: 0 !important;
    border-left: 1px solid rgba(0, 0, 0, 0.18) !important;
}
.compact-header {
    padding: 8px 16px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    color: white;
}
.compact-header, .compact-header * {
    color: white !important;
}
.step-details {
    max-height: 600px;
    overflow-y: auto;
    padding: 12px;
}
.step-details pre {
    max-height: 300px;
    overflow-y: auto;
}
.error-box {
    background: #fee2e2;
    border: 1px solid #ef4444;
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 8px;
}
.info-panel {
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
.info-panel-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 4px 10px;
    color: #6b7280;
}
.info-panel-body {
    padding: 6px 10px;
    max-height: 100px;
    overflow-y: auto;
    font-size: 13px;
    line-height: 1.5;
}
.info-panel-body code {
    background: rgba(0,0,0,0.05);
    border-radius: 3px;
    padding: 1px 4px;
    font-size: 12px;
}
code {
    white-space: pre-wrap;
}
.help-content {
    max-height: 260px;
    overflow-y: auto;
    padding-right: 8px;
}
th {
    white-space: normal !important;
    word-wrap: break-word !important;
}
#timeline_click_input {
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* Experiments table: hide entire first header cell (checkbox col has no label) */
#exp_table table thead tr th:first-child {
    visibility: hidden !important;
}
/* Experiments table: narrow the checkbox column */
#exp_table table th:first-child,
#exp_table table td:first-child {
    width: 36px !important;
    min-width: 36px !important;
    max-width: 36px !important;
}
/* Experiments table: cap experiment name column, overflow with ellipsis */
#exp_table table th:nth-child(2),
#exp_table table td:nth-child(2) {
    max-width: 300px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}
/* Experiments table: fixed widths for metadata columns */
#exp_table table th:nth-child(3),
#exp_table table td:nth-child(3) { width: 130px !important; white-space: nowrap !important; }
#exp_table table th:nth-child(4),
#exp_table table td:nth-child(4) { width: 120px !important; overflow: hidden !important; text-overflow: ellipsis !important; white-space: nowrap !important; }
#exp_table table th:nth-child(5),
#exp_table table td:nth-child(5) { width: 160px !important; overflow: hidden !important; text-overflow: ellipsis !important; white-space: nowrap !important; }
#exp_table table th:nth-child(6),
#exp_table table td:nth-child(6) { width: 130px !important; overflow: hidden !important; text-overflow: ellipsis !important; white-space: nowrap !important; }
#exp_table table th:nth-child(7),
#exp_table table td:nth-child(7) { width: 130px !important; }
#exp_table table th:nth-child(8),
#exp_table table td:nth-child(8) { width: 120px !important; white-space: nowrap !important; }
/* Experiments table: hide the cell context menu (Add row/Delete row) entirely */
#exp_table .cell-menu {
    display: none !important;
}
/* Experiments table: prevent renaming column headers */
#exp_table table thead th [contenteditable] {
    pointer-events: none !important;
}
/* Experiments table: non-checkbox cells are read-only — block click-to-edit */
#exp_table td:not(:first-child) span[data-editable] {
    pointer-events: none !important;
    cursor: default !important;
    user-select: text !important;
}
/* Experiments table: stronger contrast for unchecked checkboxes */
#exp_table input[type="checkbox"] {
    -webkit-appearance: none;
    appearance: none;
    width: 16px !important;
    height: 16px !important;
    border: 2px solid #6b7280 !important;
    border-radius: 3px !important;
    background: white !important;
    cursor: pointer !important;
    position: relative !important;
    flex-shrink: 0 !important;
}
#exp_table input[type="checkbox"]:checked {
    background: #6366f1 !important;
    border-color: #6366f1 !important;
}
#exp_table input[type="checkbox"]:checked::after {
    content: "" !important;
    display: block !important;
    width: 4px !important;
    height: 8px !important;
    border: 2px solid white !important;
    border-top: none !important;
    border-left: none !important;
    transform: rotate(45deg) !important;
    position: absolute !important;
    top: 1px !important;
    left: 4px !important;
}
"""

# Runs once on app load (Blocks js=): force light theme + bind keyboard event
# navigation. Plain arrows (←/→ or ↑/↓) step to the previous/next event; Shift+↑/↓
# (or Home/End) jump to the first/last step. preventDefault + the input/textarea
# guard keep Shift+arrow from extending a browser text selection. Gradio
# puts `elem_id` on the <button> itself, so the selectors are `#xray_prev_btn`,
# NOT `#xray_prev_btn button`. Tooltips advertise the shortcut. (Rail scroll is
# preserved by the CSS overflow living on the stable `#xray_rail` container, so
# no scroll-restore JS is needed.)
_INIT_JS = """
() => {
    document.body.classList.remove('dark');
    if (window.__xrayInit) return;
    window.__xrayInit = true;
    const TIPS = {
        '#xray_first_btn': 'Jump to first step (Shift+↑ or Home)',
        '#xray_prev_btn': 'Previous event (← or ↑)',
        '#xray_next_btn': 'Next event (→ or ↓)',
        '#xray_last_btn': 'Jump to last step / end (Shift+↓ or End)',
        '#exp_browse_btn': 'Pick a different results directory',
        '#exp_refresh_btn': 'Re-scan the results directory (cached — fast)',
        '#exp_archive_btn': 'Archive all checked experiments (moves them to _archive/)',
        '#exp_pick_archivable_btn': 'Auto-select broken + rejected + explicit-debug (is_official=False) experiments to archive',
        '#exp_submit_registry_btn': 'Submit checked experiments to the cube-registry reproducibility journal — opens an auto-validating, auto-merging PR. For publishing REFERENCE values (cross-infra drift detection), NOT a leaderboard.',
        '#exp_submit_eee_btn': 'Submit checked experiments to EEE (the eval results store) — for showcasing agent/model performance. Runs scripts/submit_to_eee.py.',
        '#exp_pick_submittable_btn': 'Auto-select submittable, not-yet-submitted experiments (applies to whichever submit button you click next)',
    };
    const setTip = () => {
        for (const [sel, tip] of Object.entries(TIPS)) {
            const el = document.querySelector(sel);
            if (el) el.title = tip;
        }
    };
    setTip();
    setTimeout(setTip, 1000);
    document.addEventListener('keydown', (e) => {
        const t = e.target, tag = (t.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select' || t.isContentEditable) return;
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        let sel = null;
        // Shift+↑/↓ and Home/End jump to the first/last step; plain arrows step.
        if (e.shiftKey) {
            if (e.key === 'ArrowUp') sel = '#xray_first_btn';
            else if (e.key === 'ArrowDown') sel = '#xray_last_btn';
        } else if (e.key === 'Home') sel = '#xray_first_btn';
        else if (e.key === 'End') sel = '#xray_last_btn';
        else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') sel = '#xray_prev_btn';
        else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') sel = '#xray_next_btn';
        if (!sel) return;
        const b = document.querySelector(sel);
        if (b) { e.preventDefault(); b.click(); }
    }, true);
}
"""


# ---------------------------------------------------------------------------
# HTML rendering helpers (tables + info panels)
# ---------------------------------------------------------------------------


def _render_goal_panel(text: str) -> str:
    """Render the task goal as a styled HTML panel with a fixed title bar."""
    safe = html_lib.escape(text)
    # Preserve newlines
    safe = safe.replace("\n", "<br>")
    return (
        '<div class="info-panel" style="background:#f0f4ff; border-color:#c7d2fe;">'
        '<div class="info-panel-title" style="background:#e0e7ff; color:#4338ca;">📋 Goal</div>'
        f'<div class="info-panel-body">{safe}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# run_xray — main entry point
# ---------------------------------------------------------------------------


def run_xray(
    results_dir: Path,
    debug: bool = False,
    port: int | None = None,
    share: bool = False,
) -> None:
    """Launch the cube-harness XRay Gradio viewer.

    Args:
        results_dir: Path to the root results directory containing experiment subdirectories.
        debug: Enable Gradio debug mode with hot reloading.
        port: Server port. If None, Gradio picks an available port.
        share: Enable Gradio share link for remote access.
    """
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    # Single state instance captured by all handler closures below
    state = XRayState(results_dir=results_dir)

    # ------------------------------------------------------------------
    # Handler functions (closures capturing `state`)
    # ------------------------------------------------------------------

    def _make_tab_labels(
        agent_rows: list[dict[str, Any]],
        traj_rows: list[dict[str, Any]],
    ) -> tuple[gr.Tab, gr.Tab]:
        """Return gr.Tab updates with counts embedded in labels."""
        return (
            gr.Tab(label=f"Agents ({len(agent_rows)})"),
            gr.Tab(label=f"Trajectories ({len(traj_rows)})"),
        )

    def _load_and_build_hierarchy() -> tuple[str, Any, Any, StepId, gr.Tab, gr.Tab, str, str]:
        """Build experiment stats + agent/trajectory tables after state.load_experiments().

        Auto-selects the first agent and first trajectory when available.
        Returns the 8-tuple expected by both on_experiments_change and on_bg_load_tick callers.
        """
        exp_stats = xray_utils.compute_experiment_stats(state.trajectories)
        agent_rows = xray_utils.build_agent_table(state.trajectories)

        if not agent_rows:
            tab_labels = _make_tab_labels(agent_rows, [])
            return exp_stats, _rows_to_table(agent_rows), [], StepId(), *tab_labels, *state.get_config_jsons()
        first_agent_key = agent_rows[0]["agent_name"]
        state.select_agent(first_agent_key)
        agent_table_data = _rows_to_table(agent_rows, first_agent_key, "agent_name")

        traj_rows = xray_utils.build_trajectory_table(state.trajectories, first_agent_key)
        state._traj_row_ids = [r["_traj_id"] for r in traj_rows]
        if not traj_rows:
            tab_labels = _make_tab_labels(agent_rows, traj_rows)
            return (
                exp_stats,
                agent_table_data,
                _rows_to_table(traj_rows),
                StepId(),
                *tab_labels,
                *state.get_config_jsons(),
            )
        first_traj_id = traj_rows[0]["_traj_id"]
        state.select_trajectory(first_traj_id)
        traj_table_data = _rows_to_table(traj_rows, first_traj_id, "_traj_id")

        tab_labels = _make_tab_labels(agent_rows, traj_rows)
        return (
            exp_stats,
            agent_table_data,
            traj_table_data,
            StepId(step=state.selected),
            *tab_labels,
            *state.get_config_jsons(),
        )

    def on_experiments_change(
        exp_df: Any,
    ) -> tuple[str, Any, Any, StepId, gr.Tab, gr.Tab, str, str, gr.Timer]:
        """Handle checkbox changes in the Experiments table.

        Extracts selected experiment names, loads them (merging trajectories), and
        rebuilds the agent/trajectory hierarchy. Returns gr.skip() for all outputs
        when the selection hasn't actually changed (avoids spurious Gradio events).
        """
        _empty = (
            "",
            None,
            None,
            StepId(),
            gr.Tab(label="Agents (0)"),
            gr.Tab(label="Trajectories (0)"),
            "",
            "",
            gr.Timer(active=False),
        )
        if exp_df is None or len(exp_df) == 0:
            return _empty
        selected_names = [str(exp_df.iloc[i, 1]) for i in range(len(exp_df)) if exp_df.iloc[i, 0]]
        if set(selected_names) == set(state._selected_exp_names):
            return tuple(gr.skip() for _ in range(9))  # type: ignore[return-value]
        if not selected_names:
            state._selected_exp_names = []
            state.trajectories = []
            state.selected_agent_key = None
            return _empty
        exp_dirs = [state.results_dir / name for name in selected_names]
        state.load_experiments(exp_dirs)
        hierarchy = _load_and_build_hierarchy()
        return (*hierarchy, gr.Timer(active=state.should_poll()))

    def on_archive_selected() -> tuple[
        Any, str, Any, Any, Any, StepId, gr.Tab, gr.Tab, gr.Tab, str, str, gr.Timer, Any
    ]:
        """Archive all currently selected experiments and reset state."""
        names = list(state._selected_exp_names)
        for name in names:
            # Stamp a durable rejection for broken runs before moving them, so the
            # verdict travels into _archive/ (mirrors scan_experiments --persist-broken).
            xray_utils.persist_broken_rejection(state.results_dir / name)
            xray_utils.archive_experiment(state.results_dir, name)
        state._selected_exp_names = []
        state.trajectories = []
        state.selected_agent_key = None
        _empty_hierarchy = (
            "",
            None,
            None,
            StepId(),
            gr.Tab(label="Agents (0)"),
            gr.Tab(label="Trajectories (0)"),
            "",
            "",
            gr.Timer(active=False),
        )
        # Replace the now-stale "Selected N…" line with a result (or clear it).
        status = (
            gr.update(value=f"🗃 Archived **{len(names)}** experiment(s) to `_archive/`.", visible=True)
            if names
            else gr.update(value="", visible=False)
        )
        return (_exp_table_rows(), *_empty_hierarchy, status)

    def on_select_agent(evt: gr.SelectData, agent_df: Any) -> tuple[Any, Any, StepId, gr.Tab, gr.Tab, str, str]:
        if evt is None or evt.index is None or agent_df is None or len(agent_df) == 0:
            return (
                [],
                [],
                StepId(),
                gr.Tab(label="Agents (0)"),
                gr.Tab(label="Trajectories (0)"),
                "",
                "",
            )
        row = evt.index[0]
        agent_key = re.sub(r"<[^>]+>", "", str(agent_df.iloc[row, 0]))
        state.select_agent(agent_key)
        agent_rows = xray_utils.build_agent_table(state.trajectories)
        agent_table_data = _rows_to_table(agent_rows, agent_key, "agent_name")
        traj_rows = xray_utils.build_trajectory_table(state.trajectories, agent_key)
        state._traj_row_ids = [r["_traj_id"] for r in traj_rows]
        if not traj_rows:
            tab_labels = _make_tab_labels(agent_rows, traj_rows)
            return agent_table_data, _rows_to_table(traj_rows), StepId(), *tab_labels, *state.get_config_jsons()
        first_traj_id = traj_rows[0]["_traj_id"]
        state.select_trajectory(first_traj_id)
        traj_table_data = _rows_to_table(traj_rows, first_traj_id, "_traj_id")
        tab_labels = _make_tab_labels(agent_rows, traj_rows)
        return (
            agent_table_data,
            traj_table_data,
            StepId(step=state.selected),
            *tab_labels,
            *state.get_config_jsons(),
        )

    def on_select_trajectory(evt: gr.SelectData, traj_df: Any) -> tuple[Any, StepId]:
        if evt is None or evt.index is None or traj_df is None or len(traj_df) == 0:
            return [], StepId(step=0)
        row = evt.index[0]
        # Recompute traj_ids from the live state rather than reading the shared
        # _traj_row_ids snapshot, which on_bg_load_tick may overwrite concurrently.
        agent_key = state.selected_agent_key
        if agent_key is None:
            return _rows_to_table([]), StepId(step=0)
        current_traj_rows = xray_utils.build_trajectory_table(state.trajectories, agent_key)
        current_traj_row_ids = [r["_traj_id"] for r in current_traj_rows]
        state._traj_row_ids = current_traj_row_ids
        if row >= len(current_traj_row_ids):
            return _rows_to_table([]), StepId(step=0)
        traj_id = current_traj_row_ids[row]
        state.select_trajectory(traj_id)
        return _rows_to_table(current_traj_rows, traj_id, "_traj_id"), StepId(step=state.selected)

    def on_bg_load_tick() -> tuple[Any, Any, Any, Any, str, gr.Timer, gr.Tab, gr.Tab, gr.Tab]:
        """Periodic live-poll: pick up new/changed trajectory files from a running experiment.

        Deactivates the timer once the experiment is complete (all trajectories terminal)
        or stale (no file changes for a long time — runner likely crashed).
        """
        state.refresh_experiment()

        exp_stats = xray_utils.compute_experiment_stats(state.trajectories)
        agent_rows = xray_utils.build_agent_table(state.trajectories)
        agent_key = state.selected_agent_key
        active_agent = agent_rows[0]["agent_name"] if (agent_rows and agent_key is None) else agent_key
        agent_table_data = _rows_to_table(agent_rows, active_agent, "agent_name")

        traj_rows = xray_utils.build_trajectory_table(state.trajectories, active_agent) if active_agent else []
        state._traj_row_ids = [r["_traj_id"] for r in traj_rows]
        traj_id = state.current_trajectory.id if state.current_trajectory else None
        traj_table_data = _rows_to_table(traj_rows, traj_id, "_traj_id")

        n_total = len(state.trajectories)
        n_completed = sum(
            1 for t in state.trajectories if xray_utils.trajectory_status(t) in xray_utils.TERMINAL_OUTCOME_STATUSES
        )
        n_running = sum(
            1 for t in state.trajectories if xray_utils.trajectory_status(t) in xray_utils.IN_FLIGHT_STATUSES
        )
        # Per-agent breakdown (shown when > 1 agent loaded, e.g. multi-experiment)
        agent_names = sorted({t.metadata.get("agent_name", "unknown") for t in state.trajectories})
        per_agent: list[tuple[str, int, int, int]] | None = None
        if len(agent_names) > 1:
            per_agent = []
            for aname in agent_names:
                atrajs = [t for t in state.trajectories if t.metadata.get("agent_name", "unknown") == aname]
                per_agent.append(
                    (
                        aname,
                        sum(
                            1 for t in atrajs if xray_utils.trajectory_status(t) in xray_utils.TERMINAL_OUTCOME_STATUSES
                        ),
                        len(atrajs),
                        sum(1 for t in atrajs if xray_utils.trajectory_status(t) in xray_utils.IN_FLIGHT_STATUSES),
                    )
                )
        progress_html = xray_utils.build_progress_html(
            n_completed,
            n_total,
            n_running,
            per_agent,
            state._selected_exp_names or None,
            ray_dashboard_urls=state.ray_dashboard_links() or None,
        )

        experiment_done = state.is_experiment_complete() or state.is_experiment_stale()
        timer_update = gr.Timer(active=not experiment_done)
        tab_labels = _make_tab_labels(agent_rows, traj_rows)
        return exp_stats, agent_table_data, traj_table_data, progress_html, timer_update, *tab_labels

    def navigate_prev() -> StepId:
        """Select the previous logical group (not the previous raw event), so a
        single press moves a whole step. Reads state.selected from closure so the
        JS keyboard shortcut button.click() works without losing the gr.State."""
        if state.current_events is None:
            return StepId(step=state.selected)
        state.selected = state.current_events.prev_group_root(state.selected)
        return StepId(step=state.selected)

    def navigate_next() -> StepId:
        """Select the next logical group."""
        if state.current_events is None:
            return StepId(step=state.selected)
        state.selected = state.current_events.next_group_root(state.selected)
        return StepId(step=state.selected)

    def navigate_first() -> StepId:
        """Jump to the first step (group after the initial observation)."""
        if state.current_events is None or len(state.current_events) == 0:
            return StepId(step=state.selected)
        state.selected = state.current_events.first_group_root()
        return StepId(step=state.selected)

    def navigate_last() -> StepId:
        """Jump to the last group (end of the episode)."""
        if state.current_events is None or len(state.current_events) == 0:
            return StepId(step=state.selected)
        state.selected = state.current_events.last_group_root()
        return StepId(step=state.selected)

    def handle_timeline_click(clicked_index: int | None) -> StepId:
        """Card-rail click: select the clicked event index (clamped)."""
        if clicked_index is not None and state.current_events is not None:
            sel = int(max(0, min(clicked_index, state.n_events() - 1)))
            state.selected = sel
            return StepId(step=sel)
        return StepId(step=state.selected)

    # ------------------------------------------------------------------
    # Always-rendered handlers (update on every event selection change)
    # ------------------------------------------------------------------

    def get_compact_header_info() -> str:
        if not state.current_trajectory:
            return "No trajectory selected"
        task_id = state.current_trajectory.metadata.get("task_id", "unknown")
        agent_name = state.current_trajectory.metadata.get("agent_name", "")
        status = xray_utils.trajectory_status(state.current_trajectory)
        status_label = xray_utils._STATUS_LABEL[status]
        header = f"**{task_id}**"
        if agent_name:
            header += f" │ {agent_name}"
        header += f" │ {status_label}"
        n = state.n_events()
        if n > 0:
            header += f" │ Event {state.selected + 1}/{n}"
        return header

    def update_timeline() -> str:
        return xray_utils.render_event_rail_html(state.current_events, state.selected)

    def update_trajectory_stats() -> str:
        """Structured, multi-line stats block for the header's left column."""
        if not state.current_trajectory:
            return ""
        stats = xray_utils.compute_trajectory_stats(state.current_trajectory)
        prompt_tokens = int(stats["prompt_tokens"])
        completion_tokens = int(stats["completion_tokens"])
        cached_tokens = int(stats["cached_tokens"])
        cache_creation_tokens = int(stats["cache_creation_tokens"])
        cost = float(stats["cost"])

        lines: list[str] = []
        top = []
        if stats["duration"] is not None:
            top.append(f"⏱️ **{xray_utils.format_duration(stats['duration'])}**")
        if cost > 0:
            top.append(f"💰 **${cost:.4f}**")
        if top:
            lines.append(" &nbsp;&nbsp;&nbsp; ".join(top))

        if prompt_tokens > 0:
            total = prompt_tokens + completion_tokens
            lines.append(
                f"📥 prompt **{prompt_tokens:,}** &nbsp; 📤 completion **{completion_tokens:,}** &nbsp; Σ **{total:,}**"
            )
            cache_bits = []
            if cached_tokens > 0:
                cache_bits.append(f"⚡ cached **{cached_tokens:,}** ({cached_tokens / prompt_tokens * 100:.0f}%)")
            if cache_creation_tokens > 0:
                cache_bits.append(f"cache_created **{cache_creation_tokens:,}**")
            if cache_bits:
                lines.append(" &nbsp; ".join(cache_bits))

        return "<br>".join(lines)

    def get_task_goal() -> str:
        """Return the task goal as a rendered HTML panel."""
        return _render_goal_panel(xray_utils.goal_from_events(state.current_events))

    def get_agent_action_md() -> str:
        """Return the selected group's dispatched action(s) as an HTML panel."""
        if state.current_events is None or state.selected_group() is None:
            body = "<em>No event selected</em>"
        else:
            body = xray_utils.render_group_action_html(state.current_events, state.selected_group())
        return (
            '<div class="info-panel" style="background:#f0fdf4; border-color:#bbf7d0;">'
            '<div class="info-panel-title" style="background:#dcfce7; color:#15803d;">🤖 Action</div>'
            f'<div class="info-panel-body">{body}</div>'
            "</div>"
        )

    def get_agent_reasoning_md() -> str:
        """Return the selected group's LLM reasoning as a panel (beside Action)."""
        if state.current_events is None or state.selected_group() is None:
            body = "<em>No event selected</em>"
        else:
            body = xray_utils.render_group_reasoning_html(state.current_events, state.selected_group())
        return (
            '<div class="info-panel" style="background:#eff6ff; border-color:#bfdbfe;">'
            '<div class="info-panel-title" style="background:#dbeafe; color:#1d4ed8;">🧠 Reasoning</div>'
            f'<div class="info-panel-body">{body}</div>'
            "</div>"
        )

    # ------------------------------------------------------------------
    # Lazy tab render handlers (only run when their tab is active).
    # Each reads state via closure and takes no arguments.
    # ------------------------------------------------------------------

    def _render_observation() -> tuple[Any, str]:
        """Observation tab: screenshots (gallery) + text contents for the
        selected group's observation(s). Parallel siblings are stacked.

        The gallery is hidden entirely when the group has no screenshots (most
        non-browser tasks) so it doesn't render an empty placeholder."""
        group = state.selected_group()
        if group is None or state.current_events is None:
            return gr.update(value=[], visible=False), "<em>No event selected.</em>"
        images, html = xray_utils.render_group_observation_html(state.current_events, group)
        return gr.update(value=images, visible=bool(images)), html

    def _render_chat() -> str:
        """Chat tab: the selected group's LLM call (prompt + response + tokens)."""
        group = state.selected_group()
        if group is None or state.current_events is None:
            return "<em>No event selected.</em>"
        return xray_utils.render_group_chat_html(state.current_events, group)

    def _render_evaluation() -> str:
        group = state.selected_group()
        if group is None or state.current_events is None:
            return "No event selected."
        return xray_utils.render_group_evaluation_md(state.current_events, group)

    def _render_error() -> str:
        group = state.selected_group()
        if group is None or state.current_events is None:
            return "No event selected."
        return xray_utils.render_group_error_md(state.current_events, group)

    def _render_logs() -> str:
        traj = state.current_trajectory
        storage = state.current_storage()
        log_content = storage.load_logs(traj.id) if storage and traj else ""
        return xray_utils.get_logs_tab_markdown(traj, log_content)

    def _render_retries() -> str:
        traj = state.current_trajectory
        storage = state.current_storage()
        if not traj or not storage:
            return "No trajectory selected."
        if not isinstance(storage, FileStorage):
            return "Retry history is only available for filesystem-backed experiments."
        ep_dir = storage._episode_dir(traj.id)
        history = xray_utils.load_retry_history(ep_dir)
        return xray_utils.render_retry_history_md(history, traj)

    def _render_debug() -> str:
        """Debug tab: raw JSON dump of every event in the selected group."""
        group = state.selected_group()
        if group is None or state.current_events is None:
            return "No event selected"
        return xray_utils.render_group_debug_json(state.current_events, group)

    # ------------------------------------------------------------------
    # Experiment-level analysis tabs (lazy, rendered on tab select)
    # ------------------------------------------------------------------

    def _render_constants_variables() -> tuple[pd.DataFrame, pd.DataFrame]:
        # Collect one (agent_name, config_dict) pair per loaded storage.
        # Agent name is taken from the first trajectory belonging to that storage so it
        # matches exactly what is displayed in the Agents table (including timestamp tag).
        storage_to_agent: dict[int, str] = {}
        for traj, storage in zip(state.trajectories, state._traj_storages):
            sid = id(storage)
            if sid not in storage_to_agent:
                storage_to_agent[sid] = traj.metadata.get("agent_name", "unknown")

        agents: list[tuple[str, dict]] = []
        for storage in state._storages:
            sid = id(storage)
            agent_cfg_json, _ = state._storage_configs.get(sid, (None, None))
            if not agent_cfg_json:
                continue
            try:
                cfg = json.loads(agent_cfg_json)
            except Exception:
                continue
            agents.append((storage_to_agent.get(sid, "unknown"), cfg))

        if not agents:
            return pd.DataFrame(columns=["parameter", "value"]), pd.DataFrame(columns=["parameter"])

        df = inspect_results.agent_configs_to_df(agents)
        if df is None:
            return pd.DataFrame(columns=["parameter", "value"]), pd.DataFrame(columns=["parameter"])
        return inspect_results.format_agent_comparison(df)

    def _render_error_report() -> str:
        if not state.trajectories:
            return "No trajectories loaded."
        df = inspect_results.trajectories_to_df(state.trajectories)
        if df is None:
            return "No data."
        return inspect_results.error_report(df)

    # ------------------------------------------------------------------
    # Tab activation helpers — no-arg named functions avoid lambda warnings.
    # Gradio tab.select fires with no extra inputs, so these take no args.
    # ------------------------------------------------------------------

    def _activate_observation() -> str:
        return "Observation"

    def _activate_chat() -> str:
        return "Chat"

    def _activate_evaluation() -> str:
        return "Evaluation"

    def _activate_error() -> str:
        return "Error"

    def _activate_logs() -> str:
        return "Logs"

    def _activate_retries() -> str:
        return "Retries"

    def _activate_debug() -> str:
        return "Debug"

    # ------------------------------------------------------------------
    # Build the Gradio UI
    # ------------------------------------------------------------------

    with gr.Blocks(theme=gr.themes.Soft(), css=_CSS, js=_INIT_JS) as demo:  # type: ignore[attr-defined]
        active_tab = gr.State(value="Chat")
        step_id = gr.State(value=StepId())

        with gr.Tabs(selected="experiments_tab"):
            with gr.Tab("Help"):
                gr.Markdown(
                    """\
## cube-harness XRay

### Loading experiments
1. Open the **Experiments** tab — check one or more rows to load them simultaneously.
2. Use **↺ Refresh** to pick up new experiments; **🗃 Archive selected** moves them to `_archive/`.
3. When an experiment is running, the viewer polls for new trajectories every second until complete.

### Browsing results
4. Drill down via the **Agents → Trajectories** tabs to select a specific episode.
5. The **Dashboard** tab shows a live progress bar and aggregate stats (reward, tokens, cost).
6. **Agent Config** / **Exp Config** tabs display the configuration used for the experiment.

### Inspecting a trajectory
7. The **event rail** (left) lists every event, coloured by kind
   (🧠 LLM blue · 🖥️ observation green · 🏁 evaluation purple · ⚠️ error red).
   Card height scales with the event's duration; a left stripe marks profiled events.
   - Click a card to select it: a solid border marks the active event, a dashed
     border marks its group-mates (the LLM call, observation(s), reward, and error
     that belong to the same logical step).
8. The **🤖 Action** panel shows the action(s) the selected group dispatched.
9. **Navigate events** with the ◀ / ▶ buttons or **Shift + ← / →** arrow keys.

### Tabs (lazy — only the active tab re-renders on selection change; show the selected group)
- **Chat**: the group's full LLM prompt + response + token usage.
- **Observation**: screenshot(s) + text contents; parallel siblings stacked.
- **AXTree**: raw accessibility tree of the group's observation.
- **Evaluation**: per-step / terminal reward and info.
- **Error**: any LLM / tool / agent error in the group.
- **Logs**: full episode log file (all logger output from the run).
- **Debug**: raw JSON for every event in the group.

### Status icons

| Icon | Meaning |
|------|---------|
| ✓ | Completed — success, fail, or max-steps (all terminal outcomes) |
| ▶️ | Running — episode in progress |
| 🕐 | Queued — not yet started |
| 🎬 | Max steps reached (shown in Trajectories tab) |
| ⛔ | Failed — episode errored |
| 👻 | Stale — no activity for too long |
| 🚫 | Cancelled |
| ✕ | System error — crashed before trajectory was written |
""",
                    elem_classes="help-content",
                )
            with gr.Tab("Experiments", id="experiments_tab"):
                # One toolbar row: directory controls on the left, the action
                # clusters (Archive 🤖✓ · Registry EEE 🤖✓) pushed to the right by
                # the growing directory label. Each 🤖✓ auto-selects the rows its
                # action applies to; the user reviews, then clicks Registry or EEE.
                # Tooltips are set in _INIT_JS.
                with gr.Row(elem_classes="xray-exp-toolbar"):
                    exp_browse_btn = gr.Button(
                        "📁 Browse…", scale=0, size="sm", variant="secondary", elem_id="exp_browse_btn"
                    )
                    exp_refresh_btn = gr.Button("↺", scale=0, size="sm", elem_id="exp_refresh_btn", min_width=0)
                    results_dir_md = gr.Markdown(f"📂 `{state.results_dir}`", elem_id="exp_dir_label")
                    exp_archive_btn = gr.Button(
                        "🗃 Archive", scale=0, size="sm", variant="secondary", elem_id="exp_archive_btn"
                    )
                    exp_pick_archivable_btn = gr.Button(
                        "🤖✓", scale=0, size="sm", elem_id="exp_pick_archivable_btn", min_width=0
                    )
                    exp_submit_registry_btn = gr.Button(
                        "⬆️ Registry", scale=0, size="sm", variant="primary", elem_id="exp_submit_registry_btn"
                    )
                    exp_submit_eee_btn = gr.Button(
                        "⬆️ EEE", scale=0, size="sm", variant="primary", elem_id="exp_submit_eee_btn"
                    )
                    exp_pick_submittable_btn = gr.Button(
                        "🤖✓", scale=0, size="sm", elem_id="exp_pick_submittable_btn", min_width=0
                    )
                exp_action_status = gr.Markdown("", visible=False)
                exp_table = gr.DataFrame(
                    headers=[
                        "",
                        "experiment",
                        "date",
                        "agent",
                        "model",
                        "benchmark",
                        "status",
                        "avg_reward",
                        "eligibility",
                    ],
                    datatype=["bool", "str", "str", "str", "str", "str", "html", "str", "html"],
                    col_count=(9, "fixed"),
                    interactive=True,
                    static_columns=[1, 2, 3, 4, 5, 6, 7, 8],
                    max_height=260,
                    show_label=False,
                    elem_id="exp_table",
                )
            with gr.Tab("Dashboard"):
                progress_bar = gr.HTML("")
                experiment_stats = gr.Markdown("")
            with gr.Tab("Agents") as agents_tab:
                agent_table = gr.DataFrame(
                    headers=["agent_name", "avg_reward", "status", "total_cost"],
                    datatype="html",
                    max_height=260,
                    show_label=False,
                    interactive=False,
                    elem_id="agent_table",
                )
            with gr.Tab("Trajectories") as trajs_tab:
                traj_table = gr.DataFrame(
                    datatype="html",
                    max_height=260,
                    show_label=False,
                    interactive=False,
                    elem_id="traj_table",
                )
            with gr.Tab("Agent Config"):
                agent_config_code = gr.Code(language="json", show_label=False)
            with gr.Tab("Exp Config"):
                exp_config_code = gr.Code(language="json", show_label=False)
            with gr.Tab("Constants & Variables") as cv_tab:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Constants** (identical across all selected experiments)")
                        cv_const_table = gr.DataFrame(
                            headers=["parameter", "value"],
                            max_height=400,
                            show_label=False,
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("**Variables** (differ between agents — one column per agent)")
                        cv_var_table = gr.DataFrame(
                            max_height=400,
                            show_label=False,
                            interactive=False,
                        )
            with gr.Tab("Error Report") as err_report_tab:
                err_report_md = gr.Markdown()

        # Timer: ticks every 1s to bulk-load stubs and then live-poll for new/changed trajectories.
        # Starts inactive; activated on experiment select; deactivates when experiment is complete.
        bg_timer = gr.Timer(value=1.0, active=False)

        # Header: episode identity + structured stats on the left, the task
        # goal beside it on the right (instead of two stacked full-width bars).
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=300, variant="panel", elem_classes="compact-header"):
                header_info = gr.Markdown("**Select a trajectory**")
                stats_display = gr.Markdown("")
            with gr.Column(scale=3):
                task_goal_md = gr.HTML(value="")

        # Hidden Number the card rail writes its clicked event index into.
        with gr.Row(visible=True, elem_id="timeline_click_input"):
            timeline_click_input = gr.Number(show_label=False, container=False)

        # Left: the vertical event-card rail (navigation + profiler).
        # Right: the grouped detail tabs for the selected event's group.
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=240):
                with gr.Row(elem_classes="xray-nav-row"):
                    first_btn = gr.Button("⤒", size="sm", elem_id="xray_first_btn", min_width=0, scale=0)
                    prev_btn = gr.Button("◀", size="sm", elem_id="xray_prev_btn", min_width=0, scale=0)
                    next_btn = gr.Button("▶", size="sm", elem_id="xray_next_btn", min_width=0, scale=0)
                    last_btn = gr.Button("⤓", size="sm", elem_id="xray_last_btn", min_width=0, scale=0)
                timeline_html = gr.HTML(elem_id="xray_rail")
            with gr.Column(scale=3):
                # Reasoning (the LLM's thinking) beside the dispatched action.
                with gr.Row(equal_height=True):
                    agent_reasoning_md = gr.HTML(value="")
                    agent_action_md = gr.HTML(value="")
                with gr.Tabs():
                    with gr.Tab("Chat") as chat_tab:
                        chat_act_md = gr.HTML()

                    # Observation folds in the screenshot gallery AND any text
                    # contents (incl. AXTree) — there is no separate AXTree tab.
                    with gr.Tab("Observation") as screenshots_tab:
                        observation_gallery = gr.Gallery(
                            label="Screenshots",
                            show_label=True,
                            columns=2,
                            height=420,
                            object_fit="contain",
                            visible=False,  # shown only when the group has screenshots
                        )
                        observation_text = gr.HTML()

                    with gr.Tab("Evaluation") as evaluation_tab:
                        evaluation_md = gr.Markdown()

                    with gr.Tab("Error") as error_tab:
                        error_md = gr.Markdown()

                    with gr.Tab("Logs") as logs_tab:
                        logs_md = gr.Markdown()

                    with gr.Tab("Retries") as retries_tab:
                        retries_md = gr.Markdown()

                    with gr.Tab("Debug") as debug_tab:
                        debug_code = gr.Code(language="json", show_label=False)

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------

        def _to_exp_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
            return [
                [
                    r["selected"],
                    r["experiment"],
                    r["date"],
                    r["agent"],
                    r["model"],
                    r["benchmark"],
                    r["status"],
                    r.get("avg_reward", "—"),
                    r.get("eligibility", "—"),
                ]
                for r in rows
            ]

        def _exp_table_rows(auto_select_first: bool = False) -> list[list[Any]]:
            rows = xray_utils.get_experiments_table_rows(state.results_dir)
            if auto_select_first and rows:
                rows[0]["selected"] = True
            return _to_exp_table(rows)

        def _exp_table_value() -> list[list[Any]]:
            return _exp_table_rows(auto_select_first=False)

        def _select_rows(predicate: Callable[[dict[str, Any], Path], bool], label: str) -> tuple[list[list[Any]], Any]:
            """Tick rows for which ``predicate(row, exp_dir)`` is true. Routes
            through the same cached `get_experiments_table_rows` as Refresh (status +
            ghost heartbeat + eligibility, all cached), so it is as fast as a refresh."""
            rows = xray_utils.get_experiments_table_rows(state.results_dir)
            n = 0
            for r in rows:
                hit = predicate(r, state.results_dir / r["experiment"])
                r["selected"] = hit
                n += int(hit)
            msg = f"🎯 Selected **{n}** {label} experiment(s). Review the ticks, then click the action button."
            return _to_exp_table(rows), gr.update(value=msg, visible=True)

        def on_pick_archivable() -> tuple[list[list[Any]], Any]:
            """Auto-tick non-keepers for Archive: broken runs, runs recorded as
            rejected (e.g. all-ghost), and explicit debug runs (is_official=False).
            The user reviews the ticks before archiving."""
            return _select_rows(
                lambda r, d: xray_utils.is_archivable(d, r.get("_category", "broken"), r.get("_is_official")),
                "broken / rejected / debug",
            )

        def on_pick_submittable() -> tuple[list[list[Any]], Any]:
            """Auto-tick submittable experiments that aren't already submitted or
            mid-submission (reads submissions.json fresh, so a just-submitted run
            is not re-ticked even if its cached category lags)."""
            return _select_rows(
                lambda r, d: xray_utils.is_submittable_pick(d, r.get("_category", "broken")), "submittable"
            )

        def _selected_exp_dirs(table: Any) -> list[Path]:
            """Experiment dirs whose checkbox is ticked in the current table value."""
            records = table.values.tolist() if hasattr(table, "values") else (table or [])
            return [state.results_dir / row[1] for row in records if row and bool(row[0])]

        def _run_submitter(script: str, exp_dir: Path, extra: list[str]) -> tuple[bool, str]:
            """Invoke a submit script for one experiment; return (ok, last-meaningful-line).

            On failure the tail is taken from stderr (where the traceback /
            CalledProcessError lands) so the recorded failure reason is the actual
            error, not the last incidental stdout line."""
            cmd = [sys.executable, str(Path(__file__).resolve().parents[3] / "scripts" / script), str(exp_dir), *extra]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            ok = proc.returncode == 0
            stream = proc.stdout if ok else (proc.stderr or proc.stdout)
            tail = next((ln for ln in reversed(stream.strip().splitlines()) if ln.strip()), "")
            return ok, tail

        def on_submit(table: Any, destination: str) -> tuple[Any, ...]:
            """Submit the checked experiments to EEE or the cube-registry journal.

            Outputs: [exp_table, *_hierarchy_outputs, exp_action_status]. After
            submitting, the just-submitted rows are no longer ticked, so we re-
            select the first experiment and rebuild the detail panel — otherwise
            the table would show no selection while the panel still displays the
            previous experiment."""
            dirs = _selected_exp_dirs(table)
            if not dirs:
                skip_hierarchy = tuple(gr.skip() for _ in _hierarchy_outputs)
                return (
                    _exp_table_value(),
                    *skip_hierarchy,
                    gr.update(value="Nothing selected to submit.", visible=True),
                )
            if destination == "eee":
                # Clicking Submit → EEE is an explicit publish: actually open the
                # HF-dataset PR (the CLI defaults to --no-upload for safety).
                script, extra = "submit_to_eee.py", ["--upload"]
            else:
                script, extra = "submit_to_journal.py", ["--auto-pr", "--i-understand-this-is-not-a-leaderboard"]
            dest_key = "eee" if destination == "eee" else "journal"
            lines = [f"### Submit → {destination.upper()} ({len(dirs)} experiment(s))"]
            for d in dirs:
                # Mark in-progress so the row reads "submitting…" and the auto-
                # selector won't re-tick it. The submitter writes `submitted` on
                # success (overwriting pending); we record `failed` otherwise so
                # the failure persists in submissions.json instead of vanishing.
                submissions.record_pending(d, dest_key)
                ok, tail = _run_submitter(script, d, extra)
                if not ok:
                    submissions.record_failed(d, dest_key, reason=tail or "submission failed")
                lines.append(f"- {'✅' if ok else '❌'} `{d.name}` — {tail or ('done' if ok else 'failed')}")
            status = gr.update(value="\n".join(lines), visible=True)
            # Re-select the first experiment so the table + detail panel stay in sync.
            rows = xray_utils.get_experiments_table_rows(state.results_dir)
            if not rows:
                state._selected_exp_names = []
                state.trajectories = []
                state.selected_agent_key = None
                empty_hierarchy = (
                    "",
                    None,
                    None,
                    StepId(),
                    gr.Tab(label="Agents (0)"),
                    gr.Tab(label="Trajectories (0)"),
                    "",
                    "",
                    gr.Timer(active=False),
                )
                return (_to_exp_table(rows), *empty_hierarchy, status)
            rows[0]["selected"] = True
            first = rows[0]["experiment"]
            state._selected_exp_names = [first]
            state.load_experiments([state.results_dir / first])
            return (_to_exp_table(rows), *_load_and_build_hierarchy(), gr.Timer(active=state.should_poll()), status)

        def on_browse_dir() -> tuple[list[list[Any]], str]:
            """Open a native folder picker; on choice, switch the results dir and
            reload the experiments table (the table .change cascade clears the
            current selection/hierarchy). No-op if the user cancels."""
            picked = xray_utils.pick_directory(state.results_dir)
            if picked is not None:
                state.results_dir = picked
            return _exp_table_value(), f"📂 `{state.results_dir}`"

        _hierarchy_outputs = [
            experiment_stats,
            agent_table,
            traj_table,
            step_id,
            agents_tab,
            trajs_tab,
            agent_config_code,
            exp_config_code,
            bg_timer,
        ]

        exp_table.change(fn=on_experiments_change, inputs=exp_table, outputs=_hierarchy_outputs)
        exp_browse_btn.click(fn=on_browse_dir, outputs=[exp_table, results_dir_md])
        exp_refresh_btn.click(fn=_exp_table_value, outputs=exp_table)
        exp_pick_archivable_btn.click(fn=on_pick_archivable, outputs=[exp_table, exp_action_status])
        exp_pick_submittable_btn.click(fn=on_pick_submittable, outputs=[exp_table, exp_action_status])
        exp_submit_registry_btn.click(
            fn=lambda t: on_submit(t, "journal"),
            inputs=exp_table,
            outputs=[exp_table, *_hierarchy_outputs, exp_action_status],
        )
        exp_submit_eee_btn.click(
            fn=lambda t: on_submit(t, "eee"),
            inputs=exp_table,
            outputs=[exp_table, *_hierarchy_outputs, exp_action_status],
        )
        exp_archive_btn.click(fn=on_archive_selected, outputs=[exp_table, *_hierarchy_outputs, exp_action_status])

        bg_timer.tick(
            fn=on_bg_load_tick,
            outputs=[
                experiment_stats,
                agent_table,
                traj_table,
                progress_bar,
                bg_timer,
                agents_tab,
                trajs_tab,
            ],
        )

        agent_table.select(
            fn=on_select_agent,
            inputs=agent_table,
            outputs=[
                agent_table,
                traj_table,
                step_id,
                agents_tab,
                trajs_tab,
                agent_config_code,
                exp_config_code,
            ],
        )
        traj_table.select(fn=on_select_trajectory, inputs=traj_table, outputs=[traj_table, step_id])

        # Timeline click
        timeline_click_input.change(fn=handle_timeline_click, inputs=timeline_click_input, outputs=step_id)

        # Navigation buttons — handlers read state.step from closure (inputs=[]) so that
        # JS button.click() also works without Gradio losing the gr.State value.
        first_btn.click(fn=navigate_first, inputs=[], outputs=step_id)
        prev_btn.click(fn=navigate_prev, inputs=[], outputs=step_id)
        next_btn.click(fn=navigate_next, inputs=[], outputs=step_id)
        last_btn.click(fn=navigate_last, inputs=[], outputs=step_id)

        # Always-rendered on step change
        step_id.change(fn=get_compact_header_info, outputs=header_info)
        step_id.change(fn=update_timeline, outputs=timeline_html)
        step_id.change(fn=update_trajectory_stats, outputs=stats_display)
        step_id.change(fn=get_task_goal, outputs=task_goal_md)
        step_id.change(fn=get_agent_action_md, outputs=agent_action_md)
        step_id.change(fn=get_agent_reasoning_md, outputs=agent_reasoning_md)

        # Lazy renders on event-selection change (active_tab checked by if_active;
        # step_id is the trigger).
        step_id.change(
            fn=if_active("Observation", 2)(_render_observation),
            inputs=[active_tab, step_id],
            outputs=[observation_gallery, observation_text],
        )
        step_id.change(
            fn=if_active("Chat")(_render_chat),
            inputs=[active_tab, step_id],
            outputs=chat_act_md,
        )
        step_id.change(
            fn=if_active("Evaluation")(_render_evaluation),
            inputs=[active_tab, step_id],
            outputs=evaluation_md,
        )
        step_id.change(
            fn=if_active("Error")(_render_error),
            inputs=[active_tab, step_id],
            outputs=error_md,
        )
        step_id.change(
            fn=if_active("Logs")(_render_logs),
            inputs=[active_tab, step_id],
            outputs=logs_md,
        )
        step_id.change(
            fn=if_active("Retries")(_render_retries),
            inputs=[active_tab, step_id],
            outputs=retries_md,
        )
        step_id.change(
            fn=if_active("Debug")(_render_debug),
            inputs=[active_tab, step_id],
            outputs=debug_code,
        )

        # Tab selection: update active_tab state AND immediately re-render the newly visible tab.
        # Tab .select fires with no extra inputs — handlers take no arguments.
        screenshots_tab.select(fn=_activate_observation, outputs=active_tab)
        screenshots_tab.select(fn=_render_observation, outputs=[observation_gallery, observation_text])

        chat_tab.select(fn=_activate_chat, outputs=active_tab)
        chat_tab.select(fn=_render_chat, outputs=chat_act_md)

        evaluation_tab.select(fn=_activate_evaluation, outputs=active_tab)
        evaluation_tab.select(fn=_render_evaluation, outputs=evaluation_md)

        error_tab.select(fn=_activate_error, outputs=active_tab)
        error_tab.select(fn=_render_error, outputs=error_md)

        logs_tab.select(fn=_activate_logs, outputs=active_tab)
        logs_tab.select(fn=_render_logs, outputs=logs_md)

        retries_tab.select(fn=_activate_retries, outputs=active_tab)
        retries_tab.select(fn=_render_retries, outputs=retries_md)

        debug_tab.select(fn=_activate_debug, outputs=active_tab)
        debug_tab.select(fn=_render_debug, outputs=debug_code)

        cv_tab.select(fn=_render_constants_variables, outputs=[cv_const_table, cv_var_table])
        err_report_tab.select(fn=_render_error_report, outputs=err_report_md)

        def _auto_load_first_experiment() -> tuple:
            rows = xray_utils.get_experiments_table_rows(state.results_dir)
            if not rows:
                return (
                    "",
                    None,
                    None,
                    StepId(),
                    gr.Tab(label="Agents (0)"),
                    gr.Tab(label="Trajectories (0)"),
                    "",
                    "",
                    gr.Timer(active=False),
                )
            state.load_experiments([state.results_dir / rows[0]["experiment"]])
            hierarchy = _load_and_build_hierarchy()
            return (*hierarchy, gr.Timer(active=state.should_poll()))

        # Two independent demo.load calls: one populates the exp table with the
        # first experiment row already checked (so the selection is visible), the
        # other pre-loads that experiment so the viewer is immediately usable.
        demo.load(fn=lambda: _exp_table_rows(auto_select_first=True), outputs=exp_table)
        demo.load(fn=_auto_load_first_experiment, outputs=_hierarchy_outputs)

    demo.queue()
    demo.launch(server_port=port, share=share, debug=debug)


def _rows_to_table(rows: list[dict[str, Any]], active_key: str | None = None, key_col: str = "") -> pd.DataFrame:
    """Convert a list of dicts to a Gradio-ready DataFrame.

    Keys starting with '_' are hidden metadata and excluded from displayed columns.
    When active_key and key_col are provided, cells in the matching row are
    wrapped in a highlight span (used with datatype='html' DataFrames).
    """
    display_keys = [k for k in rows[0] if not k.startswith("_")] if rows else []
    if not rows:
        return pd.DataFrame()
    result = []
    for row in rows:
        is_active = active_key is not None and re.sub(r"<[^>]+>", "", str(row.get(key_col, ""))) == str(active_key)
        display_values = [row[k] for k in display_keys]
        if is_active:
            cells = [f'<span style="font-weight:600;color:#1d4ed8">{v}</span>' for v in display_values]
        else:
            cells = [str(v) for v in display_values]
        result.append(cells)
    return pd.DataFrame(result, columns=display_keys)


def main() -> None:
    """CLI entry point for ch-xray."""
    parser = argparse.ArgumentParser(description="cube-harness XRay Experiment Viewer")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(EXP_DIR),
        help="Path to results directory containing experiments",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: auto)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    run_xray(Path(args.results_dir), debug=args.debug, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
