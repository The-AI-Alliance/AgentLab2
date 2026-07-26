"""The portal's viewer shows the judge's evidence and none of its verdict.

Two claims are tested. First, *evidence parity*: the annotator gets the goal, the
event rail, screenshots and the LLM conversation -- the whole episode, which is
the entire reason the static text packet was replaced. Second, *blinding*: no
rendered surface leaks the taxonomy the annotator is being asked to apply.

No Gradio server is started; the renderers are pure functions, so this follows
`test_xray_ui_integration.py` and drives them directly.
"""

from pathlib import Path

import pytest

from cube_harness.analyze.annotate.panel import EpisodeViewer
from cube_harness.analyze.annotate.store import AnnotationStore, Episode
from cube_harness.eval_log import BlameCategory, Outcome
from tests.xray_fixture import DEMO_TRAJ_ID, build_demo_experiment


@pytest.fixture
def episode(tmp_path: Path) -> Episode:
    """One study episode backed by the synthetic XRay fixture experiment."""
    exp_dir = tmp_path / "results" / "exp"
    build_demo_experiment(exp_dir)
    store = AnnotationStore(tmp_path / "study")
    store.seed(
        [
            {
                "uid": "E000",
                "experiment": "exp",
                "trajectory_id": DEMO_TRAJ_ID,
                "episode_dir": str(exp_dir / "episodes" / DEMO_TRAJ_ID),
                "modality": "Web",
                "benchmark": "miniwob",
                "task_id": "demo_task_0",
                "score": 0.0,
            }
        ]
    )
    result = store.get("E000")
    assert result is not None
    return result


@pytest.fixture
def viewer() -> EpisodeViewer:
    return EpisodeViewer()


def test_episode_loads_through_the_production_path(episode: Episode, viewer: EpisodeViewer) -> None:
    events = viewer.events(episode)
    assert events is not None and len(events) > 0


def test_render_shows_the_full_episode(episode: Episode, viewer: EpisodeViewer) -> None:
    """Evidence parity: goal, rail, reasoning, action, screenshots, LLM detail."""
    render = viewer.render(episode, viewer.initial_index(episode))
    assert "search for shoes" in render.goal, "the task goal must be visible"
    n_events = viewer.n_events(episode)
    assert render.rail.count("xray-event-card") == n_events, "every event gets a clickable card"
    assert "Reasoning" in render.reasoning and "Action" in render.action
    assert render.images, "browser episodes must surface their screenshots"
    assert "gpt-4o-mini" in render.chat, "the LLM call the judge read must be shown"
    assert "E000" in render.header


def test_header_carries_the_blind_uid_not_the_results_path(episode: Episode, viewer: EpisodeViewer) -> None:
    render = viewer.render(episode, 1)
    assert "E000" in render.header
    assert episode.episode_dir not in render.all_text()
    assert "episodes/" not in render.header


def test_nothing_rendered_leaks_the_judges_verdict(episode: Episode, viewer: EpisodeViewer) -> None:
    """Blinding: an annotator must not be able to anchor on the judge's answer.

    Checked over every event, since a leak could hide in one group's panes.
    The needles are strings only the investigator's output uses -- the blame
    categories, the compound outcome values, and the record's field names. Bare
    `success` / `failure` are deliberately excluded: a benchmark's own
    `reward_info` says `{"success": true}`, which is environment evidence the
    judge read too, not the judge's verdict.
    """
    needles = [b.value for b in BlameCategory if b is not BlameCategory.none]
    needles += [o.value for o in Outcome if "_" in o.value]
    needles += ["primary_blame", "findings", "investigator"]
    for index in range(viewer.n_events(episode)):
        text = viewer.render(episode, index).all_text().lower()
        for needle in needles:
            assert needle not in text, f"event {index} leaks {needle!r}"


def test_navigation_walks_group_roots(episode: Episode, viewer: EpisodeViewer) -> None:
    first_step = viewer.step(episode, 5, "first")
    last = viewer.step(episode, 0, "last")
    assert first_step < last
    assert viewer.step(episode, first_step, "next") > first_step
    assert viewer.step(episode, last, "prev") < last
    # `first` lands on the first *step*, so stepping back once more reaches the
    # reset observation (index 0) and then clamps there.
    reset = viewer.step(episode, first_step, "prev")
    assert reset == 0
    assert viewer.step(episode, reset, "prev") == reset
    assert viewer.step(episode, last, "next") == last


def test_selection_is_clamped_to_the_episode(episode: Episode, viewer: EpisodeViewer) -> None:
    render = viewer.render(episode, 10_000)
    assert f"/{viewer.n_events(episode)}" in render.header


def test_unreadable_episode_degrades_instead_of_raising(tmp_path: Path, viewer: EpisodeViewer) -> None:
    """One corrupt episode must not take the portal down mid-study."""
    missing = Episode(
        uid="E999",
        experiment="gone",
        trajectory_id="nope_ep0",
        episode_dir=str(tmp_path / "results" / "gone" / "episodes" / "nope_ep0"),
        modality="Web",
        task_id="nope",
        stratum="Web",
        benchmark="miniwob",
        score=0.0,
    )
    render = viewer.render(missing, 0)
    assert viewer.events(missing) is None
    assert "Skip" in render.goal
    assert render.images == []


def test_events_are_cached_across_renders(episode: Episode, viewer: EpisodeViewer) -> None:
    """Rail clicks re-render constantly; re-decoding each time would crawl."""
    assert viewer.events(episode) is viewer.events(episode)
