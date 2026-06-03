"""Integration test: XRay renders a real event-format trajectory end-to-end.

Loads the synthetic fixture through the production `FileStorage.load_episode`
path (so legacy adaptation, if any, runs in the loader), builds the
`EpisodeEvents` view model, and asserts the rail + grouped detail renderers
produce the right HTML — including the parallel-tool-call siblings and the
LLM data that the pre-rewrite band-aids used to blank. No Gradio server.
"""

import json
from pathlib import Path

from cube_harness.analyze import xray_utils
from cube_harness.analyze.xray_events import KIND_LLM, EpisodeEvents
from cube_harness.core import LLMCallEvent
from cube_harness.storage import FileStorage
from tests.xray_fixture import build_demo_experiment


def _events(tmp_path: Path) -> EpisodeEvents:
    build_demo_experiment(tmp_path / "exp")
    view = FileStorage(tmp_path / "exp").load_episode("demo_task_0_ep0")
    return EpisodeEvents.from_view(view)


def _parallel_llm_index(ep: EpisodeEvents) -> int:
    """Index of the LLM call that dispatched >1 tool call (the parallel turn)."""
    for i in range(len(ep)):
        if isinstance(ep.output(i), LLMCallEvent) and len(ep.child_indices(i)) > 1:
            return i
    raise AssertionError("fixture should contain a parallel LLM turn")


def test_event_rail_renders_every_card(tmp_path: Path) -> None:
    ep = _events(tmp_path)
    html = xray_utils.render_event_rail_html(ep, selected=0)
    # One clickable card per event, all carrying the wiring class.
    assert html.count("xray-event-card") == len(ep)
    # Action names from the fixture show up as card titles.
    for name in ("click", "type", "read_page", "final_step"):
        assert name in html
    # The reset observation is the active card (solid border), not dashed.
    assert "Initial observation" in html


def test_goal_extracted_from_first_observation(tmp_path: Path) -> None:
    ep = _events(tmp_path)
    assert "search for shoes" in xray_utils.goal_from_events(ep)


def test_chat_pane_shows_llm_conversation(tmp_path: Path) -> None:
    # Selecting the parallel turn surfaces its LLM prompt + response (the data
    # the band-aids used to blank).
    ep = _events(tmp_path)
    llm_i = _parallel_llm_index(ep)
    assert ep.cards()[llm_i].kind == KIND_LLM
    chat = xray_utils.render_group_chat_html(ep, ep.group_for(llm_i))
    assert "web agent" in chat  # system prompt
    assert "gpt-4o-mini" in chat  # model config
    assert "parallel" in chat.lower()  # assistant content


def test_observation_pane_stacks_parallel_siblings(tmp_path: Path) -> None:
    ep = _events(tmp_path)
    group = ep.group_for(_parallel_llm_index(ep))
    images, html = xray_utils.render_group_observation_html(ep, group)
    assert len(images) == 2  # two parallel observations, two screenshots
    assert "Sibling 1/2" in html and "Sibling 2/2" in html


def test_axtree_pane_reads_observation_content(tmp_path: Path) -> None:
    ep = _events(tmp_path)
    group = ep.group_for(_parallel_llm_index(ep))
    assert "value=shoes" in xray_utils.render_group_axtree(ep, group)


def test_evaluation_and_error_panes(tmp_path: Path) -> None:
    ep = _events(tmp_path)
    # Terminal eval lives in the last group.
    last_group = ep.group_for(len(ep) - 1)
    eval_md = xray_utils.render_group_evaluation_md(ep, last_group)
    assert "Terminal evaluation" in eval_md and "**Reward:** 1" in eval_md
    # The LLM-error event group surfaces the error.
    err_idx = next(i for i in range(len(ep)) if ep.error(i) is not None and isinstance(ep.output(i), LLMCallEvent))
    assert "RateLimitError" in xray_utils.render_group_error_md(ep, ep.group_for(err_idx))


def test_debug_pane_is_valid_json(tmp_path: Path) -> None:
    ep = _events(tmp_path)
    dump = json.loads(xray_utils.render_group_debug_json(ep, ep.group_for(0)))
    assert isinstance(dump, list) and dump and "output" in dump[0] and "kind" in dump[0]
