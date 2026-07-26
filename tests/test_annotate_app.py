"""The portal's handler wiring, and the flow through it.

Gradio only checks that a handler returns as many values as it declared outputs
*when the handler fires*, so an arity mismatch in a rarely-taken branch (pool
exhausted, session lost, missing radio selection) is a runtime error no other
test would reach. Every registered handler is introspected off `demo.fns` and
called directly -- no server, no browser.

Results are read back by the output component's `elem_id`, never by position, so
reordering `page_outputs` cannot silently invert what these tests assert.
"""

from pathlib import Path
from typing import Any

import pytest

from cube_harness.analyze.annotate.app import build_app, progress_line
from cube_harness.analyze.annotate.panel import EpisodeViewer
from cube_harness.analyze.annotate.store import AnnotationStore
from tests.xray_fixture import DEMO_TRAJ_ID, build_demo_experiment

HANDLERS = {"on_load", "on_start", "on_submit", "on_skip", "_navigate", "on_card_click"}


class _Request:
    """Stand-in for `gr.Request` — the handlers only read `query_params`."""

    def __init__(self, **params: str) -> None:
        self.query_params = params


class Portal:
    """A built app whose handlers can be called and whose results are keyed.

    `call(name, *args)` runs a handler and returns `(values, by_id)` where
    `by_id` maps an output component's `elem_id` to the value that handler
    assigned it. The three `gr.State`s have no `elem_id`, so they are exposed
    under the stable keys `annotator`, `uid` and `selected` by position within
    the state prefix -- the one ordering assumption, and it is asserted.
    """

    STATE_KEYS = ("annotator", "uid", "selected")

    def __init__(self, store: AnnotationStore, **kwargs: Any) -> None:
        self.store = store
        self.demo = build_app(store, EpisodeViewer(), **kwargs)
        self._fns: dict[str, Any] = {}
        for fn in self.demo.fns.values():
            name = getattr(fn.fn, "__name__", "?")
            # The four rail-nav buttons share one lambda shape; keep the first.
            self._fns.setdefault("_navigate" if name == "<lambda>" else name, fn)

    @property
    def names(self) -> set[str]:
        return set(self._fns)

    def n_outputs(self, name: str) -> int:
        return len(self._fns[name].outputs)

    def call(self, name: str, *args: Any) -> tuple[list[Any], dict[str, Any]]:
        fn = self._fns[name]
        values = list(fn.fn(*args))
        assert len(values) == len(fn.outputs), (
            f"{name} returned {len(values)} values for {len(fn.outputs)} declared outputs"
        )
        by_id: dict[str, Any] = {}
        for component, value in zip(fn.outputs, values, strict=True):
            elem_id = getattr(component, "elem_id", None)
            if elem_id:
                by_id[elem_id] = value
        for i, key in enumerate(self.STATE_KEYS):
            if i < len(values) and getattr(fn.outputs[i], "elem_id", None) is None:
                by_id[key] = values[i]
        return values, by_id


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    exp_dir = tmp_path / "results" / "exp"
    build_demo_experiment(exp_dir)
    store = AnnotationStore(tmp_path / "study", target_coverage=2)
    store.seed(
        [
            {
                "uid": f"E{i:03d}",
                "experiment": "exp",
                "trajectory_id": DEMO_TRAJ_ID,
                "episode_dir": str(exp_dir / "episodes" / DEMO_TRAJ_ID),
                "modality": "Web",
                "benchmark": "miniwob",
                "task_id": "demo_task_0",
                "score": 0.0,
            }
            for i in range(3)
        ]
    )
    return store


@pytest.fixture
def portal(store: AnnotationStore) -> Portal:
    return Portal(store)


def test_every_handler_is_registered(portal: Portal) -> None:
    assert portal.names == HANDLERS


def test_every_handler_returns_its_declared_output_count(portal: Portal) -> None:
    """`Portal.call` asserts the arity; this exercises each handler's happy path."""
    calls: dict[str, tuple[Any, ...]] = {
        "on_load": ("", _Request()),
        "on_start": ("alice", "", ""),
        "on_submit": ("alice", "E000", 1, "failure", "tool_failure", 4, "note"),
        "on_skip": ("alice", "E000"),
        "_navigate": ("E000", 1),
        "on_card_click": (3, "E000", 1),
    }
    assert set(calls) == HANDLERS
    for name, args in calls.items():
        portal.call(name, *args)


def test_error_branches_keep_their_arity(portal: Portal) -> None:
    """The branches a real annotator actually hits by accident."""
    cases = [
        ("on_load", ("", _Request()), "no name and no query param → landing"),
        ("on_load", ("alice", _Request()), "remembered name → straight to work"),
        ("on_load", ("", _Request(annotator="bob")), "personal link"),
        ("on_start", ("", "", ""), "empty name"),
        ("on_start", ("   ", "", ""), "whitespace-only name"),
        ("on_submit", ("alice", "E000", 1, None, None, 3, ""), "nothing selected"),
        ("on_submit", ("", "", 0, "failure", "tool_failure", 3, ""), "session lost"),
        ("on_skip", ("alice", ""), "skip with no episode held"),
        ("_navigate", ("", 0), "nav with no episode held"),
        ("on_card_click", (None, "E000", 0), "click with no index"),
    ]
    for name, args, _why in cases:
        portal.call(name, *args)


def test_personal_link_skips_the_prompt(portal: Portal) -> None:
    _, by_id = portal.call("on_load", "", _Request(annotator="bob"))
    assert by_id["annotator"] == "bob"
    assert by_id["uid"], "an episode is served straight away"
    assert by_id["ann_landing"]["visible"] is False
    assert by_id["ann_work"]["visible"] is True


def test_load_with_passphrase_always_shows_the_landing_gate(store: AnnotationStore) -> None:
    """A remembered name is not evidence of access on an open share tunnel."""
    portal = Portal(store, passphrase="hunter2")
    _, by_id = portal.call("on_load", "alice", _Request(annotator="alice"))
    assert by_id["ann_landing"]["visible"] is True
    assert by_id["annotator"] == ""
    assert by_id["uid"] == "", "no episode is claimed before the gate is passed"
    # The name still prefills, so a personal link only costs the passphrase.
    assert by_id["ann_name"]["value"] == "alice"


def test_wrong_passphrase_serves_nothing(store: AnnotationStore) -> None:
    portal = Portal(store, passphrase="hunter2")
    _, by_id = portal.call("on_start", "alice", "wrong", "alice")
    assert by_id["annotator"] == "", "no identity is established"
    assert by_id["ann_landing"]["visible"] is True
    assert "passphrase" in by_id["ann_status"].lower()
    assert by_id["ann_name"]["value"] == "alice", "the typed name is not thrown away"
    assert store.progress().per_annotator == {}


def test_right_passphrase_serves_an_episode(store: AnnotationStore) -> None:
    portal = Portal(store, passphrase="hunter2")
    _, by_id = portal.call("on_start", "alice", "hunter2", "")
    assert by_id["annotator"] == "alice"
    assert by_id["uid"]
    assert by_id["ann_work"]["visible"] is True


def test_submit_records_the_label_and_advances(portal: Portal) -> None:
    _, served = portal.call("on_start", "alice", "", "")
    first_uid = served["uid"]
    assert first_uid

    _, after = portal.call("on_submit", "alice", first_uid, 1, "almost", "eval_brittle", 5, "close")
    assert portal.store.export_labels()["alice"][first_uid] == {
        "outcome": "almost",
        "blame": "eval_brittle",
        "confidence": 5,
        "notes": "close",
    }
    assert after["uid"] and after["uid"] != first_uid, "the next episode is served immediately"
    assert "recorded" in after["ann_status"]
    assert "you: 1" in after["ann_progress"]


def test_incomplete_submit_writes_nothing_and_keeps_the_episode(portal: Portal) -> None:
    _, by_id = portal.call("on_submit", "alice", "E000", 4, None, "tool_failure", 3, "")
    assert portal.store.export_labels() == {}
    assert by_id["uid"] == "E000", "the annotator keeps the trajectory they were reading"
    assert by_id["selected"] == 4, "and their place in it"
    assert "outcome" in by_id["ann_status"]


def test_skip_returns_the_episode_and_serves_another(portal: Portal) -> None:
    _, served = portal.call("on_start", "alice", "", "")
    _, after = portal.call("on_skip", "alice", served["uid"])
    assert portal.store.export_labels() == {}, "skipping records no label"
    assert after["uid"], "another episode is served"
    assert "pool" in after["ann_status"]


def test_exhausted_pool_shows_the_done_banner(portal: Portal) -> None:
    for uid in ("E000", "E001", "E002"):
        portal.store.submit(uid, "alice", outcome="failure", blame="none", confidence=3)
    _, by_id = portal.call("on_start", "alice", "", "")
    assert by_id["uid"] == "", "no episode is held"
    assert by_id["ann_work"]["visible"] is False
    assert by_id["ann_done"]["visible"] is True
    assert "Nothing left" in by_id["ann_done"]["value"]


def test_two_annotators_get_different_episodes(portal: Portal) -> None:
    """The whole point of the shared pool, exercised through the handlers."""
    _, alice = portal.call("on_start", "alice", "", "")
    _, bob = portal.call("on_start", "bob", "", "")
    assert alice["uid"] and bob["uid"]
    assert alice["uid"] != bob["uid"]


def test_navigation_moves_the_selection(portal: Portal) -> None:
    _, served = portal.call("on_start", "alice", "", "")
    uid, start = served["uid"], served["selected"]
    _, forward = portal.call("_navigate", uid, start)
    assert forward["selected"] != start


def test_progress_line_reports_you_and_the_pool(store: AnnotationStore) -> None:
    store.submit("E000", "alice", outcome="failure", blame="none", confidence=3)
    store.submit("E000", "bob", outcome="failure", blame="none", confidence=3)
    line = progress_line(store, "alice")
    assert "you: 1" in line
    assert "1/3" in line and "≥2 labels" in line
    assert "2 annotators" in line
