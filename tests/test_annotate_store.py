"""The annotation pool's concurrency and ordering guarantees.

The portal hands episodes to several annotators at once, so the store is the one
place where a bug silently corrupts the study: a double-served episode inflates
apparent agreement, a lost lease drops an episode from the sample, and ordering
that never overlaps yields no inter-annotator kappa at all. Each of those is
pinned here.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from cube_harness.analyze.annotate.store import AnnotationStore


def _sample(n: int) -> list[dict[str, Any]]:
    return [
        {
            "uid": f"E{i:03d}",
            "experiment": "exp",
            "trajectory_id": f"task_{i}_ep0",
            "episode_dir": f"/results/exp/episodes/task_{i}_ep0",
            "modality": "Web",
            "benchmark": "miniwob",
            "task_id": f"task_{i}",
            "task_description": "do the thing",
            "score": 0.0,
        }
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    s = AnnotationStore(tmp_path / "study", target_coverage=2)
    s.seed(_sample(6))
    return s


def _label(store: AnnotationStore, uid: str, annotator: str) -> None:
    store.submit(uid, annotator, outcome="failure", blame="tool_failure", confidence=4)


def test_seed_is_idempotent(store: AnnotationStore) -> None:
    _label(store, "E000", "alice")
    store.seed(_sample(6))
    assert store.progress().total == 6
    # Re-seeding must not wipe work already done.
    assert store.export_labels()["alice"]["E000"]["blame"] == "tool_failure"


def test_seed_extends_the_pool(store: AnnotationStore) -> None:
    store.seed(_sample(9))
    assert store.progress().total == 9


def test_concurrent_claims_never_collide(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path / "study", target_coverage=2)
    store.seed(_sample(8))
    names = [f"ann{i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        claimed = list(pool.map(store.claim_next, names))
    uids = [e.uid for e in claimed if e is not None]
    assert len(uids) == 8, "every annotator should get an episode"
    assert len(set(uids)) == 8, f"two annotators were served the same episode: {uids}"


def test_annotator_is_never_re_served_their_own_work(store: AnnotationStore) -> None:
    seen: set[str] = set()
    while (episode := store.claim_next("alice")) is not None:
        assert episode.uid not in seen
        seen.add(episode.uid)
        _label(store, episode.uid, "alice")
    # Alice exhausts the pool exactly once, never twice.
    assert seen == {f"E{i:03d}" for i in range(6)}


def test_partially_covered_episodes_are_served_first(store: AnnotationStore) -> None:
    """Ordering must make kappa pairs materialise, not maximise disjointness.

    Alice labels three episodes. Bob must then be handed those same three
    first -- each of his labels completes an inter-annotator pair -- rather than
    the three untouched ones, which would leave the study with zero overlap.
    """
    alice_uids = []
    for _ in range(3):
        episode = store.claim_next("alice")
        assert episode is not None
        alice_uids.append(episode.uid)
        _label(store, episode.uid, "alice")

    bob_uids = []
    for _ in range(3):
        episode = store.claim_next("bob")
        assert episode is not None
        bob_uids.append(episode.uid)
        _label(store, episode.uid, "bob")

    assert set(bob_uids) == set(alice_uids)


def test_episodes_at_target_are_served_last(store: AnnotationStore) -> None:
    """A third label on E000 only happens once nothing below target is left."""
    _label(store, "E000", "alice")
    _label(store, "E000", "bob")
    episode = store.claim_next("carol")
    assert episode is not None
    assert episode.uid != "E000"


def test_live_lease_hides_an_episode_from_others(store: AnnotationStore) -> None:
    first = store.claim_next("alice")
    assert first is not None
    other = store.claim_next("bob")
    assert other is not None and other.uid != first.uid


def test_expired_lease_returns_to_the_pool(tmp_path: Path) -> None:
    """A closed browser tab must not cost the study an episode."""
    store = AnnotationStore(tmp_path / "study", lease_ttl=60.0, target_coverage=2)
    store.seed(_sample(1))
    abandoned = store.claim_next("alice", now=1000.0)
    assert abandoned is not None
    # Bob is locked out while the lease is live...
    assert store.claim_next("bob", now=1030.0) is None
    # ...and gets the episode once it lapses.
    reissued = store.claim_next("bob", now=1000.0 + 61.0)
    assert reissued is not None and reissued.uid == abandoned.uid


def test_skip_releases_immediately(store: AnnotationStore) -> None:
    """Skip must not wait for the TTL — the episode is free for anyone at once."""
    episode = store.claim_next("alice")
    assert episode is not None
    store.release(episode.uid, "alice")
    reissued = store.claim_next("bob")
    assert reissued is not None and reissued.uid == episode.uid


def test_resubmit_updates_rather_than_duplicates(store: AnnotationStore) -> None:
    store.submit("E000", "alice", outcome="failure", blame="tool_failure", confidence=2, notes="first pass")
    store.submit("E000", "alice", outcome="almost", blame="eval_brittle", confidence=5, notes="looked again")
    labels = store.export_labels()
    assert list(labels["alice"]) == ["E000"]
    assert labels["alice"]["E000"] == {
        "outcome": "almost",
        "blame": "eval_brittle",
        "confidence": 5,
        "notes": "looked again",
    }
    assert store.progress().per_annotator == {"alice": 1}


def test_submit_drops_the_lease(store: AnnotationStore) -> None:
    episode = store.claim_next("alice")
    assert episode is not None
    _label(store, episode.uid, "alice")
    # With the lease gone, bob can be served the same episode for the second label.
    for _ in range(6):
        if (candidate := store.claim_next("bob")) is None:
            break
        if candidate.uid == episode.uid:
            return
        _label(store, candidate.uid, "bob")
    pytest.fail("submitting must release the lease so the episode can reach target coverage")


def test_progress_reports_coverage_histogram(store: AnnotationStore) -> None:
    _label(store, "E000", "alice")
    _label(store, "E000", "bob")
    _label(store, "E001", "alice")
    p = store.progress()
    assert p.total == 6
    assert p.per_annotator == {"alice": 2, "bob": 1}
    assert p.coverage == {0: 4, 1: 1, 2: 1}
    assert p.at_target == 1


def test_export_matches_the_shape_score_parses(store: AnnotationStore) -> None:
    """`judge_validation.py score` indexes `[annotator][uid]["blame"]`."""
    _label(store, "E000", "alice")
    store.submit("E001", "bob", outcome="almost", blame="none", confidence=1, notes="")
    labels = store.export_labels()
    assert sorted(labels) == ["alice", "bob"]
    assert labels["alice"]["E000"]["blame"] == "tool_failure"
    assert labels["bob"]["E001"]["outcome"] == "almost"
    for per_uid in labels.values():
        for record in per_uid.values():
            assert set(record) == {"outcome", "blame", "confidence", "notes"}


def test_dry_pool_returns_none(store: AnnotationStore) -> None:
    while (episode := store.claim_next("alice")) is not None:
        _label(store, episode.uid, "alice")
    assert store.claim_next("alice") is None


def test_store_survives_reopen(tmp_path: Path) -> None:
    """The SQLite file outlives the process, so a dead share link is recoverable."""
    first = AnnotationStore(tmp_path / "study", target_coverage=2)
    first.seed(_sample(3))
    _label(first, "E000", "alice")
    second = AnnotationStore(tmp_path / "study", target_coverage=2)
    assert second.export_labels()["alice"]["E000"]["confidence"] == 4
    assert second.progress().total == 3
