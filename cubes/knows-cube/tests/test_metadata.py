"""Metadata registry tests. No credentials, no network."""

from knows_cube.benchmark import KnowsBenchmarkConfig
from knows_cube.task import KnowsTaskMetadata


def test_registry_has_110_tasks() -> None:
    assert KnowsBenchmarkConfig().num_tasks == 110


def test_no_legacy_alias_and_no_slides_25() -> None:
    """Upstream registers a suffix-less gym alias and a stale slides_25 name; neither is a task."""
    ids = set(KnowsBenchmarkConfig().tasks())
    assert "knows.docs_1_formal_letter" not in ids
    assert not any(".slides_25" in task_id for task_id in ids)


def test_subclass_fields_survive_json_load() -> None:
    """A missing _type discriminator silently downgrades entries to plain TaskMetadata."""
    for tm in KnowsBenchmarkConfig().tasks().values():
        assert isinstance(tm, KnowsTaskMetadata)
        assert tm.task_family_folder
        assert tm.task_id_prefix
        assert tm.workspace_kind in {"docs", "sheets", "slides"}
        assert 1 <= tm.instance_number <= 5


def test_id_is_prefix_plus_instance() -> None:
    for task_id, tm in KnowsBenchmarkConfig().tasks().items():
        assert task_id == f"{tm.task_id_prefix}.{tm.instance_number}"


def test_prefix_is_not_derived_from_folder() -> None:
    """9 families diverge from their folder name; 6 are renames no transform recovers.

    Guards against a "cleanup" that starts computing the prefix from the folder.
    """
    tasks = KnowsBenchmarkConfig().tasks()
    divergent = {
        tm.task_family_folder for tm in tasks.values() if tm.task_id_prefix != f"knows.{tm.task_family_folder}"
    }
    assert len(divergent) == 9


def test_known_bad_task_is_flagged() -> None:
    """docs_37 instance 4 has an unparseable upstream evaluator; it must not look healthy."""
    tm = KnowsBenchmarkConfig().tasks()["knows.docs_37_reference_list.4"]
    assert tm.is_gradeable is False
    assert tm.known_issues


def test_kind_counts() -> None:
    """22 families = 5 docs + 9 sheets + 8 slides, 5 instances each."""
    kinds = [tm.workspace_kind for tm in KnowsBenchmarkConfig().tasks().values()]
    assert (kinds.count("docs"), kinds.count("sheets"), kinds.count("slides")) == (25, 45, 40)


def test_every_family_has_five_instances() -> None:
    counts: dict[str, int] = {}
    for tm in KnowsBenchmarkConfig().tasks().values():
        counts[tm.task_family_folder] = counts.get(tm.task_family_folder, 0) + 1
    assert len(counts) == 22
    assert set(counts.values()) == {5}


def test_named_subsets_resolve() -> None:
    for name in ("docs", "sheets", "slides", "gradeable"):
        assert KnowsBenchmarkConfig().named_subset(name).num_tasks > 0
    assert KnowsBenchmarkConfig().named_subset("gradeable").num_tasks == 109
