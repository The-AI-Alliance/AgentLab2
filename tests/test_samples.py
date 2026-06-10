"""Tests for cube_harness.reproducibility.samples (the per-task bundle)."""

from __future__ import annotations

import gzip
import json
import statistics
from pathlib import Path

import pytest

from cube_harness.eval_log import EvalLog
from cube_harness.reproducibility import samples

# Reuse the record fixtures from the reproducibility test module.
from tests.test_reproducibility import _ep_record, _exp_record


def _save_eval_log(exp_dir: Path, scores: list[float]) -> None:
    exp = _exp_record(n_tasks=len(scores))
    episodes = [_ep_record(f"t{i}", s) for i, s in enumerate(scores)]
    EvalLog(experiment=exp, episodes=episodes).save(exp_dir)


class TestSamplesBundle:
    def test_bundle_is_deterministic(self, tmp_path: Path) -> None:
        _save_eval_log(tmp_path, [1.0, 0.0, 1.0])
        a = samples.build_samples_bundle(tmp_path)
        b = samples.build_samples_bundle(tmp_path)
        assert a == b  # mtime=0 + sorted keys → byte-identical
        assert samples.sha256_hex(a) == samples.sha256_hex(b)

    def test_bundle_is_gzipped_jsonl_one_line_per_episode(self, tmp_path: Path) -> None:
        _save_eval_log(tmp_path, [1.0, 0.0, 0.5])
        raw = gzip.decompress(samples.build_samples_bundle(tmp_path)).decode()
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["sample_id"] == "t0" and first["score"] == 1.0

    def test_roundtrip_iter_samples(self, tmp_path: Path) -> None:
        _save_eval_log(tmp_path, [1.0, 0.0])
        s = samples.iter_samples(samples.build_samples_bundle(tmp_path))
        assert [x["score"] for x in s] == [1.0, 0.0]

    def test_empty_experiment_bundles_cleanly(self, tmp_path: Path) -> None:
        EvalLog(experiment=_exp_record(n_tasks=0), episodes=[]).save(tmp_path)
        bundle = samples.build_samples_bundle(tmp_path)
        assert samples.iter_samples(bundle) == []
        assert samples.aggregate_from_samples([]) == {
            "n_samples": 0,
            "n_scored": 0,
            "avg_score": 0.0,
            "std_err": 0.0,
        }


class TestAggregateFromSamples:
    def test_matches_hand_computed_mean_and_stderr(self, tmp_path: Path) -> None:
        scores = [1.0, 0.0, 1.0, 1.0, 0.0]
        _save_eval_log(tmp_path, scores)
        agg = samples.aggregate_from_samples(samples.iter_samples(samples.build_samples_bundle(tmp_path)))
        assert agg["n_samples"] == 5 and agg["n_scored"] == 5
        assert agg["avg_score"] == round(sum(scores) / len(scores), 6)
        assert agg["std_err"] == round(statistics.stdev(scores) / (len(scores) ** 0.5), 6)

    def test_matches_journal_record(self, tmp_path: Path) -> None:
        # The whole point: the bundle re-derives the summary's headline numbers.
        from cube_harness.reproducibility.journal import build_journal_record  # noqa: PLC0415

        _save_eval_log(tmp_path, [1.0, 0.0, 1.0, 0.5])
        agg = samples.aggregate_from_samples(samples.iter_samples(samples.build_samples_bundle(tmp_path)))
        rec = build_journal_record(tmp_path, submitter="tester")
        assert agg["avg_score"] == rec["results"]["avg_score"]
        assert agg["std_err"] == rec["results"]["std_err"]


class TestJournalSubmission:
    def test_summary_points_at_paired_bundle(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility.journal import build_journal_submission  # noqa: PLC0415

        _save_eval_log(tmp_path, [1.0, 0.0, 1.0])
        sub = build_journal_submission(tmp_path, submitter="tester")
        # Paired filenames share a stem; summary references the bundle by name + hash.
        assert sub.summary_filename.endswith(".json")
        assert sub.bundle_filename == sub.summary_filename[: -len(".json")] + samples.SAMPLES_SUFFIX
        dr = sub.record["detailed_results"]
        assert dr["file"] == sub.bundle_filename
        assert dr["format"] == "jsonl.gz"
        assert dr["n_samples"] == 3
        assert dr["sha256"] == samples.sha256_hex(sub.bundle)

    def test_bundle_re_derives_the_summary(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility.journal import build_journal_submission  # noqa: PLC0415

        _save_eval_log(tmp_path, [1.0, 0.0, 1.0, 1.0])
        sub = build_journal_submission(tmp_path, submitter="tester")
        agg = samples.aggregate_from_samples(samples.iter_samples(sub.bundle))
        assert agg["avg_score"] == sub.record["results"]["avg_score"]

    def test_duplicate_sample_ids_rejected(self, tmp_path: Path) -> None:
        # Two episode records for the same task (e.g. a stale retry attempt that
        # leaked into the episode set) must refuse to build — averaging both
        # silently misreports the score and no downstream gate can catch it.
        from cube_harness.reproducibility.journal import build_journal_submission  # noqa: PLC0415

        exp = _exp_record(n_tasks=2)
        episodes = [_ep_record("t0", 1.0), _ep_record("t1", 0.0), _ep_record("t0", 0.0)]
        episodes[2].trajectory_id = "t0_ep1"  # distinct dir, same task
        EvalLog(experiment=exp, episodes=episodes).save(tmp_path)
        with pytest.raises(ValueError, match="duplicate sample_id"):
            build_journal_submission(tmp_path, submitter="tester")

    def test_more_samples_than_tasks_rejected(self, tmp_path: Path) -> None:
        from cube_harness.reproducibility.journal import build_journal_submission  # noqa: PLC0415

        exp = _exp_record(n_tasks=2)
        episodes = [_ep_record(f"t{i}", 1.0) for i in range(3)]
        EvalLog(experiment=exp, episodes=episodes).save(tmp_path)
        with pytest.raises(ValueError, match="outnumber"):
            build_journal_submission(tmp_path, submitter="tester")
