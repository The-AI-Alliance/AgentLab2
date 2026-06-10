"""Per-task *samples* bundle for the cube-registry journal.

The journal record (:mod:`cube_harness.reproducibility.journal`) is the small,
human-reviewable **summary** that lives in the registry git repo and drives the
UI table. This module produces its companion **detailed** bundle: one JSON line
per episode (the eval-level :class:`~cube_harness.eval_log.EpisodeRecord` — score,
usage, steps, etc.; *no* trajectory blobs), gzipped into a single file.

`aggregate_from_samples` is the integrity check shared by *both* sides: the
harness uses it to assert the summary it emits matches the bundle, and
cube-registry's ``results_check.py`` re-runs the same computation in CI to prove
a submitted summary is consistent with its raw per-task data. Keeping the one
function here (and a dependency-light mirror in the registry) is what lets the
self-reported headline number become verifiable.

The gzip stream is written with ``mtime=0`` so the same experiment always
produces byte-identical bundles — the ``sha256`` is stable, and a re-submission
is a no-op rather than a spurious diff.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from cube_harness.eval_log import EvalLog

SAMPLES_FORMAT = "jsonl.gz"
SAMPLES_SUFFIX = ".samples.jsonl.gz"


def build_samples_bundle(experiment_dir: Path) -> bytes:
    """Return the gzipped JSONL bundle for *experiment_dir*.

    One line per episode — the full eval-level ``EpisodeRecord`` (no trajectory
    blobs, which live separately in ``episodes/<id>/steps``). Deterministic:
    ``mtime=0`` makes the bytes (and thus the sha256) reproducible.
    """
    eval_log = EvalLog.load(Path(experiment_dir))
    lines = [json.dumps(ep.model_dump(mode="json"), separators=(",", ":"), sort_keys=True) for ep in eval_log.episodes]
    raw = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    return gzip.compress(raw, mtime=0)


def iter_samples(bundle_gz: bytes) -> list[dict[str, Any]]:
    """Decode a gzipped JSONL bundle into a list of per-task sample dicts."""
    raw = gzip.decompress(bundle_gz).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def aggregate_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the headline aggregate from per-task samples.

    The shared integrity check: the harness asserts this matches the summary it
    builds, and the registry verifier re-runs it in CI to confirm a submitted
    summary derives from its bundle. ``avg_score`` / ``std_err`` mirror the
    journal record's ``results`` block exactly (same rounding, same stdev).
    """
    scores = [float(s["score"]) for s in samples if s.get("score") is not None]
    n_scored = len(scores)
    avg = sum(scores) / n_scored if n_scored else 0.0
    std_err = statistics.stdev(scores) / (n_scored**0.5) if n_scored > 1 else 0.0
    return {
        "n_samples": len(samples),
        "n_scored": n_scored,
        "avg_score": round(avg, 6),
        "std_err": round(std_err, 6),
    }


def sha256_hex(data: bytes) -> str:
    """Hex sha256 of *data* — recorded in the summary, re-checked by the verifier."""
    return hashlib.sha256(data).hexdigest()
