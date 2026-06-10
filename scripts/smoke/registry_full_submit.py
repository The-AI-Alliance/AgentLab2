#!/usr/bin/env python3
"""Smoke: a full-data registry submission validates against the real cube-registry verifier.

Builds a synthetic-but-consistent experiment, runs ``build_journal_submission`` to
produce the summary + samples bundle, then invokes cube-registry's
``scripts/results_check.py`` (in its own venv) on the pair — proving the harness
output is exactly what the registry CI accepts, and that a tampered bundle is
rejected by the consistency / sha256 gates.

Discovers the registry checkout via ``$CUBE_REGISTRY_DIR`` (default: the sibling
``../cube-registry``). SKIPs cleanly when the checkout or its venv is absent.

    SMOKE OK / FAIL / SKIP: registry_full_submit   (exit 0 / 1 / 2)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cube_harness.eval_log import EvalLog
from cube_harness.reproducibility.journal import build_journal_submission

NAME = "registry_full_submit"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _skip(msg: str) -> None:
    print(f"SMOKE SKIP: {NAME} — {msg}")
    sys.exit(2)


def _registry_dir() -> Path:
    d = Path(os.environ.get("CUBE_REGISTRY_DIR", Path(__file__).resolve().parents[2].parent / "cube-registry"))
    if not (d / "scripts" / "results_check.py").exists():
        _skip(f"cube-registry not found at {d} (set $CUBE_REGISTRY_DIR)")
    py = d / ".venv" / "bin" / "python"
    if not py.exists():
        _skip(f"cube-registry venv not found at {py}")
    return d


def _build_experiment(exp_dir: Path, scores: list[float]) -> None:
    """A minimal, internally-consistent EvalLog (one COMPLETED episode per score).

    Reuses the reproducibility test fixtures so the records stay valid as the
    models evolve."""
    sys.path.insert(0, str(_REPO_ROOT))
    from tests.test_reproducibility import _ep_record, _exp_record, _write_status  # noqa: PLC0415

    exp = _exp_record(n_tasks=len(scores))
    EvalLog(experiment=exp, episodes=[_ep_record(f"t{i}", s) for i, s in enumerate(scores)]).save(exp_dir)
    for i, s in enumerate(scores):
        _write_status(exp_dir, f"t{i}", "COMPLETED", reward=s)


_VALIDATE_SNIPPET = r"""
import sys, gzip, glob
from pathlib import Path
sys.path.insert(0, "scripts")
import results_check as rc
SMOKE = Path(sys.argv[1])
for attr, val in [("REPO_ROOT", SMOKE), ("SCHEMA_PATH", SMOKE/"results-schema.json"),
                  ("SAMPLES_SCHEMA_PATH", SMOKE/"samples-schema.json"),
                  ("ENTRIES_DIR", SMOKE/"entries"), ("RESULTS_DIR", SMOKE/"results")]:
    setattr(rc, attr, val)
schema, samples = rc._load_schema(), rc._load_samples_schema()
summary = Path(glob.glob(str(SMOKE/"results/miniwob/*.json"))[0])
ok = rc.check_file(summary, schema, samples_schema=samples)
bundle = next(SMOKE.glob("results/miniwob/*.samples.jsonl.gz"))
bundle.write_bytes(gzip.compress(b'{"sample_id":"t0","score":1.0}\n', mtime=0))
bad = rc.check_file(summary, schema, samples_schema=samples)
print("valid_ok=%s tamper_rejected=%s" % (ok == [], bool(bad)))
if ok: print("  valid failures:", ok)
sys.exit(0 if (ok == [] and bad) else 1)
"""


def main() -> int:
    registry = _registry_dir()
    with tempfile.TemporaryDirectory(prefix="cubereg-smoke-") as tmp:
        smoke = Path(tmp)
        (smoke / "results" / "miniwob").mkdir(parents=True)
        (smoke / "entries").mkdir()
        shutil.copy(registry / "results-schema.json", smoke / "results-schema.json")
        shutil.copy(registry / "samples-schema.json", smoke / "samples-schema.json")

        scores = [1.0, 0.0, 1.0, 1.0, 0.0]
        _build_experiment(smoke / "_exp", scores)
        sub = build_journal_submission(smoke / "_exp", submitter="tester", cube_id="miniwob")
        cube_dir = smoke / "results" / "miniwob"
        (cube_dir / sub.summary_filename).write_text(json.dumps(sub.record, indent=2) + "\n")
        (cube_dir / sub.bundle_filename).write_bytes(sub.bundle)
        ver = sub.record["benchmark_version"]
        (smoke / "entries" / "miniwob.yaml").write_text(
            f"id: miniwob\nname: MiniWob\nversion: {ver}\ndescription: smoke\npackage: miniwob-cube\n"
            "authors:\n  - github: tester\nlegal:\n  wrapper_license: MIT\ntask_count: 125\n"
        )

        proc = subprocess.run(
            [str(registry / ".venv" / "bin" / "python"), "-c", _VALIDATE_SNIPPET, str(smoke)],
            cwd=registry,
            capture_output=True,
            text=True,
        )
        print(proc.stdout.strip())
        if proc.returncode != 0:
            print(proc.stderr.strip())
            print(f"SMOKE FAIL: {NAME}")
            return 1

    print(f"SMOKE OK: {NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
