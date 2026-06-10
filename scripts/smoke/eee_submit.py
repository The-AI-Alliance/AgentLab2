#!/usr/bin/env python3
"""Smoke: an EEE record built from an experiment validates against EEE's own schema.

Builds a synthetic-but-consistent experiment, runs ``build_eee_record``, and
validates the result with the real ``every-eval-ever`` package — proving the
emitted record is exactly what EEE accepts. Also confirms the upload path is
wired (``huggingface_hub`` importable + ``HfApi.upload_file`` present); it does
*not* actually open a PR (that needs auth and would spam the community dataset —
verified manually instead).

SKIPs cleanly when ``every-eval-ever`` isn't installed.

    SMOKE OK / FAIL / SKIP: eee_submit   (exit 0 / 1 / 2)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from cube_harness.eval_log import EvalLog
from cube_harness.reproducibility.eee import build_eee_record

NAME = "eee_submit"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _skip(msg: str) -> None:
    print(f"SMOKE SKIP: {NAME} — {msg}")
    sys.exit(2)


def main() -> int:
    if subprocess.run([sys.executable, "-c", "import every_eval_ever"], capture_output=True).returncode != 0:
        _skip("every-eval-ever not installed (pip install every-eval-ever)")

    sys.path.insert(0, str(_REPO_ROOT))
    from tests.test_reproducibility import _ep_record, _exp_record, _write_status  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="eee-smoke-") as tmp:
        exp_dir = Path(tmp) / "exp"
        scores = [1.0, 0.0, 1.0, 1.0, 0.0]
        EvalLog(
            experiment=_exp_record(n_tasks=len(scores)), episodes=[_ep_record(f"t{i}", s) for i, s in enumerate(scores)]
        ).save(exp_dir)
        for i, s in enumerate(scores):
            _write_status(exp_dir, f"t{i}", "COMPLETED", reward=s)

        record = build_eee_record(exp_dir, source_organization_name="smoke")
        target = Path(tmp) / "record.json"
        target.write_text(json.dumps(record, indent=2))

        proc = subprocess.run(
            [sys.executable, "-m", "every_eval_ever", "validate", str(target)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout + proc.stderr)
            print(f"SMOKE FAIL: {NAME} — EEE validation failed")
            return 1

    # Upload path wired?
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        assert callable(HfApi().upload_file)
    except Exception as e:  # noqa: BLE001
        print(f"SMOKE FAIL: {NAME} — upload path not wired: {e}")
        return 1

    print("  ✓ record validates against EEE schema; upload path wired (huggingface_hub)")
    print(f"SMOKE OK: {NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
