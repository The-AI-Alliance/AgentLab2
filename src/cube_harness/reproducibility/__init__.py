"""Reproducibility — convert an experiment's :class:`EvalLog` into the formats
consumed by external community journals.

Two destinations, one source of truth:

- :mod:`.journal` builds the cube-registry community-journal record
  (per-cube, typed, drives the per-cube UI on the registry site).
- :mod:`.eee` builds an `Every Eval Ever <https://evalevalai.com>`_-shaped
  record for upstream submission to the EvalEval HuggingFace dataset.

Both read the same :class:`cube_harness.eval_log.EvalLog` + per-episode
``status.json`` files; neither depends on the other.
"""

from cube_harness.reproducibility import submissions
from cube_harness.reproducibility.eee import EEE_SCHEMA_VERSION, build_eee_record
from cube_harness.reproducibility.journal import (
    JOURNAL_SCHEMA_VERSION,
    JournalSubmission,
    Outcomes,
    build_journal_record,
    build_journal_submission,
    sanitize_filename,
)

__all__ = [
    "EEE_SCHEMA_VERSION",
    "JOURNAL_SCHEMA_VERSION",
    "JournalSubmission",
    "Outcomes",
    "build_eee_record",
    "build_journal_record",
    "build_journal_submission",
    "sanitize_filename",
    "submissions",
]
