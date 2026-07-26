"""SQLite-backed shared pool for the human judge-validation study.

Several annotators hit one Gradio portal at the same time, so handing out
episodes needs a serialization point. That point is a single SQLite file in WAL
mode at ``<study_dir>/annotations.db``: ``claim_next`` runs one
``BEGIN IMMEDIATE`` transaction, which makes "pick an unclaimed episode and
lease it" atomic without adding a dependency or a server process.

``sample.json`` stays the source of truth for *which* episodes are in the study;
``seed`` copies its blind fields in idempotently. The judge's own labels live in
``judge_key.json`` and are never read here — the store holds only what
annotators produce.

Three tables:

    episode  the blind pool (uid, where to load it from, display fields)
    lease    who is currently holding which uid, and until when
    label    one row per (uid, annotator) — the study's output

Every method opens its own short-lived connection. Gradio serves requests from a
thread pool, and a connection per call is both thread-safe by construction and
cheap under WAL — far simpler than sharing one handle behind a lock.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DB_FILENAME = "annotations.db"

#: How long a claimed episode stays reserved for one annotator, in seconds.
#: A closed browser tab never releases its lease explicitly, so the pool must
#: reclaim it on a timer or the episode is lost for the rest of the study.
DEFAULT_LEASE_TTL = 45 * 60

#: How many independent labels each episode should collect before any episode
#: is served a third (or later) time. Two is the minimum that yields an
#: inter-annotator kappa pair for every episode in the study.
DEFAULT_TARGET_COVERAGE = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episode (
    uid           TEXT PRIMARY KEY,
    experiment    TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    episode_dir   TEXT NOT NULL,
    modality      TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    stratum       TEXT NOT NULL,
    benchmark     TEXT NOT NULL DEFAULT '',
    score         REAL NOT NULL DEFAULT 0.0,
    position      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS lease (
    uid        TEXT NOT NULL,
    annotator  TEXT NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (uid, annotator)
);
CREATE TABLE IF NOT EXISTS label (
    uid        TEXT NOT NULL,
    annotator  TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    blame      TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    notes      TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    PRIMARY KEY (uid, annotator)
);
CREATE INDEX IF NOT EXISTS label_by_uid ON label (uid);
CREATE INDEX IF NOT EXISTS label_by_annotator ON label (annotator);
"""


@dataclass(frozen=True)
class Episode:
    """One blind episode as served to an annotator.

    Carries no judge label and no results-directory path in anything the UI
    displays — the annotator sees ``uid`` ("E017"), never the experiment name.
    """

    uid: str
    experiment: str
    trajectory_id: str
    episode_dir: str
    modality: str
    task_id: str
    stratum: str
    benchmark: str
    score: float


@dataclass(frozen=True)
class Progress:
    """Study-wide progress, for the portal header and the CLI."""

    total: int
    per_annotator: dict[str, int]
    #: label count → how many episodes have exactly that many labels
    coverage: dict[int, int]
    at_target: int
    target_coverage: int


def _episode_from_row(row: sqlite3.Row) -> Episode:
    return Episode(
        uid=row["uid"],
        experiment=row["experiment"],
        trajectory_id=row["trajectory_id"],
        episode_dir=row["episode_dir"],
        modality=row["modality"],
        task_id=row["task_id"],
        stratum=row["stratum"],
        benchmark=row["benchmark"],
        score=row["score"],
    )


class AnnotationStore:
    """The study's shared pool: seed it, claim from it, submit into it."""

    def __init__(
        self,
        study_dir: str | Path,
        *,
        lease_ttl: float = DEFAULT_LEASE_TTL,
        target_coverage: int = DEFAULT_TARGET_COVERAGE,
    ) -> None:
        self.study_dir = Path(study_dir)
        self.db_path = self.study_dir / DB_FILENAME
        self.lease_ttl = lease_ttl
        self.target_coverage = max(1, target_coverage)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # WAL lets readers (progress polls, other tabs) run while a claim
            # holds its write transaction.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A short-lived connection with row access by name.

        ``isolation_level=None`` turns off pysqlite's implicit transactions so
        ``claim_next`` can issue its own ``BEGIN IMMEDIATE`` and mean it.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # --- seeding ----------------------------------------------------------

    def seed(self, entries: list[dict[str, Any]]) -> int:
        """Insert the study's episodes from ``sample.json``; return the count.

        Idempotent: re-seeding an existing study leaves labels and leases alone
        and refreshes the episode rows, so re-running ``sample`` with a wider
        draw extends the pool rather than restarting the study.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for position, entry in enumerate(entries):
                conn.execute(
                    """INSERT INTO episode
                           (uid, experiment, trajectory_id, episode_dir, modality,
                            task_id, stratum, benchmark, score, position)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(uid) DO UPDATE SET
                           experiment=excluded.experiment,
                           trajectory_id=excluded.trajectory_id,
                           episode_dir=excluded.episode_dir,
                           modality=excluded.modality,
                           task_id=excluded.task_id,
                           stratum=excluded.stratum,
                           benchmark=excluded.benchmark,
                           score=excluded.score,
                           position=excluded.position""",
                    (
                        entry["uid"],
                        entry.get("experiment", ""),
                        entry["trajectory_id"],
                        entry["episode_dir"],
                        entry.get("modality", "Other"),
                        entry.get("task_id", ""),
                        # The sample is stratified by (modality, judge blame);
                        # blame is the withheld key, so the blind half of the
                        # cell is all the store may hold.
                        entry.get("stratum") or entry.get("modality", "Other"),
                        entry.get("benchmark", ""),
                        float(entry.get("score", 0.0)),
                        position,
                    ),
                )
            conn.execute("COMMIT")
        return len(entries)

    def seed_from_sample(self, sample_path: str | Path) -> int:
        """Seed from a ``sample.json`` file on disk."""
        return self.seed(json.loads(Path(sample_path).read_text()))

    # --- serving ----------------------------------------------------------

    def expire_stale(self, *, now: float | None = None) -> int:
        """Drop leases past their TTL; return how many were reclaimed."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM lease WHERE expires_at <= ?", (now if now is not None else time.time(),))
            return cur.rowcount or 0

    def claim_next(self, annotator: str, *, now: float | None = None) -> Episode | None:
        """Lease the next episode for ``annotator``, or None if the pool is dry.

        One ``BEGIN IMMEDIATE`` transaction sweeps expired leases, picks a
        candidate and writes the lease, so two annotators arriving together can
        never be handed the same uid.

        Ordering is what makes the study scorable. Episodes already carrying
        *some* labels but short of ``target_coverage`` come first, because
        finishing one completes an inter-annotator pair immediately; untouched
        episodes come next; episodes already at target are served only when
        nothing else is left. Ordering purely by fewest-labels-first would give
        concurrent annotators disjoint sets and yield no overlap at all until
        someone had labelled the entire pool.
        """
        now = now if now is not None else time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM lease WHERE expires_at <= ?", (now,))
                row = conn.execute(
                    """SELECT e.*, COUNT(l.uid) AS n_labels
                         FROM episode e
                         LEFT JOIN label l ON l.uid = e.uid
                        WHERE e.uid NOT IN (SELECT uid FROM label WHERE annotator = ?)
                          AND e.uid NOT IN (SELECT uid FROM lease WHERE annotator <> ?)
                        GROUP BY e.uid
                        ORDER BY CASE
                                     WHEN COUNT(l.uid) = 0 THEN 1
                                     WHEN COUNT(l.uid) < ? THEN 0
                                     ELSE 2
                                 END,
                                 COUNT(l.uid) ASC,
                                 e.position ASC
                        LIMIT 1""",
                    (annotator, annotator, self.target_coverage),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    """INSERT INTO lease (uid, annotator, expires_at) VALUES (?, ?, ?)
                       ON CONFLICT(uid, annotator) DO UPDATE SET expires_at=excluded.expires_at""",
                    (row["uid"], annotator, now + self.lease_ttl),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return _episode_from_row(row)

    def get(self, uid: str) -> Episode | None:
        """Look up one episode by uid."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM episode WHERE uid = ?", (uid,)).fetchone()
        return _episode_from_row(row) if row is not None else None

    def release(self, uid: str, annotator: str) -> None:
        """Give a leased episode back to the pool without labelling it (Skip)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM lease WHERE uid = ? AND annotator = ?", (uid, annotator))

    def submit(
        self,
        uid: str,
        annotator: str,
        *,
        outcome: str,
        blame: str,
        confidence: int,
        notes: str = "",
        now: float | None = None,
    ) -> None:
        """Record one annotator's label and drop their lease.

        Idempotent on the (uid, annotator) pair: re-submitting corrects the
        earlier answer instead of double-counting it.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """INSERT INTO label (uid, annotator, outcome, blame, confidence, notes, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(uid, annotator) DO UPDATE SET
                           outcome=excluded.outcome, blame=excluded.blame,
                           confidence=excluded.confidence, notes=excluded.notes,
                           created_at=excluded.created_at""",
                    (uid, annotator, outcome, blame, int(confidence), notes, now if now is not None else time.time()),
                )
                conn.execute("DELETE FROM lease WHERE uid = ? AND annotator = ?", (uid, annotator))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # --- reporting --------------------------------------------------------

    def progress(self) -> Progress:
        """Per-annotator counts plus the coverage histogram."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM episode").fetchone()["n"]
            per_annotator = {
                r["annotator"]: r["n"]
                for r in conn.execute(
                    "SELECT annotator, COUNT(*) AS n FROM label GROUP BY annotator ORDER BY annotator"
                )
            }
            labelled = {r["uid"]: r["n"] for r in conn.execute("SELECT uid, COUNT(*) AS n FROM label GROUP BY uid")}
        coverage: dict[int, int] = {0: total - len(labelled)}
        for n in labelled.values():
            coverage[n] = coverage.get(n, 0) + 1
        at_target = sum(count for n, count in coverage.items() if n >= self.target_coverage)
        return Progress(
            total=total,
            per_annotator=per_annotator,
            coverage=dict(sorted(coverage.items())),
            at_target=at_target,
            target_coverage=self.target_coverage,
        )

    def export_labels(self) -> dict[str, dict[str, dict[str, Any]]]:
        """``{annotator: {uid: {outcome, blame, confidence, notes}}}``.

        This is the shape ``judge_validation.py score`` consumes — it used to
        come from one ``labels-<annotator>.json`` per person.
        """
        out: dict[str, dict[str, dict[str, Any]]] = {}
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM label ORDER BY annotator, uid"):
                out.setdefault(row["annotator"], {})[row["uid"]] = {
                    "outcome": row["outcome"],
                    "blame": row["blame"],
                    "confidence": row["confidence"],
                    "notes": row["notes"],
                }
        return out
