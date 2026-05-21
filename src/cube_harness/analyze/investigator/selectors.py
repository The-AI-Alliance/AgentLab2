"""Related-trajectory selectors for the investigator.

A `Selector` answers the question: "given the main episode the investigator is
analysing, which *other* episodes should it also have read access to for
context?" Plugged into `investigate_episode` / `investigate_experiment` via the
`selector=` kwarg on `InvestigationConfig`.

Selectors are sync — they should be cheap (filename / on-disk record
inspection), not extra LLM calls. Heavy similarity scorers should
pre-compute embeddings off-line and load them here.

Episode discovery / `--ids` / `--sample` filtering lives in
`episode_discovery.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from cube_harness.analyze.investigator.episode_discovery import EpisodeRef


@runtime_checkable
class Selector(Protocol):
    """Pluggable rule for picking *related* episodes the investigator should also read.

    `select` is sync: selectors are expected to be cheap (look at filenames or a
    few JSON files), not run extra LLMs. Heavy similarity scorers should
    pre-compute embeddings off-line and load them here.
    """

    name: str
    k: int

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]: ...


def _filter_main_out(refs: Sequence[EpisodeRef], main: EpisodeRef) -> list[EpisodeRef]:
    """Drop the main episode from a candidate list."""
    return [r for r in refs if r.episode_dir != main.episode_dir]


@dataclass
class SameTaskDifferentAgent:
    """Pick episodes with the same `sample_id` but a different `evaluation_id`.

    Useful for "did other agents fail this task the same way?" Reads each
    candidate's `episode_record.json` to compare; cheap because the records
    are already on disk.
    """

    k: int = 3
    name: str = "same_task_different_agent"

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]:
        if main_episode.record is None:
            return []
        main_sample = main_episode.record.sample_id
        main_eval = main_episode.record.evaluation_id
        out: list[Path] = []
        for ref in _filter_main_out(all_refs, main_episode):
            if ref.record is None:
                continue
            if ref.record.sample_id == main_sample and ref.record.evaluation_id != main_eval:
                out.append(ref.episode_dir)
                if len(out) >= self.k:
                    break
        return out


@dataclass
class SameAgentPreviousIteration:
    """Pick episodes from the same agent on the same task at earlier timestamps.

    Useful for "did the agent already try this and learn anything?" Compares
    `evaluation_id` (same agent run) and `sample_id` (same task), keeps the K
    earliest by `timestamp`."""

    k: int = 2
    name: str = "same_agent_previous_iteration"

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]:
        if main_episode.record is None:
            return []
        main_eval = main_episode.record.evaluation_id
        main_sample = main_episode.record.sample_id
        main_ts = main_episode.record.timestamp
        candidates: list[tuple[float, Path]] = []
        for ref in _filter_main_out(all_refs, main_episode):
            if ref.record is None:
                continue
            if ref.record.evaluation_id != main_eval or ref.record.sample_id != main_sample:
                continue
            if ref.record.timestamp >= main_ts:
                continue
            candidates.append((ref.record.timestamp, ref.episode_dir))
        # Earliest first; bound by k.
        candidates.sort(key=lambda t: t[0])
        return [p for _, p in candidates[: self.k]]


@dataclass
class TopKBySimilarityStub:
    """Placeholder for a real similarity scorer.

    Today: deterministic hash of trajectory_id, sorted ascending — it returns a
    stable list per main episode but the choice has no semantic meaning. The
    concrete scorer (e.g. via cosine similarity over task descriptions) is a
    follow-up; this stub keeps the call-site contract working."""

    k: int = 3
    name: str = "top_k_by_similarity_stub"

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]:
        candidates = _filter_main_out(all_refs, main_episode)
        # Stable ranking by SHA-256(trajectory_id) — placeholder for real cosine
        # similarity over task-description embeddings.
        ranked = sorted(
            candidates,
            key=lambda r: hashlib.sha256(r.trajectory_id.encode()).hexdigest(),
        )
        return [r.episode_dir for r in ranked[: self.k]]


__all__ = [
    "Selector",
    "SameTaskDifferentAgent",
    "SameAgentPreviousIteration",
    "TopKBySimilarityStub",
]
