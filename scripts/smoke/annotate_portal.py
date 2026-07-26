#!/usr/bin/env python3
"""SMOKE: the judge-validation annotation portal, end to end.

Unit tests cover the store's ordering and the panel's blinding against a
synthetic fixture. What they cannot cover is the thing that actually decides
whether the study can run: do *real* episodes -- 60 of them, drawn from real
results directories across modalities -- load and render, and does the whole
sample → serve → label → score pipeline close?

Verifies, against a throwaway copy of a real study directory:

  - every episode in `sample.json` loads through `FileStorage.load_episode`
    and renders a goal, an event rail, and per-step LLM detail;
  - screenshot coverage, since evidence parity with the judge is the entire
    reason the static text packet was replaced;
  - no rendered surface, on any event of any episode, leaks the judge's
    verdict (blame category, compound outcome, or record field name);
  - concurrent annotators are never served the same episode, and coverage
    converges to the target with inter-annotator overlap on every episode;
  - `judge_validation.py score` reads the resulting store and emits its table.

The labels this smoke submits are random, so the kappa it prints is noise --
the point is that the pipeline closes, not what it says.

Run:
    .venv/bin/python scripts/smoke/annotate_portal.py
    .venv/bin/python scripts/smoke/annotate_portal.py --study some_other_study

Prints SMOKE OK|FAIL|SKIP: annotate_portal  (exit 0|1|2).
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cube_harness.analyze.annotate.panel import EpisodeViewer
from cube_harness.analyze.annotate.store import AnnotationStore
from cube_harness.eval_log import BlameCategory, Outcome

NAME = "annotate_portal"
DEFAULT_STUDY = Path("judge_study")
ANNOTATORS = ["alice", "bob", "carol"]
TARGET_COVERAGE = 2

# Strings only the investigator's output uses. Bare `success` / `failure` are
# excluded on purpose: a benchmark's own reward_info says `{"success": true}`,
# which is environment evidence the judge read too, not the judge's verdict.
LEAK_NEEDLES = (
    [b.value for b in BlameCategory if b is not BlameCategory.none]
    + [o.value for o in Outcome if "_" in o.value]
    + ["primary_blame", "findings", "investigator"]
)


def fail(message: str) -> int:
    print(f"SMOKE FAIL: {NAME} — {message}")
    return 1


def skip(message: str) -> int:
    print(f"SMOKE SKIP: {NAME} — {message}")
    return 2


def check_rendering(store: AnnotationStore, uids: list[str]) -> tuple[int, list[str]]:
    """Render every episode; return screenshot count and any problems found."""
    viewer = EpisodeViewer()
    problems: list[str] = []
    with_screenshots = 0
    for uid in uids:
        episode = store.get(uid)
        if episode is None:
            problems.append(f"{uid}: not in the store after seeding")
            continue
        if viewer.events(episode) is None:
            problems.append(f"{uid}: episode did not load ({episode.episode_dir})")
            continue
        render = viewer.render(episode, viewer.initial_index(episode))
        n_events = viewer.n_events(episode)
        if render.rail.count("xray-event-card") != n_events:
            problems.append(f"{uid}: rail has {render.rail.count('xray-event-card')} cards for {n_events} events")
        if "No goal text found" in render.goal or "could not be loaded" in render.goal:
            problems.append(f"{uid}: no task goal rendered")
        if render.images:
            with_screenshots += 1

        # Blinding sweep over every event, not just the landing one.
        for index in range(n_events):
            text = viewer.render(episode, index).all_text().lower()
            for needle in LEAK_NEEDLES:
                if needle in text:
                    problems.append(f"{uid} event {index}: leaks {needle!r}")
    return with_screenshots, problems


def check_serving(store: AnnotationStore, total: int) -> list[str]:
    """Drive concurrent annotators through the pool until it is dry."""
    problems: list[str] = []
    rng = random.Random(7)
    blames = [b.value for b in BlameCategory]
    outcomes = [o.value for o in Outcome]

    def work(annotator: str) -> list[str]:
        claimed: list[str] = []
        while (episode := store.claim_next(annotator)) is not None:
            claimed.append(episode.uid)
            store.submit(
                episode.uid,
                annotator,
                outcome=rng.choice(outcomes),
                blame=rng.choice(blames),
                confidence=rng.randint(1, 5),
                notes="",
            )
        return claimed

    with ThreadPoolExecutor(max_workers=len(ANNOTATORS)) as pool:
        per_person = list(pool.map(work, ANNOTATORS))

    for annotator, claimed in zip(ANNOTATORS, per_person, strict=True):
        if len(claimed) != len(set(claimed)):
            problems.append(f"{annotator} was served a duplicate episode")

    labels = store.export_labels()
    if sorted(labels) != sorted(ANNOTATORS):
        problems.append(f"expected labels from {ANNOTATORS}, got {sorted(labels)}")

    progress = store.progress()
    if progress.at_target != total:
        problems.append(f"only {progress.at_target}/{total} episodes reached ≥{TARGET_COVERAGE} labels")

    # Every episode must carry at least one inter-annotator pair, or there is
    # nothing to compute inter-annotator kappa from.
    per_uid: dict[str, int] = {}
    for per_annotator in labels.values():
        for uid in per_annotator:
            per_uid[uid] = per_uid.get(uid, 0) + 1
    unpaired = [uid for uid, count in per_uid.items() if count < 2]
    if unpaired:
        problems.append(f"{len(unpaired)} episode(s) have no second label: {unpaired[:5]}")
    return problems


def check_scoring(study: Path) -> list[str]:
    """`judge_validation.py score` must read the store and emit its table."""
    tex = study / "judge_validation.tex"
    result = subprocess.run(
        [sys.executable, "scripts/paper/judge_validation.py", "score", str(study), "--out", str(tex)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"score failed (exit {result.returncode}): {result.stderr.strip()[-500:]}"]
    if not tex.exists() or r"\begin{tabular}" not in tex.read_text():
        return ["score wrote no LaTeX table"]
    if "inter-annotator" not in result.stdout:
        return ["score reported no inter-annotator kappa"]
    print("  score output:")
    for line in result.stdout.strip().splitlines()[:6]:
        print(f"    {line}")
    return []


def main() -> int:
    source = Path(sys.argv[sys.argv.index("--study") + 1]) if "--study" in sys.argv else DEFAULT_STUDY
    if not (source / "sample.json").exists():
        return skip(f"no study at {source}/sample.json — run `judge_validation.py sample` first")

    # Work on a copy: the smoke submits junk labels and must never touch the
    # real study's annotations.db.
    with tempfile.TemporaryDirectory() as tmp:
        study = Path(tmp) / "study"
        study.mkdir()
        shutil.copy(source / "sample.json", study / "sample.json")
        if (source / "judge_key.json").exists():
            shutil.copy(source / "judge_key.json", study / "judge_key.json")
        else:
            return skip(f"{source}/judge_key.json missing — nothing to score against")

        store = AnnotationStore(study, target_coverage=TARGET_COVERAGE)
        uids = [e["uid"] for e in json.loads((study / "sample.json").read_text())]
        store.seed_from_sample(study / "sample.json")
        print(f"seeded {len(uids)} episodes from {source}/sample.json")

        with_screenshots, problems = check_rendering(store, uids)
        if problems:
            return fail("; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
        print(f"rendered all {len(uids)} episodes; {with_screenshots} carry screenshots; no verdict leaked")

        problems = check_serving(store, len(uids))
        if problems:
            return fail("; ".join(problems[:5]))
        progress = store.progress()
        print(
            f"served to {len(ANNOTATORS)} concurrent annotators with no collisions; "
            f"{progress.at_target}/{progress.total} at ≥{TARGET_COVERAGE} labels "
            f"({', '.join(f'{a}={c}' for a, c in progress.per_annotator.items())})"
        )

        problems = check_scoring(study)
        if problems:
            return fail("; ".join(problems))

    print(f"SMOKE OK: {NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
