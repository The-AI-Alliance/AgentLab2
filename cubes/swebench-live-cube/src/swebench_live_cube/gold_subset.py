"""Official ``lite-gold`` subset: the lite (300) tasks whose gold patch resolves
under the root (Daytona) oracle.

The committed id list (``lite_solvable_daytona_2026-05-25.json``, 275 tasks) is the
source of truth. ``create_task_metadata.py`` stamps the ``lite-gold`` marker into each
gold-solvable task's ``splits`` so ``named_subset("lite-gold")`` can glob it the same
way as ``lite`` / ``verified`` / ``full``. No cube-harness dependency — pure metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

# Root/Daytona oracle result; supersedes the non-root EAI list (223). See the JSON header.
LITE_GOLD_SOLVABLE_JSON = Path(__file__).parent / "lite_solvable_daytona_2026-05-25.json"
LITE_GOLD_SPLIT = "lite-gold"


def lite_gold_ids() -> set[str]:
    """Task IDs of the gold-solvable lite subset (root/Daytona oracle)."""
    return set(json.loads(LITE_GOLD_SOLVABLE_JSON.read_text())["task_ids"])


def tag_lite_gold(rows: list[dict]) -> int:
    """Append the ``lite-gold`` marker to the ``splits`` of each gold-solvable task.

    Mutates ``rows`` (the list-of-dicts form of ``task_metadata.json``) in place and
    returns the number of tasks newly tagged. Idempotent. Raises if a gold-solvable id
    is absent from the registry.
    """
    gold = lite_gold_ids()
    by_id = {r["id"]: r for r in rows}
    missing = gold - by_id.keys()
    if missing:
        raise ValueError(f"{len(missing)} gold-solvable ids missing from task registry, e.g. {sorted(missing)[:5]}")
    tagged = 0
    for tid in gold:
        splits = by_id[tid].setdefault("splits", [])
        if LITE_GOLD_SPLIT not in splits:
            splits.append(LITE_GOLD_SPLIT)
            tagged += 1
    return tagged
