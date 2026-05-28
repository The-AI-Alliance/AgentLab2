"""Regenerate src/drbench_cube/task_metadata.json from the live drbench package.

Usage:
    uv run python scripts/generate_task_metadata.py [--force]

The drbench package (installed as a dependency) is the source of truth. Run this
script whenever the upstream drbench task set changes to keep task_metadata.json
in sync.

Fields written per task:
    id, split, abstract_description, recommended_max_steps, container_config,
    domain, difficulty, company_name, company_industry, persona_name,
    persona_role, insight_count, _type
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "src" / "drbench_cube" / "task_metadata.json"
_TYPE = "drbench_cube.benchmark.DrBenchTaskMetadata"
_DEFAULT_MAX_STEPS = 50
_CONTAINER_PORTS = [8080, 8081, 8082, 8085, 8090, 8099, 1143]
_CONTAINER_RAM_GB = 6.0
_CONTAINER_CPU_CORES = 2.0
_CONTAINER_DISK_GB = 10.0


def _build_entry(task_id: str, split: str) -> dict:
    from drbench.task_loader import get_task_from_id

    task = get_task_from_id(task_id)
    cfg = task.get_task_config()
    persona = cfg.get("persona", {})
    company = cfg.get("company_info", {})

    eval_cfg = task.get_eval_config() or {}
    insight_count = len(eval_cfg.get("dr_report_evaluation_qa", []))

    dr_question = task.get_dr_question()
    domain = cfg.get("domain", "") or cfg.get("category", "")
    difficulty = cfg.get("level", "") or cfg.get("difficulty", "")

    return {
        "id": task_id,
        "split": split,
        "abstract_description": dr_question,
        "recommended_max_steps": _DEFAULT_MAX_STEPS,
        "container_config": {
            "image": f"drbench-services:{task_id}",
            "ports": _CONTAINER_PORTS,
            "ram_gb": _CONTAINER_RAM_GB,
            "cpu_cores": _CONTAINER_CPU_CORES,
            "disk_gb": _CONTAINER_DISK_GB,
        },
        "domain": domain,
        "difficulty": difficulty,
        "company_name": company.get("name", ""),
        "company_industry": company.get("industry", ""),
        "persona_name": persona.get("name", ""),
        "persona_role": persona.get("department", ""),
        "insight_count": insight_count,
        "_type": _TYPE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite without prompting")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        answer = input(f"{OUTPUT_PATH} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    from drbench.task_loader import get_task_ids_from_subset

    entries = []
    for split in ("val", "train", "test"):
        try:
            task_ids = get_task_ids_from_subset(split)
        except FileNotFoundError:
            continue
        for task_id in sorted(task_ids):
            print(f"  {task_id} ({split})")
            entries.append(_build_entry(task_id, split))

    OUTPUT_PATH.write_text(json.dumps(entries, indent=2) + "\n")
    size = OUTPUT_PATH.stat().st_size
    print(f"\nWrote {len(entries)} tasks to {OUTPUT_PATH}")
    print(f"  {size} bytes  ({size // len(entries)} bytes/task)")


if __name__ == "__main__":
    main()
