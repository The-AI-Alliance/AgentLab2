#!/usr/bin/env python3
"""SMOKE: grade a KNOWS gold-instance doc end-to-end against the real evaluators.

The only check that exercises the real KNOWS grading path. Requires full Google +
Gemini credentials and SKIPs cleanly without them.

Asserts a THRESHOLD, not 1.0. The VLM scoring steps flake between runs, and
docs_1 has a deterministic ceiling of 8/10 because `verify_image_in_region` has
no non-VLM tier. A smoke that demanded 1.0 would fail constantly and be ignored.

WARNING: gold docs are shared 'anyone-with-link: writer' in a third party's
Drive. One stray edit permanently changes this smoke's score, and upstream's
`--reset` mints new ids so the pinned id goes stale by design. Investigate a
drop before assuming a regression in this cube.

    SMOKE OK | FAIL | SKIP: knows_gold_eval   (exit 0 | 1 | 2)
"""

import sys
from typing import Annotated

import typer
from knows_cube.benchmark import KnowsBenchmarkConfig
from knows_cube.evaluator import missing_credentials
from knows_cube.task import KnowsTaskConfig

_TASK_ID = "knows.docs_1_formal_letter.1"


def main(
    gold_doc_id: Annotated[str, typer.Option(help="Gold-instance document id to grade.")],
    threshold: Annotated[float, typer.Option(help="Minimum acceptable reward.")] = 0.8,
) -> None:
    """Grade a KNOWS gold instance and assert reward >= threshold."""
    missing = missing_credentials()
    if missing:
        print(f"SMOKE SKIP: knows_gold_eval ({'; '.join(missing)})")
        raise typer.Exit(2)

    metadata = KnowsBenchmarkConfig().tasks()[_TASK_ID]
    task = KnowsTaskConfig(metadata=metadata, existing_doc_id=gold_doc_id).make()
    try:
        reward, info = task._grade(gold_doc_id)
    finally:
        task.close()

    if error := info.get("evaluation_error"):
        print(f"SMOKE FAIL: knows_gold_eval (evaluation_error: {error})")
        raise typer.Exit(1)
    if reward < threshold:
        print(f"SMOKE FAIL: knows_gold_eval (reward {reward:.3f} < {threshold})")
        raise typer.Exit(1)
    print(
        f"SMOKE OK: knows_gold_eval (reward {reward:.3f}, {info.get('eval.score_result')}/{info.get('eval.score_total')})"
    )
    sys.exit(0)


if __name__ == "__main__":
    typer.run(main)
