"""Reset episode statuses so the experiment runner retries them.

Sets ``status="FAILED"`` and ``retry_count=0`` on matching episode directories,
making them eligible for retry when the recipe is relaunched with ``resume=True``.

Only episodes whose task_id matches a requested task are touched.  COMPLETED
episodes with reward=1.0 (genuinely solved) are skipped unless ``--force`` is
passed.

CLI: ``ch-reset-episodes <experiment_dir> --tasks <task_id> [<task_id> ...]``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cube_harness.episode_status import RETRIABLE_STATUSES, STATUS_FILENAME, EpisodeStatus

logger = logging.getLogger(__name__)

_RESET_ERROR_TYPE = "ManualReset"
_RESET_ERROR_MESSAGE = "Reset by ch-reset-episodes for retry"


def reset_episodes(
    experiment_dir: Path,
    task_ids: set[str],
    *,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Reset matching episodes under *experiment_dir*.

    Returns the number of episodes actually reset (or that would be, on dry-run).
    """
    episodes_dir = experiment_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"No 'episodes' directory found under {experiment_dir}")

    reset_count = 0
    skipped_solved = 0
    skipped_already_retriable = 0
    not_found: set[str] = set(task_ids)

    for ep_dir in sorted(episodes_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        status_path = ep_dir / STATUS_FILENAME
        status = EpisodeStatus.read(status_path)
        if status is None:
            continue
        if status.task_id not in task_ids:
            continue

        not_found.discard(status.task_id)

        if status.status in RETRIABLE_STATUSES:
            logger.debug("Skipping %s — already retriable (%s)", ep_dir.name, status.status)
            skipped_already_retriable += 1
            continue

        if status.reward == 1.0 and not force:
            logger.info("Skipping %s — already solved (reward=1.0); use --force to reset", ep_dir.name)
            skipped_solved += 1
            continue

        print(
            f"{'[dry-run] ' if dry_run else ''}Reset {ep_dir.name} "
            f"({status.status}, retry={status.retry_count}, reward={status.reward})"
        )

        if not dry_run:
            status.status = "FAILED"
            status.retry_count = 0
            status.error_type = _RESET_ERROR_TYPE
            status.error_message = _RESET_ERROR_MESSAGE
            status.ended_at = None
            status.write(status_path)

        reset_count += 1

    if not_found:
        logger.warning("Task IDs not found in experiment: %s", ", ".join(sorted(not_found)))
    if skipped_solved:
        logger.info("Skipped %d solved episode(s) (reward=1.0); use --force to reset", skipped_solved)
    if skipped_already_retriable:
        logger.debug("Skipped %d already-retriable episode(s)", skipped_already_retriable)

    return reset_count


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ch-reset-episodes",
        description="Reset episode statuses to FAILED so the runner retries them.",
    )
    p.add_argument("path", type=Path, help="Experiment directory.")
    p.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK_ID",
        required=True,
        help="One or more task IDs to reset (e.g. psf__requests-1142).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would change without modifying files.")
    p.add_argument("--force", action="store_true", help="Also reset episodes with reward=1.0.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    try:
        n = reset_episodes(
            args.path.resolve(),
            set(args.tasks),
            dry_run=args.dry_run,
            force=args.force,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    label = "Would reset" if args.dry_run else "Reset"
    print(f"{label} {n} episode(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
