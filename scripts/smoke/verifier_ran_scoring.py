"""SMOKE: a completed-but-ungraded episode (reward_info.verifier_ran is False) is
excluded from accuracy — not counted as a graded zero — on BOTH score paths:
the live ``FileStorage.update_experiment_summary`` and the ``experiments_report``
``_scan_episodes`` table path.

Builds a synthetic experiment with three completed episodes — a graded pass (1.0),
a graded fail (0.0), and an ungraded one (verifier didn't run) — and asserts the
mean is 0.5 over the two GRADED episodes (not 1/3 over all three), with the
ungraded one surfaced separately.

    python scripts/smoke/verifier_ran_scoring.py

Prints `SMOKE OK/FAIL: verifier_ran_scoring` and exits 0/1.
"""

import json
import sys
import tempfile
from pathlib import Path

from cube_harness.core import Trajectory
from cube_harness.episode_status import EpisodeStatus
from cube_harness.storage import FileStorage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_report import _scan_episodes  # noqa: E402

from cube_harness.results import ExperimentResult  # noqa: E402

# (task_id, reward, verifier_ran)
EPISODES = [
    ("graded_pass", 1.0, True),
    ("graded_fail", 0.0, True),
    ("ungraded", 0.0, False),  # verifier never ran → must NOT count as a graded 0
]
EXPECTED_AVG = 0.5  # mean(1.0, 0.0) over the two graded episodes
EXPECTED_UNGRADED = 1


def _fail(msg: str) -> None:
    print(f"SMOKE FAIL: verifier_ran_scoring — {msg}")
    sys.exit(1)


def _check_live_summary(root: Path) -> None:
    storage = FileStorage(root)
    for tid, reward, verifier_ran in EPISODES:
        traj = Trajectory(
            id=f"{tid}_ep0",
            metadata={"task_id": tid, "agent_name": "A"},
            reward_info={"reward": reward, "verifier_ran": verifier_ran},
            summary_stats={"final_reward": reward},
        )
        storage.update_experiment_summary(traj)
    summary = json.loads((root / "experiment_summary.json").read_text())
    if summary["n_ungraded"] != EXPECTED_UNGRADED:
        _fail(f"live: n_ungraded={summary['n_ungraded']} != {EXPECTED_UNGRADED}")
    if summary["avg_reward"] != EXPECTED_AVG:
        _fail(f"live: avg_reward={summary['avg_reward']} != {EXPECTED_AVG} (ungraded leaked into accuracy?)")
    print(f"  live summary: avg_reward={summary['avg_reward']} n_ungraded={summary['n_ungraded']} ✓")


def _check_report_path(root: Path) -> None:
    storage = FileStorage(root)
    for tid, reward, verifier_ran in EPISODES:
        ep_dir = root / "episodes" / f"{tid}_ep0"
        ep_dir.mkdir(parents=True, exist_ok=True)
        storage.write_episode_status(
            f"{tid}_ep0",
            EpisodeStatus(status="COMPLETED", task_id=tid, episode_id=0, started_at=1.0, reward=reward),
        )
        (ep_dir / "episode.metadata.json").write_text(
            json.dumps({"reward_info": {"reward": reward, "verifier_ran": verifier_ran}})
        )
    _counts, rewards, _cost, _ctx, n_ungraded = _scan_episodes(ExperimentResult(root))
    if n_ungraded != EXPECTED_UNGRADED:
        _fail(f"report: n_ungraded={n_ungraded} != {EXPECTED_UNGRADED}")
    if len(rewards) != 2 or sum(rewards) / len(rewards) != EXPECTED_AVG:
        _fail(f"report: graded rewards={rewards} (mean != {EXPECTED_AVG}; ungraded leaked in?)")
    print(f"  report path: graded_rewards={rewards} mean={sum(rewards) / len(rewards)} n_ungraded={n_ungraded} ✓")


def main() -> None:
    with tempfile.TemporaryDirectory() as live_root, tempfile.TemporaryDirectory() as report_root:
        _check_live_summary(Path(live_root))
        _check_report_path(Path(report_root))
    print("SMOKE OK: verifier_ran_scoring")


if __name__ == "__main__":
    main()
