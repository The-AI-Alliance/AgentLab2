"""Convert an experiment :class:`EvalLog` into an Every Eval Ever (EEE) record.

EEE (`<https://evalevalai.com/events/shared-task-every-eval-ever/>`_) is a
community shared task building a unified HF-hosted database of LLM
evaluations. cube-harness becomes a peer of HELM / lm-eval / Inspect AI by
emitting records that conform to ``every_eval_ever/schemas/eval.schema.json``
(v0.2.2 at time of writing).

The schema is strict (``additionalProperties: false`` at the top), so
cube-specific provenance (git hashes, dep versions, agent config hash, cube-
standard checkout state) goes into ``source_metadata.additional_details`` as
string KVs. Aggregate score + uncertainty go into one
``evaluation_results[]`` entry; per-episode data is not emitted (EEE has
``detailed_evaluation_results`` for that — left optional for V1).

This module is pure — no I/O beyond the experiment directory.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from cube_harness.eval_log import EvalLog

EEE_SCHEMA_VERSION = "0.2.2"


def _to_string_kvs(d: dict[str, Any]) -> dict[str, str]:
    """EEE's ``additional_details`` requires string-typed values."""
    return {k: str(v) for k, v in d.items() if v is not None}


def _llm_developer(llm_model: str | None) -> str:
    """Best-effort extraction of the model developer from a LiteLLM-style id.

    ``azure/gpt-5.4-mini`` → ``Azure``;  ``anthropic/claude-opus-4-7`` →
    ``Anthropic``.  Falls back to ``""`` when the prefix is missing.
    """
    if not llm_model or "/" not in llm_model:
        return ""
    return llm_model.split("/", 1)[0].title()


def build_eee_record(
    experiment_dir: Path,
    *,
    source_organization_name: str = "cube-harness submitter",
    source_organization_url: str | None = None,
    evaluator_relationship: str = "third_party",
) -> dict[str, Any]:
    """Build one EEE-shaped record from a completed experiment.

    Args:
        experiment_dir: Path to the experiment output directory.
        source_organization_name: Free-text identifier for the submitter (a
            person or org); shown in EEE's UI.
        source_organization_url: Optional URL for the org.
        evaluator_relationship: One of ``first_party``, ``third_party``,
            ``collaborative``, ``other``. Defaults to ``third_party`` because
            the typical cube-harness user is not the model developer.

    Returns:
        A dict matching EEE ``eval.schema.json`` v0.2.2.
    """
    experiment_dir = Path(experiment_dir)
    eval_log = EvalLog.load(experiment_dir)
    exp = eval_log.experiment
    episodes = eval_log.episodes

    scores = [ep.score for ep in episodes]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    std_err = statistics.stdev(scores) / (len(scores) ** 0.5) if len(scores) > 1 else 0.0

    bench_name = exp.benchmark_subset.name
    cube_id = bench_name.split("[", 1)[0]

    provenance = _to_string_kvs(
        {
            "cube_harness_version": exp.eval_library.version,
            "cube_harness_git_commit": exp.agent.git_commit,
            "cube_harness_git_remote_url": exp.agent.git_remote_url,
            "cube_harness_git_is_dirty": exp.agent.git_is_dirty,
            "cube_standard_git_commit": exp.agent.cube_standard_git_commit,
            "cube_standard_git_is_dirty": exp.agent.cube_standard_git_is_dirty,
            "agent_id": exp.agent.agent_id,
            "agent_config_type": exp.agent.config_type,
            **{f"dep:{k}": v for k, v in exp.agent.dependency_versions.items()},
        }
    )

    source_metadata: dict[str, Any] = {
        "source_type": "evaluation_run",
        "source_organization_name": source_organization_name,
        "evaluator_relationship": evaluator_relationship,
        "additional_details": provenance,
    }
    if source_organization_url:
        source_metadata["source_organization_url"] = source_organization_url

    model_info: dict[str, Any] = {
        "name": exp.agent.llm_model or "",
        "id": exp.agent.llm_model or "",
    }
    developer = _llm_developer(exp.agent.llm_model)
    if developer:
        model_info["developer"] = developer

    record: dict[str, Any] = {
        "schema_version": EEE_SCHEMA_VERSION,
        "evaluation_id": f"{cube_id}/{exp.agent.llm_model}/{exp.evaluation_id}",
        "retrieved_timestamp": str(int(exp.evaluation_timestamp)),
        "evaluation_timestamp": str(int(exp.evaluation_timestamp)),
        "source_metadata": source_metadata,
        "eval_library": {
            "name": exp.eval_library.name,
            "version": exp.eval_library.version,
        },
        "model_info": model_info,
        "evaluation_results": [
            {
                "evaluation_result_id": f"{cube_id}.reward",
                "evaluation_name": bench_name,
                "source_data": {
                    "dataset_name": bench_name,
                    "source_type": "other",
                    "additional_details": {
                        "benchmark_version": exp.benchmark_version or "unknown",
                        "n_tasks": str(exp.benchmark_subset.n_tasks),
                    },
                },
                "metric_config": {
                    "metric_id": f"{cube_id}.reward",
                    "metric_name": "Mean episode reward",
                    "metric_kind": "accuracy",
                    "lower_is_better": False,
                    "score_type": "continuous",
                    "min_score": 0,
                    "max_score": 1,
                },
                "score_details": {
                    "score": round(avg_score, 6),
                    "uncertainty": {
                        "standard_error": {
                            "value": round(std_err, 6),
                            "method": "analytic",
                        },
                        "num_samples": len(scores),
                    },
                },
            }
        ],
    }
    return record
