# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "math-tool-use",
#     "typer",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# math-tool-use = { path = "../cubes/math-tool-use", editable = true }
# ///
"""Inference-only pass@K reflection probe (Experiment B).

Question: does the current policy *already* extract signal from its own feedback?
For N math problems, against an ALREADY-RUNNING OpenAI-compatible vLLM endpoint
(no training), compare three numbers:

  - pass@1            : fraction correct on the first attempt
  - pass@K_independent: fraction with >=1 correct over K independent attempts (resampling)
  - pass@K_reflection : fraction with >=1 correct over K attempts, where attempt i+1 is
                        conditioned on a reflection generated over attempt i's outcome

Decision rule (see ~/dev/plans_rl/research/meta-rl-self-reflection.md):
  pass@K_reflection > pass@K_independent  -> the model uses its feedback; RL can amplify it.
  pass@K_reflection ~= pass@K_independent -> the model ignores feedback; RL faces a cold start.

Both arms run through ``run_multi_episode_rollout`` — the SAME helper PipelineRL training uses
(``cube_rl.domain._rollout``): the independent arm with a plain ``TirAgent`` (no memory -> K
i.i.d. resamples), the reflection arm with a ``LaMerTirAgent`` (file-backed cross-episode memory +
``finalize`` reflection). So the reflection arm measures the EXACT training rollout. Consequences:
each arm runs its own attempt 1 (no shared attempt-1; pass@1 is the independent arm's first
attempt), the refl-vs-indep comparison is unpaired, and early-stop is the cube reward (not the
probe's content-aware score) so a content-correct-but-unboxed attempt still triggers a
format-fixing reflection. Scoring stays content-aware (see _score_attempt).

Run with the live training env's interpreter, e.g.:
  /home/toolkit/dev/rl/PipelineRL/.venv/bin/python recipes/passk_reflection_probe.py \
      --n-problems 50 --k 3 --api-base http://localhost:8080/v1
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import typer

# The cube's module-level registry reads this cap at IMPORT time, so it must be set before
# importing math_tool_use. We cap at HOLDOUT_END so install() only materialises the first
# HOLDOUT_END tasks/dataset — enough to cover the held-out test block, no 130k-file blow-up.
#
# Train/test split is defined HERE (recipe-side, not in the cube framework): training consumes
# ORZ-57k from index 0 (MATH_TOOL_MAX_TASKS_PER_DATASET<=HOLDOUT_START), so reserving the
# contiguous block [HOLDOUT_START, HOLDOUT_END) as the held-out TEST set keeps it disjoint from
# training by construction. Convention to uphold: training must NOT consume indices >= HOLDOUT_START.
_HOLDOUT_START = 2000  # training uses [0, 2000); test is reserved above it
_HOLDOUT_END = 4000  # → ~2000 held-out test problems
os.environ.setdefault("MATH_TOOL_MAX_TASKS_PER_DATASET", str(_HOLDOUT_END))

from math_tool_use import (  # noqa: E402
    MATH_TIR_SYSTEM_PROMPT,
    REFLECTION_VARIANTS,
    MathReflectionProviderConfig,
    extract_attempt_summary,
)
from math_tool_use.benchmark import MathToolUseBenchmarkConfig  # noqa: E402
from math_tool_use.task import _verify_answer_status  # noqa: E402
from math_tool_use.tool import MathToolUseToolConfig  # noqa: E402

from cube_harness.agents.lamer import LaMerTirAgentConfig, run_multi_episode_rollout  # noqa: E402
from cube_harness.agents.tir import TirAgentConfig  # noqa: E402
from cube_harness.core import AgentOutput, Trajectory  # noqa: E402
from cube_harness.llm import DummyRouter, RoutedLLMConfig  # noqa: E402

logger = logging.getLogger("passk_probe")

# Probe and training share ONE system prompt (math_tool_use.MATH_TIR_SYSTEM_PROMPT) so the probe
# measures the policy under the exact prompt it's trained with. Training pulls the same constant via
# the math_tir_system_prompt Hydra factory in conf/cube_math_lamer_rl.yaml. Edit it in the cube.
BASE_SYSTEM_PROMPT = MATH_TIR_SYSTEM_PROMPT


def _final_assistant_content(trajectory: Any) -> str:
    """The last non-empty assistant message content in the episode — where base models
    often place their final \\boxed{} answer without ever calling the MathAnswer tool.

    Skips the LaMer agent's end-of-episode reflection (tag=="reflection"): that step is appended
    AFTER the solve and would otherwise shadow the real solve content for content-based scoring.
    """
    last = ""
    for step in getattr(trajectory, "steps", []):
        if not isinstance(step.output, AgentOutput):
            continue
        for call in step.output.llm_calls:
            if getattr(call, "tag", None) == "reflection":
                continue
            content = getattr(call.output, "content", None)
            if isinstance(content, str) and content.strip():
                last = content.strip()
    return last


def _score_attempt(trajectory: Any, reward_info: dict, expected: str, scoring: str = "content") -> dict:
    """Score an attempt. ``cube_success`` and ``content_correct`` are ALWAYS recorded (so a run is
    re-scorable either way); ``solved`` — what the pass@K metrics use — depends on ``scoring``:

    - ``"content"`` (default): solved = cube success OR a content-correct \\boxed{} in the final
      message. Decouples math/reflection ability from MathAnswer-protocol compliance — base models
      often answer correctly in chat text without calling MathAnswer (cube scores it no_answer), and
      this rescues those so the probe measures reasoning, not tool-protocol adherence.
    - ``"strict"``: solved = cube reward only (correct AND submitted via MathAnswer in \\boxed{}).
    """
    submitted, _reasoning = extract_attempt_summary(trajectory)
    final_content = _final_assistant_content(trajectory)
    cube_success = bool(reward_info.get("success", False))
    content_correct = (
        _verify_answer_status(final_content, expected, strict=True, max_prediction_length=2000) == "correct"
    )
    solved = cube_success if scoring == "strict" else (cube_success or content_correct)
    return {
        "cube_success": cube_success,
        "content_correct": content_correct,
        "solved": solved,
        "submitted_mathanswer": submitted,
        "final_content": final_content[-1500:],
    }


def _save_transcripts(transcript_dir: Path, task_id: str, expected: str, attempts: list[dict]) -> None:
    """Persist per-attempt scoring + final content so a run is auditable / re-scorable."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{task_id}.json").write_text(
        json.dumps({"task_id": task_id, "expected": expected, "attempts": attempts}, indent=2)
    )


def _task_index(task_id: str) -> int:
    """Trailing integer index of a math-tool-use task id (e.g. 'open_reasoner_zero_57k_2317' -> 2317)."""
    try:
        return int(str(task_id).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def _make_agent_config(llm_config: RoutedLLMConfig, system_prompt: str, max_actions: int) -> TirAgentConfig:
    return TirAgentConfig(llm_config=llm_config, system_prompt=system_prompt, max_actions=max_actions)


def _reflection_text(trajectory: Trajectory) -> str | None:
    """The end-of-episode reflection prose, if the LaMer agent emitted one this episode."""
    for step in getattr(trajectory, "steps", []):
        if isinstance(step.output, AgentOutput) and any(
            getattr(c, "tag", None) == "reflection" for c in step.output.llm_calls
        ):
            return step.output.thoughts
    return None


def _safe_rollout(
    agent_config: Any,
    task_config: Any,
    runtime_context: Any,
    k: int,
    exp_name: str,
) -> list[Trajectory] | None:
    """One arm = up to K sequential episodes via the shared training helper.

    Returns ``None`` if the rollout raised (counted as one arm-level error). Early-stop is the
    helper default (cube reward > 0) — the SAME rule training uses — so a content-correct-but-
    unboxed attempt (cube reward 0) does NOT stop the arm, giving the reflection arm a chance to
    fix the format on the next attempt; content-aware scoring happens afterwards in _score_attempt.
    """
    try:
        return run_multi_episode_rollout(
            agent_config, task_config, runtime_context, k, exp_name=exp_name, persist_episode=False
        )
    except Exception:
        logger.exception("%s rollout raised", exp_name)
        return None


def _probe_one_problem(
    task_config: Any,
    runtime_context: Any,
    base_agent_config: TirAgentConfig,
    lamer_agent_config: LaMerTirAgentConfig,
    k: int,
    transcript_dir: Path,
    scoring: str = "content",
) -> dict:
    """Independent arm + reflection arm for one problem, each a full run_multi_episode_rollout.

    Both arms run through the SAME helper PipelineRL training uses: the independent arm with a
    plain ``TirAgent`` (no memory -> K i.i.d. resamples), the reflection arm with a
    ``LaMerTirAgent`` (file-backed cross-episode memory + finalize reflection). There is no shared
    attempt 1 — each arm runs its own — so pass@1 is the independent arm's first (base-prompt)
    attempt. The solved criterion follows ``scoring`` (see _score_attempt); per-attempt scoring is
    persisted under ``transcript_dir/<task_id>.json`` for auditing / re-scoring.
    """
    exec_info = task_config.load_task_execution_info()
    expected = exec_info["expected"]
    task_id = str(getattr(task_config.metadata, "id", "?"))
    rec: dict[str, Any] = {"task_id": task_id, "errors": 0}

    indep_trajs = _safe_rollout(base_agent_config, task_config, runtime_context, k, "passk_indep")
    refl_trajs = _safe_rollout(lamer_agent_config, task_config, runtime_context, k, "passk_refl")
    rec["errors"] += int(indep_trajs is None) + int(refl_trajs is None)
    indep_trajs, refl_trajs = indep_trajs or [], refl_trajs or []

    indep_scores = [_score_attempt(t, t.reward_info or {}, expected, scoring) for t in indep_trajs]
    refl_scores = [_score_attempt(t, t.reward_info or {}, expected, scoring) for t in refl_trajs]

    attempts: list[dict] = []
    for i, s in enumerate(indep_scores):
        attempts.append({"arm": "independent", "attempt": i + 1, **s})
    for i, (t, s) in enumerate(zip(refl_trajs, refl_scores)):
        attempts.append({"arm": "reflection", "attempt": i + 1, "reflection_emitted": _reflection_text(t), **s})

    indep_first = next((i + 1 for i, s in enumerate(indep_scores) if s["solved"]), None)
    refl_first = next((i + 1 for i, s in enumerate(refl_scores) if s["solved"]), None)
    rec.update(
        pass1=bool(indep_scores and indep_scores[0]["solved"]),
        indep_success=indep_first is not None,
        indep_first=indep_first,
        refl_success=refl_first is not None,
        refl_first=refl_first,
        content_only_solve=any(a.get("content_correct") and not a.get("cube_success") for a in attempts),
    )
    _save_transcripts(transcript_dir, task_id, expected, attempts)
    return rec


def _summarize(records: list[dict], k: int, scoring: str) -> dict:
    n = len(records)
    pass1 = sum(r["pass1"] for r in records) / n
    pass_k_indep = sum(r["indep_success"] for r in records) / n
    pass_k_refl = sum(r["refl_success"] for r in records) / n
    indep_firsts = [r["indep_first"] for r in records if r["indep_success"]]
    refl_firsts = [r["refl_first"] for r in records if r["refl_success"]]
    # Cumulative pass@m curve for m=1..k: fraction of problems solved within m attempts. The
    # independent arm = i.i.d. resampling (the only way to get pass@m>1 for a single-episode/baseline
    # model); the reflection arm = reflection-conditioned retries (LaMer's trained mode). Use
    # first-solved attempt index (1-based, None if never), so pass@m = fraction with first <= m.
    pass_curve: dict[str, float] = {}
    for m in range(1, k + 1):
        pass_curve[f"pass@{m}_independent"] = round(
            sum(1 for r in records if r["indep_first"] is not None and r["indep_first"] <= m) / n, 4
        )
        pass_curve[f"pass@{m}_reflection"] = round(
            sum(1 for r in records if r["refl_first"] is not None and r["refl_first"] <= m) / n, 4
        )
    return {
        "n_problems": n,
        "k": k,
        "scoring": scoring,
        "pass@1": round(pass1, 4),
        "pass@K_independent": round(pass_k_indep, 4),
        "pass@K_reflection": round(pass_k_refl, 4),
        "delta_reflection_minus_independent": round(pass_k_refl - pass_k_indep, 4),
        "mean_first_success_independent": round(sum(indep_firsts) / len(indep_firsts), 3) if indep_firsts else None,
        "mean_first_success_reflection": round(sum(refl_firsts) / len(refl_firsts), 3) if refl_firsts else None,
        "problems_with_content_only_solve": sum(1 for r in records if r.get("content_only_solve")),
        "episodes_errored": sum(r["errors"] for r in records),
        **pass_curve,
    }


def main(
    n_problems: int = typer.Option(50, help="Number of math problems to probe."),
    k: int = typer.Option(3, help="Max attempts per problem (pass@K)."),
    api_base: str = typer.Option("http://localhost:8000/v1", help="OpenAI-compatible vLLM endpoint."),
    model: str = typer.Option("openai/Qwen3-4B-Instruct-2507", help="served_model_name (LiteLLM form)."),
    tokenizer: str = typer.Option("/home/toolkit/models/Qwen3-4B-Instruct-2507", help="Tokenizer path/name."),
    sandbox_endpoint: str = typer.Option(
        "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-sbxf1780318122",
        help="SandboxFusion endpoint for run_python_code.",
    ),
    dataset: str = typer.Option("open_reasoner_zero_57k", help="Task-id prefix to sample problems from."),
    holdout_start: int = typer.Option(_HOLDOUT_START, help="Held-out TEST block start index (inclusive)."),
    holdout_end: int = typer.Option(_HOLDOUT_END, help="Held-out TEST block end index (exclusive)."),
    temperature: float = typer.Option(0.7, help="Sampling temperature for attempts."),
    max_actions: int = typer.Option(3, help="Agent action budget per episode."),
    max_completion_tokens: int = typer.Option(4096, help="Per-generation token budget."),
    max_model_len: int = typer.Option(32000, help="Context window (match the endpoint)."),
    seed: int = typer.Option(0, help="RNG seed for problem sampling."),
    reflection_variant: str = typer.Option(
        "v0_baseline",
        help=f"Self-reflection prompt variant. One of: {', '.join(REFLECTION_VARIANTS)}.",
    ),
    use_verdict: bool = typer.Option(
        False,
        help="Use --use-verdict to inject a hand-built failure diagnosis into the reflection prompt. "
        "Default (off): the reflector self-diagnoses, given the task + format contract.",
    ),
    scoring: str = typer.Option(
        "content",
        help="Solved criterion for pass@K: 'content' (cube success OR content-correct \\boxed{} in "
        "the final message — rescues unboxed-correct, measures reasoning) or 'strict' (cube reward "
        "only: correct AND submitted via MathAnswer). Both are recorded per-attempt regardless.",
    ),
    enable_thinking: bool | None = typer.Option(
        None,
        help="Hybrid-thinking toggle passed via chat_template_kwargs (Qwen3 originals). "
        "Use --no-enable-thinking to force non-thinking; omit for models without the toggle.",
    ),
    out: Path = typer.Option(
        Path("/home/toolkit/dev/plans_rl/research/passk_probe_results.json"),
        help="Where to write full results JSON.",
    ),
) -> None:
    """Run the pass@1 / pass@K-independent / pass@K-reflection probe and write results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if reflection_variant not in REFLECTION_VARIANTS:
        raise typer.BadParameter(
            f"unknown --reflection-variant {reflection_variant!r}; choose from {list(REFLECTION_VARIANTS)}"
        )
    if scoring not in ("content", "strict"):
        raise typer.BadParameter(f"--scoring must be 'content' or 'strict', got {scoring!r}")

    # Forward enable_thinking to the chat template only when explicitly set (models without the
    # toggle — the 2507 anchor, Nemotron instruct — get no kwarg, avoiding template errors).
    extra_body: dict = {}
    if enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    llm_config = RoutedLLMConfig(
        model_name=model,
        router=DummyRouter(),
        tokenizer_name=tokenizer,
        api_key="EMPTY",
        api_base=api_base,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        max_model_len=max_model_len,
        extra_body=extra_body,
        training=True,  # disable caching for honest sampling
        skip_special_tokens=True,
        include_stop_str_in_output=False,
    )
    base_agent_config = _make_agent_config(llm_config, BASE_SYSTEM_PROMPT, max_actions)
    # Reflection arm = the exact training rollout: LaMerTirAgent (file memory + finalize reflection)
    # with the SAME base system prompt, so its episode-0 matches the independent arm's base attempt.
    lamer_agent_config = LaMerTirAgentConfig(
        llm_config=llm_config,
        system_prompt=BASE_SYSTEM_PROMPT,
        max_actions=max_actions,
        reflection_provider=MathReflectionProviderConfig(
            reflection_variant=reflection_variant, use_verdict=use_verdict
        ),
    )

    tool_config = MathToolUseToolConfig(sandbox_endpoint=sandbox_endpoint)
    bench_config = MathToolUseBenchmarkConfig(tool_config=tool_config)
    bench_config.install()

    all_task_configs = list(bench_config.get_task_configs())
    # Held-out TEST set = reserved contiguous index block [holdout_start, holdout_end) of `dataset`.
    # Disjoint from training (which consumes from index 0 up to <= holdout_start).
    pool = [
        tc
        for tc in all_task_configs
        if str(getattr(tc.metadata, "id", "")).startswith(dataset)
        and holdout_start <= _task_index(getattr(tc.metadata, "id", "")) < holdout_end
    ]
    if not pool:
        raise typer.BadParameter(
            f"No held-out tasks for {dataset!r} in index block [{holdout_start},{holdout_end}); "
            f"have {len(all_task_configs)} task configs (is MATH_TOOL_MAX_TASKS_PER_DATASET >= {holdout_end}?)."
        )
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = pool[:n_problems]
    logger.info(
        "held-out TEST block [%d,%d) of %s: %d problems, sampling %d (seed=%d)",
        holdout_start,
        holdout_end,
        dataset,
        len(pool),
        min(n_problems, len(pool)),
        seed,
    )
    logger.info("Probing %d/%d problems from %r, K=%d, temp=%.2f", len(chosen), len(pool), dataset, k, temperature)

    transcript_dir = out.parent / f"transcripts_{out.stem}"
    records: list[dict] = []
    t0 = time.time()
    with bench_config.make() as benchmark:
        runtime_context = benchmark._runtime_context
        for idx, tc in enumerate(chosen):
            rec = _probe_one_problem(
                tc,
                runtime_context,
                base_agent_config,
                lamer_agent_config,
                k,
                transcript_dir,
                scoring,
            )
            records.append(rec)
            logger.info(
                "[%d/%d] %s pass1=%s indep=%s refl=%s errors=%d",
                idx + 1,
                len(chosen),
                rec["task_id"],
                rec["pass1"],
                rec["indep_success"],
                rec["refl_success"],
                rec["errors"],
            )

    summary = _summarize(records, k, scoring)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["model"] = model
    summary["dataset"] = dataset
    summary["temperature"] = temperature
    summary["holdout_block"] = [holdout_start, holdout_end]
    summary["enable_thinking"] = enable_thinking
    summary["reflection_variant"] = reflection_variant
    summary["use_verdict"] = use_verdict
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))

    print("\n" + "=" * 60)
    print(f"pass@K reflection probe  (N={summary['n_problems']}, K={k}, model={model})")
    print("=" * 60)
    print(f"  pass@1               : {summary['pass@1']:.3f}")
    print(f"  pass@K independent   : {summary['pass@K_independent']:.3f}")
    print(f"  pass@K reflection    : {summary['pass@K_reflection']:.3f}")
    print(
        f"  delta (refl - indep) : {summary['delta_reflection_minus_independent']:+.3f}  [variant={reflection_variant}]"
    )
    print("  pass@m curve (cumulative, m=1..K):")
    print(f"    {'m':>3}  {'independent':>12}  {'reflection':>12}")
    for m in range(1, k + 1):
        print(f"    {m:>3}  {summary[f'pass@{m}_independent']:>12.3f}  {summary[f'pass@{m}_reflection']:>12.3f}")
    print(
        f"  mean 1st-success ep  : indep={summary['mean_first_success_independent']}  refl={summary['mean_first_success_reflection']}"
    )
    print(
        f"  content-only solves  : {summary['problems_with_content_only_solve']}  (correct \\boxed in chat, cube scored no_answer)"
    )
    print(f"  episodes errored     : {summary['episodes_errored']}")
    print(f"  elapsed              : {summary['elapsed_s']}s")
    print(f"  scoring              : {summary['scoring']}")
    print(f"  results -> {out}   transcripts -> {transcript_dir}")
    print("=" * 60)


if __name__ == "__main__":
    typer.run(main)
