"""Atlas EvalLog: two-level structured evaluation records for community-scale agent benchmarking.

Two files per experiment:
    experiment_record.json  — ExperimentRecord (once per experiment): agent, benchmark, git provenance
    episodes/<id>/episode_record.json  — one EpisodeRecord per episode, written after each episode

Records are plain JSON, no cube-harness dependency to read.

Classes:
    EvalLibrary       — library descriptor (name, version)
    UsageSummary      — aggregated LLM token/cost stats across an episode
    AgentInfo         — agent descriptor: config, dependency versions, git provenance
    BenchmarkSubset   — benchmark subset descriptor for MNAR propensity correction
    InvestigatorLLMConfig       — configuration of the investigator LLM (optional)
    BlameCategory     — closed-world taxonomy of failure causes
    Outcome           — outcome of the episode as investigated
    EvidenceItem      — step-indexed transcript quote backing a blame attribution
    BaseFindings   — per-episode investigator assessment, base shape (recipes extend it)
    Findings       — deprecated alias for BaseFindings
    InvestigationMetadata     — billing/provenance for a single investigator invocation (optional)
    Verifier          — task verifier reference (optional)
    ExperimentRecord  — experiment-level record written to experiment_record.json
    EpisodeRecord     — episode-level record written after each episode completes
    EvalLog           — two-level container with save/load
"""

import hashlib
import importlib.metadata
import json
import logging
import re
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any

import cube
from cube.benchmark import BenchmarkConfig
from cube.core import TypedBaseModel
from pydantic import Field

from cube_harness.core import Trajectory
from cube_harness.storage import EPISODES_DIR as _EPISODES_DIR

logger = logging.getLogger(__name__)

EPISODE_RECORD_FILENAME = "episode_record.json"
EXPERIMENT_RECORD_FILENAME = "experiment_record.json"

_TRACKED_PACKAGES: list[str] = [
    "cube-harness",
    "cube",
    "litellm",
    "anthropic",
    "openai",
    "browsergym-core",
    "playwright",
    "pydantic",
    "ray",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_dependency_versions() -> dict[str, str]:
    """Return installed versions for all tracked packages that are present."""
    return {pkg: v for pkg in _TRACKED_PACKAGES if (v := _get_package_version(pkg)) is not None}


def _to_github_url(remote_url: str, commit: str) -> str | None:
    """Convert a git remote URL (HTTPS or SSH) to a permanent GitHub commit URL."""
    ssh = re.match(r"git@github\.com:(.+?)(?:\.git)?$", remote_url)
    if ssh:
        return f"https://github.com/{ssh.group(1)}/tree/{commit}"
    https = re.match(r"https://github\.com/(.+?)(?:\.git)?$", remote_url)
    if https:
        return f"https://github.com/{https.group(1)}/tree/{commit}"
    return None


def _get_git_info(cwd: str | None = None) -> tuple[str | None, str | None, bool | None]:
    """Return (commit_sha, github_permalink, is_dirty) for the repo at cwd.

    All three values are None when git is unavailable or cwd is not inside a repo.
    is_dirty is True when uncommitted changes exist (result may not reproduce exactly
    from git_commit alone).
    """
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        return None, None, None

    try:
        remote = (
            subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=cwd, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        github_url = _to_github_url(remote, commit)
    except Exception:
        github_url = None

    try:
        is_dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL) != 0
    except Exception:
        is_dirty = None

    return commit, github_url, is_dirty


def _extract_llm_model(config_dict: dict) -> str | None:
    """Walk a serialized agent config dict looking for a model name field."""
    for key in ("model_name", "model", "llm_model"):
        if key in config_dict and isinstance(config_dict[key], str):
            return config_dict[key]
    for nested_key in ("llm_config", "llm"):
        nested = config_dict.get(nested_key)
        if isinstance(nested, dict):
            for key in ("model_name", "model"):
                if key in nested and isinstance(nested[key], str):
                    return nested[key]
    return None


def _extract_tool_names(tools: list[dict]) -> list[str]:
    """Extract action names from serialized action schemas.

    Handles both litellm format ({"type": "function", "function": {"name": ...}})
    and flat format ({"name": ...}).
    """
    names = []
    for tool in tools:
        fn = tool.get("function", {})
        if isinstance(fn, dict) and "name" in fn:
            names.append(fn["name"])
        elif "name" in tool:
            names.append(tool["name"])
    return names


def _extract_error_type(trajectory: Trajectory) -> str | None:
    """Return the error_type of the first StepError in the trajectory, or None."""
    for step in trajectory.steps:
        if hasattr(step.output, "error") and step.output.error is not None:
            return step.output.error.error_type
    return None


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class EvalLibrary(TypedBaseModel):
    """Library that produced the evaluation."""

    name: str = "cube-harness"
    version: str


class UsageSummary(TypedBaseModel):
    """Aggregated LLM token usage and cost across a complete episode."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_cache_read: int = 0
    input_tokens_cache_write: int = 0
    total_cost_usd: float = 0.0
    n_llm_calls: int = 0

    @classmethod
    def from_summary_stats(cls, stats: dict | None) -> "UsageSummary":
        """Build from the summary_stats dict stored on a Trajectory."""
        if not stats:
            return cls()
        prompt = stats.get("prompt_tokens", 0)
        completion = stats.get("completion_tokens", 0)
        return cls(
            input_tokens=prompt,
            output_tokens=completion,
            total_tokens=prompt + completion,
            input_tokens_cache_read=stats.get("cached_tokens", 0),
            input_tokens_cache_write=stats.get("cache_creation_tokens", 0),
            total_cost_usd=stats.get("cost", 0.0),
            n_llm_calls=stats.get("total_llm_calls", 0),
        )


class AgentInfo(TypedBaseModel):
    """Agent descriptor for Atlas embedding and reproducibility.

    Tools are NOT included here — they vary per episode due to task-level action filtering.
    See EpisodeRecord.tool_names for the per-episode tool list.
    """

    agent_id: str = Field(description="SHA-256 of the serialized agent config — stable unique identifier across runs.")
    config_type: str = Field(description="Agent config class name (from _type discriminator field).")
    config: dict = Field(description="Full serialized agent config (model_dump with serialize_as_any=True).")
    llm_model: str | None = Field(default=None, description="LLM model name extracted from config.")
    framework_version: str = Field(description="cube-harness version at eval time.")
    dependency_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Installed versions of tracked packages (cube-harness, litellm, anthropic, openai, ...).",
    )
    git_commit: str | None = Field(default=None, description="Git SHA-1 of the repo HEAD at eval time.")
    git_remote_url: str | None = Field(
        default=None,
        description="Permanent GitHub URL pointing to the exact commit (tree view). None if not on GitHub.",
    )
    git_is_dirty: bool | None = Field(
        default=None,
        description=(
            "True when uncommitted changes exist at eval time — result may not reproduce exactly "
            "from git_commit alone. None when git info is unavailable."
        ),
    )
    cube_standard_git_commit: str | None = Field(
        default=None,
        description=(
            "Git SHA-1 of the installed cube-standard (`cube`) package's repo HEAD. Populated only "
            "when cube-standard is an editable/source checkout (the common case while it tracks an "
            "unreleased branch); None for a released wheel — use dependency_versions['cube'] then."
        ),
    )
    cube_standard_git_is_dirty: bool | None = Field(
        default=None,
        description="True when the cube-standard checkout had uncommitted changes. None when unavailable.",
    )
    description: str | None = Field(
        default=None,
        description="Free-form prose description of the agent for Atlas LLM embedding warm-start.",
    )

    @classmethod
    def from_agent_config(
        cls,
        agent_config: Any,
        git_cwd: str | None = None,
    ) -> "AgentInfo":
        """Build AgentInfo from an agent config object.

        Args:
            agent_config: Any AgentConfig (TypedBaseModel subclass).
            git_cwd: Working directory for git commands. Defaults to CWD.
        """
        harness_version = _get_package_version("cube-harness") or "unknown"
        config_dict = json.loads(agent_config.model_dump_json(serialize_as_any=True))
        agent_id = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
        config_type = config_dict.get("_type", type(agent_config).__name__)
        llm_model = _extract_llm_model(config_dict)
        git_commit, git_remote_url, git_is_dirty = _get_git_info(cwd=git_cwd)
        # cube-standard is a separate repo; capturing its hash matters because it is
        # frequently run from an unreleased branch (dependency_versions only has the
        # PyPI version string). Probe the installed `cube` package's source dir —
        # yields a commit only for an editable/source checkout, None for a wheel.
        cube_standard_dir = str(Path(cube.__file__).resolve().parent)
        cube_standard_git_commit, _, cube_standard_git_is_dirty = _get_git_info(cwd=cube_standard_dir)

        return cls(
            agent_id=agent_id,
            config_type=config_type,
            config=config_dict,
            llm_model=llm_model,
            framework_version=harness_version,
            dependency_versions=_collect_dependency_versions(),
            git_commit=git_commit,
            git_remote_url=git_remote_url,
            git_is_dirty=git_is_dirty,
            cube_standard_git_commit=cube_standard_git_commit,
            cube_standard_git_is_dirty=cube_standard_git_is_dirty,
        )


class BenchmarkSubset(TypedBaseModel):
    """Benchmark subset descriptor for MNAR propensity correction.

    Automatically derived from the benchmark object. The name field captures any subset
    suffix applied via subset_from_glob (e.g., "[level=l1]") or subset_from_list.
    n_tasks is the denominator for computing completion rate without requiring the benchmark.
    """

    name: str = Field(description="Benchmark name including any subset suffix (benchmark_metadata.name).")
    n_tasks: int = Field(description="Total tasks in this subset — denominator for completion rate.")
    filter: str | None = Field(
        default=None,
        description="Glob expression if the subset was created via subset_from_glob.",
    )

    @classmethod
    def from_benchmark_config(cls, benchmark_config: BenchmarkConfig) -> "BenchmarkSubset":
        """Derive BenchmarkSubset from a cube BenchmarkConfig object."""
        name = benchmark_config.benchmark_metadata.name
        n_tasks = len(benchmark_config.task_metadata)
        return cls(name=name, n_tasks=n_tasks)


class InvestigatorLLMConfig(TypedBaseModel):
    """Configuration of the LLM investigator used for post-hoc episode assessment."""

    model: str = Field(description="Investigator model identifier (e.g. 'claude-opus-4-7').")
    prompt_version: str = Field(description="Version or hash of the investigator prompt template.")
    investigated_at: str | None = Field(default=None, description="ISO-8601 timestamp when the investigation was run.")


FINDINGS_SCHEMA_VERSION = "v1"


class BlameCategory(str, Enum):
    """Closed-world taxonomy of failure causes. The investigator must pick from this set or `none`."""

    task_unclear = "task_unclear"
    model_capability = "model_capability"
    tool_failure = "tool_failure"
    env_failure = "env_failure"
    agent_scaffolding = "agent_scaffolding"
    action_space_limited = "action_space_limited"
    insufficient_observation = "insufficient_observation"
    eval_brittle = "eval_brittle"
    submission_format = "submission_format"
    none = "none"


class Outcome(str, Enum):
    """What happened in the episode, beyond the binary reward."""

    success = "success"
    success_lucky = "success_lucky"
    almost = "almost"
    failure = "failure"
    should_have_been_rewarded = "should_have_been_rewarded"


class EvidenceItem(TypedBaseModel):
    """A step-indexed transcript quote backing a blame attribution."""

    step: int = Field(description="Step index in the trajectory.")
    quote: str = Field(description="Verbatim excerpt from the agent or environment output.")


class BaseFindings(TypedBaseModel):
    """Base per-episode assessment from a post-hoc LLM investigator.

    Each investigator recipe (general_blame, profiling, agent_scaffolding, ...) extends this
    base with use-case-specific fields. The cross-recipe core fields (analysis,
    outcome, summary, primary_blame, primary_blame_confidence) live here so that
    aggregate views (CSV report, cross-experiment joins) can flatten any recipe's
    output to a common schema.

    Field order is CoT-deliberate. Models token-emit in declared order, so:

      1. `analysis` — free-form scratchpad, full reasoning before commitment.
      2. `evidence` — cite specific transcript quotes that ground what comes next.
      3. `summary` — narrative of what happened, before categorizing.
      4. `outcome` — categorical commitment.
      5. `primary_blame` — attribute the dominant cause.
      6. `primary_blame_confidence` — score the attribution (after making it).
      7. `other_blames` — secondary causes (knowing the primary).
      8. `hypothesis` — propose the fix.
      9. `hypothesis_confidence` — score the fix (after proposing it).

    Pydantic accepts JSON keys in any order on parse, so this reorder does not
    break existing on-disk records.
    """

    analysis: str = Field(
        description="Multi-paragraph reasoning scratchpad. Filled first; grounds all structured fields below."
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Step-indexed quotes from the transcript. Required when primary_blame != 'none'.",
    )
    summary: str = Field(description="1-3 sentence description of what happened.")
    outcome: Outcome = Field(description="What happened in the episode beyond the binary reward.")
    primary_blame: BlameCategory = Field(description="Dominant failure cause; `none` for clean successes.")
    primary_blame_confidence: int = Field(
        ge=0, le=5, description="Confidence in primary_blame (0=no basis, 5=certain)."
    )
    other_blames: list[BlameCategory] = Field(
        default_factory=list,
        description="Secondary contributing causes. Must not repeat primary_blame.",
    )
    hypothesis: str = Field(description="1-2 sentences: what change would most likely fix this class of failure.")
    hypothesis_confidence: int = Field(
        ge=0, le=5, description="Confidence in the proposed fix (0=pure guess, 5=certain)."
    )


# Deprecated alias — `Findings` was renamed to `BaseFindings` to make the
# extension contract explicit. The `general_blame` recipe's `OutputModel` is the
# direct successor (identical on-disk shape). Kept for one release window so
# existing callers (investigation_report.py, downstream consumers) keep working.
Findings = BaseFindings


class InvestigationMetadata(TypedBaseModel):
    """Billing and provenance for a single investigator invocation. Sibling to `findings`."""

    model: str = Field(description="Investigator model identifier (e.g. 'claude-opus-4-7').")
    prompt_tokens: int = Field(default=0, description="Input tokens consumed by the investigator call.")
    completion_tokens: int = Field(default=0, description="Output tokens produced by the investigator call.")
    cost_usd: float = Field(default=0.0, description="Total USD cost of the investigator call.")
    duration_s: float = Field(default=0.0, description="Wall-clock duration of the investigator call in seconds.")
    timestamp: float = Field(description="Wall-clock time the investigator ran (Unix seconds).")
    findings_schema_version: str = Field(
        default=FINDINGS_SCHEMA_VERSION,
        description="Schema version of the Findings record produced — for forward compatibility.",
    )


class Verifier(TypedBaseModel):
    """Task verifier reference for reproducibility and post-hoc inspection."""

    ref: str | None = Field(
        default=None,
        description="Permanent GitHub URL pointing to the verifier function at the exact commit.",
    )
    source: str | None = Field(
        default=None,
        description="Verifier source code at eval time (for auditing without git access).",
    )


class ExperimentRecord(TypedBaseModel):
    """Experiment-level record. Written once to experiment_record.json at experiment start.

    Contains all fields shared across every episode: agent description, benchmark
    metadata, git provenance. EpisodeRecord links to this via evaluation_id.
    """

    evaluation_id: str = Field(
        description="output_dir.name — unique per run (includes UUID suffix from make_experiment_output_dir)."
    )
    experiment_name: str = Field(description="Experiment name as set in Experiment.name.")
    evaluation_timestamp: float = Field(description="Experiment start time as Unix timestamp.")
    eval_library: EvalLibrary = Field(description="Library that produced the evaluation.")
    agent: AgentInfo = Field(description="Agent descriptor (config, dependency versions, git provenance).")
    benchmark_name: str = Field(description="Benchmark name from benchmark_metadata.name.")
    benchmark_version: str | None = Field(default=None, description="Benchmark version string.")
    benchmark_subset: BenchmarkSubset = Field(description="Subset descriptor for MNAR propensity correction.")
    investigator_llm_config: InvestigatorLLMConfig | None = Field(
        default=None,
        description="Investigator configuration if a post-hoc LLM investigator was run on these episodes.",
    )

    @classmethod
    def from_experiment(
        cls,
        exp_name: str,
        output_dir: Path,
        agent_config: Any,
        benchmark_config: BenchmarkConfig,
        git_cwd: str | None = None,
    ) -> "ExperimentRecord":
        """Build ExperimentRecord from experiment parameters."""
        harness_version = _get_package_version("cube-harness") or "unknown"
        agent_info = AgentInfo.from_agent_config(agent_config, git_cwd=git_cwd)
        bm_metadata = benchmark_config.benchmark_metadata
        bm_name = bm_metadata.name
        bm_version = bm_metadata.version

        return cls(
            evaluation_id=Path(output_dir).name,
            experiment_name=exp_name,
            evaluation_timestamp=time.time(),
            eval_library=EvalLibrary(version=harness_version),
            agent=agent_info,
            benchmark_name=bm_name,
            benchmark_version=bm_version,
            benchmark_subset=BenchmarkSubset.from_benchmark_config(benchmark_config),
        )

    def write(self, output_dir: Path) -> None:
        """Write experiment_record.json to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / EXPERIMENT_RECORD_FILENAME
        path.write_text(self.model_dump_json(indent=2))
        logger.info(f"Saved experiment record to {path}")


class EpisodeRecord(TypedBaseModel):
    """Episode-level record. Written to episodes/<trajectory_id>/episode_record.json after each episode.

    Links to ExperimentRecord via evaluation_id. Contains all episode-specific fields:
    task identity, per-episode tool list, outcome, usage, and optional investigator output.
    """

    evaluation_id: str = Field(description="FK → ExperimentRecord.evaluation_id.")
    sample_id: str = Field(description="Unique task identifier within the benchmark.")
    sample_hash: str | None = Field(
        default=None,
        description="SHA-256 of TaskConfig JSON. Detects task drift across benchmark versions.",
    )
    seed: int | None = Field(default=None, description="Random seed for this task instance.")
    split: str | None = Field(default=None, description="Dataset split: 'train', 'val', or 'test'.")
    task_description: str | None = Field(
        default=None,
        description="Abstract task description from TaskMetadata.abstract_description.",
    )
    tool_names: list[str] = Field(
        default_factory=list,
        description=(
            "Action names available during this episode. Episode-specific: same agent gets "
            "different tools on different tasks due to task-level action filtering."
        ),
    )
    is_correct: bool = Field(description="True when final score > 0.")
    score: float = Field(description="Final reward from the last EnvironmentOutput.")
    error: str | None = Field(
        default=None,
        description="Exception class name if any step raised an error. None for clean episodes.",
    )
    num_turns: int = Field(description="Total trajectory steps (agent + env combined).")
    n_agent_steps: int = Field(description="Agent steps (LLM decision turns).")
    n_env_steps: int = Field(description="Environment steps (tool executions).")
    wall_time_s: float | None = Field(default=None, description="Episode wall-clock duration in seconds.")
    usage: UsageSummary = Field(
        default_factory=UsageSummary,
        description="Aggregated LLM token usage and cost for the episode.",
    )
    trajectory_id: str = Field(description="Trajectory ID as stored on disk.")
    timestamp: float = Field(description="Episode start time as Unix timestamp.")
    verifier: Verifier | None = Field(
        default=None,
        description="Task verifier reference for reproducibility and post-hoc inspection.",
    )
    findings: BaseFindings | None = Field(
        default=None,
        description=(
            "Per-episode LLM investigator assessment (outcome, blame, evidence, hypothesis). "
            "Concrete shape depends on the recipe used; the base fields are always present."
        ),
    )
    investigation_metadata: InvestigationMetadata | None = Field(
        default=None,
        description="Billing/provenance for the investigator invocation. None until a investigator has run.",
    )

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        evaluation_id: str,
        task_metadata: Any | None = None,
        task_config: Any | None = None,
    ) -> "EpisodeRecord":
        """Assemble an EpisodeRecord from a completed trajectory."""
        sample_id = str(trajectory.metadata.get("task_id", ""))
        action_schemas: list[dict] = trajectory.metadata.get("action_schemas", [])
        tool_names = _extract_tool_names(action_schemas)

        last_env = trajectory.last_env_step()
        score = last_env.reward
        stats = trajectory.summary_stats or {}

        wall_time_s: float | None = None
        if trajectory.start_time is not None and trajectory.end_time is not None:
            wall_time_s = trajectory.end_time - trajectory.start_time

        sample_hash: str | None = None
        seed: int | None = None
        if task_config is not None:
            sample_hash = hashlib.sha256(task_config.model_dump_json(serialize_as_any=True).encode()).hexdigest()
            seed = getattr(task_config, "seed", None)

        split: str | None = None
        task_description: str | None = None
        if task_metadata is not None:
            split = getattr(task_metadata, "split", None)
            task_description = getattr(task_metadata, "abstract_description", None) or None

        return cls(
            evaluation_id=evaluation_id,
            sample_id=sample_id,
            sample_hash=sample_hash,
            seed=seed,
            split=split,
            task_description=task_description,
            tool_names=tool_names,
            is_correct=score > 0,
            score=score,
            error=_extract_error_type(trajectory),
            num_turns=len(trajectory.steps),
            n_agent_steps=stats.get("n_agent_steps", trajectory.n_agent_steps),
            n_env_steps=stats.get("n_env_steps", trajectory.n_env_steps),
            wall_time_s=wall_time_s,
            usage=UsageSummary.from_summary_stats(stats),
            trajectory_id=trajectory.id,
            timestamp=trajectory.start_time or 0.0,
        )

    def write(self, output_dir: Path) -> None:
        """Write episode_record.json to episodes/<trajectory_id>/ inside output_dir."""
        ep_dir = Path(output_dir) / _EPISODES_DIR / self.trajectory_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / EPISODE_RECORD_FILENAME).write_text(self.model_dump_json(indent=2))


class EvalLog(TypedBaseModel):
    """Two-level eval log container.

    Experiment-level data goes to experiment_record.json (written once at experiment start).
    Episode-level data goes to episodes/<trajectory_id>/episode_record.json
    (one file per episode, co-located with the trajectory, written after each episode).

    All files are plain JSON, readable without a cube-harness dependency.

    For ATLAS submission, call to_jsonl() to aggregate episode records into a
    single flat JSONL file.
    """

    experiment: ExperimentRecord
    episodes: list[EpisodeRecord] = Field(default_factory=list)

    def save(self, output_dir: Path) -> None:
        """Write experiment_record.json and per-trajectory episode_record.json files."""
        output_dir = Path(output_dir)
        self.experiment.write(output_dir)
        for record in self.episodes:
            record.write(output_dir)
        logger.info(f"Saved {len(self.episodes)} episode records under {output_dir / _EPISODES_DIR}")

    @classmethod
    def load(cls, output_dir: Path) -> "EvalLog":
        """Load experiment_record.json and all per-trajectory episode_record.json files."""
        output_dir = Path(output_dir)
        experiment = ExperimentRecord.model_validate_json((output_dir / EXPERIMENT_RECORD_FILENAME).read_text())
        episodes: list[EpisodeRecord] = []
        episodes_dir = output_dir / _EPISODES_DIR
        if episodes_dir.exists():
            for ep_dir in sorted(episodes_dir.iterdir()):
                record_path = ep_dir / EPISODE_RECORD_FILENAME
                if ep_dir.is_dir() and record_path.exists():
                    episodes.append(EpisodeRecord.model_validate_json(record_path.read_text()))
        return cls(experiment=experiment, episodes=episodes)

    def to_jsonl(self, path: Path) -> None:
        """Write all episode records as a flat JSONL file for ATLAS submission.

        Each line is a self-contained EpisodeRecord JSON object. No cube-harness
        dependency required to read the output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self.episodes:
                f.write(record.model_dump_json() + "\n")
        logger.info(f"Exported {len(self.episodes)} episode records to {path}")
