# EvalLog

**Module:** `cube_harness.eval_log`

## Purpose

Exports two structured files per experiment, together forming the Atlas EvalLog:

- **`experiment_record.json`** — one JSON object, written once per experiment. Holds
  agent description, benchmark metadata, and git provenance. Does not repeat per episode.
- **`episodes/<trajectory_id>/episode_record.json`** — one JSON file per completed
  episode, co-located with trajectory data. Holds outcome, usage, trajectory summary,
  and optional investigator output. Links to `experiment_record.json` via `experiment_id` FK.
  Retried episodes overwrite stale records naturally.
- **`to_jsonl(path)`** — submission helper on `EvalLog` that assembles all per-trajectory
  records into a flat JSONL for ATLAS upload. Call explicitly after `export_eval_log()`.

Both files are plain JSON, readable without any cube-harness dependency.

The primary consumer is **Project ATLAS** (Agent-Task Latent Analysis System), which
builds the community matrix **M[agent, task] = reward** from these records via sparse
matrix factorization and IRT. Secondary consumers include leaderboards, cost trackers,
and any framework that wants a stable per-episode data contract.

Fields are structured to map cleanly to the two-level
[Every Eval Ever (EEE)](https://github.com/evaleval/every_eval_ever) schema:
`ExperimentRecord` ≈ EEE aggregate record, `EpisodeRecord` ≈ EEE instance-level record.

---

## Public API

### `UsageSummary`

```python
class UsageSummary(TypedBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    total_cost_usd: float = 0.0
    n_llm_calls: int = 0

    @classmethod
    def from_summary_stats(cls, stats: dict | None) -> "UsageSummary"
```

Aggregated LLM token usage and cost across a complete episode. Built from
`Trajectory.summary_stats` — no re-scanning of steps required.

---

### `AgentInfo`

```python
class AgentInfo(TypedBaseModel):
    # Identity
    agent_id: str                        # SHA-256(sorted config JSON)
    config_type: str                     # AgentConfig._type discriminator
    config: dict                         # full serialized agent config
    llm_model: str | None                # extracted from config

    # Runtime environment
    framework_version: str               # cube-harness version
    dependency_versions: dict[str, str]  # 9 tracked packages

    # Git provenance
    git_commit: str | None
    git_remote_url: str | None           # permanent GitHub permalink
    git_is_dirty: bool | None
    cube_standard_git_commit: str | None   # cube-standard repo HEAD (editable/source checkout only)
    cube_standard_git_is_dirty: bool | None

    # LLM warm-start embedding (ATLAS cold-start)
    description: str | None

    @classmethod
    def from_agent_config(
        cls,
        agent_config: AgentConfig,
        git_cwd: str | None = None,
    ) -> "AgentInfo"
```

**`agent_id`** is the primary stable row key for the ATLAS matrix. It is the SHA-256 of
the agent config serialized to JSON with sorted keys. Two runs of the same config produce
the same `agent_id`, regardless of wall time or machine.

**No `tools` field.** Tools vary per episode (the same agent gets different action schemas
on different tasks due to task-level action filtering). Tools are captured at the episode
level in `EpisodeRecord.tool_names`.

**`description`** is optional free-form prose intended for ATLAS's LLM warm-start embedding
(cold-start for new agents with zero observed scores). May be human-authored or synthesized
from the structured fields above. Never auto-populated by `from_agent_config()`.

Tracked packages: `cube-harness`, `cube`, `litellm`, `anthropic`, `openai`,
`browsergym-core`, `playwright`, `pydantic`, `ray`.

---

### `BenchmarkSubset`

```python
class BenchmarkSubset(TypedBaseModel):
    name: str           # benchmark_metadata.name (includes subset suffix like "[level=l1]")
    n_tasks: int        # len(benchmark.task_metadata) — denominator for completion rate
    filter: str | None  # glob expression if subset_from_glob was used

    @classmethod
    def from_benchmark(cls, benchmark: Any) -> "BenchmarkSubset"
```

Automatically derived from the benchmark object. Used by ATLAS for MNAR propensity
correction: `n_tasks` tells ATLAS what fraction of the benchmark was run without requiring
submitters to fill in subjective fields.

**`name`** captures any subset suffix applied via `subset_from_glob` (e.g.,
`"WorkArena_[level=l1]"`) or `subset_from_list`. It is `benchmark_metadata.name` verbatim.

**`filter`** is `None` unless manually populated — there is currently no standard way to
extract the glob pattern from a benchmark object automatically.

---

### `InvestigatorLLMConfig`

```python
class InvestigatorLLMConfig(TypedBaseModel):
    model: str           # e.g. "claude-opus-4-7"
    prompt_version: str  # version or hash of the investigator prompt template
    investigated_at: str | None  # ISO-8601 timestamp
```

Configuration of the LLM investigator used for post-hoc episode assessment. Stored in
`ExperimentRecord.investigator_llm_config`; `None` if no investigator was run.

---

### `Findings`

```python
class Findings(TypedBaseModel):
    difficulty: str | None         # estimated task difficulty (free-form or enum)
    feasible: bool | None          # whether the task was deemed completable
    failure_root_cause: str | None # short description of why the agent failed
```

Per-episode LLM investigator assessment. Stored in `EpisodeRecord.findings`; `None` if no
investigator was run. Populated in a post-processing step, not during the episode run.

---

### `Verifier`

```python
class Verifier(TypedBaseModel):
    ref: str | None     # permanent GitHub URL to the verifier function at the exact commit
    source: str | None  # verifier source code at eval time
```

Task verifier reference for reproducibility and post-hoc inspection. Stored in
`EpisodeRecord.verifier`; `None` if not populated.

---

### `ExperimentRecord`

```python
class ExperimentRecord(TypedBaseModel):
    experiment_id: str              # SHA-256(experiment_name + output_dir)[:16]
    experiment_name: str
    timestamp: float                # export time, Unix
    framework_version: str
    agent: AgentInfo
    benchmark_name: str             # benchmark_metadata.name
    benchmark_version: str | None
    benchmark_subset: BenchmarkSubset
    investigator_llm_config: InvestigatorLLMConfig | None = None

    @classmethod
    def from_experiment(
        cls,
        exp_name: str,
        output_dir: Path,
        agent_config: Any,
        benchmark: Any,
        git_cwd: str | None = None,
    ) -> "ExperimentRecord"
```

Written once per experiment to `experiment_record.json`. Contains all fields shared
across every episode: agent description, benchmark metadata, git provenance.

**`experiment_id`** links every `EpisodeRecord` back to this record. It is
SHA-256(experiment_name + str(output_dir))[:16] — stable for the same run (output_dir is
unique per experiment), deterministic across repeated calls.

---

### `EpisodeRecord`

```python
class EpisodeRecord(TypedBaseModel):
    # FK
    experiment_id: str

    # Task identity
    task_id: str
    task_version_hash: str | None   # SHA-256 of TaskConfig JSON
    seed: int | None
    split: str | None               # "train" | "val" | "test"
    task_description: str | None    # TaskMetadata.abstract_description

    # Episode-specific tools
    tool_names: list[str]           # from trajectory.metadata["action_schemas"]

    # Outcome
    success: bool                   # reward > 0
    reward: float
    error_type: str | None          # exception class name if any step errored

    # Trajectory summary
    n_steps: int
    n_agent_steps: int
    n_env_steps: int
    wall_time_s: float | None
    usage: UsageSummary

    # Provenance
    trajectory_id: str
    timestamp: float                # episode start, Unix

    # Optional post-hoc fields
    verifier: Verifier | None = None
    findings: Findings | None = None

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        experiment_id: str,
        task_metadata: Any | None = None,
        task_config: Any | None = None,
    ) -> "EpisodeRecord"
```

One file per episode at `episodes/<trajectory_id>/episode_record.json`. Links to `ExperimentRecord` via `experiment_id`. Retried episodes overwrite stale records since the new trajectory occupies the same directory.

**`task_version_hash`** is the SHA-256 of `TaskConfig.model_dump_json(serialize_as_any=True)`.
It changes whenever the task config changes, even if `task_id` is unchanged. ATLAS uses it
to detect silent benchmark drift: if the same `task_id` has two different hashes across
submissions, the records cannot be naively merged in the matrix.

**`tool_names`** is read from `trajectory.metadata["action_schemas"]` at export time.
Returns `[]` for trajectories produced before the `action_schemas` field was added to
metadata. The list is episode-specific — the same agent gets different tools on different
tasks due to task-level action filtering.

**`error_type`** is the exception class name (e.g. `"TimeoutError"`) of the first
`StepError` found in the trajectory. It is `None` for clean episodes that simply scored
zero — a zero-reward, no-error episode is a valid failure, not a crash.

---

### `EvalLog`

```python
class EvalLog(TypedBaseModel):
    experiment: ExperimentRecord
    episodes: list[EpisodeRecord] = []

    def save(self, output_dir: Path) -> None
    # Writes experiment_record.json and episodes/<trajectory_id>/episode_record.json.

    @classmethod
    def load(cls, output_dir: Path) -> "EvalLog"
    # Reads experiment_record.json and all episodes/*/episode_record.json.

    def to_jsonl(self, path: Path) -> None
    # Aggregates all episode records into a flat JSONL file for ATLAS submission.
    # Each line is a self-contained EpisodeRecord; no cube-harness dependency to read.
```

Two-level container. Episode records are co-located with trajectory data in
`episodes/<trajectory_id>/` — retried episodes naturally overwrite stale records
since the new trajectory occupies the same directory. `to_jsonl()` is the submission
helper: call it after `export_eval_log()` or after loading an existing eval log to
produce a flat file for ATLAS upload.

---

## Integration Points

### `Episode.run` → `trajectory.metadata["action_schemas"]`

In `Episode.run`, immediately after the action set is resolved:

```python
extra_metadata = {"action_schemas": [a.as_dict() for a in action_set]}
return self._run_loop(setup_fn, step_fn, close_fn, agent, extra_metadata=extra_metadata)
```

`_run_loop` merges `extra_metadata` into `trajectory.metadata`. This is the only change
to the episode loop. Action schemas are captured once at task reset time and persisted so
`export_eval_log()` can reconstruct `EpisodeRecord.tool_names` without re-instantiating tasks.

### `Experiment.export_eval_log`

```python
def export_eval_log(
    self,
    output_dir: Path | None = None,
    git_cwd: str | None = None,
) -> EvalLog
```

Called after `run_sequentially()` or `run_with_ray()` completes. Reads all data from
persisted files — no task or benchmark re-instantiation required.

Resolution order for `EpisodeRecord.tool_names`:
`trajectory.metadata["action_schemas"]` → `[]` if absent.

Resolution order for `task_metadata` fields:
`benchmark.task_metadata[task_id]` → `None` values (split, description) if absent.

---

## On-disk output

```
<output_dir>/
├── experiment_config.json
├── experiment_summary.json
├── experiment_record.json          ← written by export_eval_log()
└── episodes/
    └── <trajectory_id>/
        ├── episode_config.json
        ├── episode_record.json     ← written by export_eval_log(), one per episode
        └── ...
```

`EvalLog.save(output_dir)` writes `experiment_record.json` at the top level and
`episode_record.json` inside each trajectory directory. Episode records are co-located
with trajectory data: if an episode is retried, the new trajectory's record naturally
replaces the old one without leaving stale flat-file entries.

For ATLAS submission, call `eval_log.to_jsonl(path)` to assemble a single flat JSONL
from the per-trajectory records. This is a separate step to keep the submission artifact
distinct from the experiment's working files.

---

## Invariants

1. `agent_id` is deterministic: same `AgentConfig` → same hash, regardless of run time,
   machine, or framework version. Never include timestamps or random values in the config.
2. `experiment_id` is stable for the same (experiment_name, output_dir) pair. Different
   output directories produce different IDs even for experiments with the same name.
3. `task_version_hash` covers the full `TaskConfig` JSON, not just the prompt. A task
   whose environment setup changes produces a new hash even if the written instructions
   are identical.
4. All `EpisodeRecord` files in an experiment share the same `experiment_id`, matching
   `ExperimentRecord.experiment_id` in the companion `experiment_record.json`.
5. Lines in the JSONL produced by `to_jsonl()` are self-contained. Each line is a
   complete `EpisodeRecord`; no cube-harness dependency is required to read the file.

## Gotchas

- `export_eval_log()` loads all trajectories into memory. For experiments with thousands
  of episodes, this can be slow.
- Old trajectories (before the `action_schemas` metadata field was added) yield
  `EpisodeRecord.tool_names = []`. Records are still valid for reward/cost stats.
- `git_is_dirty = True` means the eval may not reproduce exactly from `git_commit` alone.
- `AgentInfo.description` is never auto-populated by `from_agent_config()`. Set it
  manually when preparing ATLAS submissions.
- `BenchmarkSubset.filter` is `None` unless manually populated after calling
  `BenchmarkSubset.from_benchmark()`.
