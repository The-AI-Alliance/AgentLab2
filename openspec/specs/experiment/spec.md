# Experiment

**Module:** `cube_harness.experiment`, `cube_harness.exp_runner`

## Purpose

An `Experiment` pairs an `AgentConfig` with a `BenchmarkConfig` and produces one
`Episode` per task. The runners in `exp_runner.py` instantiate the live
`Benchmark` via `benchmark_config.make(infra)` and execute episodes sequentially
or in parallel via Ray, with OpenTelemetry benchmark-level spans wrapping the run.

## Public API

### `Experiment` (serializable)
```python
class Experiment(TypedBaseModel):
    name: str
    output_dir: Path
    agent_config: AgentConfig
    benchmark_config: BenchmarkConfig    # cube.benchmark.BenchmarkConfig
    infra: InfraConfig | None = None     # forwarded to benchmark_config.make(infra)
    resume: bool = False
    max_steps: int = MAX_STEPS
    max_retries: int = 3                 # per-episode retry cap

    @property
    def config(self) -> dict             # model_dump(serialize_as_any=True)

    def get_episodes_to_run(
        self,
        benchmark: Benchmark | None = None,
        *,
        step_timeout_s: float = 1800.0,
        cancel_grace_s: float = 120.0,
        orphan_threshold_s: float = 3600.0,
        process_start_s: float | None = None,    # current driver's wall-clock start
    ) -> list[Episode]
    def save_config(self) -> None        # writes experiment_config.json
    @classmethod
    def load_config(cls, path: str) -> Experiment

    def print_stats(self, results: ExpResult) -> None
```

`benchmark_config` is the serialisable side; `Benchmark` is no longer Pydantic
in cube-standard, so it cannot live as a field on `Experiment`. The runners
build the live `Benchmark` for the duration of a run via
`with exp.benchmark_config.make(exp.infra) as benchmark:` and pass it to
`get_episodes_to_run` so episodes pick up its `_runtime_context` and
`config.container_backend`. Tests that only enumerate episodes without running
may omit `benchmark`; the resulting episodes carry no `runtime_context` and no
`container_backend`.

### Resume / retry semantics

| `resume` | Episodes returned                                                                                                                         |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `False`  | All episodes from scratch                                                                                                                 |
| `True`   | Episodes with no `status.json` (never started), plus retriable statuses (`FAILED`, `CANCELLED`, `STALE`) with `retry_count < max_retries` |

`RUNNING` / `QUEUED` (in-flight) are never returned. `COMPLETED`,
`MAX_STEPS_REACHED`, and `INVALID_CONFIG` (terminal, non-retriable) are always
skipped. `INVALID_CONFIG` is set when the episode died from a permanent provider
error (typo'd model name, bad key, malformed/oversized/policy-violating request —
classified by `is_permanent_llm_error` in `llm.py`); retrying it would fail
identically, so it terminates with `retry_count=0`.

When `resume=True`, `sweep_stale_statuses` runs first so orphaned `RUNNING`/`QUEUED`
entries become `STALE` and are eligible for retry. The runner threads its own
`process_start_s` into the sweep so RUNNING episodes whose heartbeat predates the
new process are force-swept regardless of heartbeat age — fixes the kill-and-retry
race where a freshly killed worker still has a recent heartbeat.

The shared "is this RUNNING worker dead?" predicate
(`should_sweep_running_to_stale` in `episode_status.py`) is the single source of
truth used by both `sweep_stale_statuses` (runner-side) and
`_promote_ghost_episodes` (XRay-side).

### `ExpResult`
```python
class ExpResult(TypedBaseModel):
    exp_id: str
    tasks_num: int
    config: dict = {}
    trajectories: dict[str, Trajectory] = {}    # task_id → Trajectory
    failures: dict[str, str] = {}               # task_id → error message
```

### Runners

#### `run_sequentially(exp, debug_limit=None, *, step_timeout_s=1800.0, cancel_grace_s=120.0, orphan_threshold_s=3600.0, max_retry_rounds=3, otlp_endpoint=None, model=None, agent_name=None) -> ExpResult`
Runs episodes one at a time in-process. `debug_limit` caps the list (useful for
`make debug`). The driver IS the worker — `experiment_status.json` is heartbeated
*between* episodes, so a long episode can leave the experiment heartbeat stale even
though the driver is alive (XRay falls back to per-episode heartbeats in this case).

#### `run_with_ray(exp, *, n_cpus=4, ray_poll_timeout=2.0, step_timeout_s=1800.0, setup_timeout_s=7200.0, cancel_grace_s=120.0, orphan_threshold_s=3600.0, max_retry_rounds=3, otlp_endpoint=None, model=None, agent_name=None) -> ExpResult`
Parallel via Ray `@remote`. Initializes Ray with dashboard enabled.

Driver-side step timeout is enforced in `_kill_stale_workers` by reading each
active episode's `status.json` and comparing `now - last_heartbeat_at` against a
**phase-aware budget**: episodes still in setup (`current_step == 0`) get
`setup_timeout_s` (default `4 × step_timeout_s = 7200s`; container scheduling /
toolkit queuing can be slow); episodes inside the agent loop get the full
`step_timeout_s`. Workers exceeding their budget are `ray.cancel(force=True)`-ed
and stamped `CANCELLED` with `error_type="SetupTimeout"` or `"StepTimeout"`.

Both runners:
1. Enter `tracer.benchmark(exp.name)` span
2. Enter `_experiment_lifecycle(exp.output_dir, mode=...)` — writes RUNNING to
   `experiment_status.json` on enter, COMPLETED (or INTERRUPTED on exception) on
   exit. The driver heartbeats the file: every ~30 s in Ray mode (from the
   `_poll_ray` loop), and before each `episode.run()` in sequential mode.
3. `exp.save_config()`
4. `with exp.benchmark_config.make(exp.infra) as benchmark:` — `make` provisions
   declared resources, instantiates the runtime pair, and calls `setup()`. The
   context manager guarantees `close()` on exit.
5. Call the internal `_run_*_impl` with the live `benchmark`
6. `tracer.shutdown()`

Per-episode stdout/stderr is redirected to `<output_dir>/episodes/<traj_id>/logs/`
via `redirect_output_to_log` from `episode_logs.py`.

### Recipes (declarative config files)

A recipe imports canonical configs by name, tweaks attributes, builds one or
more `Experiment` objects, and ends with `run(...)`. No per-recipe argparse.

- `ConfigRegistry` (`cube_harness.config_registry`) — a `Mapping`; every
  `reg[name]` returns a `model_copy(deep=True)`, so a recipe cannot mutate the
  shared canonical instance. Unknown name → `KeyError` listing valid names.
- Canonical registries: `GENNY_CONFIGS`, `REACT_CONFIGS`
  (`cube_harness.agents.*_configs`); per-cube `*_CONFIGS` (e.g.
  `swebench_verified_cube.SWEBENCH_CONFIGS`).
- `cube_harness.infra.INFRA_CONFIGS` — built-in `"local"` plus entries from
  `~/.cube/infra.py` (a `dict[str, InfraConfig]`, machine-local, never
  committed; credentials resolved from env, never fields).
- `cube_harness.recipe.run(*exps)` — the only CLI, fixed for every recipe:
  `--limit N` (first N tasks via `run_sequentially`), `--ray N`
  (`run_with_ray` worker count), `--set dotted.path=value` (repeatable;
  JSON-parsed, then assigned — `ValidatedConfig` validates the type).
- Config attribute assignment validates at the assignment site because the
  config ABCs subclass `cube.core.ValidatedConfig`.

## Invariants

1. `Experiment` is itself a `TypedBaseModel` — JSON-serializable. It holds a
   `BenchmarkConfig` (Pydantic, picklable), not a live `Benchmark`. Live
   benchmarks are constructed by the runner via `benchmark_config.make(infra)`
   inside a context manager.
2. `save_config()` is called before every run. `experiment_config.json` is
   authoritative for resume/retry workflows.
3. The runner's `with benchmark_config.make(infra) as benchmark:` block wraps
   every run. `make()` calls `setup()` internally, and the context manager calls
   `close()` on exit — resource cleanup is guaranteed on exceptions.
4. Ray disrupts signal handling (Ctrl+C) — known limitation, no fix yet.
5. `experiment_status.json` (sibling of `experiment_config.json`) is written by
   the driver, never by Ray workers. Its lifecycle is RUNNING on entry →
   COMPLETED on clean return → INTERRUPTED on exception. Heartbeats update
   `last_heartbeat_at` and counters (`completed`, `failed`). Old experiments
   without this file behave as before — XRay treats "no file" as "driver alive"
   for backward compat.

## Contracts for implementers

- New runners should preserve the setup/close/tracer wrapping so benchmarks that
  create L2 resources don't leak.
- Telemetry metadata (model, agent_name) flows from runner args into the tracer
  span attributes — populate these when writing custom runners.
- Resume requires `episode_config.json` to exist — writes always go through
  `save_episode_config()` before the episode runs (see episode spec).

## Gotchas

- `run_with_ray` sets `dashboard_host="0.0.0.0"` — the Ray dashboard is exposed on
  all interfaces when running locally. Fine on a workstation; consider in multi-user
  environments.
- Ray workers inherit `env_vars` from `get_trace_env_vars()` — if you need extra env
  vars in workers, they must be added explicitly (not automatic from the driver).
- Episode timeouts are enforced by reading per-episode `status.json` heartbeats
  from disk (filesystem-driven, no Ray dashboard dependency). Workers exceeding
  their phase-aware budget are `ray.cancel(force=True)`-ed.
- Cancellation is best-effort: `cancel_grace_s` (default 120s) is added to the
  budget before the kill. An episode that hangs in a C extension may not respect
  cancel.
- `_promote_ghost_episodes` (XRay) only sweeps QUEUED → STALE when
  `experiment_status.json` reports the driver dead — never on every refresh.
  Sequential mode falls back to per-episode heartbeats to avoid false-positive
  promotions during long episodes.
