# Deprecated — cube-harness

Code and artifacts flagged for future cleanup. Each item is actionable — remove,
replace, rewrite, or consolidate. Nothing here is currently in progress; the list
exists so we can pick up simplification work when timing allows.

---

## Dead / broken code

### Legacy XML-tag agent and its tests
- [src/cube_harness/agents/legacy_generic_agent.py](src/cube_harness/agents/legacy_generic_agent.py) (~1260 lines)
- [tests/test_legacy_generic_agent.py](tests/test_legacy_generic_agent.py) (~910 lines)
- Re-exported from [src/cube_harness/agents/__init__.py](src/cube_harness/agents/__init__.py) — `GenericAgent`, `GenericPromptFlags`

Superseded by ReAct ([agents/react.py](src/cube_harness/agents/react.py)) and
Genny ([agents/genny.py](src/cube_harness/agents/genny.py)). Confirm no active
recipe or cube imports it:

```bash
grep -r "legacy_generic_agent\|GenericAgent\|GenericPromptFlags" recipes/ cubes/
```

**Action:** once grep comes up clean, remove file + test + exports. ~2100 LOC gone.

---

## Legacy parameters / migration debt

### V1 storage format read paths
[src/cube_harness/storage.py](src/cube_harness/storage.py) — all `_v1_*` methods (~150 LOC)

| Method | Line |
|--------|------|
| `_v1_metadata_files` | [111](src/cube_harness/storage.py#L111) |
| `_v1_traj_id_from_file` | [123](src/cube_harness/storage.py#L123) |
| `_v1_resolve_trajectory_paths` | [127](src/cube_harness/storage.py#L127) |
| `_v1_load_trajectory` | [228](src/cube_harness/storage.py#L228) |
| `_v1_resolve_llm_call_refs` | [254](src/cube_harness/storage.py#L254) |
| `_v1_load_all_metadata` | [316](src/cube_harness/storage.py#L316) |
| `_v1_list_ids`, `_v1_list_ids_with_mtime`, `_v1_load_all_trajectories` | 332-387 |

Writes have been V2-only for some time. Read paths exist to load old experiment
dirs.

**Action:** ship a migration script (V1 → V2 layout), then delete all `_v1_*`
methods and the V2/V1 fallback branches in `load_trajectory`, `load_logs`, etc.

### Legacy log path fallback
[src/cube_harness/storage.py:391-406](src/cube_harness/storage.py#L391-L406)

```python
def load_logs(self, trajectory_id: str) -> str:
    log_path = self.get_log_path(trajectory_id)
    if not log_path.exists():
        legacy_log_path = self.output_dir / "logs" / f"{trajectory_id}.log"
        if not legacy_log_path.exists():
            return ""
        log_path = legacy_log_path
    return log_path.read_text()
```

**Action:** remove when V1 support is dropped (same PR as the `_v1_*` removal).

---

## Known limitations (document, don't fix)

### Ray breaks Ctrl+C signal handling
[src/cube_harness/exp_runner.py:73](src/cube_harness/exp_runner.py#L73)

```python
)  # TODO: Ray breaks signal handling, we cannot react to Ctrl+C here,
   # still cannot find a workaround
```

Hard blocker upstream. **Action:** document as a known limitation in the
experiment spec; revisit on Ray major upgrades.

### XRay background metadata loading
[src/cube_harness/analyze/xray.py](src/cube_harness/analyze/xray.py) — background thread fills in `n_steps`, `tokens`, `cost`, `duration` because V2 `episode.metadata.json` lacks them.

**Action:** persist these at episode completion
([episode.py:208-210](src/cube_harness/episode.py#L208-L210) `trajectory.summary_stats`
is computed but not flattened into the top-level metadata file). Once metadata is
complete, delete the background loader.

---

## Low-value abstractions

### `EpisodeConfig.benchmark` optional parameter with double guard
[src/cube_harness/episode.py:64-104](src/cube_harness/episode.py#L64-L104)

`load_episode_from_config` accepts an optional `benchmark`, then checks
`if benchmark is not None` to decide whether to forward `runtime_context`.

**Action:** split into two classmethods — `load_from_config(path)` (bare) and
`load_from_config_with_benchmark(path, benchmark)` (full). Clearer contract, no
behavior change.
