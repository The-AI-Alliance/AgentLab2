# DRBench CUBE Integration

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2510.00172-8512DA)](https://arxiv.org/abs/2510.00172)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E)](https://huggingface.co/datasets/ServiceNow/drbench)
[![Discord](https://img.shields.io/badge/Discord-Community-7289DA?logo=discord)](https://discord.gg/9rQ6HgBbkd)

## What is DRBench?

**DRBench** is the first benchmark designed to evaluate deep research agents on complex,
open-ended **enterprise research tasks**. It tests an agent's ability to conduct multi-hop,
insight-driven research across realistic enterprise data sources — just like a real analyst.

Developed at [ServiceNow Research](https://www.servicenow.com/research/), DRBench provides
100 validation tasks spanning compliance, HR, IT, facilities, and finance domains. Each task
runs in a self-contained Docker environment with Nextcloud, Mattermost, email (IMAP),
FileBrowser, and web search, and is scored by an LLM judge that evaluates the agent's
submitted report for insight recall and factual accuracy.

For full details, see the [paper on arXiv](https://arxiv.org/abs/2510.00172) and the
[main DRBench repository](https://github.com/ServiceNow/drbench).

## CUBE Integration

This package wraps DRBench as a [CUBE-standard](https://github.com/ServiceNow/cube-standard) benchmark.
Agents interact with the enterprise environment through CUBE's Task/Tool/Container interfaces and must submit a research report to complete each task.

## Setup

**Prerequisites:** Python 3.12+, Docker

All commands below should be run from the **drbench-cube** directory.

```bash
# Install drbench-cube (pulls drbench + task data from GitHub automatically)
uv pip install -e .

# Set required API keys (in your shell or a .env file)
export OPENAI_API_KEY=...       # required — used by the LLM judge for scoring
export SERPER_API_KEY=...       # optional — enables the web_search action
export DRBENCH_DOCKER_REGISTRY=ghcr.io/mmunozm  # enables auto-pull of Docker images
```

**Task data:** The `drbench` package (installed as a dependency) bundles all 100 task
definitions under `drbench/data/`. No extra download is needed. If you prefer to use
the [HuggingFace dataset](https://huggingface.co/datasets/ServiceNow/drbench) instead,
set `DRBENCH_DATA_DIR` to point to your local copy.

**Docker images:**

Pre-built images are currently available for `arm64` (Apple Silicon). `amd64` images
are coming soon — for other architectures, build locally using the
[main DRBench repository](https://github.com/ServiceNow/drbench#2-quick-run-with-docker).

Set `DRBENCH_DOCKER_REGISTRY=ghcr.io/mmunozm` in your `.env` and the CUBE container backend
will automatically pull images on first use. You can also pull manually:

```bash
docker pull ghcr.io/mmunozm/drbench-services:DR0001   # per-task image
docker pull ghcr.io/mmunozm/drbench-services:latest    # base image (fallback)
```

To build images locally instead, see the
[main DRBench repository](https://github.com/ServiceNow/drbench#2-quick-run-with-docker).

See [Container Images](#container-images) for details on the two startup modes.

### Quickstart

```python
from drbench_cube.benchmark import DrBenchBenchmark
from drbench_cube.container import DrBenchContainerBackend
from cube.core import Action

# 1. Instantiate benchmark
benchmark = DrBenchBenchmark(container_backend=DrBenchContainerBackend())

# 2. Pick a task config (iterates all 100 val tasks)
task_config = next(benchmark.get_task_configs())

# 3. Create task (launches Docker container automatically)
task = task_config.make()

# 4. Reset: get initial observation (persona prompt + research question)
obs, info = task.reset()
print(obs.contents[0].data)   # the system prompt shown to the agent
print(info["task_id"])        # e.g. "DR0001"

# 5. See what actions are available
for schema in task.action_set:
    print(schema.name, "—", schema.description[:60])

# 6. Execute actions
env_out = task.step(Action(name="list_nextcloud_directory", arguments={"path": "/"}))
print(env_out.obs.contents[0].data)  # JSON list of files
print(env_out.done)                  # False — task still ongoing

# 7. Submit report to end the episode
env_out = task.step(Action(
    name="submit_report",
    arguments={"report_text": "Based on my research, ..."}
))
print(env_out.done)    # True
print(env_out.reward)  # float in [0, 1] — insights_recall score

# 8. Always close to stop the container
task.close()
```

#### Running all 100 tasks

```python
from drbench_cube.benchmark import DrBenchBenchmark
from drbench_cube.container import DrBenchContainerBackend
from cube.core import Action

benchmark = DrBenchBenchmark(container_backend=DrBenchContainerBackend())

results = []
for task_config in benchmark.get_task_configs():
    task = task_config.make()
    try:
        obs, info = task.reset()
        task_id = info["task_id"]

        # --- your agent loop here ---
        # Use obs.contents[0].data as the prompt, call task.step(Action(...))
        # until env_out.done is True or you hit your step budget.
        # Example: a no-op agent that immediately submits an empty report
        env_out = task.step(Action(
            name="submit_report",
            arguments={"report_text": "No findings."}
        ))

        results.append({"task_id": task_id, "reward": env_out.reward})
        print(f"{task_id}: reward={env_out.reward:.3f}")
    finally:
        task.close()
```

#### Running a specific task

```python
from drbench_cube.task import DrBenchTaskConfig
from drbench_cube.container import DrBenchContainerBackend

config = DrBenchTaskConfig(task_id="DR0001")
task = config.make(container_backend=DrBenchContainerBackend())
obs, info = task.reset()
# ... agent loop ...
task.close()
```

#### Listing available actions

```python
from unittest.mock import MagicMock
from drbench_cube.tool import DrBenchTool

mock = MagicMock()
mock.get_url.return_value = "http://localhost:9999"
mock.forward_port.return_value = 9999
tool = DrBenchTool(mock, username="x", password="x")
for a in tool.action_set:
    print(f"{a.name}: {a.description[:80]}")
```

## Environment Variables

All variables can be set in `.env` (loaded automatically) or as shell environment variables.
Shell variables take precedence over `.env`.

### Required

| Variable         | Used by               | Notes                                                                                |
| ---------------- | --------------------- | ------------------------------------------------------------------------------------ |
| `OPENAI_API_KEY` | `evaluate()` (reward) | Used by `score_report()` to judge submitted reports via LLM                          |
| `SERPER_API_KEY` | `web_search` action   | Google search via Serper API. Missing key returns a JSON error — task does not crash |

### Optional / overrides

| Variable                  | Default            | Notes                                                        |
| ------------------------- | ------------------ | ------------------------------------------------------------ |
| `OPENAI_BASE_URL`         | _(OpenAI default)_ | Override to route through OpenRouter or a proxy              |
| `DRBENCH_DOCKER_REGISTRY` | _(none)_           | Registry prefix (e.g. `ghcr.io/mmunozm`). Enables auto-pull |
| `DRBENCH_DOCKER_IMAGE`    | `drbench-services` | Docker image name prefix                                     |
| `DRBENCH_DOCKER_TAG`      | `latest`           | Fallback tag when per-task baked image is not found           |

### Changing the reward model

The judge model is set per-task via `DrBenchTaskConfig`, not an env var. This keeps it
explicit in serialized configs and experiment logs:

```python
# Default — canonical DRBench judge (requires OPENAI_API_KEY)
# reward = harmonic_mean(insights_recall, factuality)
config = DrBenchTaskConfig(task_id="DR0001")

# Cheaper judge for debugging (scores not comparable with official results)
config = DrBenchTaskConfig(task_id="DR0001", eval_model="gpt-4o-mini")

# Insights recall only — faster, no embedding calls
config = DrBenchTaskConfig(task_id="DR0001", eval_metrics=["insights_recall"])

# Via OpenRouter (requires OPENROUTER_API_KEY + OPENROUTER_API_URL)
config = DrBenchTaskConfig(task_id="DR0001", eval_model="openrouter/openai/gpt-4o")
```

Supported model prefixes in `AIAgentManager` (the underlying LLM client):
- `"gpt-4o"`, `"gpt-4o-mini"` → OpenAI direct (`OPENAI_API_KEY`)
- `"openrouter/..."` → OpenRouter (`OPENROUTER_API_KEY`, `OPENROUTER_API_URL`)
- `"vllm/..."` or vllm model names → self-hosted vLLM (`VLLM_API_URL`, `VLLM_API_KEY`)

**Tip:** The OpenAI client reads `OPENAI_BASE_URL` from the environment automatically. To route
`gpt-4o` through OpenRouter without changing the model name:
```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=<your-openrouter-key>
```

**Note:** `eval_embedding_model` is used by `factuality` for semantic retrieval (defaults to
`text-embedding-3-large`, matching the paper). It has no effect if `factuality` is not in
`eval_metrics`.

## Available Actions (14 total)

| Action                       | Description                                                 |
| ---------------------------- | ----------------------------------------------------------- |
| `search_nextcloud_files`     | Search filenames in Nextcloud                               |
| `list_nextcloud_directory`   | List directory contents in Nextcloud                        |
| `download_nextcloud_file`    | Download and extract text from a Nextcloud file             |
| `search_filebrowser_files`   | Search filenames in FileBrowser                             |
| `list_filebrowser_directory` | List directory contents in FileBrowser                      |
| `download_filebrowser_file`  | Download and extract text from a FileBrowser file           |
| `search_mattermost`          | Full-text search across Mattermost channels                 |
| `list_mattermost_teams`      | List accessible Mattermost teams                            |
| `search_emails`              | Search emails by keyword                                    |
| `get_email`                  | Fetch a specific email by folder + message ID               |
| `list_email_folders`         | List mailbox folders with message counts                    |
| `web_search`                 | Google search via Serper API                                |
| `fetch_url`                  | Fetch a URL and extract text                                |
| `submit_report`              | Submit the research report (ends episode, triggers scoring) |

## Benchmark Structure

- **100 tasks** (`val` split) across domains: compliance, HR, IT, facilities, finance
- **Difficulty levels**: easy, medium, hard
- **Recommended max steps**: 50 per task

## Container Images

Each task runs in its own Docker container. `DrBenchContainerBackend` looks for a
pre-baked image named `drbench-services:<task_id>` (e.g. `drbench-services:DR0001`).
This name comes from `TaskMetadata.container_config.image`, which is set for all 100
tasks in `task_metadata.json`.

**Two startup paths:**

| Path             | Image                                 | Startup time | When used                                    |
| ---------------- | ------------------------------------- | ------------ | -------------------------------------------- |
| Pre-baked (fast) | `drbench-services:DR0001`             | ~8s          | Image exists locally or pulled from registry |
| Fallback (slow)  | `drbench-services:latest` + task load | ~45s         | Image not found — loads task data at runtime |

The fallback means the benchmark works out of the box without pre-built images, just slower.

**Pre-built images from registry (recommended):**

Set `DRBENCH_DOCKER_REGISTRY` and the CUBE container backend will automatically pull
per-task images on first use:

```bash
export DRBENCH_DOCKER_REGISTRY=ghcr.io/mmunozm

# Or pull manually
docker pull ghcr.io/mmunozm/drbench-services:DR0001
docker pull ghcr.io/mmunozm/drbench-services:latest   # base image (fallback)
```

When `DRBENCH_DOCKER_REGISTRY` is set, `DrBenchContainerBackend.launch()` will:
1. Check for the image locally (`ghcr.io/mmunozm/drbench-services:DR0001`)
2. If not found, pull it from the registry automatically
3. If pull fails, fall back to the base image + runtime task loading (slower)

**Building images locally:**

To build images from source instead of pulling, see the
[main DRBench repository](https://github.com/ServiceNow/drbench#2-quick-run-with-docker).

## Using a Different Container Backend

`DrBenchContainerBackend` runs containers on your local Docker daemon, but
`DrBenchTaskConfig.make()` accepts **any** CUBE `ContainerBackend`. As long as
the pre-baked task images (`drbench-services:DR0001`, …) are available in a
registry that the backend can pull from, you can swap in Daytona, Modal, or any
other CUBE-compatible backend:

```python
from drbench_cube.benchmark import DrBenchBenchmark
from cube.backends.daytona import DaytonaContainerBackend

# Use Daytona (or Modal, Toolkit, etc.) instead of local Docker
backend = DaytonaContainerBackend(...)
benchmark = DrBenchBenchmark(container_backend=backend)

for task_config in benchmark.get_task_configs():
    task = task_config.make(container_backend=backend)
    # ... agent loop ...
    task.close()
```

`DrBenchTool` communicates with container services over HTTP and IMAP using
only the standard CUBE `Container` interface (`get_url()`, `forward_port()`),
so no DRBench-specific wiring is needed on the backend side.

> **Note:** The runtime fallback path (base image + `add_task()`) only works
> with local Docker. Remote backends require pre-baked per-task images in a
> registry the backend can reach.

## Caveats / Open Questions for CUBE Devs

These are known deviations or ambiguities to resolve before submission:

### 1. `reset_isolation = "restart"`

Each task gets its own Docker container launched from a per-task baked image (`drbench-services:DR0001`, etc.). A "reset" currently restarts the *tool state* (clears `_submitted_report`, re-runs `discover_capabilities`) but does **not** restart the container.

- Container state (files, messages) is read-only from the agent's perspective, so this is functionally correct.
- If CUBE's `restart` isolation implies a full container restart between seeds, we would need to stop and relaunch the container in `DrBenchTask.reset()`.
- **Question**: Does `ResetIsolation.RESTART` require a container restart on every `reset()` call, or just that the environment is logically reset?

### 2. `evaluate()` calls an external LLM

`DrBenchTask.evaluate()` calls `score_report()` for both `insights_recall` and `factuality`, hitting the OpenAI API (`gpt-4o` + `text-embedding-3-large` by default). Reward is their harmonic mean, matching the composite score in the DRBench paper. This means:
- Reward is **stochastic** (small variance run-to-run from the LLM judge)
- Reward depends on `OPENAI_API_KEY` being set — returns `0.0` with an error dict if missing
- Reward computation takes ~2–10s depending on report length and API latency

This is intentional (LLM-as-judge is the evaluation methodology), but it violates "deterministic reward" if CUBE assumes that.

### 3. `get_debug_task_configs()` vs `debug.py`

`DrBenchBenchmark.get_debug_task_configs()` returns `[DrBenchTaskConfig(task_id="DR0001")]`. This method is separate from the `debug.py` module protocol (`get_debug_benchmark()` / `make_debug_agent()`) used by `cube.testing.run_debug_suite`. 

- **Question**: Is `get_debug_task_configs()` part of the CUBE spec? If so, what is it used for vs `debug.py`?

### 4. `ToolConfig` passed through `TaskConfig` vs loaded at runtime

`DrBenchTaskConfig` does not store a `tool_config` — it loads the persona credentials from the task data directory at `make()` time. This means `tool_config` field (inherited from `TaskConfig`) is always `None` in serialized form.

This works correctly but may confuse harness code that introspects `task_config.tool_config`.

### 5. Ports in `container_config`

`TaskMetadata.container_config.ports` is populated with `[8080, 8081, 8082, 8090, 1143]` for documentation purposes, but `DrBenchContainerBackend.launch()` ignores this list and uses `auto_ports=True` (Docker assigns ephemeral host ports). The actual host ports are queried at runtime via `container.forward_port(container_port)`.

- **Question**: Does CUBE's harness infrastructure use `container_config.ports` to pre-allocate or validate ports? If so, we need to align.

### 6. Web search requires external API

`web_search` depends on `SERPER_API_KEY` (external paid API). When the key is absent, the action returns `{"error": "SERPER_API_KEY not set"}` rather than raising. This is intentional — the agent can still complete tasks using only internal sources. However it means benchmark results will differ depending on whether web search is available.

## Running the Debug Suite

The CUBE debug suite verifies structural compliance (container starts, actions work,
done triggers). Run from the **drbench-cube** directory:

```bash
# Requires Docker + at least drbench-services:DR0001 image
uv run python -m drbench_cube.debug

# Or via the CUBE CLI
cube test drbench-cube
```

### Unit tests (no Docker required)

```bash
pytest tests/
```

Covers benchmark loading, `DrBenchTaskMetadata` field access, config roundtrip,
subset operations, and action-set presence — no container needed.

### Known non-determinism in `reset()`

`cube test`'s `test_reset_reproducibility` check will fail for DRBench. This is
**expected and intentional**: `reset()` embeds dynamically-allocated host port URLs
(e.g. `http://localhost:55023`) in the agent's initial prompt. These ports are
assigned by Docker at container launch time, so two resets of the same task produce
structurally identical but textually different observations.

The non-determinism is confined to port numbers — the research question, persona,
and credentials are always the same. Agents should treat the URLs as opaque
endpoints and not rely on port values.

### `evaluate()` requires `OPENAI_API_KEY`

`cube test`'s `test_full_episode` check will also fail without `OPENAI_API_KEY` set,
because the LLM judge (`score_report`) is called during `evaluate()`. The debug agent
submits a placeholder report that intentionally scores 0.0. Set `OPENAI_API_KEY` and
`SERPER_API_KEY` to run a meaningful end-to-end evaluation.

### Regenerating `task_metadata.json`

If the upstream `drbench` package updates its task set, regenerate the metadata file:

```bash
uv run python scripts/generate_task_metadata.py
```

For the full DRBench test suite, see the
[main DRBench repository](https://github.com/ServiceNow/drbench).
