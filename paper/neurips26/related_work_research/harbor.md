# Harbor: Detailed Assessment for CUBE Related Work

**Source repositories inspected:** `~/dev/harbor` (v0.1.19) and `~/dev/AgentHarbor` (v0.1.45, the live branch)
**Paper:** Terminal-Bench 2.0 — Merrill, Shaw, Carlini et al., arXiv:2601.11868 (2026), 87 co-authors
**Official site:** https://harborframework.com / https://hub.harborframework.com
**License:** Apache-2.0

---

## Background

Harbor is a framework from the creators of Terminal-Bench (Alex Shaw and collaborators at the Laude Institute, Snorkel AI, Stanford, and others). It was released alongside Terminal-Bench 2.0 in early 2026. Its stated goals are: (1) evaluate arbitrary coding agents at scale; (2) build and share benchmarks in a standardized format; (3) run experiments in thousands of parallel cloud-hosted container environments; (4) generate rollouts for RL/SFT post-training.

Harbor grew out of the Terminal-Bench harness: every Terminal-Bench 2.0 task is specified in Harbor format and run by the Harbor harness. The framework can be seen as "Terminal-Bench generalized into a universal benchmark runner for coding and terminal-capable agents."

---

## 1. Schema Flexibility

### The task.toml + instruction.md + Dockerfile + test.sh contract

Every Harbor task is a directory with the following canonical structure:

```
<task_id>/
├── task.toml          # configuration and metadata
├── instruction.md     # natural-language task prompt for the agent
├── environment/
│   └── Dockerfile     # (or docker-compose.yaml)
├── tests/
│   └── test.sh        # writes reward to /logs/verifier/reward.txt or reward.json
└── solution/
    └── solve.sh        # oracle solution (optional)
```

The `task.toml` schema (Pydantic-validated, version "1.0") covers:

| Section | Key fields |
|---|---|
| `[metadata]` | `author_name`, `author_email`, `difficulty`, `category`, `tags` |
| `[verifier]` | `timeout_sec`, `env`, `user` |
| `[agent]` | `timeout_sec`, `user` |
| `[environment]` | `docker_image`, `os` (linux/windows), `cpus`, `memory_mb`, `storage_mb`, `gpus`, `gpu_types`, `allow_internet`, `mcp_servers[]`, `skills_dir`, `healthcheck`, `build_timeout_sec` |
| `[solution]` | `env` |
| root | `multi_step_reward_strategy` ("mean" \| "final" \| null) |

The MCP server config (`environment.mcp_servers`) is natively expressed in task.toml with `name`, `transport` (sse / streamable-http / stdio), `url`/`command`/`args`.

### Structural limitations by benchmark type

**(a) Benchmarks requiring shared persistent servers across tasks (e.g., WebArena)**

Harbor's model is strictly one-container-per-task, provisioned fresh for each trial and torn down afterwards. The `TaskConfig` and `EnvironmentConfig` classes have no concept of a "shared service" or multi-container orchestration beyond a single Dockerfile. There is no docker-compose support for side-car services that persist across tasks in the same job. WebArena requires one shared web server (e.g., a live GitLab instance, a shopping site) to be reused across 812 tasks — this is architecturally incompatible with Harbor's per-task container model. The AgentHarbor code confirms this: the only compose files found are internal Docker-compose templates for the agent and verifier within a single sandbox, not for external service dependencies. **WebArena and benchmarks with globally-shared state cannot be accommodated without fundamental changes to the task model.**

**(b) CUA / desktop / OSWorld**

Harbor does support a `windows` target OS in `task.toml` (`environment.os = "windows"`), but the agent interface is a terminal exec interface (`environment.exec(command)`). There is no VNC, display server, screenshot API, or pixel-level action space in the base environment contract. OSWorld tasks require a running desktop session with screenshot-and-click interactions; the Harbor `BaseEnvironment.exec()` is a shell exec over Docker/kubectl/E2B, not a GUI abstraction. **Desktop/CUA benchmarks are structurally unsupported without a new environment backend providing display access.** There is no evidence of such a backend in the codebase (environments: docker, daytona, e2b, modal, runloop, gke — all terminal-only).

**(c) Multi-agent scenarios**

ATIF v1.2+ added `SubagentTrajectoryRef` schema to record delegated subagent trajectories by reference, and the `Observation` object can contain an array of `subagent_trajectory_ref` entries. So the trajectory format can *represent* multi-agent interaction. However, at the orchestration level, each `Trial` runs one agent against one task. There is no native mechanism to launch two cooperating agents within a single trial that interact via a shared environment or message bus. Multi-agent evaluation would require embedding multi-agent logic inside a single agent implementation (as e.g. Claude Code subagents running inside the container). **Multi-agent as a first-class evaluation primitive is not supported.**

**(d) Streaming observations**

There is no streaming API between the sandbox environment and the orchestrator. Agent execution is a batch operation: the agent runs to completion (or timeout) and the verifier then runs. The `BaseEnvironment.exec()` returns an `ExecResult` object (stdout, stderr, return_code) synchronously after the command completes. There is no websocket, event stream, or incremental observation feed. Benchmarks requiring real-time observation streams (e.g., a live game state) would need custom wrapping. **Streaming observations are not supported in the current architecture.**

---

## 2. Auto-Provisioning

Harbor has native auto-provisioning for five cloud backends plus local Docker:

| Backend | How it works |
|---|---|
| **docker** (local) | Calls `docker build` + docker-compose, manages the container lifecycle directly |
| **Daytona** | Uses `daytona>=0.121.0` Python SDK; creates a Daytona workspace per trial, uploads the environment directory, executes commands via the workspace API |
| **E2B** | Uses `e2b>=2.4.2`; sandboxes are created on-demand, files uploaded over the E2B API |
| **Modal** | Uses `modal>=1.3.2`; builds a Modal image from the Dockerfile, runs the agent and verifier as Modal functions |
| **Runloop** | Uses `runloop-api-client>=1.2.0`; creates Runloop devboxes on demand |
| **GKE** | Full Kubernetes implementation: calls `gcloud builds submit` to build and push the Docker image to Artifact Registry (GCR), then schedules a Kubernetes pod per trial. A singleton `KubernetesClientManager` is shared across concurrent trials. Supports Standard and Autopilot clusters. Resource requests/limits are driven by `task.toml` fields. |

Image caching is implemented for GKE (checks if the image already exists in Artifact Registry before rebuilding) and implicitly via Docker's layer cache for local runs.

There is **no AWS/EKS or Azure AKS backend**. The `pyproject.toml` has no boto3 or azure-sdk dependency. The parity API infrastructure uses AWS ELB endpoints (visible in `parity_api_instructions.md`) but that is an inference proxy, not a compute backend.

---

## 3. RL Support

### ATIF trajectory format (v1.6 as of the current codebase)

ATIF is Harbor's JSON-based trajectory specification. Key RL-relevant fields:

| Field (location) | RL relevance |
|---|---|
| `metrics.logprobs` (per-step) | Per-token log probabilities of the completion |
| `metrics.completion_token_ids` (per-step) | Integer token IDs for the generated response, "enabling RL training without retokenization drift" |
| `metrics.prompt_token_ids` (per-step) | Full prompt token IDs including chat history |
| `metrics.prompt_tokens`, `completion_tokens`, `cached_tokens` | Token accounting |
| `agent.tool_definitions` | OpenAI function-calling schema for tools available to the agent, needed for SFT training pipelines |
| `step.reasoning_content` | Explicit chain-of-thought reasoning text |
| `step.is_copied_context` (v1.5+) | Flag to exclude context-summarization steps from training data |
| `subagent_trajectory_ref` | References to delegated sub-trajectories for hierarchical agents |
| `final_metrics.total_cost_usd`, `total_steps` | Episode-level summary |

The Terminus-2 agent is the only built-in agent with first-class ATIF+RL support (`SUPPORTS_ATIF = True`, collects logprobs and token IDs per turn via LiteLLM). The Claude Code, OpenHands, Gemini CLI adapters also set `SUPPORTS_ATIF = True` but reconstruct ATIF post-hoc from agent logs (e.g., Claude Code parses its `.jsonl` session files after the run; logprobs and token IDs are not populated unless the underlying LLM API returns them).

### QueueOrchestrator

The `QueueOrchestrator` (`src/harbor/orchestrators/queue.py`) is an async producer-consumer pool. It:
- Maintains N concurrent async worker coroutines (configurable via `n_concurrent_trials`)
- Accepts `submit(trial_config)` and `submit_batch(configs)` calls that return `asyncio.Future[TrialResult]`
- Supports `start()` / `shutdown(wait=True/False)` for programmatic lifecycle control
- Implements exponential backoff retry logic with configurable include/exclude exception lists
- Enforces a `CONTAINER_LAUNCH_GRACE_PERIOD_SEC = 2.0` stagger between container launches to avoid overwhelming cloud orchestration APIs

This is the key component enabling online RL loops: a training process can call `submit()` for individual rollouts and await results asynchronously, interleaving policy updates with trajectory collection. The `LocalOrchestrator` is a simpler non-queue alternative for sequential runs.

### RL / SFT pipeline connections

- **SFT export**: `harbor traces export --filter success --sharegpt --push` converts ATIF trajectories to ShareGPT format for instruction-tuning. Supports filtering by success/failure, episode selection, and direct push to HuggingFace Hub.
- **RL**: ATIF's `logprobs` + `completion_token_ids` fields are explicitly designed for policy optimization (the RFC states "avoiding retokenization drift"). Harbor's documentation mentions RL as a training workflow module. There is no built-in PPO/GRPO trainer — Harbor generates the rollouts and rewards; an external RL framework (e.g., trl, verl) would consume the exported trajectories.
- **Phoenix/Arize integration**: As of April 2026, Arize Phoenix added `upload_atif_trajectories_as_spans` to ingest Harbor trajectories as OpenTelemetry spans for observability.

---

## 4. Separation from Platform (Adapter Portability)

Harbor "adapters" are Python scripts that convert an upstream benchmark's data format into the Harbor task directory structure. The adapter itself (e.g., `adapters/swebench/adapter.py`) is a standalone Python script with no Harbor runtime dependency — it simply reads from HuggingFace datasets and writes files to disk following the `task.toml` + `instruction.md` + `Dockerfile` + `test.sh` layout.

However, **the output artifacts are not framework-agnostic**:

- `tests/test.sh` writes a reward to `/logs/verifier/reward.txt` — a Harbor-specific convention (path, format, and semantics)
- `task.toml` uses Harbor's proprietary schema (not an open standard like an OCI label or SWE-bench JSON)
- The `environment/Dockerfile` is generic Docker, which is portable
- `instruction.md` is a plain markdown file, fully portable

In practice, reusing a Harbor adapter in a different framework requires: (1) ignoring `task.toml` and re-reading metadata from scratch; (2) substituting the `/logs/verifier/reward.txt` convention with the target framework's reward signal; (3) the `Dockerfile` and `test.sh` logic are the most reusable parts. There is no claim of cross-framework portability in Harbor's documentation.

The parity experiment requirement is strong: every accepted adapter must demonstrate that Harbor scores statistically match the original benchmark scores (overlapping run ranges; minimum 3 runs each side). The `parity_experiment.json` files in each adapter directory record these results. This is a meaningful quality gate that CUBE lacks as a standard.

---

## 5. Implemented Benchmarks (Adapters)

As of v0.1.45 (AgentHarbor, May 2026):

**In the Hub registry (confirmed):**
- terminal-bench/terminal-bench-2 (89 tasks)
- swe-bench/swe-bench-verified (500 tasks)
- scale-ai/swe-bench-pro (731 tasks)
- lawbench/lawbench (1,000 tasks)
- replicationbench/replicationbench (90 tasks)
- stanford/medagentbench (300 tasks)

**Adapters in the codebase (adapters/ directory in AgentHarbor):**
SWE-Bench family: `swebench`, `swebenchpro`, `swesmith`, `swtbench`
Code generation: `aider_polyglot`, `autocodebench`, `compilebench`, `livecodebench`, `humanevalfix`, `evoeval`, `deveval`
Data/reasoning: `dabstep`, `gpqa-diamond`, `aime`, `usaco`, `satbench`
Science/research: `mlgym-bench`, `replicationbench`, `codepde`, `bixbench`, `labbench`, `mmau`
Finance/law: `kumo`, `lawbench`, `financeagent`
Other: `sldbench`, `crustbench`, `quixbugs`, `bird_bench`, `gaia`, `ds1000`, `mmmlu`, `adebench`, `ineqmath`, `algotune`, `qcircuitbench`, `reasoning-gym`, `strongreject`, `swelancer`, `bigcodebench_hard`, `arc_agi_2`, `simpleqa`

**Confirmed absent:** WebArena, OSWorld, WorkArena, Mind2Web, VisualWebArena, AssistGUI — none of these can be accommodated due to the shared-server and/or GUI interface constraints described in Section 1.

The benchmark corpus is heavily skewed toward **coding, terminal tasks, and SWE-style code repair**. Data analysis, reasoning, and science benchmarks are growing but represent a minority. Web browsing and GUI benchmarks are structurally absent.

---

## 6. Tool Reconfigurability

Harbor provides two mechanisms for configuring agent tools:

**Per-task MCP server injection (task.toml):**
The `[environment].mcp_servers` array in `task.toml` declares MCP servers available to the agent for that task. The `BaseAgent.__init__` accepts `mcp_servers: list[MCPServerConfig]` and `skills_dir: str | None`. In the Claude Code adapter, `_build_register_mcp_servers_command()` writes a `~/.claude.json` file containing the MCP server list before the agent runs. This means **MCP servers can be task-scoped**: a task that needs a database tool can declare it; a task that needs no tools can omit it.

**Per-task skills injection:**
The `skills_dir` field in `task.toml` points to a directory inside the container whose contents are copied to the agent's skills configuration directory (e.g., `$CLAUDE_CONFIG_DIR/skills/`). This enables task-specific "skills" (slash commands, custom prompts) for agents that support the concept.

**Tool allow/disallow lists:**
The Claude Code adapter exposes `allowed_tools` and `disallowed_tools` as `CliFlag` entries, passed as `--allowedTools` / `--disallowedTools` flags. These are job-level or agent-instance-level settings, not per-task (they come from the job config's agent arguments, not from `task.toml`).

**What "wrapping a full coding agent" means:**
Harbor does not call LLM APIs directly for installed agents (Claude Code, OpenHands, Codex CLI, etc.). Instead, it installs the agent binary inside the container and runs it as a subprocess. The agent then manages its own LLM calls, tool execution, context management, and stopping criteria. Harbor only provides: (1) the task instruction as a CLI argument; (2) the container environment; (3) capture of stdout for logging; (4) post-hoc ATIF trajectory parsing from agent logs.

By contrast, Terminus-2 is a Harbor-native agent that calls LLMs directly via LiteLLM, giving Harbor full control over prompting, token tracking, logprob collection, and tool definition injection.

**Implication:** For installed agents (Claude Code, OpenHands), Harbor cannot modify what tools the agent has access to beyond what the agent itself exposes as CLI flags. You cannot, for instance, add a custom tool to Claude Code's tool set from within Harbor — you can only configure MCP servers or allowlists that Claude Code already supports.

---

## 7. Other Differentiating Aspects

### Parity experiment validation as a standard

Every benchmark adapter must pass a mandatory parity experiment before acceptance into the registry. The criterion is that the score distributions from the Harbor adapter and the original benchmark must overlap (neither side's minimum exceeds the other's maximum). This is enforced by the adapter review process and documented in per-adapter `parity_experiment.json` files. Example: the SLDBench adapter shows Terminal-Bench R²=0.381±0.095 vs. Harbor R²=0.392±0.107 — close enough to pass. This is a quality guarantee that Harbor results are directly comparable to published benchmark scores.

**CUBE does not have an analogous standard.** CUBE benchmarks are expected to be the canonical implementations; Harbor adapters must replicate an existing canonical implementation.

### "Wrapping a full coding agent" vs. wrapping a raw execution API

The distinction is meaningful for evaluation fidelity. When Harbor evaluates Claude Code, it runs the actual Claude Code binary with all its built-in behaviors (file editing heuristics, tool selection, context management, multi-turn loop). The benchmark score reflects the agent as users actually deploy it. When CUBE (or AgentLab/BrowserGym) wraps a raw LLM API to evaluate on WorkArena/WebArena, the scaffold is part of the experiment — different scaffolds on the same model produce different scores.

Harbor's approach removes scaffold variance for installed-agent evals but introduces a different problem: you cannot ablate individual agent components (you cannot easily swap the planning module while keeping the tool set constant). It is better suited for "how does Agent X perform on Benchmark Y" than for "how does design choice Z affect performance."

### Multi-backend concurrency at scale

The combination of QueueOrchestrator + GKE/Daytona/Modal backends genuinely enables 50–100 concurrent trials (demonstrated in the GKE config: `n_concurrent_trials: 50`). This is a real operational advantage over frameworks that are primarily Docker-local.

### No concept of benchmark-level state or shared infrastructure

Each Harbor trial is a hermetic, stateless container. There is no Harbor equivalent of CUBE's `environment.reset()` that can return an existing shared server to a known state without teardown. This is both a strength (complete isolation, no cross-contamination) and a limitation (shared-server benchmarks are excluded).

### Hub / Registry ecosystem

The Harbor Hub (hub.harborframework.com) provides a benchmark sharing registry comparable to HuggingFace Hub for datasets. Tasks are versioned by git commit ID, fetched on demand, and cached locally. This is a deliberate ecosystem play to become the "npm for agent benchmarks."

---

## Key Claim for Paper

**What CUBE does that Harbor doesn't:** CUBE supports benchmarks with shared persistent environments across tasks (e.g., WebArena's single web server, WorkArena's ServiceNow instance) and GUI/CUA benchmarks (OSWorld, WorkArena), covering the full spectrum of agentic evaluation including web browsing and desktop tasks — categories that Harbor structurally cannot accommodate due to its per-task hermetic container model.

**What Harbor does that CUBE doesn't:** Harbor provides a first-class RL/SFT data pipeline (ATIF trajectory format with logprobs and token IDs, QueueOrchestrator for online rollout collection, ShareGPT export) and native cloud auto-provisioning at scale (GKE, Daytona, Modal, E2B, Runloop), positioning it as an end-to-end post-training infrastructure rather than a pure evaluation standard.

---

## Summary Table

| Axis | Harbor | CUBE |
|---|---|---|
| Task interface | task.toml + instruction.md + Dockerfile + test.sh | cube.yaml + Python task class |
| Shared-server benchmarks (WebArena) | Not supported (per-task container model) | Supported |
| CUA/desktop (OSWorld) | Not supported (no GUI backend) | Supported |
| Multi-agent | Not natively (ATIF can represent, not orchestrate) | TBD |
| Streaming observations | Not supported | TBD |
| Cloud backends | Docker, GKE, Daytona, Modal, E2B, Runloop | TBD |
| RL trajectory format | ATIF (logprobs, token IDs, tool defs) | CUBE trajectory format |
| SFT export | ShareGPT via `harbor traces export` | N/A |
| Adapter portability | Harbor-specific conventions | CUBE-specific |
| Parity validation | Mandatory, quantified | Not a standard requirement |
| Implemented benchmarks | ~40 (coding/terminal/science heavy) | WebArena, WorkArena, OSWorld, etc. |
| Primary bias | Coding agents in containers | Web/GUI/enterprise agents |

---

## Sources Consulted

- `/Users/alexandre.lacoste/dev/AgentHarbor/` — v0.1.45 source (primary)
- `/Users/alexandre.lacoste/dev/harbor/` — v0.1.19 source (older branch)
- arXiv:2601.11868 — Terminal-Bench 2.0 paper
- https://harborframework.com/docs
- https://hub.harborframework.com
- https://arize.com/docs/phoenix/release-notes/04-2026/04-03-2026-atif-trajectory-upload
