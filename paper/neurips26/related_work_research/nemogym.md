# NeMo Gym: Research Assessment for CUBE Related Work

**Date:** 2026-05-03
**Repo examined:** `~/dev/NemoGym/` (github.com/NVIDIA-NeMo/Gym, Apache 2.0)
**Companion paper:** ProRL Agent (arXiv 2603.18815, March 2026) — extends NeMo Gym with a decoupled rollout-as-a-service layer
**Status:** Early development ("Beta"), APIs evolving. No companion NeurIPS/ICML paper for Gym itself; the Nemotron-3 white paper (Dec 2025) is the primary academic reference that uses it.

---

## Overview

NeMo Gym is NVIDIA's open-source library for building RL training environments for LLMs. Its stated purpose is generating verified rollout trajectories ("rollout collection") for RLVR (Reinforcement Learning from Verifiable Rewards) post-training. The architecture decomposes a training environment into three FastAPI microservices — a **Resources Server**, a **Model Server**, and an **Agent Server** — connected exclusively via HTTP. Ray is used as the distributed coordination layer.

The library is part of NVIDIA's broader NeMo ecosystem (NeMo RL for training, NeMo Evaluator for evaluation, NeMo Gym for RL environment infrastructure).

---

## Axis-by-Axis Assessment

### 1. Schema Flexibility

**Core interface.** The canonical task/environment interface is the OpenAI **Responses API** schema (`NeMoGymResponseCreateParamsNonStreaming`), a Pydantic model in `nemo_gym/openai_utils.py` that is a validated copy of OpenAI's `ResponseCreateParamsNonStreaming` TypedDict. The trajectory format is a list of `NeMoGymResponseInputItem` items, a union of:
- `NeMoGymEasyInputMessage` / `NeMoGymMessage` (text turns)
- `NeMoGymResponseOutputMessage` (assistant text response)
- `NeMoGymResponseFunctionToolCall` (tool invocation)
- `NeMoGymFunctionCallOutput` (tool result)
- `NeMoGymResponseReasoningItem` (chain-of-thought)

For training, each of these item types has a `*ForTraining` variant (e.g., `NeMoGymResponseOutputMessageForTraining`) that mixes in `TokenIDLogProbMixin`, adding `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs` fields. This is the key extension that makes the format usable for RL gradient computation. The Chat Completions API is also supported (`NeMoGymChatCompletionCreateParamsNonStreaming`) via a parallel class hierarchy.

**What it handles well.** Text/tool-call agentic tasks, multi-step (up to 65,000 concurrent requests per their documentation), multi-turn conversational scenarios, single-step MCQA/math. The three-server architecture cleanly separates orchestration from environment logic.

**What it does not accommodate (or accommodates with friction):**

- **CUA / desktop tasks.** No native abstraction for pixel-level computer-use, mouse/keyboard action spaces, or VM snapshots. The `NeMoGymResponseCreateParamsNonStreaming` schema carries no `screenshot`, `viewport`, or `action_type` concept. The MiniSWE agent (`resources_servers/mini_swe_agent/`) uses Singularity/Apptainer containers for code execution but actions are still text-only tool calls. BrowserGym is listed as "in progress" in the ecosystem doc (`docs/about/ecosystem.md`), not yet integrated.

- **Shared persistent servers.** NeMo Gym's session model uses HTTP cookies to persist per-rollout state across tool calls within a single rollout. This handles stateful multi-step tasks. However, there is no concept of a benchmark-level shared server (e.g., a single WebArena instance serving 100 concurrent rollouts against the same web app). Each `seed_session` call initializes a per-rollout session. Tasks requiring a warm, pre-populated web application or database server shared across tasks are not natively provisioned.

- **Multi-agent scenarios.** The agent loop in `SimpleAgent.responses()` is a single-model loop (one `model_server` reference per agent config). There is no native multi-agent routing, agent-to-agent communication protocol, or competitive/cooperative agent schema.

- **Streaming.** The `NeMoGymResponseCreateParamsNonStreaming` class explicitly locks `stream: Optional[Literal[False]] = None`. Streaming is architecturally excluded in the current design.

- **Schema rigidity.** The `model_config = ConfigDict(extra="forbid")` on `NeMoGymResponseCreateParamsNonStreaming` means any field not in the OpenAI Responses API schema is rejected at the server boundary. Benchmarks that need task-specific metadata in the request (e.g., a task ID, privileged ground truth) must pass it through data JSONL rows rather than the live request, or through the `metadata` field (a free-form dict). Environment-specific task parameters (e.g., `instance_id`, `expected_synonym_values`) are carried in custom `BaseRunRequest` subclasses, but these are resources-server-internal and not part of the on-wire API contract.

---

### 2. Auto-Provisioning

**No auto-provisioning framework.** NeMo Gym does not manage Docker builds, VM snapshots, or benchmark-level shared infrastructure automatically. The `ng_run` command starts FastAPI servers as OS subprocesses (with isolated `uv` virtual environments per server), but all containers or external services that an environment needs must be provisioned manually by the benchmark implementor.

The SWE-style environments (`swerl_gen`, `mini_swe_agent`) use Singularity/Apptainer containers, but this is implemented as custom Python logic inside each resources server's `app.py`, not as a framework primitive. From `docs/infrastructure/engineering-notes/swe-rl-case-study.md`:

> "Inside the Apptainer container, we will run the harness logic... After the trajectory finishes, we will spin up another instance of the same task container to run final unit tests validation."

Each benchmark author is responsible for:
- Obtaining and caching Singularity `.sif` images
- Writing `prepare.py` scripts (called by `ng_prepare_benchmark`) to download and format datasets
- Managing container lifecycle inside their `verify()` implementation

The `mini_swe_agent` config (`resources_servers/mini_swe_agent/configs/mini_swe_agent.yaml`) shows manual fields such as `env: singularity`, `cache_dir_template: ???`, `step_timeout: 600`, and `eval_timeout: 1800`, all requiring manual configuration. The `???` value for `cache_dir_template` is an OmegaConf "missing required field" sentinel, meaning this must be supplied at runtime by the operator.

**Conclusion:** Environment lifecycle is entirely user-managed. NeMo Gym provides scaffolding (server startup, health checks, Ray coordination) but no provisioning abstraction.

---

### 3. RL Support

This is NeMo Gym's defining strength.

**Training framework integrations.** From `docs/about/ecosystem.md` and `docs/reference/rl-framework-compatibility.md`:
- **NeMo RL** (NVIDIA's own GRPO/DPO/SFT toolkit) — primary, with a dedicated tutorial and version-pinned container compatibility table
- **OpenRLHF** — an example `agent_func_nemogym_executor.py` is maintained in the OpenRLHF repo
- **Unsloth** — GRPO training tutorial, tested on a specific pinned version
- **VeRL** — listed as "in progress" in the ecosystem docs; ProRL Agent (arXiv 2603.18815) validates VeRL integration

**Trajectory format for training.** The `TokenIDLogProbMixin` (fields: `prompt_token_ids: List[int]`, `generation_token_ids: List[int]`, `generation_log_probs: List[float]`) is injected at generation time by the Model Server (which must be a NeMo Gym-native model server, i.e., a vLLM endpoint via `local_vllm_model` or `vllm_model`). The integration doc makes this explicit:

> "During training, NeMo Gym models will return additional information on response messages... prompt_token_ids, generation_token_ids, and generation_log_probs. When constructing messages for the next model call in multi-step or multi-turn scenarios, propagate this information from the previous model response."

Token IDs are the canonical representation; re-tokenization is never required. This avoids tokenizer drift in multi-step rollouts and is the core of what ProRL Agent calls the "token-in/token-out protocol."

**Rollout collection throughput.** Per `docs/environment-tutorials/integrate-external-environments.md`:

> "A single NeMo Gym instance that consists of three FastAPI server instances can handle 65,000 concurrent math rollouts without crashing."

For SWE-bench style tasks (`docs/infrastructure/engineering-notes/swe-rl-case-study.md`), throughput is significantly lower: typical RL training uses batch size 512 concurrent Singularity containers, targeting ~0.37 instances/sec (from ProRL Agent paper ablations).

**Reward format.** A single `float` field on `BaseVerifyResponse`. The `/aggregate_metrics` endpoint (called post-rollout) computes `AggregateMetrics` containing `group_level_metrics` (per-task), `agent_metrics` (overall), and `key_metrics` (headline subset). W&B and MLflow logging are built in.

**Multi-environment (multi-verifier) training.** Natively supported. Multiple agent/resource server pairs can be run simultaneously, and a single training dataset can route different rows to different agents via the `agent_ref` field. See `docs/training-tutorials/multi-environment-training.md`.

**ProRL Agent extension (arXiv 2603.18815).** This companion paper (March 2026) adds a decoupled "rollout-as-a-service" API layer above NeMo Gym, handling SWE-bench-scale containerized tasks with:
- Singularity `.sif` caching (Scratch/Versioned/Lock modes)
- Near-linear throughput scaling with compute nodes
- Action execution time optimized from 0.78s to 0.42s
- VeRL and NeMo RL as validated training backends
- Thread-safe loopback IP allocation for container isolation

---

### 4. Separation from Platform

**Tight platform coupling.** An environment implemented for NeMo Gym is NOT portable without adaptation. The requirements for portability are:

1. **FastAPI server as the interface.** The Resources Server must implement `/seed_session`, `/verify`, and `/aggregate_metrics` endpoints with specific Pydantic schemas (`BaseVerifyRequest`, `BaseVerifyResponse`, `AggregateMetricsRequest`, `AggregateMetrics`). An evaluator that does not speak these endpoints cannot use the environment.

2. **OpenAI Responses API as the wire format.** The task/observation representation is an OpenAI message list. Harnesses that use non-OpenAI schemas (e.g., BrowserGym's `dict`-based obs/action, Gymnasium's `np.ndarray` observation space) cannot be connected without an adapter layer. The `docs/environment-tutorials/integrate-external-environments.md` describes exactly this adaptation cost for external libraries.

3. **NeMo Gym model server for training.** The `TokenIDLogProbMixin` fields are populated only by NeMo Gym-compatible model servers (vLLM with the NeMo Gym adapter). Plugging in a third-party inference server (LiteLLM, Anthropic client, etc.) loses token-ID propagation and breaks RL training.

4. **No standalone benchmark runner.** There is no `ng eval` or standalone evaluation command equivalent to `cube test`. Rollout collection (`ng_collect_rollouts`) is the primary evaluation mechanism, requiring all three server types to be running. This is functional but heavier than a pure benchmark execution path.

The `ng_test` CLI does run environment-level tests, but these test the server implementation, not end-to-end benchmark scores. Benchmarks (in `benchmarks/`) require `ng_prepare_benchmark` + `ng_collect_rollouts` to produce scores.

**Portability verdict:** An environment implemented natively for NeMo Gym (Resources Server subclass) cannot be trivially run on AgentLab, HAL, or CUBE without wrapping. The integration guide acknowledges this and documents the adapter pattern as the supported path.

---

### 5. Implemented Benchmarks

The `benchmarks/` directory contains 12 formal benchmark configurations:
`aalcr`, `aime24`, `aime25`, `browsecomp`, `gpqa`, `livecodebench`, `mmlu_pro`, `mmlu_prox`, `nemotron_3_ultra`, `ruler`, `spider2_lite`, `xstest`

The `resources_servers/` directory contains 40+ training environments (see README table), including:
- **Math:** `math_with_judge` (DAPO17k, OpenMathReasoning, MathStackOverflow), `math_with_code`, `math_formal_lean`, `newton_bench`, `gpqa_diamond`, `arc_agi`, `nvarc`
- **Coding:** `code_gen`, `swerl_gen`, `swerl_llm_judge`, `mini_swe_agent`, `spider2_lite`, `cvdp`
- **Agent/Tool-use:** `workplace_assistant`, `single_step_tool_use_with_argument_comparison`, `aviary`, `calendar`, `google_search`, `tavily_search`, `finance_sec_search`, `terminus_judge`
- **Knowledge/MCQ:** `mcqa`, `multichallenge`, `reasoning_gym` (100+ tasks), `vlm_eval_kit`
- **Safety:** `jailbreak_detection`, `xstest`, `over_refusal_detection`
- **Instruction following:** `instruction_following`, `structured_outputs`
- **RLHF:** `abstention`, `genrm_compare`
- **Other:** `openenv` (MCP-based echo/coding/maze), `circle_click`

**Domain coverage:** Math, coding, tool-use, knowledge QA, safety, instruction following, formal proofs. No web navigation (BrowserGym integration listed as "in progress"), no desktop/GUI tasks, no multi-agent scenarios.

---

### 6. Tool Reconfigurability

**Per-environment, not per-experiment.** Tools in NeMo Gym are HTTP endpoints registered on the Resources Server's FastAPI app in `setup_webserver()`. Each call to `app.post("/{tool_name}")` registers a tool. The list of tools available to the agent is determined by the `tools` field of `NeMoGymResponseCreateParamsNonStreaming` (a `List[ToolParam]`), which is set per-row in the input JSONL or overridden globally via the `+responses_create_params.tools=...` CLI flag.

In practice: you can inject a different tool list at rollout time via data-row overrides or CLI `responses_create_params` overrides. The Resources Server will route any tool call to `/{tool_call.name}` dynamically, so if the tool is registered on the server, it will be called. If a tool in the list is not registered, it will return an error (which the agent receives as a tool output, not a crash — this is by design per the integration guide).

**Tool sharing across benchmarks.** The agent server doc states: "Prefer defining tools in the Resources server rather than the Agent server. This separation of concerns allows different agents to share the same Resources server without duplicating tool logic." Tools are not first-class shareable objects in a registry; sharing happens by pointing multiple agent configs at the same resources server.

**Bottom line:** Tool swap is possible but requires config-level wiring, not a first-class `tool_set` experiment parameter. There is no mechanism analogous to CUBE's reconfigurable tool surfaces that can be swapped at benchmark registration time independently of the resources server implementation.

---

### 7. Other Differentiating Aspects

**Distributed scale via Ray.** Ray is a core dependency (`ray[default]`). The `ng_run` command initializes a Ray cluster (or connects to an existing one via `ray_head_node_address`). Each server subprocess can use `ray.remote` for fan-out parallelism within its rollout handler. This is how `swerl_gen` achieves concurrent Singularity container execution at scale (512 concurrent instances across GPU nodes in the SWE-RL case study).

**Independent virtual environments per server.** Each server is launched in its own `uv` venv. This isolates dependency conflicts between, e.g., a math server requiring `sympy` and a coding server requiring a specific `torch` version. This is a practical engineering advantage over monolithic harness designs.

**Session state via cookies.** Multi-step rollout state is maintained via Starlette `SessionMiddleware` and cookie propagation. The agent server propagates cookies between model and resources server calls. This is lightweight and stateless from the framework perspective — no external state store required.

**Hydra + OmegaConf configuration.** All configuration is YAML-based and composable via Hydra's `+config_paths` override syntax. This enables modular server composition, config inheritance (`_inherit_from` directives in benchmark configs), and runtime parameter overrides.

**No benchmark registry.** Unlike CUBE, there is no registry of benchmark packages. All environments live in the monorepo. Contributing a new environment means a PR to the NeMo Gym main branch, with code review by NVIDIA. The contributing guide states: "Variables are passed through NeMo Gym config and not through environment variables." This monorepo model provides quality control but does not support community-contributed external packages.

**W&B and MLflow integration.** Rollout collection (`ng_collect_rollouts`) automatically uploads rollout tables to W&B and key metrics to both W&B and MLflow if configured. This is production-grade instrumentation baked into the framework.

**No evaluation-only path.** NeMo Gym is designed for RL training. Evaluation (benchmark scoring) happens as a side-effect of rollout collection. There is no `ng_eval` command or benchmarking mode that does not also produce training trajectories.

---

## Summary Table

| Axis | NeMo Gym Assessment |
|------|---------------------|
| **Task interface** | OpenAI Responses API (`NeMoGymResponseCreateParamsNonStreaming`); `BaseVerifyRequest`/`BaseVerifyResponse` on resources server |
| **Trajectory format** | OpenAI message list + `TokenIDLogProbMixin` (token IDs + log-probs) for RL |
| **Schema flexibility** | Strong for text/tool-call agents; no CUA/desktop, no streaming, no shared persistent servers |
| **Auto-provisioning** | None from the framework; containers managed per-environment in custom `app.py` code |
| **RL support** | First-class: NeMo RL, OpenRLHF, Unsloth; VeRL via ProRL Agent; GRPO native; 65k concurrent math rollouts |
| **Platform coupling** | Tight: FastAPI endpoints + OpenAI wire format + NeMo model server required for training; not portable as-is |
| **Implemented environments** | 40+ training environments + 12 benchmark configs; math/code/tool-use/knowledge/safety; no web nav or desktop |
| **Tool reconfigurability** | Config-level (per row or CLI override of `tools` list); no first-class experiment-level tool surface swap |
| **Distributed scale** | Ray-native; per-server isolated venvs; near-linear throughput scaling (ProRL Agent) |
| **Benchmark registry** | None: monorepo, PR-based contribution |
| **Paper** | No dedicated paper; cited in Nemotron-3 (Dec 2025); ProRL Agent (arXiv 2603.18815, March 2026) extends it |

---

## Key Claim for Paper

NeMo Gym provides best-in-class RL training infrastructure for text/tool-call LLM agents at GPU-cluster scale, but it does not provide a benchmark portability standard: environments are tightly coupled to the NeMo Gym server architecture, lack auto-provisioning for the shared-server and VM-snapshot resource patterns that web and desktop benchmarks require, and cannot be run on other evaluation harnesses (AgentLab, HAL, CUBE) without a custom adapter layer; CUBE addresses exactly this portability and provisioning gap, and is designed to be harness-agnostic so that a benchmark implemented once can run on NeMo Gym or any other orchestrator.

---

## Notes for Paper Drafting

- NeMo Gym is mentioned by name in the CUBE paper's intro and standard section; position it in paragraph (2) of Related Work (harnesses/platforms) alongside AgentBeats, HAL, OpenEnv.
- The complementarity angle is strong: NeMo Gym is the RL training platform; CUBE provides the evaluation standard that feeds it. A CUBE-standard benchmark can be wrapped in a NeMo Gym resources server, giving NeMo Gym access to CUBE's community-contributed environments. This is worth one sentence in Related Work.
- BrowserGym integration is "in progress" in NeMo Gym's ecosystem page, which supports the claim that web-navigation benchmarks are not yet natively supported there.
- The ProRL Agent paper (2603.18815) is the closest to a peer-reviewed publication about this system; cite it if citing NeMo Gym academically. The GitHub repo BibTeX (`@misc{nemo-gym, ...}`) is the official citation format.
