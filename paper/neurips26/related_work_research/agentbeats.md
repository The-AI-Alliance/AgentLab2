# AgentBeats — Related Work Assessment
*For NeurIPS 2026 CUBE position paper — researched 2026-05-03*

**Primary sources consulted:**
- White paper: `~/dev/agentbeats-white-paper/AgentBeats___white_paper__sosp_format_/` (full LaTeX source, SOSP camera-ready format)
- AgentHarbor code: `~/dev/AgentHarbor/` (Harbor framework — the companion evaluation harness created by the same team; appears to be what gets described as the execution layer)
- Note: "AgentHarbor" / "Harbor" is a distinct but closely related project from the same lab (creators of Terminal-Bench). AgentBeats is the *paradigm + orchestration platform*; Harbor is a *separate container-based harness* that overlaps in the RL/execution space. The white paper discusses AgentBeats; the Harbor repo provides the concrete engineering artefacts. These are treated as a single ecosystem below but distinctions are noted.

---

## 1. Schema Flexibility

### Core interface
AgentBeats instantiates the **Agentified Agent Assessment (AAA)** paradigm. The interface between a benchmark (called a **green agent** or *assessor*) and the evaluated agent (called a **white agent** or *assessee*) is entirely mediated by two existing open protocols:

- **A2A (Agent2Agent protocol, v0.3.0)** — for task management: task specification, progress tracking, multimodal data exchange, bidirectional communication.
- **MCP (Model Context Protocol, 2025-11-25 spec)** — for tool access: structured, format-enforced calls to environment functions.

There is no benchmark-specific harness API. A benchmark becomes a runnable A2A service ("green agent") that sends task instructions and exposes MCP tools; the evaluated agent needs only to accept A2A tasks and connect to MCP servers.

### Benchmark types it claims to accommodate
The white paper explicitly asserts coverage of "all known agent types":
- Terminal-use / coding agents (via thin A2A wrapper that forwards instructions to terminal)
- Browser-use and computer-use (multimodal A2A exchange; green agent exposes browser/screen MCP tools)
- Multi-agent evaluations (adversarial and collaborative; multiple white agents receive separate A2A task streams)
- Tool-use / user-simulation style benchmarks (e.g., Tau-Bench ported as green agent)
- Gym / RL environments (Gym-to-MCP translation layer maps Gymnasium-style APIs to self-descriptive MCP tools)

### Limits and rigidity concerns

**Shared persistent infrastructure (e.g., WebArena's servers):** Not natively addressed at the protocol level. The white paper acknowledges a "Sandbox and Container Manager" extension that centralizes resource management for green agents, but this is described as an *optional platform extension*, not a core protocol guarantee. There is no standardized `ContainerConfig` / `ContainerBackend` abstraction equivalent to CUBE's. Each green agent is responsible for its own environment setup. A green agent wrapping WebArena would need to manage the WebArena servers itself and expose MCP tools pointing at them — doable but not automated by AgentBeats.

**Dynamic MCP tool loading:** Evaluated agents are not required to support runtime MCP reconfiguration. AgentBeats addresses this through an **MCP Gateway** extension with a fixed endpoint: the green agent registers tools dynamically behind the gateway; the white agent always connects to the same static URL. This is an engineering workaround, not a core schema feature.

**Streaming observations / async tasks:** The A2A protocol supports bidirectional communication and progress updates; long-running tasks with partial scores are mentioned in a TODO section (`600_migrate.tex`). Concrete streaming support is aspirational in the current white paper draft.

**CUA/desktop benchmarks:** Procedure-oriented CUA (where scoring depends on intermediate steps) requires the green agent to expose MCP tools for screen states and keyboard/mouse actions. The paper describes this pattern but does not provide a reference green agent for any desktop benchmark.

**Schema rigidity overall:** The schema is intentionally *minimal* — any A2A- and MCP-compatible agent works without additional integration. The trade-off is that task-specific semantics (scoring granularity, partial credit, intermediate checkpoints) are pushed entirely into the green agent, with no standardized cross-benchmark conventions. This makes the interface highly flexible but also means there is no shared vocabulary for e.g. "reset a task" or "evaluate a trajectory."

---

## 2. Auto-Provisioning

### AgentBeats (the paradigm/platform)
AgentBeats **does not define a standardized infrastructure lifecycle protocol**. Environment management is encapsulated entirely inside each green agent. The platform optionally provides a Sandbox and Container Manager service that green agents can use, but this is an extension, not a requirement.

In **Local Mode**, developers manually write scripts that launch all agent services on predetermined ports before running an evaluation. In **Hosted Mode**, the AgentBeats platform handles agent (white agent) instantiation from GitHub repos or Docker images, but green agent infrastructure is still self-managed.

There is no equivalent to CUBE's `cube/spawn`, `cube/reset`, or standardized `ContainerBackend` across the platform.

### Harbor (companion harness)
Harbor provides much more concrete auto-provisioning, but as a *separate system*:

- **Docker backend:** Per-task Docker Compose environments built from `environment/Dockerfile`; parallel image build locks prevent redundant rebuilds.
- **Cloud backends:** Daytona, Modal, E2B, GKE, Runloop — selected via `--env <provider>` CLI flag. Up to 100 concurrent trials on Daytona as shown in the README example.
- **Task schema:** Each task lives in a directory with `instruction.md`, `task.toml` (declaring `EnvironmentConfig` with `docker_image`, `cpus`, `memory_mb`, `storage_mb`, `gpus`, `allow_internet`, and `mcp_servers`), `environment/`, `solution/`, and `tests/`.
- **Lifecycle:** `Trial` class handles environment start → agent run → verifier → teardown automatically.

**Key limitation (Harbor):** The task format is **Harbor-specific** (`task.toml` + Dockerfile + `test.sh` pattern). Benchmarks packaged for Harbor do not automatically run on other evaluation platforms without re-packaging.

---

## 3. RL Support

### AgentBeats
RL support is described as **partial / aspirational**:
- The Gym-to-MCP translation layer ("expose GYM with MCP") lets MCP-compatible agents interact with RL environments without native Gym support.
- The white paper notes integration potential with "existing RL training infrastructure" and OpenEnv, but this is framed as future/extension work.
- AAA provides "an intermediate layer that facilitates tracking and analysis of agent behavior across training and evaluation" — but no concrete trajectory format or logprob capture is specified at the AgentBeats protocol level.
- The `340_ab_scale.tex` section on scalable assessments is marked as a TODO/placeholder in the draft.

### Harbor (companion harness)
Harbor has substantially more developed RL support:
- **ATIF (Agent Trajectory Interchange Format)**, RFC v1.6, is a JSON-based trajectory standard capturing: per-step `completion_token_ids`, `prompt_token_ids`, `logprobs`, `reasoning_content`, `tool_calls`, `observations`, `cost_usd`, and `subagent_trajectory_ref` for hierarchical agents.
- **QueueOrchestrator:** Producer-consumer orchestrator for dynamic trial submission; designed explicitly for RL training loops (SkyRL integration pattern documented). Supports `submit()` and `submit_batch()` with asyncio futures, hooks for real-time policy updates, and configurable concurrency.
- **Framework integrations (claimed):** LiteLLM, MLflow, W&B for observability; SkyRL for training. No direct RLHF/PPO framework (e.g., TRL, OpenRLHF, NeMo RL) integration is implemented in the current codebase — these are listed as integration targets.
- **Trajectory capture:** The `Terminus 2` agent in Harbor is the reference implementation of ATIF and demonstrates full token-ID + logprob logging.

**Summary:** Harbor (not AgentBeats proper) has genuine RL trajectory infrastructure. The AgentBeats white paper's RL story is thin; it depends on Harbor or similar execution engines to provide the actual rollout machinery.

---

## 4. Separation from Platform / Benchmark Portability

### AgentBeats
**By design, benchmark logic is self-contained in the green agent.** A green agent is a runnable A2A service. It can be deployed in all five operation modes (Local, Remote, Hosted, Proxy, CI) with "no or minimal changes to agent implementations." Any A2A-compatible evaluation framework can, in principle, serve as the delegator and invoke a green agent.

The white paper explicitly positions this as a differentiator: "benchmark logic is self-contained in the judge agent, portable across all five AgentBeats operation modes."

**Practical caveat:** Portability is at the *interaction protocol* level (A2A/MCP), not at the *infrastructure packaging* level. If a green agent depends on specific AgentBeats platform services (MCP Gateway, Agent Control Plane, Sandbox Manager), it is not fully portable outside that platform. In Local Mode, running without the platform is fully portable; in Hosted or Remote Mode, platform dependencies exist.

The white paper acknowledges CUBE as addressing a complementary, lower layer: "CUBE standardizes how individual benchmark environments are packaged, deployed, reset, and lifecycle-managed, independently of any evaluation framework." A green agent can use a CUBE-compliant benchmark as its internal infrastructure substrate — the two are described as composable and mutually reinforcing.

### Harbor
**Harbor-specific format.** The `task.toml` + Dockerfile + `test.sh` structure works only within Harbor's Trial/Environment machinery. Harbor lists this as "Limited: Harbor-specific task.toml + Dockerfile format" in its own comparison table. Adapters for existing benchmarks (SWE-Bench, Aider Polyglot, etc.) are Harbor-specific.

---

## 5. Implemented Benchmarks

### AgentBeats (competition-derived green agents)
Phase 1 of the AgentX-AgentBeats competition (run in CI Mode via GitHub Actions) received **approximately 250 green agent submissions** spanning **17 evaluation tracks**:

| Track | Submissions |
|---|---|
| Multi-agent evaluation | 34 |
| Coding agents | 30 |
| Agent safety | 27 |
| Research agents | 27 |
| Cybersecurity | 21 |
| Web agents | 19 |
| Game agents | 17 |
| Finance agents | 16 |
| (other tracks) | ~59 |

Among "porting" submissions, **38+ distinct existing benchmarks were agentified**, including: Tau2-Bench (7 submissions), Finance Agent Benchmark (5), CyberGym (4), Werewolf (4), MedAgentBench (4), SWE-Bench (3), Online-Mind2Web (3), GAIA (2), and OSWorld.

**Important caveat:** These are community competition submissions, not officially maintained or compliance-tested benchmarks. There is no equivalent to CUBE's compliance registry or `cube test` verification suite for green agents.

### Harbor
The `registry.json` contains **65 benchmark entries** (some duplicates for different versions of the same benchmark). Named entries include: terminal-bench, terminal-bench-pro, swebench-verified, swebenchpro, swesmith, swe-lancer-diamond, gaia, gpqa-diamond, aider-polyglot, livecodebench, usaco, mmmlu, simpleqa, arc-agi-2, medagentbench, bigcodebench-hard, bfcl, bird-bench, reasoning-gym, and more.

These are maintained benchmarks with Docker environments, not community submissions.

---

## 6. Tool Reconfigurability

### AgentBeats
Tool reconfigurability is a **first-class concern**, addressed through the MCP Gateway:
- The green agent registers task-specific MCP tools dynamically per evaluation.
- The white agent connects to a **fixed MCP Gateway URL** once during setup; the gateway transparently proxies to whichever tools the green agent has granted access for that session.
- The white agent does *not* need to be modified to receive different tool sets across benchmarks.
- This design is explicitly intended to support **generalist agents**: "AgentBeats targets general-purpose agents that can compete across multiple benchmarks under a unified interface."

Per-experiment tool reconfiguration is handled by the green agent's design (which tools to expose), not by the white agent or the platform CLI.

### Harbor
Tool configuration is per-task through `task.toml`'s `mcp_servers` list. Tasks can declare named MCP servers with `sse`, `streamable-http`, or `stdio` transports. Agents that support MCP (configured via `skills_dir`) can consume these. However, tool sets are baked into task configurations at authoring time; there is no gateway-style dynamic reconfiguration.

Supported agents: Claude Code, Codex, Aider, OpenHands, Gemini CLI, Goose, Kimi CLI, Cursor CLI, QwenCoder, OpenCode, Terminus 2. All are terminal/coding agents — Harbor is **weakly suited for generalist cross-benchmark agents** since task instructions are delivered to a terminal, not via A2A.

---

## 7. Other Differentiating Aspects

### The AAA Paradigm
The core intellectual contribution of AgentBeats is the **Agentified Agent Assessment (AAA)** paradigm: benchmarks themselves become agents (green agents). This reframes evaluation as an agentic task and argues it reduces N×M benchmark-agent integrations to N+M protocol-level integrations. The "benchmark as agent" framing is novel and has architectural implications (multi-agent evaluation becomes natural, LLM-as-judge is trivially integrated, benchmark logic is version-controlled and shareable as a service).

### Operation Modes (five)
AgentBeats defines five deployment modes addressing real-world constraints:
1. **Local Mode** — fully self-contained, no platform dependency.
2. **Remote Mode** — platform orchestrates; agents remain on developer infrastructure; closed-source agents supported.
3. **Hosted Mode** — platform instantiates agents from GitHub repos or Docker images.
4. **Proxy Mode** — developer's local agent connects via reverse tunnel to platform assessments; enables testing against registered agents without full deployment.
5. **CI Mode** — assessments trigger on git push via GitHub Actions; used in the AgentX competition; fully auditable.

No other system in the comparison space (HAL, Harbor, NeMo, CUBE) offers this range of deployment modes with explicit support for privacy (remote mode keeps agents private), openness (hosted mode makes blueprints public), and auditability (CI mode).

### Test-Production Alignment
A central claim: by reusing A2A and MCP — protocols already used in production agent systems — evaluated agents behave as they would in deployment. Traditional benchmark-specific harnesses create test-production mismatch because they force bespoke integrations. This is a positioning argument, not empirically validated in the paper.

### Pricing / Deployment
AgentBeats is described as an academic/community platform (associated with UC Berkeley RDI and the Agentic AI MOOC with ~40,000 registered learners). No pricing information appears in the white paper. The AgentX-AgentBeats competition is the primary public deployment. The paper describes a platform architecture but does not reference a production URL.

Harbor is open-source (MIT license based on the GitHub structure; from the Laude Institute / Terminal-Bench team). It integrates with commercial cloud providers (Daytona, Modal, E2B, Runloop, GKE) for scale; users bring their own API keys.

### Agent Control Plane
An optional extension providing: log streaming from remote agents, agent process restart, service health checks. This fills the observability gap that pure A2A services leave (no standard mechanism to restart or monitor a remote agent).

### Agent Registry + Leaderboard
Planned extensions: a publicly discoverable registry of both green agents (benchmarks) and white agents, plus a leaderboard with recommended metric reporting formats. Currently community-driven via the competition; no production registry URL cited.

---

## Summary Table

| Axis | AgentBeats | Harbor |
|---|---|---|
| **Core abstraction** | Benchmark-as-agent (AAA paradigm); A2A + MCP protocols | Container-per-task harness; terminal I/O + check scripts |
| **Schema** | A2A task message + MCP tool list (flexible, protocol-level) | `task.toml` + `instruction.md` + `environment/` + `tests/` (structured, Harbor-specific) |
| **Auto-provisioning** | Delegated to green agent; optional Sandbox Manager extension | Automated Docker/cloud per task; 6 backends |
| **RL support** | Gym-to-MCP translation (aspirational); relies on Harbor/OpenEnv for rollouts | ATIF trajectory format (logprobs, token IDs, v1.6); QueueOrchestrator for dynamic rollout submission |
| **Benchmark portability** | Green agent is portable A2A service (protocol-portable); infrastructure dependencies vary by mode | Harbor-specific packaging; limited portability |
| **Implemented benchmarks** | ~250 competition submissions (38+ ported benchmarks) across 17 tracks; community quality | 65 registry entries; maintained with Docker environments |
| **Tool reconfigurability** | Yes — MCP Gateway enables per-session dynamic tool grants; generalist agents supported | Per-task static `mcp_servers` declaration; no dynamic reconfiguration |
| **Deployment modes** | 5 modes (Local, Remote, Hosted, Proxy, CI) | Local + cloud providers via `--env` flag |
| **Shared persistent infra** | Not standardized; green agent self-manages | Not standardized; per-task containers |
| **Test-production alignment** | Central claim: A2A/MCP are production protocols | Weaker: terminal serialization diverges from production for non-CLI agents |

---

## Key Claim for Paper

AgentBeats standardizes *how evaluation campaigns are orchestrated* — which agents assess which subjects, through which protocols, and under which deployment mode — but delegates *how benchmark environments are packaged, provisioned, reset, and lifecycle-managed* entirely to each green agent; CUBE standardizes precisely that missing infrastructure layer, making CUBE-wrapped benchmarks portable across any compliant evaluation framework (including AgentBeats, where a CUBE benchmark can serve as a green agent's internal substrate) with reproducible, compliance-tested environment packaging that AgentBeats neither provides nor requires.
