"""Scratch: walk ONE tau2 task through the full env cycle, single-threaded.

Flip the knobs below to debug any domain. Run with VSCode "Debug: current file".
Breakpoints worth setting:
  tool.py  → CubeTool.reset / action_set / execute_action / _render
  task.py  → CubeTask.reset / evaluate / finished
Throwaway — not part of the package.
"""
# ruff: noqa: E402  — knobs intentionally precede imports in this scratch file

# ── knobs ──────────────────────────────────────────────────────────────────
DOMAIN = "airline"  # mock | airline | retail | telecom
TASK_INDEX = 10  # which task within the domain (0-based)
RUN_AGENT = True  # True → also run a real ReAct agent on vLLM (needs server up)
PROBE_USER = True  # True → one manual send_message_to_user turn (hits vLLM user-sim)
# ─────────────────────────────────────────────────────────────────────────────

import os

from cube.core import Action, Observation
from tau2.registry import registry

from tau2_cube.task import CubeTaskConfig, CubeTaskMetadata

# Point LiteLLM at the local vLLM for BOTH the agent and the user simulator
# (both use model id "hosted_vllm/qwen-32b"). Must be set before any LLM call,
# i.e. before PROBE_USER / RUN_AGENT.
os.environ.setdefault("HOSTED_VLLM_API_BASE", "http://localhost:8000/v1")
os.environ.setdefault("HOSTED_VLLM_API_KEY", "local-key")

# --- build one task ---------------------------------------------------------
t = registry.get_tasks_loader(DOMAIN)()[TASK_INDEX]  # type: ignore[call-arg]  # default split
task_id = f"{DOMAIN}/{t.id}"
meta = CubeTaskMetadata(id=task_id, abstract_description=t.id, recommended_max_steps=30, domain=DOMAIN, tau2_task=t)
task = CubeTaskConfig(metadata=meta).make()  # CubeTaskConfig.make → CubeTask
print(f"=== {task_id} ({DOMAIN}) ===")

# --- full env cycle ---------------------------------------------------------
obs, info = task.reset()  # ① build env + apply initial_state
print("① OBS  :", obs.to_markdown()[:300])
print("① DB   :", task.tool._env.get_db_hash())
print("① ACTIONS:", [a.name for a in task.action_set])  # dynamic, per-domain

print("② FINISHED before:", task.finished())  # ② False — db != gold yet

# ③ replay the gold (reference) assistant actions — domain-general
crit = t.evaluation_criteria
gold_actions = list(crit.actions or []) if crit else []
assistant_actions = [a for a in gold_actions if a.requestor == "assistant"]
print(f"③ replaying {len(assistant_actions)} assistant gold action(s) ({len(gold_actions)} total, rest are user-side)")
for a in assistant_actions:
    out = task.tool.execute_action(Action(name=a.name, arguments=a.arguments))  # real cube dispatch
    print(f"   {a.name}({a.arguments}) -> {str(out)[:120]}")

print("④ REWARD  :", task.evaluate())  # ④ DB-hash compare (1.0 if matches gold)
print("⑤ FINISHED:", task.finished())

# --- optional: one manual user-sim turn (needs vLLM) ------------------------
# Breakpoint in CubeTool._talk_to_user / UserSimulator.generate_next_message.
if PROBE_USER:
    reply = task.tool.execute_action(
        Action(name="send_message_to_user", arguments={"message": "Hi! How can I help you today?"})
    )
    print("👤 USER REPLY:", reply.to_markdown() if isinstance(reply, Observation) else reply)

task.close()  # ⑥ cleanup
print("⑥ closed")


# ==========================================================================
# Optionally run a REAL ReAct agent against the SAME task, driven by vLLM.
# Prereq: vLLM up (see __oo_scripts/serve_qwen32b_vllm.sh, port 8000).
# Breakpoint in CubeTool.execute_action to watch the MODEL's tool calls.
# ==========================================================================
if RUN_AGENT:
    from cube_harness.agents.react import ReactAgentConfig
    from cube_harness.exp_runner import run_sequentially
    from cube_harness.experiment import Experiment
    from cube_harness.llm import LLMConfig

    from tau2_cube.benchmark import CubeBenchmarkConfig

    exp = Experiment(
        name=f"tau2-{DOMAIN}",
        agent_config=ReactAgentConfig(
            # tool_choice="required": tau2 agents must ACT every turn — talk via
            # send_message_to_user, act via domain tools, end via final_step.
            # With "auto" the model replies in bare text, which the harness drops
            # (no tool_call → no action → episode stops). See airline/10 trace.
            llm_config=LLMConfig(model_name="hosted_vllm/qwen-32b", temperature=0.7, tool_choice="required"),
        ),
        benchmark_config=CubeBenchmarkConfig().subset_from_list([task_id]),
        max_steps=30,
    )
    run_sequentially(exp, debug_limit=1)  # in-process, 1 task → debuggable
