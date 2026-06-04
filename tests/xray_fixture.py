"""Synthesize a rich event-format experiment on disk for XRay testing.

Builds a complete `episodes/<id>/` directory straight through the `FileStorage`
event API — no live LLM, no benchmark, fully deterministic — so both the
Playwright UI smoke and the integration test exercise XRay against a trajectory
that has everything the event-card viewer must render:

  - a reset observation (no parent LLM)
  - sequential LLM-call -> observation steps with prompts, usage, screenshots
  - a PARALLEL turn: one LLM call dispatching two tool calls
  - a step-wise EvaluationEvent and a terminal EvaluationEvent
  - an LLM-call error event

The screenshots are tiny solid-colour PIL images so the Screenshot tab has
something to show without pulling in a browser.
"""

from pathlib import Path

from cube.core import Action, Content, Observation, StepError
from litellm import Message
from PIL import Image

from cube_harness.core import (
    EvaluationEvent,
    LLMCallEvent,
    ToolCallEvent,
    TrajectoryEvent,
    TrajectoryMetadata,
)
from cube_harness.llm import LLMCall, LLMConfig, Prompt, Usage
from cube_harness.storage import FileStorage

DEMO_TRAJ_ID = "demo_task_0_ep0"


def _img(color: str) -> Image.Image:
    return Image.new("RGB", (320, 180), color)


def _obs(goal: str, color: str, axtree: str) -> Observation:
    return Observation(
        contents=[
            Content.from_data(goal, name="goal"),
            Content.from_data(_img(color), name="screenshot"),
            Content.from_data(axtree, name="axtree"),
        ]
    )


def _llm_event(
    tag: str, user: str, assistant: str, ptoks: int, ctoks: int, cost: float, *, error: bool = False
) -> LLMCallEvent:
    call = LLMCall(
        tag=tag,
        llm_config=LLMConfig(model_name="openai/gpt-4o-mini", temperature=0.7),
        prompt=Prompt(
            messages=[
                {"role": "system", "content": "You are a web agent. Think, then act."},
                {"role": "user", "content": user},
            ],
            tools=[{"type": "function", "function": {"name": "click", "description": "Click an element."}}],
        ),
        output=Message(content=assistant, role="assistant"),
        usage=Usage(
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            total_tokens=ptoks + ctoks,
            cached_tokens=ptoks // 2,
            cost=cost,
        ),
    )
    err = (
        StepError(error_type="RateLimitError", exception_str="429 too many requests", stack_trace="…")
        if error
        else None
    )
    return LLMCallEvent(call=call, error=err)


def build_demo_experiment(exp_dir: Path, traj_id: str = DEMO_TRAJ_ID) -> str:
    """Write one rich event-format episode under `exp_dir`; return its id."""
    storage = FileStorage(exp_dir)
    storage.save_metadata(
        TrajectoryMetadata(
            id=traj_id,
            metadata={"task_id": "demo_task_0", "agent_name": "DemoReAct"},
            start_time=1000.0,
            end_time=1042.0,
            summary_stats={
                "n_env_steps": 4,
                "n_agent_steps": 3,
                "total_actions": 4,
                "total_llm_calls": 3,
                "duration": 42.0,
                "prompt_tokens": 900,
                "completion_tokens": 130,
                "cached_tokens": 450,
                "cache_creation_tokens": 0,
                "cost": 0.0123,
                "final_reward": 1.0,
            },
            reward_info={"reward": 1.0, "info": {"success": True}},
        )
    )

    def ev(output: object, start: float, end: float) -> None:
        storage.save_event(TrajectoryEvent(output=output, start_time=start, end_time=end), traj_id)

    # Reset observation — no parent LLM call.
    ev(
        ToolCallEvent(
            parent_event_id="__reset__",
            obs=_obs("Goal: search for shoes and add them to the cart", "#1e3a8a", "root\n  button 'Search'"),
        ),
        1000.0,
        1000.1,
    )

    # Turn 1 — sequential click.
    l1 = _llm_event("act", "Find the search box.", "I'll click the search box first.", 300, 40, 0.004)
    ev(l1, 1000.1, 1002.0)
    ev(
        ToolCallEvent(
            parent_event_id=l1.id,
            action=Action(name="click", arguments={"element_id": "search"}),
            obs=_obs("search box focused", "#075985", "textbox 'Search' focused"),
        ),
        1002.0,
        1002.3,
    )

    # Turn 2 — PARALLEL: one LLM call, two tool calls, plus a step-wise reward.
    l2 = _llm_event(
        "act", "Type the query and read results.", "Typing 'shoes' and reading the page in parallel.", 420, 60, 0.006
    )
    ev(l2, 1002.3, 1010.0)
    t2a = ToolCallEvent(
        parent_event_id=l2.id,
        action=Action(name="type", arguments={"text": "shoes"}),
        obs=_obs("typed 'shoes'", "#047857", "textbox value=shoes"),
    )
    ev(t2a, 1010.0, 1010.4)
    ev(
        ToolCallEvent(
            parent_event_id=l2.id,
            action=Action(name="read_page", arguments={}),
            obs=_obs("3 results for shoes", "#047857", "list\n  3 results"),
        ),
        1010.0,
        1010.6,
    )
    ev(
        EvaluationEvent(reward=0.5, info={"progress": "found results"}, is_terminal=False, parent_event_id=t2a.id),
        1010.6,
        1010.6,
    )

    # Turn 3 — an LLM-call error event, then a recovering final step.
    l3err = _llm_event("act", "Add first result to cart.", "(rate limited)", 0, 0, 0.0, error=True)
    ev(l3err, 1010.6, 1011.0)
    l3 = _llm_event("act", "Add first result to cart.", "Adding the first result and finishing.", 180, 30, 0.002)
    ev(l3, 1011.0, 1020.0)
    ev(
        ToolCallEvent(
            parent_event_id=l3.id,
            action=Action(name="final_step", arguments={}),
            obs=_obs("added to cart — task complete", "#065f46", "button 'Checkout'"),
        ),
        1020.0,
        1020.2,
    )

    # Terminal evaluation — no parent link (attaches to the last group by position).
    ev(EvaluationEvent(reward=1.0, info={"success": True}, is_terminal=True), 1041.0, 1041.0)

    return traj_id
