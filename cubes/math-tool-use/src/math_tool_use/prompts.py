"""Canonical prompts for the math-tool-use cube — the single source of truth.

Shared by both consumers so they can never drift:
- training: ``conf/cube_math_lamer_rl.yaml`` (PipelineRL) pulls it via the
  ``math_tir_system_prompt`` Hydra factory (``system_prompt: {_target_: math_tool_use.math_tir_system_prompt}``);
- the inference probe ``recipes/passk_reflection_probe.py`` imports ``MATH_TIR_SYSTEM_PROMPT`` directly.

Edit the prompt HERE only.
"""

MATH_TIR_SYSTEM_PROMPT = """\
You are a math-focused AI Agent. Solve problems by combining clear symbolic reasoning
with short, deterministic Python code.
Keep replies concise. Always present the final answer in LaTeX \\boxed{}.

Workflow:
1. Draft a brief plan in plain text.
2. Execute one run_python_code call to compute or verify the result.
3. Finalize by calling MathAnswer with the LaTeX-formatted answer.

Python policy (run_python_code): pure computation only; no network/filesystem/OS;
keep snippets minimal and print only the final result. Verify with run_python_code
before invoking MathAnswer."""


def math_tir_system_prompt() -> str:
    """Hydra ``_target_`` factory returning :data:`MATH_TIR_SYSTEM_PROMPT`.

    Lets the training YAML reference the canonical prompt without duplicating its text;
    Hydra instantiates this no-arg callable to the string when building the agent config.
    """
    return MATH_TIR_SYSTEM_PROMPT
