"""Benchmark-wide prompt overlay for terminalbench2-cube.

Loaded by `BenchmarkConfig.load_benchmark_clarifications()` and folded into
the agent config at recipe time via
`GennyConfig.with_benchmark_clarifications(benchmark_config)`. The framework
only carries this; nothing is delivered to the agent automatically.

`BENCHMARK_HINT` is appended to the system prompt; `TASK_CLARIFICATION` (per
task id) is injected as part of the goal. Keep both terse — they compete for
the LLM's attention with the task description itself.

The two patches below address the two highest-leverage failure buckets the
investigator surfaced on the gpt-5.4-mini × Daytona reference run (run
`20260603_114932…215cc0e7`, meta-analysis at
`~/auto_cube/tb2-r0-investigator/journal/`):

  1. `submits-without-verification` (15 episodes) — agent reaches `final_step`
     without ever running the canonical test command. Examples: ran pytest file
     with `python` (silent → assumed pass); observed 0/100 wins against every
     opponent and submitted anyway; hardcoded the example FEN and called it
     done.
  2. `missing-tool-not-installed` (≥8 episodes) — model has root and (mostly)
     network but pivots to hand-rolling instead of `apt-get install` / `pip
     install`. Reference solutions for several of these tasks are literally
     a few install commands.

`TASK_CLARIFICATION` is left empty here; the meta-analysis also identified
~13 subtle-implementation-bug tasks (one specific knowledge gap each), but
those are best addressed with reasoning_effort tuning or per-task hints
landed in a separate, narrower change.
"""

BENCHMARK_HINT = """\
TerminalBench-2 guidance:

VERIFY BEFORE SUBMIT. Every task is graded by a pytest suite. Before \
emitting `final_step`, run the project's tests yourself (look for `tests/`, \
`pytest.ini`, `pyproject.toml`, or `test*.sh` / `run_tests.sh`). If they \
fail, fix the failure and re-run — do NOT submit on a "should be right" \
guess. A surprising result (e.g. 0/N succeeded when you expected to win) \
is a real bug, not verification you can skip.

INSTALL, DON'T REINVENT. You have root and (usually) network access. If a \
task needs a missing library or compiler, run `apt-get install` or \
`pip install` first instead of hand-rolling a workaround. Common examples \
that ship as standard packages: stockfish, PIL/Pillow, primer3, \
gcc-mips-linux-gnu, requests — install them rather than reimplementing.\
"""

TASK_CLARIFICATION: dict[str, str] = {}
