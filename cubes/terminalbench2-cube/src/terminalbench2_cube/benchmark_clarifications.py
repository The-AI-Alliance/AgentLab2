"""Benchmark-wide prompt overlay for terminalbench2-cube.

Loaded by `BenchmarkConfig.load_benchmark_clarifications()`; fold in at
recipe time with `GennyConfig.with_benchmark_clarifications(benchmark)`.

Targets the two highest-leverage failure buckets from the gpt-5.4-mini
reference run (15 + 8 of 55 investigated episodes).
"""

BENCHMARK_HINT = """\
TerminalBench-2 guidance:

VERIFY BEFORE SUBMIT. Every task is graded by a pytest suite. Run the \
project's tests yourself (`tests/`, `pytest.ini`, `test*.sh`) before \
`final_step`. A surprising result (e.g. 0/N when expecting wins) is a \
real bug, not skippable verification.

INSTALL, DON'T REINVENT. You have root and (usually) network. If a \
library or compiler is missing, `apt-get install` / `pip install` it \
rather than hand-rolling a workaround.\
"""

TASK_CLARIFICATION: dict[str, str] = {}
