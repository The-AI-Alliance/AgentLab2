"""Task and TaskConfig for swebench-live-cube.

Extends the SWE-bench Verified task with SWE-bench Live specifics:
- Per-instance test_cmds (no heuristic test command generation needed)
- At least one FAIL_TO_PASS test must pass (not all) on Linux
"""

from __future__ import annotations

import base64
import logging
import re
import shlex
from typing import Any

from cube.container import relocate_if_readonly
from cube.core import ActionSchema, Observation
from cube.task import STOP_ACTION, RuntimeContext, Task, TaskConfig, TaskExecutionInfo, TaskMetadata

from cube.tools.terminal import ContainerTerminalTool, TerminalToolConfig

logger = logging.getLogger(__name__)

# POSIX-compatible: use `.` instead of `source`, skip silently if conda is absent.
# Works with both bash (Daytona/Modal/Toolkit backends) and sh/dash (LocalContainer).
CONDA_ACTIVATE = "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then . /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; fi"

# ── Scoped-eval helper (see SWEBenchLiveTask.scoped_eval) ───────────────────────
# The dataset's ``test_cmds`` typically runs the WHOLE repo test suite (e.g.
# ``pytest keras -rA``), even though ``evaluate()`` only cares about a known
# ``fail_to_pass`` and ``pass_to_pass`` list. For heavy repos that doesn't fit
# in the per-run ``eval_timeout`` budget and even a correct gold patch silently
# scores 0. ``scoped_eval`` rewrites such commands to run only the relevant
# node IDs via ``xargs``, preserving env vars, flags, and the pytest wrapper
# (``poetry run pytest``, ``uv run pytest``, etc.).
#
# Truncated IDs: a small fraction of the dataset's parametrized IDs are
# silently cut mid-string (commas inside brackets, unbalanced quotes/parens —
# e.g. ``[formats1-Vinyl-7",``). Pytest's strict node-ID matching rejects
# these with ``ERROR: not found:`` and aborts the whole run. We filter them
# out before scoping (``_is_truncated_id``). The f2p tests added by
# ``test_patch`` are almost never parametrized this way, so filtering doesn't
# typically drop the resolution-gating tests.

# Wrappers we recognise around a pytest invocation.
_PYTEST_WRAPPERS: tuple[tuple[str, ...], ...] = (
    ("poetry", "run", "pytest"),
    ("uv", "run", "pytest"),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
)

# Bare pytest flags whose VALUE is the NEXT token (rather than ``--key=value``).
# The ``--key=value`` form is a single shlex token and needs no special handling.
_PYTEST_VALUE_FLAGS: frozenset[str] = frozenset(
    {
        "-n",
        "-k",
        "-m",
        "-p",
        "-c",
        "-W",
        "-o",
        "--timeout",
        "--rootdir",
        "--basetemp",
        "--junitxml",
        "--cov",
        "--cov-report",
        "--ignore",
        "--ignore-glob",
        "--deselect",
        "--maxfail",
        "--dist",
    }
)

_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _is_truncated_id(node_id: str) -> bool:
    """Heuristic: ``node_id`` looks like a CSV-truncated parametrized ID.

    Pytest's parametrized IDs are ``file.py::Class::test[arg0-arg1-…]`` (or
    ``test_(arg,…)`` for some custom collectors). The SWE-bench-Live dataset
    extraction sometimes cuts mid-string when a parametrize arg contains a
    literal comma, paren, or double quote — or just at an arbitrary boundary —
    leaving trailers like ``[formats1-Vinyl-7",``, ``[365_day-(1,``,
    ``[extra_persistent_ds-failing_node_names0-No`` (no closing bracket), or
    ``test_split_dataset_tensorflow_(100,`` (no brackets at all). Pytest
    rejects all such IDs with ``ERROR: not found:`` and aborts the whole run.

    We flag any ID where the whole string ends in ``,`` or has an unbalanced
    ``[``/``]`` / ``(``/``)`` / ``"`` count. f2p tests added by ``test_patch``
    are rarely parametrized in these unfortunate shapes, so filtering doesn't
    typically drop the resolution-gating tests.
    """
    if node_id.endswith(","):
        return True
    if node_id.count("[") != node_id.count("]"):
        return True
    if node_id.count("(") != node_id.count(")"):
        return True
    if node_id.count('"') % 2 != 0:
        return True
    return False


def _scope_pytest_cmd(cmd: str, node_ids_file: str) -> str | None:
    """Rewrite a pytest test_cmd to run only the lines in ``node_ids_file``.

    Strips broad positional path selectors (e.g. ``keras`` in
    ``pytest keras -rA``), preserves env-var prefixes, the pytest wrapper, and
    every flag (including value-bearing ones like ``-n 4`` or ``-m unit``).
    Returns ``None`` when the command is not a pytest invocation we recognise —
    the caller falls back to the unscoped command.

    The rewritten form is ``xargs -a <file> -d '\\n' --no-run-if-empty -s 2000000
    <env> <pytest_head> <flags>`` so the test selectors (file paths — see the
    module-top rationale) come from the file (avoids ARG_MAX) and ``xargs``
    runs pytest exactly once (``-s 2000000`` is just under Linux ARG_MAX).
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    i = 0
    env_prefix: list[str] = []
    while i < len(tokens) and _ENV_VAR_RE.match(tokens[i]):
        env_prefix.append(tokens[i])
        i += 1
    pytest_head: list[str] | None = None
    for wrapper in _PYTEST_WRAPPERS:
        if tuple(tokens[i : i + len(wrapper)]) == wrapper:
            pytest_head = list(wrapper)
            i += len(wrapper)
            break
    if pytest_head is None:
        if i < len(tokens) and tokens[i] in ("pytest", "py.test"):
            pytest_head = [tokens[i]]
            i += 1
        else:
            return None
    # Walk the rest, keeping flags (and their values for known value-flags),
    # dropping bare positionals (the broad test-path selectors).
    kept_flags: list[str] = []
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("-"):
            kept_flags.append(t)
            # Consume the next token as a value only when the flag takes one
            # AND the next token doesn't start with `-` (mirrors argparse's
            # nargs="?" handling for optional flags like ``--cov``).
            if "=" not in t and t in _PYTEST_VALUE_FLAGS and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                kept_flags.append(tokens[i + 1])
                i += 1
        # else: bare positional → drop
        i += 1
    # Env vars must go BEFORE ``xargs`` — placed after, xargs would treat
    # ``KEY=VAL`` as the command to run. xargs inherits its env to pytest.
    env_str = " ".join(shlex.quote(t) for t in env_prefix)
    cmd_str = " ".join(shlex.quote(t) for t in pytest_head + kept_flags)
    prefix = f"{env_str} " if env_str else ""
    return f"{prefix}xargs -a {shlex.quote(node_ids_file)} -d '\\n' --no-run-if-empty -s 2000000 {cmd_str}"


# Appended to every task description so the agent knows evaluation constraints,
# how to verify the fix, and how to submit.
_TASK_INSTRUCTIONS_TEMPLATE = """\
Modify source files only — do not modify test files or configuration files \
(pyproject.toml, setup.cfg, etc.).

Verify your fix by running:
  {test_cmd}

When ready to submit:
1. Check: `git diff > patch.txt && cat patch.txt`
2. Confirm the patch only modifies source files, then call `final_step`.\
"""


class SWEBenchLiveTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for SWE-bench Live tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (problem_statement, patch, test_patch, etc.) lives on
    ``SWEBenchLiveExecutionInfo`` and is loaded lazily by
    ``SWEBenchLiveTaskConfig.make()``.
    """

    repo: str
    """GitHub repository name, e.g. 'django/django'."""

    base_commit: str
    """Git commit hash the agent's solution must be applied on top of."""

    splits: list[str]
    """SWE-bench Live splits this task belongs to, e.g. ['verified', 'full']."""

    log_parser: str
    """Test log parser to use during evaluation, e.g. 'pytest'."""


class SWEBenchLiveExecutionInfo(TaskExecutionInfo):
    """Heavy per-task execution data for SWE-bench Live — populated on the worker.

    Loaded by ``SWEBenchLiveTaskConfig.make()`` from the per-task execution cache
    written by ``SWEBenchLiveBenchmarkConfig.install()``.

    Mirrors the SWE-bench Verified execution info but adds ``test_cmds``: SWE-bench
    Live ships explicit per-instance test commands rather than relying on a
    repo-aware heuristic.
    """

    problem_statement: str
    """The agent-facing GitHub issue text."""

    hints_text: str = ""
    """Optional hint text (only surfaced when ``SWEBenchLiveTaskConfig.include_hints`` is True)."""

    patch: str
    """Gold patch — written to /tmp/gold_patch.diff in oracle_mode."""

    test_patch: str
    """Test patch applied during evaluation."""

    fail_to_pass: list[str]
    """Test directives that must pass after the fix (Live: at least one)."""

    pass_to_pass: list[str]
    """Test directives that must remain passing after the fix (Live: zero failures)."""

    test_cmds: list[str] = []
    """Explicit shell commands to run during evaluation; replaces the
    repo-aware test command heuristic used by SWE-bench Verified."""

    eval_timeout: int = 1800
    """Wall-clock seconds allowed for the evaluation test commands."""


class SWEBenchLiveTask(Task[SWEBenchLiveTaskMetadata, ContainerTerminalTool]):
    """A single SWE-bench Live task with test-based validation."""

    validate_per_step: bool = False
    accept_agent_stop: bool = True

    include_hints: bool = False
    """If True, append hints_text to the problem statement in reset()."""

    oracle_mode: bool = False
    """If True, write the gold patch to /tmp/gold_patch.diff in reset()."""

    scoped_eval: bool = False
    """If True, rewrite each pytest ``test_cmd`` in ``evaluate()`` to run only
    the **test files** containing the relevant node IDs (``pass_to_pass`` for
    the baseline, ``fail_to_pass ∪ pass_to_pass`` after the fix) instead of
    the whole suite. Recovers heavy repos (keras / xarray / sympy) whose
    dataset cmd doesn't fit in ``eval_timeout``. Non-pytest cmds and parses
    we can't recognise fall back to the unscoped path, so existing behaviour
    is bit-for-bit unchanged when the rewrite doesn't apply. See the module-
    top ``_scope_pytest_cmd`` comment for the rationale on file- vs
    node-ID-level scoping."""

    append_submission_instructions: bool = True
    """If True, append evaluation constraints, test command, and final_step
    submission instructions to the problem statement. Set False for raw-benchmark
    comparisons where the task description must match the original exactly."""

    @property
    def _exec(self) -> SWEBenchLiveExecutionInfo:
        """Typed view on execution_info — fails fast if it was not populated."""
        if not isinstance(self.execution_info, SWEBenchLiveExecutionInfo):
            raise RuntimeError(
                f"SWEBenchLiveTask {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected SWEBenchLiveExecutionInfo. "
                f"Construct via SWEBenchLiveTaskConfig.make() so it is populated."
            )
        return self.execution_info

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        # TODO: remove once cube-standard auto-includes STOP_ACTION in Task.action_set
        # (upstream fix: Task.action_set appends STOP_ACTION when accept_agent_stop=True,
        # and STOP_ACTION constant gets the Anthropic-compatible parameters schema).
        stop = ActionSchema(
            name=STOP_ACTION.name,
            description=STOP_ACTION.description,
            parameters={"type": "object", "properties": {}},
        )
        return actions + [stop]

    def _build_tool(self) -> None:
        """Ensure /testbed files are writable and git-safe, then build the tool.

        NON-ROOT DOCKER WORKAROUND. Upstream SWE-bench-Live images assume
        `USER root` (matched by Daytona, local Docker, AWS, Azure). On non-root
        infras — the EAI Toolkit enforces uid 13011 by cluster policy — the
        runtime user can't `chmod` root-owned files in place, can't write to a
        read-only /testbed, and Git 2.35.2+ rejects repos owned by a different
        user. This block normalises all three before the tool is constructed.
        Root-running infras short-circuit at the writability probes.

        Two pre-flight fixes applied unconditionally (mirrors swebench-verified-cube):
        1. git safe.directory: Git 2.35.2+ refuses to operate in repos owned by a
           different user. Configure /testbed as safe so agents can run `git diff`.
        2. chmod via cp/mv: some live containers ship root-owned 644 .py files inside
           a world-writable /testbed. mv unlinks via the writable parent and recreates
           with the runtime user's ownership, making every file writable without sudo.
           Running before relocate_if_readonly keeps conda editable-install paths stable.

        The trailing ``relocate_if_readonly`` call falls back to /tmp/testbed
        when /testbed itself is not writable (toolkit case); on root-running
        infras the probe returns immediately with the original path.
        """
        self._container.exec(
            f"git config --global --add safe.directory {self.tool_config.working_dir}",
            timeout=30,
        )
        self._container.exec(
            f"find {self.tool_config.working_dir} -not -path '*/.git/*' -name '*.py' ! -writable"
            f' -exec sh -c \'cp "$1" "$1.tmp" && mv "$1.tmp" "$1"\' _ {{}} \\;'
            f" 2>/dev/null || true",
            timeout=120,
        )
        new_wd = relocate_if_readonly(
            self._container,
            self.tool_config.working_dir,
            "/tmp/testbed",
            extra_setup="git config --global --add safe.directory /tmp/testbed",
        )
        if new_wd != self.tool_config.working_dir:
            # After cp -a, the conda editable install still points to the original /testbed.
            # Use the testbed Python to locate the real site-packages and update every
            # .egg-link / .pth file that references the old path, so Python imports from
            # the relocated copy (where patches will be applied).
            orig_wd = self.tool_config.working_dir
            py_script = (
                "import site, os; "
                "dirs = site.getsitepackages() + [site.getusersitepackages()]; "
                "updated = []; "
                "[updated.append(p) or open(p, 'w').write(c.replace(orig, new)) "
                " for d in dirs "
                " for root, _, files in os.walk(d) "
                " for fname in files if fname.endswith(('.egg-link', '.pth')) "
                " for p in [os.path.join(root, fname)] "
                " for c in [open(p).read()] if orig in c]; "
                "print('editable-install paths updated:', len(updated), updated)"
            )
            result = self._container.exec(
                f"{CONDA_ACTIVATE} && python -c \"orig='{orig_wd}'; new='{new_wd}'; {py_script}\" 2>/dev/null || true",
                timeout=30,
            )
            logger.info("Editable-install path update: %s", result.stdout.strip())
        self._tool = self.tool_config.model_copy(update={"working_dir": new_wd}).make(container=self._container)

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()

        # Oracle mode: write gold patch for debug/baseline use
        if self.oracle_mode and self._exec.patch:
            b64 = base64.b64encode(self._exec.patch.encode()).decode()
            self.tool.bash(f"echo '{b64}' | base64 -d > /tmp/gold_patch.diff")

        instruction = self._exec.problem_statement
        if self.include_hints and self._exec.hints_text:
            instruction += f"\n\n## Hints\n{self._exec.hints_text}"
        instruction += f"\n\n[Working directory: {self.tool._config.working_dir}]"
        if self.append_submission_instructions:
            test_cmd = self._exec.test_cmds[0] if self._exec.test_cmds else "pytest"
            instruction += f"\n\n{_TASK_INSTRUCTIONS_TEMPLATE.format(test_cmd=test_cmd)}"

        return Observation.from_text(instruction), {
            "instance_id": self.metadata.id,
            "repo": self.metadata.repo,
        }

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:

        fail_to_pass = self._exec.fail_to_pass
        pass_to_pass = self._exec.pass_to_pass
        test_cmds = self._exec.test_cmds
        eval_timeout = self._exec.eval_timeout

        # Step 1: Baseline run — identify pre-existing p2p failures BEFORE test_patch.
        # Some containers have broken environment-level tests (SSL errors, deprecated
        # imports, etc.) completely unrelated to the task. Without a baseline, these
        # show up as p2p regressions and cause correct fixes to score 0.
        # Running before _apply_patch(test_patch) means f2p tests don't exist yet,
        # so the baseline is scoped to p2p only (f2p would just be collection errors).
        baseline_output = self._run_test_cmds(
            test_cmds, timeout=eval_timeout, scoped_node_ids=list(pass_to_pass) if self.scoped_eval else None
        )
        pre_existing_p2p = self._get_failing_test_ids(baseline_output, pass_to_pass, self.metadata.log_parser)
        if pre_existing_p2p:
            logger.info(
                "Pre-existing p2p failures (excluded from scoring): %d — %s",
                len(pre_existing_p2p),
                sorted(pre_existing_p2p),
            )

        # Step 2: Apply test patch (adds new f2p test cases)
        self._apply_patch(self._exec.test_patch)

        # Step 3: Run tests with agent's fix + test_patch applied (scoped to
        # f2p ∪ p2p when scoped_eval is on; the test_patch has added the f2p
        # node IDs to the codebase so they now collect).
        test_output = self._run_test_cmds(
            test_cmds,
            timeout=eval_timeout,
            scoped_node_ids=list(fail_to_pass) + list(pass_to_pass) if self.scoped_eval else None,
        )

        # Step 4: Score — exclude pre-existing p2p failures from the count
        f2p_passed, p2p_failed = self._check_test_results(
            test_output,
            fail_to_pass,
            pass_to_pass,
            self.metadata.log_parser,
            exclude_p2p=pre_existing_p2p,
        )

        # SWE-bench Live Linux: at least one FAIL_TO_PASS must pass, zero net PASS_TO_PASS failures
        resolved = f2p_passed > 0 and p2p_failed == 0
        reward = 1.0 if resolved else 0.0

        return reward, {
            "done": True,
            "resolved": resolved,
            "fail_to_pass_passed": f2p_passed,
            "fail_to_pass_total": len(fail_to_pass),
            "pass_to_pass_failed": p2p_failed,
            "pass_to_pass_total": len(pass_to_pass),
            "pre_existing_p2p_failures": len(pre_existing_p2p),
            "test_output": test_output
            if len(test_output) <= 30000
            else test_output[:5000] + "\n...[truncated]...\n" + test_output[-25000:],
        }

    # ── Private helpers ────────────────────────────────────────────

    def _apply_patch(self, patch: str) -> str:
        """Apply a unified diff patch to /testbed using git apply with fallbacks."""
        b64 = base64.b64encode(patch.encode()).decode()
        self.tool.bash_unlimited(f"echo '{b64}' | base64 -d > /tmp/patch.diff")

        # Try git apply first
        # Commands run in tool.working_dir (may be relocated to writable copy).
        result = self.tool.bash_unlimited("git apply /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result and "[error]" not in result:
            return result

        result = self.tool.bash_unlimited("git apply --reject /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result and "[error]" not in result:
            return result

        result = self.tool.bash_unlimited("patch --batch --forward --fuzz=5 -p1 -i /tmp/patch.diff 2>&1", timeout=60)
        if "[exit_code:" in result or "[error]" in result:
            logger.warning("_apply_patch: all methods failed.\npatch output:\n%s", result)
        return result

    # Plain-content chunk size for writing the scoped-tests file. Empirically
    # (Daytona, 2026-05-28) the container's exec arg cap sits around ~64 KB;
    # ``write_file``'s single-call approach fails silently above ~30 KB. We
    # stay well under at 16 KB plain (≈22 KB base64 on the cmdline).
    _SCOPED_TESTS_CHUNK_BYTES = 16_000

    def _write_scoped_tests_file(self, node_ids: list[str], path: str) -> None:
        """Write ``node_ids`` (one per line) to ``path`` inside the container.

        ``tool.write_file`` interpolates content onto the shell cmdline and
        silently fails above ~30 KB on Daytona — the container's exec arg cap
        is tighter than ARG_MAX. Chunked base64-append keeps each individual
        shell command small (see ``_SCOPED_TESTS_CHUNK_BYTES``) so every chunk
        lands; we check each chunk's exit code and verify the final line count
        matches — fail loud rather than silently mis-score.
        """
        quoted = shlex.quote(path)
        self._exec_or_raise(f"rm -f {quoted} && touch {quoted}", timeout=30, context=f"init {path}")
        chunk: list[str] = []
        size = 0
        chunk_idx = 0
        for nid in node_ids:
            if chunk and size + len(nid) + 1 > self._SCOPED_TESTS_CHUNK_BYTES:
                self._append_b64_chunk(chunk, quoted, chunk_idx)
                chunk_idx += 1
                chunk = []
                size = 0
            chunk.append(nid)
            size += len(nid) + 1
        if chunk:
            self._append_b64_chunk(chunk, quoted, chunk_idx)
        # Verify: line count must match.
        result = self._exec_or_raise(f"wc -l < {quoted}", timeout=30, context=f"verify {path}")
        try:
            actual = int(result.split()[0])
        except (IndexError, ValueError):
            raise RuntimeError(f"could not parse line count for {path}: {result!r}")
        if actual != len(node_ids):
            raise RuntimeError(f"scoped tests file {path}: wrote {len(node_ids)} ids, container reports {actual} lines")

    def _append_b64_chunk(self, ids: list[str], quoted_path: str, chunk_idx: int) -> None:
        """Append ``ids`` (one per line) to ``quoted_path`` via base64."""
        b64 = base64.b64encode(("\n".join(ids) + "\n").encode()).decode()
        self._exec_or_raise(
            f"printf '%s' '{b64}' | base64 -d >> {quoted_path}",
            timeout=60,
            context=f"chunk #{chunk_idx} ({len(ids)} ids, {len(b64)} b64 bytes)",
        )

    def _exec_or_raise(self, cmd: str, timeout: int, context: str) -> str:
        """Run ``cmd`` and raise if its output marks a non-zero exit code.

        Returns the bash output stripped of the ``[exit_code: 0]`` trailer.
        ``bash_unlimited`` always appends an ``[exit_code: N]`` marker; we use
        it as the explicit success signal instead of trusting that a failed
        sub-pipe leaves a visible error on stdout.
        """
        out = self.tool.bash_unlimited(cmd, timeout=timeout)
        m = re.search(r"\[exit_code:\s*(\d+)\]", out)
        if m is None:
            return out  # no marker (e.g. plain output) — assume OK
        code = int(m.group(1))
        if code != 0:
            # Trim to first ~400 chars of pre-marker output for the error msg.
            head = out[: m.start()].rstrip()[-400:]
            raise RuntimeError(f"{context}: exit {code} — output tail: {head!r}")
        return out[: m.start()].rstrip()

    def _run_test_cmds(
        self,
        test_cmds: list[str],
        timeout: int = 1800,
        scoped_node_ids: list[str] | None = None,
    ) -> str:
        """Run the explicit test commands from the dataset.

        When ``scoped_node_ids`` is provided (caller has ``scoped_eval`` on),
        each pytest cmd is rewritten via ``_scope_pytest_cmd`` to run only
        those node IDs (read from a file via xargs, so ARG_MAX is a non-issue).
        Cmds we can't recognise as pytest fall back to running unscoped — same
        behaviour as before. See ``_scope_pytest_cmd`` for the scoping spec.
        """
        if not test_cmds:
            return "(no test commands)"

        node_ids_file = ""
        if scoped_node_ids:
            # Drop CSV-truncated IDs (see _is_truncated_id) — pytest's strict
            # node-ID matching aborts the whole run on these. The f2p tests
            # almost never get caught by the filter.
            clean_ids = [nid for nid in scoped_node_ids if not _is_truncated_id(nid)]
            dropped = len(scoped_node_ids) - len(clean_ids)
            if dropped:
                logger.info("scoped_eval: dropped %d/%d truncated IDs", dropped, len(scoped_node_ids))
            node_ids_file = "/tmp/cube_scoped_tests.txt"
            self._write_scoped_tests_file(clean_ids, node_ids_file)

        outputs = []
        working_dir = self.tool._config.working_dir
        # Activate conda first, then set PYTHONPATH so that conda activation cannot
        # clear it. Prepending working_dir (and src/ for src-layout packages) ensures
        # source files patched in working_dir take precedence over site-packages.
        pythonpath = f"export PYTHONPATH={working_dir}:{working_dir}/src:${{PYTHONPATH:-}}"
        # Redirect tika log to a fresh writable directory to avoid PermissionError when
        # the container image pre-bakes /tmp/tika.log with root-only permissions.
        # TIKA_LOG_PATH is a directory; tika appends /tika.log to it.
        tika_log = "mkdir -p /tmp/tika_cube_eval && export TIKA_LOG_PATH=/tmp/tika_cube_eval"
        for cmd in test_cmds:
            effective_cmd = cmd
            if node_ids_file:
                scoped = _scope_pytest_cmd(cmd, node_ids_file)
                if scoped is not None:
                    logger.info("scoped_eval: %r → %r", cmd, scoped)
                    effective_cmd = scoped
                else:
                    logger.info("scoped_eval: cannot scope %r — running unscoped", cmd)
            full_cmd = f"{CONDA_ACTIVATE} && {pythonpath} && {tika_log} && {effective_cmd}"
            output = self.tool.bash_unlimited(full_cmd, timeout=timeout)
            outputs.append(output)
        return "\n".join(outputs)

    @staticmethod
    def _get_failing_test_ids(output: str, test_ids: list[str], log_parser: str) -> set[str]:
        """Return the subset of test_ids that appear as FAILED or ERROR in output."""
        failing: set[str] = set()
        if log_parser != "pytest":
            return failing
        _NO_WORD = r"(?![a-zA-Z0-9_])"
        for test_id in test_ids:
            tid = re.escape(test_id)
            if (
                f"{test_id} FAILED" in output
                or f"{test_id} ERROR" in output
                or re.search(r"FAILED " + tid + _NO_WORD, output)
                or re.search(r"ERROR " + tid + _NO_WORD, output)
            ):
                failing.add(test_id)
        return failing

    @staticmethod
    def _check_test_results(
        output: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        log_parser: str,
        exclude_p2p: set[str] | None = None,
    ) -> tuple[int, int]:
        """Check test results: count FAIL_TO_PASS successes and net PASS_TO_PASS failures.

        Args:
            exclude_p2p: test IDs to skip in the p2p check (pre-existing failures
                identified by a baseline run before test_patch was applied).

        Returns:
            (fail_to_pass_passed, pass_to_pass_failed)
        """
        f2p_passed = 0
        p2p_failed = 0

        if log_parser == "pytest":
            # Support multiple pytest output formats:
            #   verbose (-v):  "test_id PASSED [ X%]"   (test_id then status)
            #   summary (-rA): "PASSED test_id"          (status then test_id)
            #   legacy/other:  "test_id::PASSED"
            # test_ids may be truncated prefix strings (e.g. "test_validate[Invalid")
            # that match any parameterized variant. Use a negative lookahead on
            # word chars to avoid false positives when one test name is a plain
            # identifier prefix of another (e.g. "test_foo" matching "test_foo_bar").
            _NO_WORD = r"(?![a-zA-Z0-9_])"
            for test_id in fail_to_pass:
                tid = re.escape(test_id)
                if (
                    f"{test_id} PASSED" in output
                    or f"{test_id}::PASSED" in output
                    or re.search(r"PASSED " + tid + _NO_WORD, output)
                ):
                    f2p_passed += 1
            for test_id in pass_to_pass:
                if exclude_p2p and test_id in exclude_p2p:
                    continue  # pre-existing failure; not caused by agent's change
                tid = re.escape(test_id)
                if (
                    f"{test_id} FAILED" in output
                    or f"{test_id} ERROR" in output
                    or re.search(r"FAILED " + tid + _NO_WORD, output)
                    or re.search(r"ERROR " + tid + _NO_WORD, output)
                ):
                    p2p_failed += 1
        else:
            # Generic fallback: check exit code patterns
            if "[exit_code:" not in output and "[error]" not in output:
                f2p_passed = len(fail_to_pass)
            else:
                p2p_failed = len(pass_to_pass)

        return f2p_passed, p2p_failed


class SWEBenchLiveTaskConfig(TaskConfig[SWEBenchLiveTaskMetadata]):
    """Serializable factory that produces a SWEBenchLiveTask.

    Loads heavy execution data (problem_statement, patch, test_patch, test_cmds, etc.)
    from the per-task execution cache populated by ``SWEBenchLiveBenchmarkConfig.install()``.
    """

    include_hints: bool = False
    """If True, append hints_text to the problem statement in reset()."""

    oracle_mode: bool = False
    """If True, write the gold patch to /tmp/gold_patch.diff in reset()."""

    scoped_eval: bool = False
    """If True, ``evaluate()`` runs only the test files containing the f2p/p2p
    node IDs (instead of the full dataset ``test_cmds``) for pytest commands we
    can rewrite. See ``SWEBenchLiveTask.scoped_eval``."""

    append_submission_instructions: bool = True
    """If True, append evaluation constraints, test command, and final_step
    submission instructions to the problem statement."""

    def verify_installed(self) -> None:
        """Fail fast if the per-task execution cache is empty."""
        cache_dir = type(self).task_execution_cache_dir()
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise RuntimeError(
                f"SWE-bench Live per-task execution cache is empty at {cache_dir}. "
                f"Run `cube install swebench-live-cube` (or "
                f"`SWEBenchLiveBenchmarkConfig.install()`) on this worker first."
            )

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> SWEBenchLiveTask:
        if runtime_context is None or "infra" not in runtime_context:
            raise ValueError("SWEBenchLiveTaskConfig.make() requires runtime_context['infra'].")

        self.verify_installed()
        raw = self.load_task_execution_info()
        execution_info = SWEBenchLiveExecutionInfo.model_validate(raw)

        return SWEBenchLiveTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config or TerminalToolConfig(working_dir="/testbed", enable_file_actions=True),
            runtime_context=runtime_context,
            include_hints=self.include_hints,
            oracle_mode=self.oracle_mode,
            scoped_eval=self.scoped_eval,
            append_submission_instructions=self.append_submission_instructions,
        )
