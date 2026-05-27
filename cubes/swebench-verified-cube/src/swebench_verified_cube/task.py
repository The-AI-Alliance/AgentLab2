"""Task and TaskConfig for swebench-verified-cube."""

from __future__ import annotations

import base64
import logging
import re
import shlex
from typing import Any

from cube.container import relocate_if_readonly
from cube.core import ActionSchema, Observation, TaskResult
from cube.task import STOP_ACTION, RuntimeContext, Task, TaskConfig, TaskExecutionInfo, TaskMetadata

from cube.tools.terminal import ContainerTerminalTool, TerminalToolConfig

logger = logging.getLogger(__name__)

# POSIX-compatible: use `.` instead of `source`, skip silently if conda is absent.
# Works with both bash (Daytona/Modal/Toolkit backends) and sh/dash (LocalContainer).
CONDA_ACTIVATE = "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then . /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; fi"

# Appended to every task description so the agent knows the evaluation constraints
# and submission protocol without requiring them in the recipe template.
_TASK_INSTRUCTIONS = """\
Do not modify test files (tests/ directory, test_*.py files) or configuration files.

When your fix is complete:
1. Verify: `git diff > patch.txt && cat patch.txt`
2. Confirm the patch only contains source file changes, then call `final_step`.\
"""


class SWEBenchVerifiedTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for SWE-bench Verified tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (problem_statement, patch, test_patch, etc.) lives on
    ``SWEBenchVerifiedExecutionInfo`` and is loaded lazily by
    ``SWEBenchVerifiedTaskConfig.make()``.
    """

    repo: str
    """GitHub repository, e.g. 'django/django'."""

    difficulty: str
    """Estimated fix time, e.g. '15 min - 1 hour'."""

    version: str
    """Repository version string, e.g. '4.3'."""

    base_commit: str
    """Git SHA of the base commit the agent starts from."""


class SWEBenchVerifiedExecutionInfo(TaskExecutionInfo):
    """Heavy per-task execution data for SWE-bench Verified — populated on the worker.

    Loaded by ``SWEBenchVerifiedTaskConfig.make()`` from the per-task execution cache
    written by ``SWEBenchVerifiedBenchmarkConfig.install()``.
    """

    problem_statement: str
    """The agent-facing GitHub issue text."""

    patch: str
    """Gold patch — written to /tmp/gold_patch.diff in oracle_mode."""

    test_patch: str
    """Test patch applied during evaluation."""

    fail_to_pass: list[str]
    """Test directives that must pass after the fix."""

    pass_to_pass: list[str]
    """Test directives that must remain passing after the fix."""

    eval_timeout: int = 1800
    """Wall-clock seconds allowed for the evaluation test commands."""


class SWEBenchVerifiedTask(Task[SWEBenchVerifiedTaskMetadata, ContainerTerminalTool]):
    """A single SWE-bench Verified task with test-based validation."""

    validate_per_step: bool = False
    accept_agent_stop: bool = True

    oracle_mode: bool = False
    """If True, write the gold patch to /tmp/gold_patch.diff in reset()."""

    append_submission_instructions: bool = True
    """If True, append evaluation constraints and final_step submission instructions
    to the problem statement. Disable for raw-benchmark comparisons where the task
    description must match the original SWE-bench problem statement exactly."""

    @property
    def _exec(self) -> SWEBenchVerifiedExecutionInfo:
        """Typed view on execution_info — fails fast if it was not populated."""
        if not isinstance(self.execution_info, SWEBenchVerifiedExecutionInfo):
            raise RuntimeError(
                f"SWEBenchVerifiedTask {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected SWEBenchVerifiedExecutionInfo. "
                f"Construct via SWEBenchVerifiedTaskConfig.make() so it is populated."
            )
        return self.execution_info

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        # Anthropic requires input_schema to have {"type": "object"}, not {}.
        # STOP_ACTION uses parameters={}, which LiteLLM passes through verbatim
        # and Anthropic rejects. Override with a valid empty-object schema.
        stop = ActionSchema(
            name=STOP_ACTION.name,
            description=STOP_ACTION.description,
            parameters={"type": "object", "properties": {}},
        )
        return actions + [stop]

    def _build_tool(self) -> None:
        """Ensure /testbed files are writable and git-safe, then build the tool.

        Two pre-flight fixes applied unconditionally:
        1. git safe.directory: Git 2.35.2+ refuses to operate in repos owned by a
           different user. Configure /testbed as safe so agents can run `git diff`.
        2. chmod via cp/mv: some containers ship root-owned 644 .py files inside a
           world-writable /testbed. mv unlinks via the writable parent and recreates
           with the runtime user's ownership, making every file writable without sudo.
           Running before relocate_if_readonly keeps conda editable-install paths stable.
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
        self._tool = self.tool_config.model_copy(update={"working_dir": new_wd}).make(container=self._container)

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()

        # Oracle mode: write gold patch for debug/baseline use
        if self.oracle_mode and self._exec.patch:
            b64 = base64.b64encode(self._exec.patch.encode()).decode()
            self.tool.bash(f"echo '{b64}' | base64 -d > /tmp/gold_patch.diff")

        instruction = self._exec.problem_statement
        if self.append_submission_instructions:
            instruction += f"\n\n{_TASK_INSTRUCTIONS}"

        return Observation.from_text(instruction), {
            "instance_id": self.metadata.id,
            "repo": self.metadata.repo,
            "difficulty": self.metadata.difficulty,
        }

    def evaluate(self, obs: Observation | None = None) -> TaskResult:

        # Apply test patch
        self._apply_patch(self._exec.test_patch)

        fail_to_pass = self._exec.fail_to_pass
        pass_to_pass = self._exec.pass_to_pass
        eval_timeout = self._exec.eval_timeout

        # Run FAIL_TO_PASS tests — these must all pass for resolution
        f2p_passed, f2p_output = self._run_tests(self.metadata.repo, fail_to_pass, timeout=eval_timeout)

        # Run PASS_TO_PASS tests — these must remain passing.
        # strict=False: exit-4 "no tests collected" treated as passed (truncated test IDs
        # in SWE-bench data cannot be collected; agent is not responsible for that).
        p2p_passed = True
        p2p_output = ""
        if pass_to_pass:
            p2p_passed, p2p_output = self._run_tests(
                self.metadata.repo, pass_to_pass, timeout=eval_timeout, strict=False
            )

        resolved = f2p_passed and p2p_passed
        reward = 1.0 if resolved else 0.0

        return TaskResult(
            reward=reward,
            checks=[],
            info={
                "done": True,
                "resolved": resolved,
                "fail_to_pass_passed": f2p_passed,
                "pass_to_pass_passed": p2p_passed,
                "fail_to_pass_output": f2p_output,
                "pass_to_pass_output": p2p_output,
            },
        )

    # ── Private helpers ────────────────────────────────────────────

    def _apply_patch(self, patch: str) -> str:
        """Apply a unified diff patch to /testbed using git apply with fallbacks."""
        b64 = base64.b64encode(patch.encode()).decode()
        self.tool.bash_unlimited(f"echo '{b64}' | base64 -d > /tmp/patch.diff")

        # Try git apply first
        # Commands run in tool.working_dir (set by SWEBenchToolConfig) — no need
        # to cd, and hardcoding '/testbed' breaks when the tool relocated to a
        # writable copy (see _maybe_relocate_testbed).
        result = self.tool.bash_unlimited("git apply /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result and "[error]" not in result:
            return result

        # Fallback: git apply --reject
        result = self.tool.bash_unlimited("git apply --reject /tmp/patch.diff 2>&1", timeout=30)
        if "[exit_code:" not in result and "[error]" not in result:
            return result

        # Final fallback: patch --forward prevents reversing an already-applied patch
        # (patch --batch otherwise treats "content already present" as a reversed patch
        # and removes it, causing test_empty_name_not_allowed-style evaluation failures
        # when the agent proactively added test content that the test_patch also adds).
        result = self.tool.bash_unlimited("patch --batch --forward --fuzz=5 -p1 -i /tmp/patch.diff 2>&1", timeout=60)
        if "[exit_code:" in result or "[error]" in result:
            logger.warning("_apply_patch: all methods failed.\npatch output:\n%s", result)
        return result

    def _run_tests(
        self,
        repo: str,
        test_directives: list[str],
        timeout: int = 1800,
        strict: bool = True,
    ) -> tuple[bool, str]:
        """Run test directives; return (all_passed, last-200-lines-of-output).

        Output is trimmed to the last 200 lines because some repos (Django) print
        tens of thousands of lines of DB-setup preamble before test results appear.

        strict=False is used for pass_to_pass checks and relaxes two edge cases
        that are not the agent's fault:
        - exit 4: pytest found no tests (SWE-bench stores some truncated test IDs
          that pytest cannot parse — the benchmark data is malformed for these).
        - non-zero exit but zero failures: old sympy containers emit import-level
          deprecation errors that inflate the exit code even when all tests passed.
        """
        if not test_directives:
            return True, ""

        test_cmd = f"{CONDA_ACTIVATE} && {self._build_test_cmd(repo, test_directives)}"
        result = self.tool._container.exec(test_cmd, timeout=timeout, workdir=self.tool._config.working_dir)

        raw = (result.stdout or "") + (result.stderr or "")
        output = "\n".join(raw.splitlines()[-200:])

        if result.exit_code == 124:  # shell timeout
            return False, output + "\n[timed out]"
        if result.exit_code == 4 and not strict:
            return True, output
        # pytest exit-4 ("no tests collected") also fires when test IDs are truncated, e.g.
        # "test_X[(1," with no closing "]" — a SWE-bench dataset artifact. No agent can fix
        # this, so treat it as pass even in strict (F2P) mode.
        if result.exit_code == 4 and strict:
            if re.search(r"\[\s*[^\]]*$", "\n".join(test_directives), re.MULTILINE):
                logger.warning(
                    "_run_tests: F2P test IDs appear truncated (no closing ']'); treating as pass. Directives: %r",
                    test_directives,
                )
                return True, output
        if result.exit_code != 0 and not strict:
            tests_ran = bool(re.search(r"\b\d+\s+passed\b", output, re.IGNORECASE))
            no_failures = not bool(re.search(r"\b\d+\s+failed\b", output, re.IGNORECASE))
            if tests_ran and no_failures:
                return True, output
        return result.exit_code == 0, output

    @staticmethod
    def _normalize_django_directive(directive: str) -> str | None:
        """Convert SWE-bench unittest verbose format to Django runtests.py format.

        SWE-bench stores test directives in Python unittest verbose output format:
            "test_method (module.path.ClassName)"
        Django's runtests.py expects:
            "module.path.ClassName.test_method"

        Returns None for malformed directives that look like human-readable test
        descriptions (spaces, special chars) — SWE-bench dataset artifacts that
        cannot be collected and would cause ModuleNotFoundError.
        """
        directive = directive.strip()
        m = re.match(r"^(\w+)\s+\(([^)]+)\)$", directive)
        if m:
            method, class_path = m.group(1), m.group(2)
            # Python 3.11+ unittest verbose output sometimes already includes the
            # method inside the parens: "test_foo (mod.Class.test_foo)". Returning
            # f"{class_path}.{method}" would double-append and Django's loader
            # would reject it. Detect and pass through unchanged.
            if class_path.endswith(f".{method}") or class_path == method:
                return class_path
            return f"{class_path}.{method}"
        # A valid Python dotted path has no spaces or special chars.
        if re.search(r"[\s#'\"]", directive):
            logger.debug("Skipping malformed P2P directive (bad test ID in dataset): %r", directive)
            return None
        return directive

    @staticmethod
    def _build_test_cmd(repo: str, test_directives: list[str]) -> str:
        """Build the test command based on repo's test framework."""
        if "django" in repo:
            normalized = [SWEBenchVerifiedTask._normalize_django_directive(t) for t in test_directives]
            valid = [t for t in normalized if t is not None]
            if not valid:
                return "true"  # all directives were malformed; treat as pass
            tests = " ".join(shlex.quote(t) for t in valid)
            # PYTHONIOENCODING=utf-8: Django's test runner emits Unicode characters
            # (e.g. U+2026 ellipsis) that fail when the container locale is ASCII-only.
            return f"PYTHONIOENCODING=utf-8 ./tests/runtests.py --verbosity 2 {tests}"
        if "sympy" in repo:
            tests = " ".join(shlex.quote(t) for t in test_directives)
            return f"bin/test -C --verbose {tests}"
        tests = " ".join(shlex.quote(t) for t in test_directives)
        # --no-header requires pytest>=6.0; many SWE-bench containers ship older versions.
        return f"python -m pytest -rN -p no:cacheprovider {tests}"


class SWEBenchVerifiedTaskConfig(TaskConfig[SWEBenchVerifiedTaskMetadata]):
    """Serializable factory that produces a SWEBenchVerifiedTask.

    Loads heavy execution data (problem_statement, patch, test_patch, etc.) from
    the per-task execution cache populated by ``SWEBenchVerifiedBenchmarkConfig.install()``.
    """

    oracle_mode: bool = False
    append_submission_instructions: bool = True

    def verify_installed(self) -> None:
        """Fail fast if the per-task execution cache is empty."""
        cache_dir = type(self).task_execution_cache_dir()
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise RuntimeError(
                f"SWE-bench Verified per-task execution cache is empty at {cache_dir}. "
                f"Run `cube install swebench-verified-cube` (or "
                f"`SWEBenchVerifiedBenchmarkConfig.install()`) on this worker first."
            )

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> SWEBenchVerifiedTask:
        if runtime_context is None or "infra" not in runtime_context:
            raise ValueError("SWEBenchVerifiedTaskConfig.make() requires runtime_context['infra'].")

        self.verify_installed()
        raw = self.load_task_execution_info()
        execution_info = SWEBenchVerifiedExecutionInfo.model_validate(raw)

        return SWEBenchVerifiedTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config or TerminalToolConfig(working_dir="/testbed", enable_file_actions=True),
            runtime_context=runtime_context,
            oracle_mode=self.oracle_mode,
            append_submission_instructions=self.append_submission_instructions,
        )
