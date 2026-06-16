"""Task and TaskConfig for swegym-lite-cube."""

from __future__ import annotations

import base64
import logging
import re
import shlex
from typing import Any

from cube.container import relocate_if_readonly
from cube.core import Observation
from cube.resource import IncompatibleInfraError
from cube.task import RuntimeContext, Task, TaskConfig, TaskExecutionInfo, TaskMetadata

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


class SWEGymLiteTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for SWE-Gym Lite tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (problem_statement, patch, test_patch, etc.) lives on
    ``SWEGymLiteExecutionInfo`` and is loaded lazily by
    ``SWEGymLiteTaskConfig.make()``.

    Note: SWE-Gym (unlike SWE-bench Verified) ships no human ``difficulty``
    annotation, so there is no ``difficulty`` field here.
    """

    repo: str
    """GitHub repository, e.g. 'getmoto/moto'."""

    version: str
    """Repository version string, e.g. '4.0'."""

    base_commit: str
    """Git SHA of the base commit the agent starts from."""


class SWEGymLiteExecutionInfo(TaskExecutionInfo):
    """Heavy per-task execution data for SWE-Gym Lite — populated on the worker.

    Loaded by ``SWEGymLiteTaskConfig.make()`` from the per-task execution cache
    written by ``SWEGymLiteBenchmarkConfig.install()``.
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


class SWEGymLiteTask(Task[SWEGymLiteTaskMetadata, ContainerTerminalTool]):
    """A single SWE-Gym Lite task with test-based validation."""

    validate_per_step: bool = False

    oracle_mode: bool = False
    """If True, write the gold patch to /tmp/gold_patch.diff in reset()."""

    append_submission_instructions: bool = True
    """If True, append evaluation constraints and final_step submission instructions
    to the problem statement. Disable for raw-benchmark comparisons where the task
    description must match the original SWE-Gym problem statement exactly."""

    @property
    def _exec(self) -> SWEGymLiteExecutionInfo:
        """Typed view on execution_info — fails fast if it was not populated."""
        if not isinstance(self.execution_info, SWEGymLiteExecutionInfo):
            raise RuntimeError(
                f"SWEGymLiteTask {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected SWEGymLiteExecutionInfo. "
                f"Construct via SWEGymLiteTaskConfig.make() so it is populated."
            )
        return self.execution_info

    def _make_tool(self, role: str | None = None) -> ContainerTerminalTool:
        """Ensure /testbed files are writable and git-safe, then build the tool.

        NON-ROOT DOCKER WORKAROUND. Upstream SWE-bench images assume `USER root`
        (matched by Daytona, local Docker, AWS, Azure) and bake conda+sources
        into a root-owned /testbed. On non-root infras — the EAI Toolkit enforces
        uid 13011 by cluster policy — the runtime user can't `chmod` root-owned
        files in place, can't write to a read-only /testbed, and Git 2.35.2+
        rejects repos owned by a different user. This block normalises all three
        before the tool is constructed. Root-running infras short-circuit at the
        writability probes and pay essentially nothing.

        Two pre-flight fixes applied unconditionally:
        1. git safe.directory: Git 2.35.2+ refuses to operate in repos owned by a
           different user. Configure /testbed as safe so agents can run `git diff`.
        2. chmod via cp/mv: some containers ship root-owned 644 .py files inside a
           world-writable /testbed. mv unlinks via the writable parent and recreates
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
        return self.tool_config.model_copy(update={"working_dir": new_wd}).make(container=self._container)

    # Non-root writability guard — fail loud (IncompatibleInfraError), never silent-0.
    _GOLD_TARGET_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)

    def _raise_if_unpatchable(self, working_dir: str) -> None:
        """Raise ``IncompatibleInfraError`` if the gold patch's target files can't be
        written after the writability normalisation in ``_make_tool``.

        On non-root infra (EAI toolkit, uid 13011) some images ship root-owned package
        subdirs (e.g. psf/requests' ``/testbed/requests/``) that ``cp/mv`` can't reparent
        without root. Any patch — gold or agent — to a file there dies with ``git apply:
        Permission denied``, and the task would silently score a *correct* fix 0. We probe
        exactly the files the **gold patch** touches (the canonical fix's source paths) —
        not the whole tree — so unrelated root-owned vendored dirs (e.g. astropy's
        ``astropy/_erfa``) don't false-positive. If even the gold patch's targets aren't
        writable, no agent patch to the same files can be either → surface the cube's
        IncompatibleInfraError (terminal + non-retriable per episode.py) instead. Root
        infras leave everything writable, so this never fires there. See #446.
        """
        targets = sorted(set(self._GOLD_TARGET_RE.findall(self._exec.patch or "")))
        if not targets:
            return
        # A target is patchable iff its parent dir is writable (git apply unlinks+recreates
        # via the dir) and the file itself is writable-or-absent.
        quoted = " ".join(shlex.quote(t) for t in targets)
        probe = (
            f"cd {shlex.quote(working_dir)} || exit 0; for f in {quoted}; do "
            'd=$(dirname "$f"); if [ ! -w "$d" ] || { [ -e "$f" ] && [ ! -w "$f" ]; }; then '
            "printf '%s\\n' \"$f\"; break; fi; done"
        )
        blocked = self._container.exec(probe, timeout=60).stdout.strip()
        if blocked:
            uid = self._container.exec("id -u", timeout=15).stdout.strip()
            raise IncompatibleInfraError(
                f"{self.metadata.id}: gold-patch target {blocked.splitlines()[0]!r} under "
                f"{working_dir} is not writable by the runtime user (uid {uid}) after writability "
                f"normalisation — this image ships root-owned package dirs a non-root infra cannot "
                f"patch (git apply would fail 'Permission denied', silently scoring a correct patch 0). "
                f"This task needs 'container:root'; run it on a root-capable infra (daytona/local/aws). "
                f"See cube-harness#446."
            )

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        # Fail loud (IncompatibleInfraError), not silent-0, if the gold patch's target
        # files aren't writable on this (non-root) infra. Called here — inside reset(),
        # which the episode runs within its try/except — so the error is classified
        # terminal & non-retriable (episode.py) rather than escaping setup.
        self._raise_if_unpatchable(self.tool._config.working_dir)

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
            "version": self.metadata.version,
        }

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:

        fail_to_pass = self._exec.fail_to_pass
        pass_to_pass = self._exec.pass_to_pass
        eval_timeout = self._exec.eval_timeout

        # Baseline p2p: run pass_to_pass BEFORE applying the test_patch so a post-patch
        # p2p failure can be attributed correctly — an agent regression vs. a pre-existing
        # environmental flake (e.g. a test that reaches the network from an egress-limited
        # container and fails with ConnectionResetError regardless of the patch). Without
        # this, any container with a flaky p2p test silently scores a correct fix as 0.
        p2p_baseline_passed = True
        if pass_to_pass:
            p2p_baseline_passed, _ = self._run_tests(
                self.metadata.repo, pass_to_pass, timeout=eval_timeout, strict=False
            )
            if not p2p_baseline_passed:
                logger.warning(
                    "evaluate: p2p baseline already fails for %s — post-patch p2p will not be counted against the agent",
                    self.metadata.id,
                )

        # Apply test patch
        self._apply_patch(self._exec.test_patch)

        # Run FAIL_TO_PASS tests — these must all pass for resolution
        f2p_passed, f2p_output = self._run_tests(self.metadata.repo, fail_to_pass, timeout=eval_timeout)

        # Run PASS_TO_PASS tests — these must remain passing.
        # strict=False: exit-4 "no tests collected" treated as passed (truncated test IDs
        # in the upstream data cannot be collected; the agent is not responsible for that).
        p2p_passed = True
        p2p_output = ""
        if pass_to_pass:
            p2p_passed, p2p_output = self._run_tests(
                self.metadata.repo, pass_to_pass, timeout=eval_timeout, strict=False
            )
            # If baseline p2p was already broken, don't punish the agent for an unchanged
            # p2p result. Coarse-grained (whole-suite) by design: a regression hidden in a
            # mostly-broken suite can slip through, but the alternative (silently scoring
            # correct fixes as 0) is worse. Per-test diffing is a possible follow-up.
            if not p2p_baseline_passed and not p2p_passed:
                logger.info(
                    "evaluate: p2p still failing post-patch but baseline was already failing for %s — treating as pass",
                    self.metadata.id,
                )
                p2p_passed = True
                p2p_output += "\n[NOTE: pre-existing p2p baseline failures; not counted as agent regression]"

        resolved = f2p_passed and p2p_passed
        reward = 1.0 if resolved else 0.0

        return reward, {
            "done": True,
            "resolved": resolved,
            "fail_to_pass_passed": f2p_passed,
            "pass_to_pass_passed": p2p_passed,
            "pass_to_pass_baseline_passed": p2p_baseline_passed,
            "fail_to_pass_output": f2p_output,
            "pass_to_pass_output": p2p_output,
        }

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

        Output is trimmed to the last 200 lines because some repos print thousands of
        lines of setup/collection preamble before the test results appear.

        strict=False is used for pass_to_pass checks and relaxes two edge cases
        that are not the agent's fault:
        - exit 4: pytest found no tests (the upstream data stores some truncated test
          IDs that pytest cannot parse — the dataset is malformed for these).
        - non-zero exit but zero failures: some containers emit import-level deprecation
          errors that inflate the exit code even when every test passed.
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
    def _build_test_cmd(repo: str, test_directives: list[str]) -> str:
        """Build the pytest command for a SWE-Gym Lite task.

        All 11 SWE-Gym Lite repos (getmoto/moto, python/mypy, conan-io/conan,
        iterative/dvc, dask/dask, pydantic/pydantic, pandas-dev/pandas,
        facebookresearch/hydra, bokeh/bokeh, Project-MONAI/MONAI, modin-project/modin)
        use pytest with standard node-id directives, so a single pytest invocation
        covers every task — no per-repo special-casing (unlike SWE-bench Verified,
        which also carries django/sympy with bespoke runners).
        """
        tests = " ".join(shlex.quote(t) for t in test_directives)
        # --no-header requires pytest>=6.0; many SWE-bench containers ship older versions.
        return f"python -m pytest -rN -p no:cacheprovider {tests}"


class SWEGymLiteTaskConfig(TaskConfig[SWEGymLiteTaskMetadata]):
    """Serializable factory that produces a SWEGymLiteTask.

    Loads heavy execution data (problem_statement, patch, test_patch, etc.) from
    the per-task execution cache populated by ``SWEGymLiteBenchmarkConfig.install()``.
    """

    oracle_mode: bool = False
    append_submission_instructions: bool = True

    def verify_installed(self) -> None:
        """Fail fast if the per-task execution cache is empty."""
        cache_dir = type(self).task_execution_cache_dir()
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise RuntimeError(
                f"SWE-Gym Lite per-task execution cache is empty at {cache_dir}. "
                f"Run `cube install swegym-lite-cube` (or "
                f"`SWEGymLiteBenchmarkConfig.install()`) on this worker first."
            )

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> SWEGymLiteTask:
        if runtime_context is None or "infra" not in runtime_context:
            raise ValueError("SWEGymLiteTaskConfig.make() requires runtime_context['infra'].")

        self.verify_installed()
        raw = self.load_task_execution_info()
        execution_info = SWEGymLiteExecutionInfo.model_validate(raw)

        return SWEGymLiteTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config or TerminalToolConfig(working_dir="/testbed", enable_file_actions=True),
            runtime_context=runtime_context,
            oracle_mode=self.oracle_mode,
            append_submission_instructions=self.append_submission_instructions,
        )
