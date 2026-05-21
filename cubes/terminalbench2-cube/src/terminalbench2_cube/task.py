"""Task and TaskConfig for terminalbench2-cube."""

import base64
import io
import logging
import re
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from cube.container import relocate_if_readonly
from cube.core import Observation
from cube.task import RuntimeContext, Task, TaskConfig, TaskExecutionInfo, TaskMetadata
from cube.tools.terminal import ContainerTerminalTool, TerminalToolConfig
from terminalbench2_cube.pytest_parser import PytestParser

logger = logging.getLogger(__name__)


class TerminalBench2TaskMetadata(TaskMetadata):
    """TaskMetadata subclass for Terminal-Bench tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (instruction, archive) lives in the per-task execution
    cache and is loaded lazily by TerminalBench2TaskConfig.make().
    """

    difficulty: str
    """Task difficulty level: 'easy', 'medium', or 'hard'."""

    category: str
    """Task category, e.g. 'scientific-computing', 'debugging'."""

    tags: list[str]
    """Task tags for fine-grained filtering."""

    max_agent_timeout_sec: int
    """Maximum wall-clock seconds the agent is allowed to run (from task.toml)."""


class TerminalBench2ExecutionInfo(TaskExecutionInfo):
    """Heavy per-task execution data for TerminalBench — populated on the worker."""

    instruction: str
    archive: str
    max_test_timeout_sec: int = 900


class TerminalBench2Task(Task[TerminalBench2TaskMetadata, ContainerTerminalTool]):
    """A single Terminal-Bench task with pytest-based validation."""

    metadata: TerminalBench2TaskMetadata  # type: ignore[assignment]

    validate_per_step: bool = False
    accept_agent_stop: bool = True
    oracle_mode: bool = False

    # Container-side paths — always under /tmp so logic works uniformly on root
    # and non-root backends (EAI Toolkit images have /tmp mode 1777).
    _working_dir: str = PrivateAttr(default="/app")
    _solution_dir: str = PrivateAttr(default="/tmp/solution")
    _tests_dir: str = PrivateAttr(default="/tmp/tests")
    _logs_verifier_dir: str = PrivateAttr(default="/tmp/logs/verifier")

    @property
    def _exec(self) -> TerminalBench2ExecutionInfo:
        """Typed view on execution_info — fails fast if it was not populated."""
        if not isinstance(self.execution_info, TerminalBench2ExecutionInfo):
            raise RuntimeError(
                f"TerminalBench2Task {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected TerminalBench2ExecutionInfo. "
                f"Construct via TerminalBench2TaskConfig.make() so it is populated."
            )
        return self.execution_info

    def _build_tool(self) -> None:
        # NON-ROOT DOCKER WORKAROUND
        # tbench2 task images assume root-by-default Docker semantics (Daytona,
        # local Docker, AWS, Azure all give the container `USER root`). On
        # non-root backends — EAI Toolkit enforces uid 13011 by cluster policy
        # — `/app` is often read-only and `git` may not be installed in the
        # image. The block below copies `/app` to `/tmp/app` only if needed,
        # and configures git identity *best-effort* so images without git
        # still set up cleanly. Daytona/local short-circuit at the writable-
        # /app probe and never execute the extra_setup chain.
        # auto-fix(418)↓
        new_wd = relocate_if_readonly(
            self._container,
            self.tool_config.working_dir,
            "/tmp/app",
            # Git refuses dirs whose ownership differs ('dubious ownership').
            # '*' disables the check globally — safe in this test-runner context.
            # uid 13011 (Toolkit) has no /etc/passwd entry, so git can't
            # auto-detect committer identity without explicit config.
            # Best-effort: tbench2 task images don't uniformly ship git (e.g.
            # nginx-request-logging, sqlite-with-gcov, configure-git-webserver
            # don't). Outer subshell so `|| true` neutralises ONLY the git
            # chain — the `&&` from relocate_if_readonly's `cp -a ... && X`
            # composition still propagates a cp failure (auto-fix(176)'s
            # invariant). Writable-/app paths (daytona, local) short-circuit
            # before this runs; this only matters for non-root infras (toolkit).
            extra_setup=(
                "( ( command -v git >/dev/null 2>&1 && "
                "git config --global --add safe.directory '*' && "
                "git config --global user.email 'cube-harness@example.com' && "
                "git config --global user.name 'Cube Harness' "
                ") || true )"
            ),
        )
        # /auto-fix(418)
        self._working_dir = new_wd
        self._tool = self.tool_config.model_copy(update={"working_dir": new_wd}).make(container=self._container)

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()

        # Fast-fail if the container sandbox is broken (all bash commands return
        # exit_code=1 with no output). Avoids wasting 100 agent steps on a dead env.
        health = self.tool.bash("echo ok", timeout=15)
        if "ok" not in health:
            raise RuntimeError(f"Container health check failed for {self.metadata.id!r}: {health!r}")

        # Extract task archive to a temp dir (kept alive until close())
        self._temp_dir = tempfile.TemporaryDirectory()
        task_path = Path(self._temp_dir.name) / self.metadata.id
        task_path.mkdir(parents=True, exist_ok=True)
        archive = self._exec.archive
        if isinstance(archive, str):
            archive = base64.b64decode(archive)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(path=task_path, filter="data")
        self._task_path = task_path

        # Oracle mode: upload solution for debugging/baselines
        if self.oracle_mode and (task_path / "solution").exists():
            solution_dir = task_path / "solution"
            if self._working_dir != "/app":
                self._rewrite_files_locally(solution_dir, {"/app/": f"{self._working_dir}/"})
            self.tool.bash(f"mkdir -p {self._solution_dir}")
            self._upload_directory(solution_dir, self._solution_dir)
            # Pre-install python3 + uv so oracle solve.sh scripts work on minimal
            # images (e.g. bare LaTeX) that ship without python3.  In non-oracle
            # runs the agent installs its own deps; we don't add overhead there.
            self._ensure_uv_preinstalled()

        instruction = self._exec.instruction
        # Mirror the test-script rewrite: when /app is read-only and we relocated
        # the working dir to /tmp/app, the agent's instructions still reference
        # /app verbatim (from terminal-bench task.yaml).  Rewrite so the agent
        # writes to the same path the evaluator checks.
        if self._working_dir != "/app":
            instruction = re.sub(r"/app(?=[\s/.,;:)\]'\"]|$)", self._working_dir, instruction)

        return Observation.from_text(instruction), {
            "task_id": self.metadata.id,
            "difficulty": self.metadata.difficulty,
            "category": self.metadata.category,
        }

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        """Run the upstream pytest verifier in the sandbox and return the reward.

        Deliberately mutates container state (uploads tests, installs `uv`).  Safe
        here because `validate_per_step=False` makes this a single terminal call
        after the episode ends — don't copy this pattern into a per-step evaluator.
        """
        # Upload test harness to the sandbox
        if self._task_path is not None:
            tests_dir = self._task_path / "tests"
            self.tool.bash(f"mkdir -p {self._tests_dir} {self._logs_verifier_dir}")
            if tests_dir.exists():
                # Rewrite hardcoded paths in local test files before uploading.
                # Done in Python (not via sed) to avoid shell-quoting pitfalls
                # (e.g. single quotes inside sed expressions) and GNU sed
                # re-scanning surprises that produce double '/tmp/' prefixes.
                path_subs: dict[str, str] = {
                    "/logs/verifier": self._logs_verifier_dir,
                    "/tests/": self._tests_dir + "/",
                    "/tests ": self._tests_dir + " ",
                    # Path("/tests") — no trailing slash, quote-boundary match
                    '"/tests"': f'"{self._tests_dir}"',
                    "'/tests'": f"'{self._tests_dir}'",
                }
                if self._working_dir != "/app":
                    path_subs["/app/"] = f"{self._working_dir}/"
                    path_subs['"/app"'] = f'"{self._working_dir}"'
                    path_subs["'/app'"] = f"'{self._working_dir}'"
                self._rewrite_files_locally(tests_dir, path_subs)
                self._upload_directory(tests_dir, self._tests_dir)
                self.tool.bash(f"chmod +x {self._tests_dir}/test.sh")

        # Pre-install `uv` + fake HOME so test.sh's
        #   curl https://astral.sh/uv/…/install.sh | sh  →  source $HOME/.local/bin/env
        # succeeds even when astral.sh is unreachable (EAI Toolkit returns 403
        # Forbidden on that host) and when $HOME is a read-only mount.
        # pypi is reachable on Toolkit; pip installs uv in ~10 s.
        self._ensure_uv_preinstalled()

        # Run test.sh → pytest → writes reward.txt in the logs-verifier dir.
        # Tool's working_dir is already set (may be /tmp/app after relocation).
        output = self.tool.bash(
            f"export HOME=/tmp/fakehome && bash {self._tests_dir}/test.sh",
            timeout=self._exec.max_test_timeout_sec,
        )
        test_results = self._parse_pytest_output(output)

        # Read reward written by test.sh
        reward_output = self.tool.bash(f"cat {self._logs_verifier_dir}/reward.txt 2>/dev/null || echo 0")
        try:
            reward = float(reward_output.strip().split()[0])
        except (ValueError, IndexError):
            reward = 0.0

        n_passed = sum(1 for r in test_results.values() if r == "passed")
        return reward, {
            "done": True,
            "passed": n_passed,
            "total": len(test_results),
            "all_passed": len(test_results) > 0 and n_passed == len(test_results),
            "test_results": test_results,
            "output_preview": output[:1000] if output else "",
        }

    def _upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file into the container (text or binary)."""
        try:
            content = local_path.read_text(encoding="utf-8")
            escaped = content.replace("'", "'\\''")
            self._container.exec(f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}", workdir=self._working_dir)
            self._container.exec(f"printf '%s' '{escaped}' > {shlex.quote(remote_path)}", workdir=self._working_dir)
        except UnicodeDecodeError:
            b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
            self._container.exec(f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}", workdir=self._working_dir)
            self._container.exec(
                f"printf '%s' {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}",
                workdir=self._working_dir,
            )

    def _upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        """Upload a local directory tree into the container via base64+tar.

        Packs ``local_dir`` into an in-memory tar.gz, writes the base64 string
        to a temp file via multi-chunk ``printf >> file`` (shell-quoting-safe
        even through nested eai CLI → remote bash layers), then decodes and
        extracts.  Uses only base64+tar which every POSIX task image ships.
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(local_dir, arcname=".")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        remote_q = shlex.quote(remote_dir)
        # Write the base64 payload in 8 KB chunks — short chunks are robust
        # through multiple shell layers (observed with eai CLI + bash -lc).
        chunk_size = 8192
        staging = "/tmp/cube-upload.tar.gz.b64"
        self._container.exec(f": > {staging}", workdir=self._working_dir)
        for i in range(0, len(b64), chunk_size):
            self._container.exec(
                f"printf %s {shlex.quote(b64[i : i + chunk_size])} >> {staging}", workdir=self._working_dir
            )
        self._container.exec(
            f"mkdir -p {remote_q} && base64 -d < {staging} | tar -xzf - -C {remote_q} && rm -f {staging}",
            workdir=self._working_dir,
        )

    @staticmethod
    def _rewrite_files_locally(directory: Path, subs: dict[str, str]) -> None:
        """Apply string substitutions to *.sh and *.py files under ``directory`` in-place.

        Preferred over sed-in-container: avoids shell-quoting pitfalls (e.g. single
        quotes inside sed expressions) and GNU sed re-scanning surprises.
        """
        for f in directory.rglob("*"):
            if f.suffix in (".sh", ".py") and f.is_file():
                text = f.read_text(errors="replace")
                new_text = text
                for k, v in subs.items():
                    new_text = new_text.replace(k, v)
                if new_text != text:
                    f.write_text(new_text)

    def _ensure_uv_preinstalled(self) -> None:
        """Pre-install ``uv`` so test.sh's ``source $HOME/.local/bin/env`` works.

        Terminal-Bench task test.sh files bootstrap ``uv`` via
            curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
            source $HOME/.local/bin/env

        On some backends (EAI Toolkit in particular), ``astral.sh`` returns HTTP
        403 (cluster IP range rejected by Cloudflare) AND ``curl`` isn't even in
        the image AND ``$HOME`` is read-only.  All three failures cascade: the
        curl is rc=127, the source finds nothing, uvx is missing, pytest can't
        run, reward=0.

        Fix: ensure python3 is present (some minimal images like LaTeX ship
        without it — install via apt if needed), then install ``uv`` via ``pip``
        from PyPI into ``/tmp/fakehome/.local/bin``, create the env file
        test.sh expects, and override ``HOME=/tmp/fakehome`` when running test.sh.

        Non-root fallback: when running as non-root (e.g. EAI Toolkit uid 13011),
        apt-get requires root.  Fall back to downloading the python3 packages
        via ``apt-get download`` (works without root, writes to /tmp) and
        extracting them with ``dpkg-deb --extract``, then use that python3 to
        bootstrap pip (via get-pip.py with SSL verification disabled for the
        bootstrap step only) and finally ``pip install uv``.
        """
        marker = "/tmp/fakehome/.local/bin/uv"
        probe = self.tool.bash(f"test -x {marker} && echo EXISTS || echo MISSING", timeout=15)
        if "EXISTS" in probe:
            return

        # Fast path: cube_data bundle mount (auto-provisioned by ToolkitInfraConfig).
        # When ToolkitInfraConfig mounts /opt/cube/ with the cube_data bundle, copy
        # the uv binaries directly — bypasses the python3-bootstrap path that fails
        # on minimal images lacking python3, curl, AND apt sources.  Note: EAI data
        # mounts are read-only and strip the execute bit (mode 0600), so we use
        # ``-f`` for the probe and ``chmod +x`` after copying into the writable HOME.
        assets_probe = self.tool.bash(
            "test -f /opt/cube/uv && test -f /opt/cube/uvx && echo YES || echo NO",
            timeout=15,
        )
        if "YES" in assets_probe:
            # NON-ROOT DOCKER WORKAROUND
            # The `/opt/cube` data mount and the surrounding fast-path exist
            # entirely to serve non-root Docker backends where the image lacks
            # `curl` / `apt` / `astral.sh` egress and the test.sh's upstream
            # `curl install.sh | sh` route cannot complete. On root-capable
            # backends (Daytona, local, AWS, Azure) `apt-get install curl`
            # + `curl install.sh` runs inside the test image directly and
            # this entire fast-path is unreached.
            # auto-fix(420)↓
            # Verify /opt/cube/uvx is new enough to handle test.sh's `uvx -w`
            # (a.k.a. `--with`) syntax. Older cube_assets bundles (uv ≲0.4) fail
            # the upstream test.sh with `uvx: error: unexpected argument '-w'`,
            # silently producing reward=0 on tasks the agent solved correctly.
            # When the probe fails we fall through to the pip-install path so a
            # stale bundle in any user's cube_assets is self-healing.
            version_probe = self.tool.bash(
                "/opt/cube/uvx --help 2>&1 | grep -q -- ' --with ' && echo OK || echo OLD",
                timeout=15,
            )
            if "OK" in version_probe:
                logger.info("Using mounted /opt/cube/uv for uv install (version supports --with)")
                self.tool.bash(
                    "export HOME=/tmp/fakehome && "
                    "mkdir -p $HOME/.local/bin && "
                    "cp /opt/cube/uv /opt/cube/uvx $HOME/.local/bin/ && "
                    "chmod +x $HOME/.local/bin/uv $HOME/.local/bin/uvx && "
                    "printf 'export PATH=\"$HOME/.local/bin:$PATH\"\\n' > $HOME/.local/bin/env",
                    timeout=30,
                )
                return
            logger.warning(
                "/opt/cube/uv missing --with flag; falling through to pip-install path "
                "(rebuild cube_assets via cube-infra-toolkit/scripts/publish-cube-assets.sh)"
            )
            # /auto-fix(420)

        # Some minimal images (e.g. bare LaTeX) ship without python3.
        # Try root apt-get first (works on Docker/local backends).
        has_python = self.tool.bash("python3 --version 2>/dev/null && echo HAS_PYTHON || echo NO_PYTHON", timeout=15)
        if "NO_PYTHON" in has_python:
            logger.info("python3 not found — trying apt-get install (root path)")
            self.tool.bash(
                "apt-get update -qq && apt-get install -y --no-install-recommends python3 python3-pip 2>&1",
                timeout=120,
            )
            has_python = self.tool.bash(
                "python3 --version 2>/dev/null && echo HAS_PYTHON || echo NO_PYTHON", timeout=15
            )

        if "NO_PYTHON" in has_python:
            # Root apt-get failed (non-root container).  Download packages without
            # root and extract them to /tmp/python3_pkg.
            logger.info("root apt-get failed — trying non-root apt download + dpkg-deb extract")
            self._install_python3_nonroot()
            has_python = self.tool.bash(
                "test -x /tmp/python3_pkg/usr/bin/python3.12 && echo HAS_PYTHON || echo NO_PYTHON", timeout=10
            )

        if "NO_PYTHON" in has_python:
            logger.warning("python3 unavailable — skipping uv pre-install; test.sh will fall back to curl")
            return

        logger.info("Pre-installing uv into /tmp/fakehome/.local/bin (backend-portable workaround)")

        use_extracted = "exists" in self.tool.bash(
            "test -x /tmp/python3_pkg/usr/bin/python3.12 && echo exists || echo missing", timeout=5
        )

        if use_extracted:
            # Bootstrap pip via get-pip.py (SSL verification disabled for this
            # one-time download from bootstrap.pypa.io; pip itself uses certifi).
            self.tool.write_file(
                "/tmp/_dl_pip.py",
                "import ssl, urllib.request as R\n"
                "ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n"
                "ctx.check_hostname=False\n"
                "ctx.verify_mode=ssl.CERT_NONE\n"
                "open('/tmp/get-pip.py','wb').write(R.urlopen('https://bootstrap.pypa.io/get-pip.py',context=ctx).read())\n",
            )
            cmd = (
                "export LD_LIBRARY_PATH=/tmp/python3_pkg/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && "
                "export HOME=/tmp/fakehome && "
                "mkdir -p $HOME/.local/bin && "
                "/tmp/python3_pkg/usr/bin/python3.12 /tmp/_dl_pip.py 2>&1 && "
                "/tmp/python3_pkg/usr/bin/python3.12 /tmp/get-pip.py --target /tmp/pip_pkg -q 2>&1 && "
                "PYTHONPATH=/tmp/pip_pkg /tmp/python3_pkg/usr/bin/python3.12 "
                "-m pip install --quiet --target /tmp/uv_pkg uv==0.9.5 2>&1 && "
                "cp /tmp/uv_pkg/bin/uv /tmp/uv_pkg/bin/uvx $HOME/.local/bin/ && "
                "printf 'export PATH=\"$HOME/.local/bin:$PATH\"\\n' > $HOME/.local/bin/env"
            )
        else:
            cmd = (
                "export HOME=/tmp/fakehome && "
                "mkdir -p $HOME/.local/bin && "
                # --trusted-host covers images where ca-certificates is absent (e.g. bare LaTeX)
                "python3 -m pip install --quiet --target /tmp/uv_pkg "
                "--trusted-host pypi.org --trusted-host files.pythonhosted.org uv && "
                "cp /tmp/uv_pkg/bin/uv /tmp/uv_pkg/bin/uvx $HOME/.local/bin/ && "
                "printf 'export PATH=\"$HOME/.local/bin:$PATH\"\\n' > $HOME/.local/bin/env"
            )

        result = self.tool.bash(cmd, timeout=300)
        if not result or "error" in result.lower():
            logger.warning("uv pre-install may have failed; test.sh will fall back to curl: %s", result[:200])

    def _install_python3_nonroot(self) -> None:
        """Download python3.12 packages via apt and extract with dpkg-deb (no root needed)."""
        logger.info("Downloading python3.12 packages via apt (non-root) and extracting to /tmp/python3_pkg")
        apt_opts = "-o Dir::State::Lists=/tmp/apt/lists -o Dir::Cache::Archives=/tmp/apt/archives"
        cmd = (
            "mkdir -p /tmp/apt/lists/partial /tmp/apt/archives/partial /tmp/python3_pkg && "
            f"apt-get {apt_opts} update -qq 2>/dev/null || true && "
            f"cd /tmp && apt-get {apt_opts} download "
            "python3.12-minimal libpython3.12-minimal libpython3.12-stdlib python3-minimal 2>&1 && "
            'for deb in /tmp/*.deb; do dpkg-deb --extract "$deb" /tmp/python3_pkg/; done'
        )
        result = self.tool.bash(cmd, timeout=180)
        logger.info("python3 nonroot install: %s", (result or "")[-300:])

    def finished(self, obs: Observation | None = None) -> bool:
        return False

    def close(self) -> None:
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._task_path = None
        super().close()

    def _parse_pytest_output(self, output: str) -> dict[str, str]:
        """Parse pytest output, falling back to regex heuristics."""
        try:
            return {name: status.value for name, status in PytestParser().parse(output).items()}
        except ValueError:
            logger.debug("PytestParser failed, falling back to heuristics")

        results: dict[str, str] = {}
        for label, status in [("passed", "passed"), ("failed", "failed")]:
            match = re.search(rf"(\d+)\s+{label}", output)
            if match:
                for i in range(int(match.group(1))):
                    results[f"test_{label}_{i}"] = status
        return results


class TerminalBench2TaskConfig(TaskConfig[TerminalBench2TaskMetadata]):
    """Serializable factory that produces a TerminalBench2Task.

    Loads heavy execution data (instruction, archive) from the per-task execution
    cache in make(), so it works correctly in Ray workers.
    """

    oracle_mode: bool = False
    """If True, upload the gold solution to /solution in reset()."""

    def verify_installed(self) -> None:
        """Fail fast if the per-task execution cache is empty."""
        cache_dir = type(self).task_execution_cache_dir()
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise RuntimeError(
                f"TerminalBench per-task execution cache is empty at {cache_dir}. "
                f"Run `cube install terminalbench2-cube` (or "
                f"`TerminalBench2BenchmarkConfig.install()`) on this worker first."
            )

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> TerminalBench2Task:
        if runtime_context is None or "infra" not in runtime_context:
            raise ValueError("TerminalBench2TaskConfig.make() requires runtime_context['infra'].")

        self.verify_installed()
        raw = self.load_task_execution_info()
        execution_info = TerminalBench2ExecutionInfo.model_validate(raw)

        return TerminalBench2Task(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config
            or TerminalToolConfig(
                working_dir="/app",
                max_timeout=900,
                enable_file_actions=True,
            ),
            runtime_context=runtime_context,
            oracle_mode=self.oracle_mode,
        )


# === auto-fix notes ===  (spec: openspec/specs/auto-fix/spec.md)
# auto-fix-note(418) {class=L1 anchor=PR#418 hash=PENDING ctx=toolkit/eai-yul101/runtime-uid-13011/tbench2:configure-git-webserver+nginx-request-logging+sqlite-with-gcov/cube-harness@1e67efdb}
# auto-fix-note(420) {class=L1 anchor=PR#420 hash=PENDING ctx=toolkit/eai-yul101/cube_assets:stale-uv<0.4/tbench2:fix-git+nginx-request-logging+sqlite-with-gcov/cube-harness@1e67efdb}
