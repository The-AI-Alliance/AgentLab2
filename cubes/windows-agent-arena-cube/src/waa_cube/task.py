"""WAATask — CUBE task for a single WindowsAgentArena desktop-automation episode.

task = WAATask(metadata=..., tool_config=ComputerConfig(...), infra=LocalInfraConfig())
obs, info = task.reset()
while not done:
    action = agent(obs, task.action_set)
    env_out = task.step(action)
    obs, done = env_out.obs, env_out.done
task.close()
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests
from cube.benchmark import RuntimeContext  # noqa: F401 — triggers WAATask.model_rebuild()
from cube.core import Observation
from cube.resource import InfraConfig, ResourceHandle
from cube.task import Task, TaskExecutionInfo
from cube_computer_tool.axtree import linearize_accessibility_tree, tag_screenshot
from PIL import Image
from pydantic import Field, PrivateAttr

from waa_cube.azure import WAA_WINDOWS_RESOURCE
from waa_cube.vm_backend.evaluator import Evaluator
from waa_cube.vm_backend.setup_controller import SetupController

if TYPE_CHECKING:
    from cube_computer_tool.computer import ComputerBase

logger = logging.getLogger(__name__)

_POST_SNAPSHOT_SLEEP = 10  # seconds to wait after QMP loadvm before taking obs

# Per-VM dump dir for dead-Flask diagnostics. Written by _health_gate_or_raise
# when the gate fast-fails or times out. Inspect after a real eval to find the
# state pattern that distinguishes dead VMs from healthy ones.
_DEAD_VM_DIAG_DIR = Path(os.environ.get("WAA_DEAD_VM_DIAG_DIR", "/tmp/dead-flask-eval-diag"))

_SSH_KEY_PATH = os.path.expanduser(os.environ.get("WAA_DEAD_VM_DIAG_SSH_KEY", "~/.ssh/id_ed25519"))
_SSH_USER = "Docker"
_SSH_OPTS = [
    "-i",
    _SSH_KEY_PATH,
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
]

# All-cmd-shell battery — powershell -EncodedCommand fails on this image
# because OpenSSH's default shell loads a profile that breaks on
# ExecutionPolicy before our -NoProfile child can run. cmd-only sidesteps it.
_DIAG_BATTERY: tuple[tuple[str, str], ...] = (
    ("LISTENING_PORTS", "netstat -ano | findstr LISTENING"),
    ("PYTHON_PROCS", 'tasklist /v /fi "imagename eq python.exe" /fo list'),
    ("CADDY_PROCS", 'tasklist /v /fi "imagename eq caddy.exe" /fo list'),
    ("CADDY_WIN_PROCS", 'tasklist /v /fi "imagename eq caddy_windows_amd64.exe" /fo list'),
    (
        "PYTHON_CMDLINES",
        "wmic process where \"name='python.exe'\" get ProcessId,ParentProcessId,CommandLine /format:list",
    ),
    (
        "CADDY_CMDLINES",
        "wmic process where \"name like 'caddy%%'\" get ProcessId,ParentProcessId,Name,CommandLine /format:list",
    ),
    ("WINDOWSARENA_TASK", "schtasks /query /tn WindowsArena_OnLogon /v /fo list 2>nul"),
    (
        "WINDOWSARENA_LOG",
        "if exist C:\\WindowsArena_OnLogon_Log.txt (type C:\\WindowsArena_OnLogon_Log.txt) else (echo NO_WINDOWSARENA_LOG_FILE)",
    ),
    (
        "FLASK_LOG_TAIL",
        "if exist C:\\oem\\server\\server.log (more +0 C:\\oem\\server\\server.log) else if exist C:\\oem\\server.log (more +0 C:\\oem\\server.log) else (echo NO_FLASK_LOG_FILE)",
    ),
    ("SYSTEM_EVT_ERR", 'wevtutil qe System "/q:*[System[(Level=1 or Level=2)]]" /c:30 /rd:true /f:text'),
    ("APP_EVT_ERR", 'wevtutil qe Application "/q:*[System[(Level=1 or Level=2)]]" /c:30 /rd:true /f:text'),
    ("RUN_HKLM", "reg query HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run 2>nul"),
    ("RUN_HKCU", "reg query HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run 2>nul"),
    ("UPTIME", 'net statistics workstation | findstr /C:"Statistics since"'),
)

# Actual VM resolution after snapshot restore.  The QEMU display adapter is
# initialised at 1920×1080 (required for the Windows accessibility API), but the
# snapshots were captured at 1280×800 so the guest reverts to that on restore.
_VM_SCREEN_WIDTH = 1280
_VM_SCREEN_HEIGHT = 800


def _reformat_axtree(raw: str) -> str:
    """Reformat linearize_accessibility_tree output into the agent-facing table.

    Input columns (from linearize_accessibility_tree):
        tag  name  text  class  description  position (top-left x&y)  size (w&h)

    Output columns:
        index  tag  name  text  x  y  w  h

    Drops class (pywinauto internal) and description (almost always empty).
    Unpacks position/size tuple strings into separate integer columns.
    """
    lines = raw.splitlines()
    if not lines:
        return raw

    out = ["index\ttag\tname\ttext\tx\ty\tw\th"]
    idx = 1
    for line in lines[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        tag, name, text = parts[0], parts[1], parts[2]
        # parts[3]=class, parts[4]=description — dropped
        pos_str, size_str = parts[5], parts[6]
        try:
            x, y = (int(v.strip()) for v in pos_str.strip("()").split(","))
            w, h = (int(v.strip()) for v in size_str.strip("()").split(","))
        except ValueError:
            continue
        out.append(f"{idx}\t{tag}\t{name}\t{text}\t{x}\t{y}\t{w}\t{h}")
        idx += 1
    return "\n".join(out)


class WAATaskExecutionInfo(TaskExecutionInfo):
    """Heavy per-task execution data for WAA tasks.

    Carries the bits that drive setup + evaluation but aren't needed for
    listing / glob-filtering: the setup-script chain, the validator config,
    related-app list, etc. Lives outside `task_metadata.json` so the
    in-tree metadata stays small (≤1 KB/task).
    """

    domain: str = "unknown"
    snapshot: str = "init_state"
    config: list[dict[str, Any]] = Field(default_factory=list)
    evaluator: dict[str, Any] = Field(default_factory=dict)
    related_apps: list[str] = Field(default_factory=list)
    test_sets: list[str] = Field(default_factory=list)


class WAATask(Task):
    """A single WAA desktop-automation task running inside a Windows 11 VM.

    WAA tasks are loaded from JSON files in evaluation_examples_windows/.
    Each task specifies: a natural-language instruction, a named QEMU snapshot
    to restore, setup scripts, and an evaluator configuration.

    Pydantic fields:
        metadata:        TaskMetadata          — id, abstract_description (light)
        execution_info:  WAATaskExecutionInfo  — heavy per-task data (config,
                                                  evaluator, snapshot, …)
        tool_config:     ToolConfig            — pass ComputerConfig(...)
        infra:           InfraConfig           — used to launch task VMs.
        validate_per_step: bool                — inherited; default False
        accept_agent_stop: bool                — inherited; default True
    """

    infra: InfraConfig | None = None
    """InfraConfig (LocalInfraConfig, AzureInfraConfig, ...)."""

    use_som: bool = False
    """If True, annotate screenshot with numbered bounding boxes (Set-of-Marks)."""

    _resource_handle: ResourceHandle | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Create the Computer tool without a VM — VM is deferred to reset()."""
        self._tool = self.tool_config.make(container=None)

    @property
    def _computer(self) -> "ComputerBase":
        return self.tool  # type: ignore[return-value]

    def _os_type(self) -> str:
        """WAA always runs Windows 11."""
        return "windows"

    def _ensure_vm(self) -> None:
        """Launch the VM via infra if not already running."""
        if self._resource_handle is not None:
            return
        if self.infra is None:
            raise RuntimeError("WAATask requires an InfraConfig — set infra= when constructing.")

        logger.info("Launching VM via %s", type(self.infra).__name__)
        self._resource_handle = self.infra.launch(WAA_WINDOWS_RESOURCE)
        # Health-gate before handing off to the agent loop. A fraction of
        # freshly-booted VMs come up with Caddy bound to port 5000 instead
        # of the WAA Flask agent (root cause TBD; investigating in parallel).
        # These are dead-on-arrival — they 502 every endpoint forever, and
        # without this gate we'd burn ~4min of in-task retries before
        # failing the episode. Fail fast so Ray re-queues on a fresh VM.
        self._health_gate_or_raise()
        # ComputerBase.attach_endpoint() takes the endpoint URL string.
        self._computer.attach_endpoint(self._resource_handle.endpoint)

    def _extract_public_ip(self) -> str:
        """Pull the VM's public IP out of the resource handle's SSH tunnel argv.
        Best-effort — returns empty string if the handle structure changes."""
        if self._resource_handle is None:
            return ""
        for proc in getattr(self._resource_handle, "_tunnels", []) or []:
            argv = getattr(proc, "args", []) or []
            if isinstance(argv, list):
                for a in argv:
                    if isinstance(a, str) and "@" in a and a.split("@", 1)[0] == _SSH_USER:
                        return a.split("@", 1)[1]
        return ""

    def _dump_dead_vm_diagnostics(self, public_ip: str, reason: str) -> None:
        """Write a per-VM diagnostic dump to ``_DEAD_VM_DIAG_DIR`` so post-eval
        analysis can find the pattern that distinguishes dead VMs from healthy
        ones. Best-effort: any failure here is logged and swallowed so the
        health gate's own teardown still runs.
        """
        if not public_ip:
            logger.warning("dead-vm diag: no public IP for %s — skipping dump", self.metadata.id)
            return
        try:
            _DEAD_VM_DIAG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("dead-vm diag: mkdir failed: %s", exc)
            return
        vm_name = getattr(self._resource_handle, "_vm_name", "unknown") if self._resource_handle else "unknown"
        out_path = _DEAD_VM_DIAG_DIR / f"{self.metadata.id}_{vm_name}.txt"
        sections: list[tuple[str, str]] = []
        for label, cmd in _DIAG_BATTERY:
            try:
                r = subprocess.run(
                    ["ssh", *_SSH_OPTS, f"{_SSH_USER}@{public_ip}", "cmd", "/c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                body = r.stdout if r.returncode == 0 else f"[exit {r.returncode}]\n{r.stderr[:400]}"
            except Exception as exc:
                body = f"[ssh exception: {type(exc).__name__}: {exc}]"
            sections.append((label, body))
        try:
            with open(out_path, "w") as f:
                f.write("# Dead-VM diagnostic dump\n")
                f.write(f"# Task: {self.metadata.id}\n")
                f.write(f"# VM: {vm_name}\n")
                f.write(f"# Public IP: {public_ip}\n")
                f.write(f"# Reason: {reason}\n")
                f.write(f"# Captured: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
                for label, body in sections:
                    f.write("=" * 72 + f"\n== {label}\n" + "=" * 72 + "\n")
                    f.write(body or "[empty]")
                    f.write("\n\n")
            logger.warning("dead-vm diag: wrote %s", out_path)
        except Exception as exc:
            logger.warning("dead-vm diag: write failed: %s", exc)

    def _health_gate_or_raise(self) -> None:
        """Probe ``/probe`` until 200, or fail-fast on the dead-Flask signature.

        Two distinct failure modes show up in the wild and this gate handles
        them differently:

          * **Dead Flask** — Caddy is up on :5000 returning ``502`` with
            ``Server: Caddy`` (Flask never bound the port). Stays that way
            indefinitely; no point waiting. **Bail after 3 consecutive
            502+Caddy responses (~6s)** and let Ray retry on a fresh VM.

          * **Slow-warming Flask** — port 5000 isn't bound yet so the SSH
            tunnel can't reach upstream and we get ``ConnectionRefused``.
            This is benign cold-boot variance and resolves on its own once
            Flask binds. **Wait the full 5-minute budget** before giving up.

        Net behaviour: dead VMs get killed in 6 s (vs 4 min of in-task
        retries we'd otherwise burn), slow VMs aren't aborted prematurely.
        """
        if self._resource_handle is None or not self._resource_handle.endpoint:
            return  # not our problem — let downstream call sites surface this
        endpoint = self._resource_handle.endpoint
        deadline = time.time() + 300  # 5 min budget for slow Flask warmup
        caddy_502_strike_limit = 3  # 502+Caddy this many times in a row → dead
        last_status: int | str = "no-response"
        last_server: str = ""
        last_exc: Exception | None = None
        attempts = 0
        consecutive_caddy_502 = 0
        while time.time() < deadline:
            attempts += 1
            try:
                resp = requests.get(endpoint + "/probe", timeout=5)
                last_status = resp.status_code
                last_server = resp.headers.get("Server", "")
                if resp.status_code == 200:
                    logger.info(
                        "Health gate passed for %s after %d attempt(s) (server=%s)",
                        self.metadata.id,
                        attempts,
                        last_server,
                    )
                    return
                # Dead-Flask signature: 502 + Server: Caddy. Streak-count so we
                # don't overreact to a single transient 502 from a recovering
                # upstream — but bail quickly on a sustained pattern.
                if resp.status_code == 502 and "caddy" in last_server.lower():
                    consecutive_caddy_502 += 1
                    if consecutive_caddy_502 >= caddy_502_strike_limit:
                        logger.warning(
                            "Health gate FAST-FAIL for %s: %d consecutive 502+Caddy responses "
                            "→ Flask never bound :5000, no point waiting %ds",
                            self.metadata.id,
                            consecutive_caddy_502,
                            int(deadline - time.time()),
                        )
                        break
                else:
                    consecutive_caddy_502 = 0
            except requests.RequestException as exc:
                last_exc = exc
                last_status = "exc"
                consecutive_caddy_502 = 0  # connection error resets the streak
            time.sleep(2)
        # Loop exited without 200 — either deadline or fast-fail break
        msg = (
            f"Health gate FAILED for {self.metadata.id} after {attempts} attempts: "
            f"last_status={last_status} server={last_server!r} exc={last_exc}"
        )
        logger.warning("%s — capturing diag, then closing handle so episode retries on fresh VM", msg)
        # Capture VM state BEFORE closing the handle — handle.close() deletes
        # the Azure VM. Diagnostic dump is best-effort and never raises.
        public_ip = self._extract_public_ip()
        self._dump_dead_vm_diagnostics(
            public_ip,
            reason=f"last_status={last_status} server={last_server!r}",
        )
        try:
            self._resource_handle.close()
        except Exception as close_exc:
            logger.warning("close() during health-gate teardown failed: %s", close_exc)
        finally:
            self._resource_handle = None
        raise RuntimeError(msg)

    def _get_vm_ports(self) -> tuple[int, int, int]:
        """Return (chromium_port, vlc_port, server_port) from the current handle.

        Cloud infra tunnels each VM port to a unique host freeport (recorded in
        ``handle.endpoints["vm_port_{N}"]``), so parallel workers don't collide
        on a fixed local port. Local infra exposes the VM ports directly so
        the hard-coded defaults still apply.
        """
        server_port = 5000
        chromium_port = 9222
        vlc_port = 8080
        if self._resource_handle is not None:
            if self._resource_handle.endpoint:
                server_port = urlparse(self._resource_handle.endpoint).port or 5000
            endpoints = getattr(self._resource_handle, "endpoints", {}) or {}
            chromium_url = endpoints.get("vm_port_9222")
            if chromium_url:
                chromium_port = urlparse(chromium_url).port or chromium_port
            vlc_url = endpoints.get("vm_port_8080")
            if vlc_url:
                vlc_port = urlparse(vlc_url).port or vlc_port
        return chromium_port, vlc_port, server_port

    def _setup_task(self, task_data: dict) -> Observation:
        """Run setup scripts, wait, return initial observation."""
        logger.info(
            "Setting up WAA task: %s. Instruction: %s",
            task_data.get("id", "unknown"),
            task_data.get("instruction", ""),
        )

        setup_steps = task_data.get("config") or []
        chromium_port, vlc_port, _ = self._get_vm_ports()
        task_cache_dir = str(Path(self._computer.config.cache_dir) / task_data.get("id", "task"))
        Path(task_cache_dir).mkdir(parents=True, exist_ok=True)
        setup_ctrl = SetupController(
            guest=self._computer._guest,
            chromium_port=chromium_port,
            vlc_port=vlc_port,
            cache_dir=task_cache_dir,
            screen_width=_VM_SCREEN_WIDTH,
            screen_height=_VM_SCREEN_HEIGHT,
        )
        # Always wait for Flask agent connectivity before proceeding — the guest
        # agent may need time to start on first boot of a cloud VM.
        reachable = setup_ctrl.setup(setup_steps)
        if not reachable:
            logger.warning("WAA VM guest agent unreachable — observation may fail")

        did_something = self._resource_handle is not None or bool(setup_steps)
        if did_something:
            logger.info("Waiting %ds for VM to stabilise...", _POST_SNAPSHOT_SLEEP)
            time.sleep(_POST_SNAPSHOT_SLEEP)

        return self._computer.get_observation()

    def _evaluate_task(self) -> tuple[float, dict[str, Any]]:
        """Run the WAA evaluator and return ``(reward, info)``.

        ``reward`` ∈ [0.0, 1.0]. ``info`` carries diagnostic context (see
        ``Evaluator.evaluate``) so harness callers can distinguish a real
        agent failure from an evaluator-side bug.
        """
        if self._computer._guest is None:
            logger.error("_evaluate_task() called with no VM attached")
            return 0.0, {"evaluation_error": {"phase": "setup", "type": "NoVMAttached", "message": "no guest agent"}}

        chromium_port, vlc_port, server_port = self._get_vm_ports()
        cache_dir_base = Path(self._computer.config.cache_dir)

        evaluator = Evaluator(
            guest=self._computer._guest,
            cache_dir_base=cache_dir_base,
            chromium_port=chromium_port,
            vlc_port=vlc_port,
            server_port=server_port,
        )
        exec_info = self._waa_execution_info()
        eval_config = {
            "id": self.metadata.id,
            "evaluator": exec_info.evaluator,
        }
        try:
            reward, info = evaluator.evaluate(eval_config, self._computer._action_history)
            logger.info("WAA task evaluation result: %f", reward)
            return reward, info
        except Exception as exc:
            logger.exception("Evaluation failed for %s", self.metadata.id)
            return 0.0, {
                "evaluation_error": {"phase": "evaluator_top_level", "type": type(exc).__name__, "message": str(exc)}
            }

    def _waa_execution_info(self) -> "WAATaskExecutionInfo":
        """Return self.execution_info coerced to WAATaskExecutionInfo.

        WAATaskConfig.make() always populates this, but defensively handle the
        case where a task is constructed directly without it (returns the
        default WAATaskExecutionInfo with empty fields, matching the old
        ``extra_info.get(..., default)`` semantics).
        """
        if isinstance(self.execution_info, WAATaskExecutionInfo):
            return self.execution_info
        return WAATaskExecutionInfo()

    def reset(self) -> tuple[Observation, dict]:
        """Run setup scripts and return the initial obs.

        Steps:
          1. Launch VM if not yet running (via infra)
          2. Build task_data dict from execution_info
          3. Run setup scripts, wait
          4. Post-process the observation (SoM or linearize axtree)
          5. Prepend task instruction as text observation
          6. Return (obs, info)
        """
        self._ensure_vm()
        exec_info = self._waa_execution_info()

        task_data = {
            "id": self.metadata.id,
            "instruction": self.metadata.abstract_description,
            "config": exec_info.config,
            "evaluator": exec_info.evaluator,
            "snapshot": exec_info.snapshot,
            "related_apps": exec_info.related_apps,
        }

        logger.info("Resetting WAATask %s (domain=%s)", self.metadata.id, exec_info.domain)

        obs = self._setup_task(task_data)
        obs = self.obs_postprocess(obs)

        goal_obs = Observation.from_text(f"Task: {self.metadata.abstract_description}")
        obs = goal_obs + obs

        info = {
            "task_id": self.metadata.id,
            "task_domain": exec_info.domain,
            "task_snapshot": exec_info.snapshot,
            "task_related_apps": exec_info.related_apps,
        }
        return obs, info

    def evaluate(self, obs: Observation) -> tuple[float, dict]:
        """Call the WAA task evaluator and return (reward, info).

        reward ∈ [0.0, 1.0]:  1.0 = task fully completed.
        """
        evaluator_cfg = self._waa_execution_info().evaluator

        if not evaluator_cfg:
            logger.warning("Task %s: no evaluator configured, returning 0.0", self.metadata.id)
            return 0.0, {"error": "no_evaluator"}

        eval_func = evaluator_cfg.get("func", "unknown")
        logger.debug("Evaluating WAA task %s with evaluator: %s", self.metadata.id, eval_func)

        reward, eval_info = self._evaluate_task()
        logger.info("WAA task %s evaluation: reward=%f, evaluator=%s", self.metadata.id, reward, eval_func)
        return reward, {
            "evaluator": eval_func,
            "expected": evaluator_cfg.get("expected", {}),
            **eval_info,
        }

    def finished(self, obs: Observation) -> bool:
        """Return True if the task has reached a terminal state."""
        return self._computer._is_done

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Post-process raw observation before returning to the agent."""
        if self.use_som:
            return self._postprocess_som(obs)
        return self._postprocess_linearize(obs)

    def _postprocess_linearize(self, obs: Observation) -> Observation:
        """Replace raw axtree XML with a clean indexed table for the agent.

        Converts the raw XML to a tab-separated table with columns:
            index  tag  name  text  x  y  w  h

        Drops the class and description columns (noise) and unpacks the
        position/size tuple strings into separate integer columns so the
        agent can compute click centres with: cx = x + w//2, cy = y + h//2.
        """
        platform = self._os_type()
        new_contents = []
        for content in obs.contents:
            if content.name == "accessibility_tree":
                try:
                    raw = linearize_accessibility_tree(content.data, platform=platform)
                    axtree_txt = _reformat_axtree(raw)
                    new_contents.append(content.model_copy(update={"data": axtree_txt, "name": "axtree_txt"}))
                except Exception as exc:
                    logger.warning("Failed to linearize accessibility tree: %s", exc)
                    new_contents.append(content)
            else:
                new_contents.append(content)
        return obs.model_copy(update={"contents": new_contents})

    def _postprocess_som(self, obs: Observation) -> Observation:
        """Annotate screenshot with numbered bounding boxes (Set-of-Marks).

        Falls back to _postprocess_linearize if screenshot or axtree are missing.
        """
        platform = self._os_type()

        screenshot_content = None
        axtree_content = None
        for content in obs.contents:
            if content.name == "screenshot" and isinstance(content.data, Image.Image):
                screenshot_content = content
            elif content.name == "accessibility_tree":
                axtree_content = content

        if screenshot_content is None or axtree_content is None:
            logger.warning("SoM requires both screenshot and accessibility_tree; falling back to linearize.")
            return self._postprocess_linearize(obs)

        try:
            buf = io.BytesIO()
            screenshot_content.data.save(buf, format="PNG")
            screenshot_bytes = buf.getvalue()

            marks, _, tagged_screenshot_bytes, element_list = tag_screenshot(
                screenshot_bytes, axtree_content.data, platform=platform
            )
            self._computer.update_marks(marks)

            tagged_img = Image.open(io.BytesIO(tagged_screenshot_bytes))
            tagged_img.load()

            new_contents = []
            for content in obs.contents:
                if content.name == "screenshot" and isinstance(content.data, Image.Image):
                    new_contents.append(content.model_copy(update={"data": tagged_img}))
                elif content.name == "accessibility_tree":
                    new_contents.append(content.model_copy(update={"data": element_list, "name": "som_elements"}))
                else:
                    new_contents.append(content)
            return obs.model_copy(update={"contents": new_contents})

        except Exception as exc:
            logger.warning("Failed to apply SoM annotation: %s", exc)
            return self._postprocess_linearize(obs)

    def close(self) -> None:
        """Clean up task resources: stop tool and release infra handle."""
        logger.info("Closing WAATask: %s", self.metadata.id)
        super().close()  # calls self.tool.close()
        if self._resource_handle is not None:
            try:
                self._resource_handle.close()
            except Exception as exc:
                logger.warning("Failed to close Azure resource handle: %s", exc)
            self._resource_handle = None
