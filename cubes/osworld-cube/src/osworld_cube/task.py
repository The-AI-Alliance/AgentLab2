"""
OSWorldTask — CUBE task for a single OSWorld desktop-automation episode.

    config = OSWorldBenchmarkConfig(tool_config=ComputerConfig(...))
    benchmark = config.make(infra=AWSInfraConfig(...))
    task = next(config.get_task_configs()).make(
        runtime_context=benchmark._runtime_context,
    )
    obs, info = task.reset()
    ...
    task.close()
"""

from __future__ import annotations

import io
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from cube.infra_utils import open_tunnel
from PIL import Image
from pydantic import PrivateAttr

from cube.benchmark import RuntimeContext  # noqa: F401 — triggers OSWorldTask.model_rebuild()
from cube.core import Observation
from cube.task import Task, TaskExecutionInfo, TaskMetadata
from cube.resource import InfraConfig, ResourceHandle, VMResourceConfig

from cube_computer_tool.axtree import linearize_accessibility_tree, tag_screenshot

from osworld_cube.vm_backend.evaluator import Evaluator
from osworld_cube.vm_backend.setup_controller import SetupController

if TYPE_CHECKING:
    from cube_computer_tool.computer import ComputerBase

logger = logging.getLogger(__name__)

OSWORLD_UBUNTU_RESOURCE = VMResourceConfig(
    name="osworld-ubuntu-vm",
    source_url="https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip",
    scope="task",
    default_ttl_seconds=60 * 60 * 24,
)

# Port constants baked into the OSWorld Ubuntu image
_CHROMIUM_PORT = 9222
_VLC_PORT = 8080
_SERVER_PORT = 5000


class OSWorldTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for OSWorld tasks.

    Public fields shipped in task_metadata.json (available at import time).
    Heavy execution data (config, evaluator) lives on ``OSWorldExecutionInfo``
    and is loaded lazily by ``OSWorldTaskConfig.make()``.
    """

    domain: str  # Desktop domain, e.g. 'chrome', 'os', 'libreoffice_calc'.
    test_sets: list[str]  # OSWorld test sets this task belongs to, e.g. ['test_all', 'test_small'].
    instruction: str  # Full agent-facing task instruction.
    snapshot: str  # VM snapshot name to restore before the task starts.
    os_type: str  # Guest OS type used for accessibility-tree linearisation ('ubuntu' or 'windows').
    related_apps: list[str]  # Applications involved in the task, e.g. ['chrome', 'libreoffice_calc'].


class OSWorldExecutionInfo(TaskExecutionInfo):
    """Heavy per-task execution data for OSWorld tasks — populated on the worker.

    Loaded by ``OSWorldTaskConfig.make()`` from the per-task execution cache
    written by ``OSWorldBenchmarkConfig.install()``.
    """

    config: list[dict] = []
    """Setup scripts to run before the task starts (paths fixed to absolute at install time)."""

    evaluator: dict = {}
    """Evaluator function + expected results."""


class OSWorldTask(Task[OSWorldTaskMetadata]):
    """
    A single OSWorld desktop-automation task running inside a VM.

    OSWorld tasks are loaded from JSON files in the OSWorld repository.
    Each task specifies: a natural-language instruction, a VM snapshot
    to restore, setup scripts, and an evaluator configuration.

    Reference: https://github.com/xlang-ai/OSWorld

    Heavy per-task execution data (setup config, evaluator) is on
    ``self.execution_info`` (an ``OSWorldExecutionInfo``); read it via
    ``self.execution_info.config`` / ``self.execution_info.evaluator``.

    Infra flows in via ``self.runtime_context["infra"]`` — published by
    ``OSWorldBenchmark._setup()``. Each task gets a fresh VM via
    ``infra.launch(OSWORLD_UBUNTU_RESOURCE)``.
    """

    use_som: bool = False
    """If True, annotate screenshot with numbered bounding boxes (Set-of-Marks)
    and replace axtree with an indexed element table before returning obs."""

    _handle: ResourceHandle | None = PrivateAttr(default=None)
    _extra_tunnels: list[subprocess.Popen] = PrivateAttr(default_factory=list)
    _chromium_port: int = PrivateAttr(default=_CHROMIUM_PORT)
    _vlc_port: int = PrivateAttr(default=_VLC_PORT)

    def model_post_init(self, __context: Any) -> None:
        """Create the Computer tool without a VM — VM is deferred to reset()."""
        self._tool = self.tool_config.make(container=None)

    @property
    def _computer(self) -> "ComputerBase":
        """Return self.tool cast to ComputerBase for type-checker satisfaction."""
        return self.tool  # type: ignore[return-value]

    @property
    def _exec(self) -> OSWorldExecutionInfo:
        """Typed view on execution_info — fails fast if it was not populated."""
        if self.execution_info is None:
            raise RuntimeError(
                f"OSWorldTask {self.metadata.id!r}: execution_info is not populated. "
                f"Construct via OSWorldTaskConfig.make() (which loads it from the per-task cache)."
            )
        if not isinstance(self.execution_info, OSWorldExecutionInfo):
            raise RuntimeError(
                f"OSWorldTask {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected OSWorldExecutionInfo."
            )
        return self.execution_info

    def _ensure_vm(self) -> None:
        """Launch the VM if not already running via infra.launch()."""
        if self._handle is not None:
            return
        infra = (self.runtime_context or {}).get("infra")
        if not isinstance(infra, InfraConfig):
            raise RuntimeError(
                "OSWorldTask requires an InfraConfig in runtime_context — "
                "construct via OSWorldBenchmarkConfig.make(infra=<InfraConfig>) so the runtime "
                "publishes infra into runtime_context['infra']."
            )
        logger.info("Launching VM via %s", type(infra).__name__)
        self._handle = infra.launch(OSWORLD_UBUNTU_RESOURCE)
        logger.info("VM ready (run_id=%s)", self._handle.run_id[:8])
        self._computer.attach_endpoint(self._handle.endpoint)
        self._open_auxiliary_tunnels()

    def _open_auxiliary_tunnels(self) -> None:
        """Open host tunnels for VM services that OSWorld accesses directly.

        Some cloud infra handles expose only the guest-agent endpoint. OSWorld setup
        and evaluation also connect to Chromium CDP and VLC from the host process,
        so reuse the handle's SSH tunnel details when available.
        """
        if self._handle is None or self._extra_tunnels:
            return

        # AzureResourceHandle exposes _tunnels (plural list of subprocess.Popen)
        # since the cube-standard SSH-tunnel refactor — the old single _tunnel
        # attribute is gone. Walk the list and pull the first valid SSH argv.
        tunnels = getattr(self._handle, "_tunnels", None) or []
        if not tunnels:
            # Fall back to legacy single-tunnel attribute for any handle that
            # still uses it.
            legacy = getattr(self._handle, "_tunnel", None)
            if legacy is not None:
                tunnels = [legacy]
        ssh_privkey = ssh_user = vm_ip = None
        for tun in tunnels:
            tunnel_args = getattr(tun, "args", None)
            if not isinstance(tunnel_args, (list, tuple)):
                continue
            try:
                identity_index = tunnel_args.index("-i") + 1
                ssh_privkey = str(tunnel_args[identity_index])
                ssh_target = str(tunnel_args[-1])
                ssh_user, vm_ip = ssh_target.split("@", 1)
                break
            except (ValueError, IndexError):
                continue
        if not (ssh_privkey and ssh_user and vm_ip):
            logger.warning(
                "Could not parse SSH tunnel arguments from handle; using default VM ports (Chrome CDP will refuse)"
            )
            return

        # cube-standard's open_tunnel now picks its own local port atomically
        # (kernel-allocated, race-free) and returns (proc, local_port). The
        # old call here was passing free_port() as remote_port and CHROMIUM
        # as max_attempts due to the signature change — this caused the
        # tunnel to point at the wrong remote port and Chrome CDP to refuse
        # connections. See cube-standard PR #99.
        chromium_tunnel, self._chromium_port = open_tunnel(vm_ip, ssh_user, ssh_privkey, _CHROMIUM_PORT)
        self._extra_tunnels.append(chromium_tunnel)
        vlc_tunnel, self._vlc_port = open_tunnel(vm_ip, ssh_user, ssh_privkey, _VLC_PORT)
        self._extra_tunnels.append(vlc_tunnel)
        logger.info(
            "Opened auxiliary VM tunnels: Chromium localhost:%d -> 9222, VLC localhost:%d -> 8080",
            self._chromium_port,
            self._vlc_port,
        )

    def _get_vm_ports(self) -> tuple[int, int, int]:
        """Return (chromium_port, vlc_port, server_port).

        Ports are baked into the OSWorld Ubuntu image.
        server_port is extracted from the handle endpoint since the SSH tunnel
        maps a free local port → VM:5000.
        """
        if self._handle is not None and self._handle.endpoint:
            server_port = urlparse(self._handle.endpoint).port or _SERVER_PORT
            return self._chromium_port, self._vlc_port, server_port
        return self._chromium_port, self._vlc_port, _SERVER_PORT

    def _setup_task(self, task_data: dict) -> Observation:
        """Run setup scripts, wait for VM to stabilise, return initial observation.

        Called from reset(). Uses SetupController for OSWorld-specific task
        configuration scripts.
        """
        logger.info(
            "Setting up task: %s. Instruction: %s", task_data.get("id", "unknown"), task_data.get("instruction", "")
        )
        # VM was launched fresh from the provisioned image — no snapshot restore needed.
        setup_steps = task_data.get("config") or []
        if setup_steps:
            chromium_port, vlc_port, _ = self._get_vm_ports()
            task_cache_dir = str(Path(self._computer.config.cache_dir) / task_data.get("id", "task"))
            Path(task_cache_dir).mkdir(parents=True, exist_ok=True)
            setup_ctrl = SetupController(
                guest=self._computer._guest,
                chromium_port=chromium_port,
                vlc_port=vlc_port,
                guest_chromium_port=_CHROMIUM_PORT,
                cache_dir=task_cache_dir,
                screen_width=1920,
                screen_height=1080,
            )
            setup_ctrl.setup(setup_steps)

        if self._handle is not None or setup_steps:
            logger.info("Waiting 60s for VM to stabilise...")
            time.sleep(60)
        return self._computer.get_observation()

    def _evaluate_task(self) -> float:
        """Run the OSWorld evaluator and return reward ∈ [0.0, 1.0]."""
        if self._computer._guest is None:
            logger.error("_evaluate_task() called with no VM attached")
            return 0.0

        chromium_port, vlc_port, server_port = self._get_vm_ports()
        cache_dir_base = Path(self._computer.config.cache_dir)

        evaluator = Evaluator(
            guest=self._computer._guest,
            cache_dir_base=cache_dir_base,
            chromium_port=chromium_port,
            vlc_port=vlc_port,
            server_port=server_port,
        )
        eval_config = {
            "id": self.metadata.id,
            "evaluator": self._exec.evaluator,
        }
        try:
            reward = evaluator.evaluate(eval_config, self._computer._action_history)
            logger.info("Task evaluation result: %f", reward)
            return reward
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            return 0.0

    def reset(self) -> tuple[Observation, dict]:
        """
        Launch the VM (if needed), run setup scripts, and return the initial obs.

        Steps:
          1. Launch VM if not yet running (via infra from runtime_context)
          2. Build task_data dict from metadata + execution_info
          3. Run setup scripts, wait for VM to stabilise
          4. Post-process the observation (SoM or linearize)
          5. Prepend task instruction as text observation
          6. Return (obs, info)
        """
        self._ensure_vm()
        self.tool.reset()
        task_data = {
            "id": self.metadata.id,
            "instruction": self.metadata.instruction,
            "config": self._exec.config,
            "evaluator": self._exec.evaluator,
            "snapshot": self.metadata.snapshot,
            "related_apps": self.metadata.related_apps,
        }

        logger.info("Resetting OSWorldTask %s (domain=%s)", self.metadata.id, self.metadata.domain)

        obs = self._setup_task(task_data)
        obs = self.obs_postprocess(obs)

        goal_obs = Observation.from_text(f"Task: {self.metadata.instruction}")
        obs = goal_obs + obs

        info = {
            "task_id": self.metadata.id,
            "task_domain": self.metadata.domain,
            "task_snapshot": self.metadata.snapshot,
            "task_related_apps": self.metadata.related_apps,
        }
        return obs, info

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        """
        Call the task evaluator and return (reward, info).

        reward ∈ [0.0, 1.0]:  1.0 = task fully completed.
        Partial credit is preserved (not rounded to binary).
        """
        evaluator_cfg = self._exec.evaluator

        if not evaluator_cfg:
            logger.warning("Task %s: no evaluator configured, returning 0.0", self.metadata.id)
            return 0.0, {"error": "no_evaluator"}

        eval_func = evaluator_cfg.get("func", "unknown")
        logger.debug("Evaluating task %s with evaluator: %s", self.metadata.id, eval_func)

        reward = self._evaluate_task()
        logger.info("Task %s evaluation: reward=%f, evaluator=%s", self.metadata.id, reward, eval_func)
        return reward, {
            "evaluator": eval_func,
            "expected": evaluator_cfg.get("expected", {}),
        }

    def finished(self, obs: Observation | None = None) -> bool:
        """Return True if the task has reached a terminal state (done() or fail() called)."""
        return self._computer._is_done

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Post-process raw observation before returning to the agent."""
        if self.use_som:
            return self._postprocess_som(obs)
        return self._postprocess_linearize(obs)

    def _postprocess_linearize(self, obs: Observation) -> Observation:
        """Replace raw axtree XML with a linearized tab-separated table."""
        platform = self.metadata.os_type.lower()
        new_contents = []
        for content in obs.contents:
            if content.name == "accessibility_tree":
                try:
                    axtree_txt = linearize_accessibility_tree(content.data, platform=platform)
                    new_contents.append(content.model_copy(update={"data": axtree_txt, "name": "axtree_txt"}))
                except Exception as e:
                    logger.warning("Failed to linearize accessibility tree: %s", e)
                    new_contents.append(content)
            else:
                new_contents.append(content)
        return obs.model_copy(update={"contents": new_contents})

    def _postprocess_som(self, obs: Observation) -> Observation:
        """Annotate screenshot with numbered bounding boxes (Set-of-Marks).

        Falls back to _postprocess_linearize if screenshot or axtree are missing,
        or if the annotation fails.
        """
        platform = self.metadata.os_type.lower()
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
            tagged_img.load()  # force load before BytesIO goes out of scope

            new_contents = []
            for content in obs.contents:
                if content.name == "screenshot" and isinstance(content.data, Image.Image):
                    new_contents.append(content.model_copy(update={"data": tagged_img}))
                elif content.name == "accessibility_tree":
                    new_contents.append(content.model_copy(update={"data": element_list, "name": "som_elements"}))
                else:
                    new_contents.append(content)
            return obs.model_copy(update={"contents": new_contents})

        except Exception as e:
            logger.warning("Failed to apply SoM annotation: %s", e)
            return self._postprocess_linearize(obs)

    def close(self) -> None:
        """Clean up task resources: stop tool, then close VM handle."""
        logger.info("Closing OSWorldTask: %s", self.metadata.id)
        super().close()  # calls self.tool.close()
        for tunnel in self._extra_tunnels:
            try:
                tunnel.terminate()
            except Exception:
                logger.warning("Failed to terminate auxiliary tunnel", exc_info=True)
        self._extra_tunnels = []
        self._chromium_port = _CHROMIUM_PORT
        self._vlc_port = _VLC_PORT
        if self._handle is not None:
            self._handle.close()
            self._handle = None
