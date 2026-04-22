"""CUBE Container implementation for DRBench.

DrBenchContainer wraps DrBenchEnterpriseSearchSpace as a CUBE Container.
DrBenchContainerBackend launches per-task baked Docker images (with base image fallback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import docker
from cube.container import (
    Container,
    ContainerBackend,
    ContainerConfig,
    ContainerStatus,
    ExecResult,
)
from drbench import config as drbench_config
from drbench.drbench_enterprise_space import DrBenchEnterpriseSearchSpace
from drbench.task_loader import get_task_from_id

logger = logging.getLogger(__name__)

_registry = drbench_config.DRBENCH_DOCKER_REGISTRY
_image = drbench_config.DRBENCH_DOCKER_IMAGE
DRBENCH_IMAGE_PREFIX = f"{_registry}/{_image}" if _registry else _image
DRBENCH_BASE_IMAGE = f"{DRBENCH_IMAGE_PREFIX}:{drbench_config.DRBENCH_DOCKER_TAG}"


class DrBenchContainer(Container):
    """Thin wrapper around a running DrBenchEnterpriseSearchSpace."""

    def __init__(self, space: DrBenchEnterpriseSearchSpace):
        self._space = space

    @property
    def id(self) -> str:
        return self._space.container_id or ""

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        try:
            output = self._space.container_manager.run_command(command)
            return ExecResult(stdout=output or "", exit_code=0)
        except Exception as e:
            return ExecResult(stderr=str(e), exit_code=1)

    def forward_port(self, container_port: int) -> int:
        return self._space.container_manager.get_host_port(container_port)

    def get_url(self, container_port: int) -> str:
        host_port = self.forward_port(container_port)
        return f"http://localhost:{host_port}"

    def stop(self, timeout: int = 10) -> None:
        try:
            self._space.stop()
            self._space.delete()
        except Exception as e:
            logger.warning(f"Error stopping container {self.id}: {e}")

    def get_status(self) -> ContainerStatus:
        try:
            status = self._space.container_manager.get_status()
            running = status.get("status") == "running"
            healthy = self._space.container_manager.is_healthy()
            return ContainerStatus(running=running, healthy=healthy, backend_info=status)
        except Exception:
            return ContainerStatus(running=False, healthy=False)


class DrBenchContainerBackend(ContainerBackend):
    """
    Launches pre-baked per-task Docker images.

    Tries image tagged as `drbench-services:<task_id>` first (fast, ~8s startup).
    Falls back to base image + add_task() if per-task image not found (~45s startup).

    The image tag is expected to come from TaskMetadata.container_config.image,
    which is set to `drbench-services:DR0042` by generate_task_metadata.py.
    """

    def launch(self, config: ContainerConfig) -> DrBenchContainer:
        image_tag = config.image  # e.g. "drbench-services:DR0042"

        # Prepend registry prefix if configured (e.g. "ghcr.io/servicenow")
        _registry = drbench_config.DRBENCH_DOCKER_REGISTRY
        full_image_tag = f"{_registry}/{image_tag}" if _registry else image_tag

        # Determine whether the per-baked image exists locally
        needs_task_load = False
        client = docker.from_env()
        try:
            client.images.get(full_image_tag)
            logger.info(f"Using pre-baked image: {full_image_tag}")
        except docker.errors.ImageNotFound:
            # Try pulling from registry if one is configured
            if _registry:
                try:
                    logger.info(f"Pulling {full_image_tag} from registry...")
                    client.images.pull(full_image_tag)
                    logger.info(f"Pulled pre-baked image: {full_image_tag}")
                except Exception as pull_err:
                    logger.warning(f"Could not pull {full_image_tag}: {pull_err}")
                    needs_task_load = True
            else:
                needs_task_load = True

            if needs_task_load:
                logger.warning(
                    f"Pre-baked image {full_image_tag!r} not found. "
                    "Falling back to base image + add_task() (slower startup)."
                )

        # Extract task_id from the image tag (e.g. "drbench-services:DR0042" → "DR0042")
        task_id = image_tag.split(":")[-1] if ":" in image_tag else None
        if task_id is None:
            raise ValueError(f"Cannot derive task_id from image tag: {image_tag!r}")
        drbench_task = get_task_from_id(task_id)
        task_path = Path(drbench_task.get_path())

        if needs_task_load:
            space = DrBenchEnterpriseSearchSpace(
                task=task_path,
                auto_ports=True,
            )
        else:
            space = DrBenchEnterpriseSearchSpace(
                image=full_image_tag,
                task=task_path,
                auto_ports=True,
                task_data_preloaded=True,
            )

        space.start()
        return DrBenchContainer(space)

    def health_check(self, container: DrBenchContainer) -> bool:
        try:
            return container._space.container_manager.is_healthy()
        except Exception:
            return False
