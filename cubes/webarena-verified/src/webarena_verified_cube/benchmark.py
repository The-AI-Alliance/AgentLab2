import logging
import urllib.error
import urllib.request
from collections.abc import Generator
from typing import ClassVar, cast

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.infra_local import LocalInfraConfig
from cube.resource import DockerServiceConfig, InfraConfig, ResourceHandle
from cube.task import TaskConfig
from cube.tool import ToolboxConfig, ToolConfig
from pydantic import SerializeAsAny
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite

from webarena_verified_cube.task import WebArenaVerifiedTaskConfig, WebArenaVerifiedTaskMetadata
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig

logger = logging.getLogger(__name__)


class WebArenaVerifiedBenchmark(Benchmark["WebArenaVerifiedBenchmarkConfig"]):
    """Runtime pair — owns the launched DockerServiceConfig handle and resolves
    ``wav_config.environments`` from the live endpoints when running in
    automatic mode.
    """

    def __init__(
        self,
        config: "WebArenaVerifiedBenchmarkConfig",
        infra: InfraConfig | None = None,
    ) -> None:
        super().__init__(config, infra=infra)
        self._handle: ResourceHandle | None = None

    def _setup(self) -> None:
        cfg = self.config
        has_environments = cfg.wav_config.environments is not None
        has_infra = self._infra is not None
        has_docker_resources = any(isinstance(r, DockerServiceConfig) for r in cfg.resources)

        if has_environments and has_infra:
            raise ValueError(
                "Ambiguous setup: provide either wav_config.environments (manual mode) "
                "or infra + resources (automatic mode) — not both."
            )
        if not has_environments and not has_infra:
            raise ValueError(
                "No setup configured. Provide either:\n"
                "  • wav_config=WebArenaVerifiedConfig(environments={...})  — manual mode\n"
                "  • config.make(infra=<InfraConfig>) with resources=[<DockerServiceConfig>]  — automatic mode"
            )
        if has_infra and not has_docker_resources:
            raise ValueError(
                "infra was passed to make() but no DockerServiceConfig found in resources=. "
                "Pass resources=[WEBARENA_SHOPPING_ADMIN] (or another site config) at construction time."
            )

        if has_infra:
            self._handle = self._launch_and_configure()
        else:
            # Manual mode: verify all configured URLs are reachable.
            for site, env_config in cfg.wav_config.environments.items():  # type: ignore[union-attr]
                url = env_config.active_url
                if url is None:
                    continue
                try:
                    urllib.request.urlopen(url, timeout=5)
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    raise RuntimeError(
                        f"Cannot reach {site} at {url}. "
                        f"Start the Docker container with: `webarena-verified env start --site {site.value}`"
                    ) from e

        self._runtime_context["wav_config"] = cfg.wav_config

    def _launch_and_configure(self) -> ResourceHandle:
        """Launch the resource and populate ``self.config.wav_config.environments``.

        Provisioning was already done by ``WebArenaVerifiedBenchmarkConfig.make(infra)``
        per the cube-standard contract — this method only launches.
        """
        assert self._infra is not None
        cfg = self.config
        docker_resources = [r for r in cfg.resources if isinstance(r, DockerServiceConfig)]
        if len(docker_resources) > 1:
            raise ValueError(
                f"infra= only supports a single DockerServiceConfig, but {len(docker_resources)} were found: "
                f"{[r.name for r in docker_resources]}. "
                "Pass exactly one DockerServiceConfig in resources=."
            )
        # Validator above guarantees at least one DockerServiceConfig is present.
        (resource,) = docker_resources

        logger.info("Launching %r …", resource.name)
        handle = self._infra.launch(resource)
        try:
            self._configure_wav_from_handle(handle, resource)
        except Exception:
            handle.close()
            raise
        return handle

    def _configure_wav_from_handle(self, handle: ResourceHandle, resource: DockerServiceConfig) -> None:
        """Translate handle.endpoints → wav_config.environments via resource.endpoint_to_site."""
        if not resource.endpoint_to_site:
            raise ValueError(
                f"DockerServiceConfig {resource.name!r} has no endpoint_to_site mapping. "
                "Add endpoint_to_site={'service_key': 'webarena_site_value', ...} to the resource."
            )
        environments: dict[WebArenaSite, EnvironmentConfig] = {}
        for service_name, url in handle.endpoints.items():
            site_value = resource.endpoint_to_site.get(service_name)
            if site_value is None:
                continue
            try:
                site = WebArenaSite(site_value)
            except ValueError:
                raise ValueError(
                    f"endpoint_to_site maps {service_name!r} → {site_value!r} which is not a valid WebArenaSite. "
                    f"Valid values: {[s.value for s in WebArenaSite]}"
                )
            if site not in environments:
                environments[site] = EnvironmentConfig(urls=[url])

        if not environments:
            raise ValueError(
                f"handle.endpoints {list(handle.endpoints)!r} produced no WebArenaSite entries. "
                "Check resource.endpoint_to_site."
            )
        # Mutate the config so get_task_configs() (called after make()) sees resolved URLs.
        self.config.wav_config = self.config.wav_config.model_copy(update={"environments": environments})
        logger.info(
            "wav_config.environments: %s",
            {s.value: e.active_url for s, e in environments.items()},
        )

    def close(self) -> None:
        if self._handle is not None:
            logger.info("Closing infra handle for run_id=%s", self._handle.run_id[:8])
            self._handle.close()
            self._handle = None


class WebArenaVerifiedBenchmarkConfig(BenchmarkConfig[WebArenaVerifiedTaskMetadata]):
    """WebArena Verified — 812 verified web automation tasks across 6 platforms.

    Exactly one of two setup modes must be configured before calling ``make()``:

    **Automatic** — pass an ``infra`` to ``make()``; it provisions and launches
    the Docker stack, and ``wav_config.environments`` is populated from the
    live endpoints::

        WebArenaVerifiedBenchmarkConfig(
            resources=[WEBARENA_SHOPPING_ADMIN],
        ).make(infra=LocalInfraConfig())

    **Manual** — caller is responsible for starting the server; ``wav_config``
    must have ``environments`` populated with reachable URLs::

        WebArenaVerifiedBenchmarkConfig(
            wav_config=WebArenaVerifiedConfig(
                environments={WebArenaSite.SHOPPING_ADMIN: EnvironmentConfig(urls=["http://..."])}
            )
        ).make()

    Mixing both modes (or providing neither) raises a ``ValueError`` at ``make()`` time.

    Filtering is done in user-land via subset_from_glob() / subset_from_list():
        config.subset_from_glob("sites", "*shopping_admin*")
        config.subset_from_glob("expected_action", "RETRIEVE")
        config.subset_from_list(["0", "1", "5"])

    To regenerate task_metadata.json (developer use only), run:
        scripts/generate_task_metadata.py
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="webarena-verified-cube",
        version="1.0.0",
        description="WebArena-Verified benchmark — 812 verified web automation tasks across 6 platforms",
        num_tasks=812,
        tags=["browser", "web", "ui", "webarena"],
    )
    """task_metadata.json is a shipped package resource containing lightweight public fields
    (sites, expected_action, intent_template_id). No heavy execution data exists."""
    task_config_class: ClassVar[type[TaskConfig]] = WebArenaVerifiedTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = WebArenaVerifiedBenchmark
    tool_config: SerializeAsAny[ToolConfig] = ToolboxConfig(
        tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()]
    )
    """Tool config applied to every task. Any ToolConfig is accepted (the canonical
    "default" config in configs.py uses a BgymToolConfig); the field default here is a
    ToolboxConfig with HAR-based observation + agent response submission."""

    wav_config: WebArenaVerifiedConfig = WebArenaVerifiedConfig()
    """WAV client configuration.

    *Automatic mode*: leave as default — ``environments`` is populated after launch.
    *Manual mode*: must have ``environments`` set with reachable URLs.
    """

    def make(self, infra: InfraConfig | None = None) -> WebArenaVerifiedBenchmark:
        """Default ``infra`` to ``LocalInfraConfig`` when DockerServiceConfig
        resources are declared in automatic mode, then delegate to the base
        ``BenchmarkConfig.make`` for provisioning + setup.

        Manual-mode users set ``wav_config.environments`` and pass no infra;
        cloud users pass their own ``infra`` explicitly; the local-default
        branch fires only for the default case where DockerServiceConfig
        resources are declared but no infra is provided.
        """
        needs_infra = self.wav_config.environments is None and any(
            isinstance(r, DockerServiceConfig) for r in self.resources
        )
        if needs_infra and infra is None:
            logger.info(
                "No infra= passed but DockerServiceConfig resources are declared; defaulting to LocalInfraConfig()"
            )
            infra = LocalInfraConfig()
        return cast(WebArenaVerifiedBenchmark, super().make(infra=infra))

    def get_task_configs(self) -> Generator[WebArenaVerifiedTaskConfig, None, None]:
        """Yield TaskConfigs with wav_config forwarded from benchmark settings."""
        for tm in self.tasks().values():
            yield WebArenaVerifiedTaskConfig(
                metadata=tm,
                tool_config=self.tool_config,
                wav_config=self.wav_config,
            )
