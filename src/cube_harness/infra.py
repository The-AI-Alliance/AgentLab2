"""User-defined infra configs, loaded from ``~/.cube/infra.py``.

Infra selection (which cluster, region, backend) is machine-local and never
committed — credentials are resolved from env vars at runtime, never stored
on the config. A recipe just names one:

    from cube_harness.infra import INFRA_CONFIGS
    infra = INFRA_CONFIGS["toolkit-yul101"]

``~/.cube/infra.py`` is a plain Python file defining a dict::

    # ~/.cube/infra.py
    from cube_infra_toolkit import ToolkitInfraConfig
    INFRA_CONFIGS = {
        "toolkit-yul101": ToolkitInfraConfig(cluster="yul101"),
    }

``"local"`` (a bare ``LocalInfraConfig``) is always available without any
config file, so a fresh machine runs the canonical recipes with zero setup.
A user-defined ``"local"`` overrides the built-in.

Starting point: copy ``recipes/infra_template.py`` to ``~/.cube/infra.py``
— it walks through the process with worked LocalInfraConfig / Toolkit /
Azure examples.
"""

import importlib.util
from pathlib import Path

from cube.core import ConfigRegistry
from cube.infra_local import LocalInfraConfig
from cube.resource import InfraConfig

USER_INFRA_PATH = Path.home() / ".cube" / "infra.py"


def _load_user_infra() -> dict[str, InfraConfig]:
    if not USER_INFRA_PATH.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_cube_user_infra", USER_INFRA_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {USER_INFRA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configs = getattr(module, "INFRA_CONFIGS", None)
    if not isinstance(configs, dict):
        raise ValueError(f"{USER_INFRA_PATH} must define INFRA_CONFIGS: dict[str, InfraConfig]")
    return configs


def load_infra_configs() -> ConfigRegistry[InfraConfig]:
    """Built-in ``local`` plus everything in ``~/.cube/infra.py`` (user wins)."""
    return ConfigRegistry({"local": LocalInfraConfig(), **_load_user_infra()})


INFRA_CONFIGS: ConfigRegistry[InfraConfig] = load_infra_configs()
