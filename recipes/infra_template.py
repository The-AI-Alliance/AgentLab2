"""TEMPLATE — copy this to ``~/.cube/infra.py`` and edit it for your setup.

This is NOT a runnable recipe. It is the worked example for the per-machine
infra file that `cube_harness.infra` loads.

How it works
------------
- `~/.cube/infra.py` is plain Python, machine-local, and **never committed**.
- It must define ``INFRA_CONFIGS: dict[str, InfraConfig]``. A recipe then
  names one: ``infra = INFRA_CONFIGS["my-cluster"]``.
- ``"local"`` (a bare ``LocalInfraConfig``) is **always available even with
  no file**, so the canonical recipes run with zero setup. Define your own
  ``"local"`` here only if you want to override it (e.g. more CPU/RAM).
- **Credentials are never written here.** Every backend resolves secrets
  from environment variables at runtime (shown below). Keep this file free
  of tokens/keys — it is just selection (which cluster / region / image).

Most users only need to (1) copy this file, (2) keep ``local`` as-is, and
(3) uncomment + edit the one cloud/cluster block they actually use.
"""

from cube.infra_local import LocalInfraConfig
from cube.resource import InfraConfig

INFRA_CONFIGS: dict[str, InfraConfig] = {
    # The zero-setup default — local Docker. Override the built-in by giving
    # it more resources, or just delete this entry to keep the built-in.
    "local": LocalInfraConfig(),
    # A bigger local box for heavier benchmarks:
    "local-big": LocalInfraConfig(cpu_cores=8, ram_gb=16),
}

# ---------------------------------------------------------------------------
# Cloud / cluster backends — uncomment the ONE you use and edit for your org.
# These live in optional packages (`pip install cube-infra-<name>`); the
# imports stay commented so this template imports even without them.
# ---------------------------------------------------------------------------

# EAI / Toolkit cluster. `cube_data` defaults to "auto" (auto-provisions the
# sidecar + uv bundle at /opt/cube on first launch — no flags needed).
#
#     from cube_infra_toolkit import ToolkitInfraConfig
#     INFRA_CONFIGS["toolkit"] = ToolkitInfraConfig(
#         profile="my-eai-profile",        # your EAI profile name
#         preemptable=True,
#         launch_timeout_seconds=3000,
#     )

# Azure VMs (one fresh VM per task). Secrets come from env:
# AZURE_RESOURCE_GROUP / AZURE_STORAGE_ACCOUNT / az-login credentials.
#
#     from cube_infra_azure import AzureInfraConfig
#     INFRA_CONFIGS["azure-westus2"] = AzureInfraConfig(
#         resource_group=os.environ["AZURE_RESOURCE_GROUP"],
#         storage_account=os.environ["AZURE_STORAGE_ACCOUNT"],
#         vnet_name="vnet-westus2",
#         nsg_name="my-nsg",
#     )
