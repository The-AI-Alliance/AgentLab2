#!/usr/bin/env python3
"""Sweep stale Azure resources tagged for a CUBE resource group.

Calls ``AzureInfraConfig.cleanup_stale()`` against a configured resource group
to delete any VMs whose ``cube:expires_at`` tag has passed. This is the L4
defense layer for orphaned-VM accumulation: it runs independently of the
harness, so a worker crash that bypasses the per-task ``finally`` block —
and a months-long gap between benchmark runs that defeats the startup-time
sweep — both still result in reclaim within one sweep interval.

Designed for invocation from a scheduled GitHub Action (or any cron-like
runner). Authentication uses ``AzureCliCredential``, so the caller must have
run ``az login`` or, in CI, the ``azure/login`` action must have populated
the credential cache before this script runs.

Usage:

    scripts/azure_orphan_sweep.py --resource-group ui_assist
    scripts/azure_orphan_sweep.py --resource-group ui_assist --max-age-seconds 86400
    scripts/azure_orphan_sweep.py --resource-group ui_assist --dry-run
"""

import logging
import sys
from typing import Annotated

import typer
from cube_infra_azure.azure import AzureInfraConfig

logger = logging.getLogger("azure_orphan_sweep")


def main(
    resource_group: Annotated[
        str,
        typer.Option(
            "--resource-group",
            "-g",
            envvar="AZURE_RESOURCE_GROUP",
            help="Azure resource group to sweep. Required.",
        ),
    ],
    max_age_seconds: Annotated[
        int | None,
        typer.Option(
            "--max-age-seconds",
            "-m",
            help="Also delete cube-tagged resources older than this (in seconds), "
            "even if they have no cube:expires_at tag. Useful for legacy resources "
            "predating TTL tagging. Omit to only delete on TTL.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Authenticate and list stale resources without deleting. "
            "Currently a no-op shim that just instantiates the infra; the "
            "underlying ``cleanup_stale()`` does not have a dry-run mode.",
        ),
    ] = False,
) -> None:
    """Reclaim stale Azure VMs in the configured resource group."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    infra = AzureInfraConfig(resource_group=resource_group)
    logger.info("Sweeping stale resources in %s (max_age_seconds=%s)", resource_group, max_age_seconds)

    if dry_run:
        logger.info("DRY-RUN: skipping cleanup_stale() call")
        sys.exit(0)

    deleted = infra.cleanup_stale(max_age_seconds=max_age_seconds)
    logger.info("Reclaimed %d expired resource(s) from %s", len(deleted), resource_group)
    if deleted:
        for resource_id in deleted:
            logger.info("  deleted: %s", resource_id)


if __name__ == "__main__":
    typer.run(main)
