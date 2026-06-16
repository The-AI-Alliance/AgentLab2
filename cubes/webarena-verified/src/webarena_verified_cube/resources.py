"""Canonical DockerServiceConfig declarations for WebArena-Verified sites.

``WEBARENA_ALL`` is the primary resource — a single DockerServiceConfig that
bundles all 6 sites (images, ports, volumes) into one provisionable unit.

Per-site resources (``WEBARENA_SHOPPING_ADMIN``, etc.) are available for
testing or running a subset of tasks against a single site.

Usage::

    from webarena_verified_cube.resources import WEBARENA_ALL

Port assignments mirror ``webarena_verified.environments.container.config.DEFAULT_CONTAINER_CONFIGS``.
"""

from __future__ import annotations

from pathlib import Path

from cube.resource import DockerServiceConfig, VolumeSpec

_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _script(name: str) -> str:
    """Return the contents of a launch script from the scripts/ directory."""
    return (_SCRIPTS_DIR / name).read_text()


# ── shopping_admin ────────────────────────────────────────────────────────────
# Magento admin portal.  184 tasks.
WEBARENA_SHOPPING_ADMIN = DockerServiceConfig(
    name="webarena-shopping-admin",
    scope="benchmark",
    docker_images=["am1n3e/webarena-verified-shopping_admin"],
    services={
        "shopping_admin": 7780,  # web UI  (container port 80)
        "shopping_admin_ctrl": 7781,  # env-ctrl (container port 8877)
    },
    endpoint_to_site={"shopping_admin": "shopping_admin"},
    launch_script=_script("shopping_admin_launch.sh"),
)

# ── shopping ──────────────────────────────────────────────────────────────────
# Magento storefront (customer-facing).  192 tasks.
WEBARENA_SHOPPING = DockerServiceConfig(
    name="webarena-shopping",
    scope="benchmark",
    docker_images=["am1n3e/webarena-verified-shopping"],
    services={
        "shopping": 7770,  # web UI  (container port 80)
        "shopping_ctrl": 7771,  # env-ctrl (container port 8877)
    },
    endpoint_to_site={"shopping": "shopping"},
    launch_script=_script("shopping_launch.sh"),
)

# ── reddit ────────────────────────────────────────────────────────────────────
# Postmill reddit clone.  129 tasks.
WEBARENA_REDDIT = DockerServiceConfig(
    name="webarena-reddit",
    scope="benchmark",
    docker_images=["am1n3e/webarena-verified-reddit"],
    services={
        "reddit": 9999,  # web UI  (container port 80)
        "reddit_ctrl": 9998,  # env-ctrl (container port 8877)
    },
    endpoint_to_site={"reddit": "reddit"},
    launch_script=_script("reddit_launch.sh"),
)

# ── gitlab ────────────────────────────────────────────────────────────────────
# GitLab CE.  204 tasks.  Slow startup (~5-10 min).
WEBARENA_GITLAB = DockerServiceConfig(
    name="webarena-gitlab",
    scope="benchmark",
    docker_images=["am1n3e/webarena-verified-gitlab"],
    services={
        "gitlab": 8023,  # web UI  (container port 8023)
        "gitlab_ctrl": 8024,  # env-ctrl (container port 8877)
    },
    endpoint_to_site={"gitlab": "gitlab"},
    launch_script=_script("gitlab_launch.sh"),
)

# ── wikipedia ─────────────────────────────────────────────────────────────────
# Kiwix Wikipedia server.  23 tasks.
# The ZIM file (~80 GB) is downloaded during provision() via VolumeSpec.
WEBARENA_WIKIPEDIA = DockerServiceConfig(
    name="webarena-wikipedia",
    scope="benchmark",
    docker_images=["ghcr.io/kiwix/kiwix-serve:3.8.0"],
    services={
        "wikipedia": 8888,  # web UI  (container port 8080)
        "wikipedia_ctrl": 8889,  # env-ctrl (container port 8874)
    },
    volumes=[
        VolumeSpec(
            name="webarena_wikipedia_data",
            mount_path="/data",
            source_url="http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim",
        ),
    ],
    endpoint_to_site={"wikipedia": "wikipedia"},
    launch_script=_script("wikipedia_launch.sh"),
)

# ── map ───────────────────────────────────────────────────────────────────────
# OpenStreetMap tile server + Nominatim + OSRM routing.  128 tasks.
# 3 large archives → 9 Docker volumes, all downloaded during provision().
_MAP_S3 = "https://webarena-map-server-data.s3.amazonaws.com"

WEBARENA_MAP = DockerServiceConfig(
    name="webarena-map",
    scope="benchmark",
    docker_images=["am1n3e/webarena-verified-map"],
    services={
        "map": 3000,  # web UI  (container port 8080)
        "map_ctrl": 3001,  # env-ctrl (container port 8877)
    },
    volumes=[
        # ── populated from osm_tile_server.tar ────────────────────────────────
        VolumeSpec(
            name="webarena_map_tile_db",
            mount_path="/data/database",
            source_url=f"{_MAP_S3}/osm_tile_server.tar",
            tar_subpath="projects/ogma3/docker/volumes/osm-data/_data",
            strip_components=6,
        ),
        # ── populated from osrm_routing.tar (3 volumes, 1 archive) ───────────
        VolumeSpec(
            name="webarena_map_routing_car",
            mount_path="/data/routing/car",
            source_url=f"{_MAP_S3}/osrm_routing.tar",
            tar_subpath="car",
            strip_components=1,
        ),
        VolumeSpec(
            name="webarena_map_routing_bike",
            mount_path="/data/routing/bike",
            source_url=f"{_MAP_S3}/osrm_routing.tar",
            tar_subpath="bike",
            strip_components=1,
        ),
        VolumeSpec(
            name="webarena_map_routing_foot",
            mount_path="/data/routing/foot",
            source_url=f"{_MAP_S3}/osrm_routing.tar",
            tar_subpath="foot",
            strip_components=1,
        ),
        # ── populated from nominatim_volumes.tar ──────────────────────────────
        VolumeSpec(
            name="webarena_map_nominatim_db",
            mount_path="/data/nominatim/postgres",
            source_url=f"{_MAP_S3}/nominatim_volumes.tar",
            tar_subpath="projects/metis2/docker/docker/volumes/nominatim-data/_data",
            strip_components=7,
        ),
        VolumeSpec(
            name="webarena_map_nominatim_flatnode",
            mount_path="/data/nominatim/flatnode",
            source_url=f"{_MAP_S3}/nominatim_volumes.tar",
            tar_subpath="projects/metis2/docker/docker/volumes/nominatim-flatnode/_data",
            strip_components=7,
        ),
        # ── empty volumes (populated at runtime) ─────────────────────────────
        VolumeSpec(name="webarena_map_website_db", mount_path="/var/lib/postgresql/14/main"),
        VolumeSpec(name="webarena_map_tiles", mount_path="/data/tiles"),
        VolumeSpec(name="webarena_map_style", mount_path="/data/style"),
    ],
    endpoint_to_site={"map": "map"},
    launch_script=_script("map_launch.sh"),
)


# ── WEBARENA_ALL — combined resource for the full benchmark ───────────────────
# All 6 sites in a single DockerServiceConfig: one provision, one launch, one VM.
WEBARENA_ALL = DockerServiceConfig(
    name="webarena-all",
    scope="benchmark",
    docker_images=[
        "am1n3e/webarena-verified-shopping_admin",
        "am1n3e/webarena-verified-shopping",
        "am1n3e/webarena-verified-reddit",
        "am1n3e/webarena-verified-gitlab",
        "ghcr.io/kiwix/kiwix-serve:3.8.0",
        "am1n3e/webarena-verified-map",
    ],
    services={
        # shopping_admin (Magento admin)
        "shopping_admin": 7780,
        "shopping_admin_ctrl": 7781,
        # shopping (Magento storefront)
        "shopping": 7770,
        "shopping_ctrl": 7771,
        # reddit (Postmill)
        "reddit": 9999,
        "reddit_ctrl": 9998,
        # gitlab
        "gitlab": 8023,
        "gitlab_ctrl": 8024,
        # wikipedia (Kiwix)
        "wikipedia": 8888,
        "wikipedia_ctrl": 8889,
        # map (OSM + Nominatim + OSRM)
        "map": 3000,
        "map_ctrl": 3001,
    },
    volumes=WEBARENA_WIKIPEDIA.volumes + WEBARENA_MAP.volumes,
    endpoint_to_site={
        "shopping_admin": "shopping_admin",
        "shopping": "shopping",
        "reddit": "reddit",
        "gitlab": "gitlab",
        "wikipedia": "wikipedia",
        "map": "map",
    },
    launch_script=_script("all_sites_launch.sh"),
)
