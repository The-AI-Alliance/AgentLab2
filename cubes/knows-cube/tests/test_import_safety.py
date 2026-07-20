"""The cube must import, register, and enumerate tasks on a machine with zero secrets."""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from knows_cube.evaluator import missing_credentials

_CREDENTIAL_VARS = (
    "SERVICE_ACCOUNT_PATH",
    "TOKEN_PATH",
    "CLIENT_SECRETS_PATH",
    "GCP_PROJECT_ID",
    "DRIVE_SA_SECRET_ID",
    "GOOGLE_AI_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
)


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


def test_package_imports_without_credentials() -> None:
    import knows_cube

    assert knows_cube.KNOWS_CONFIGS
    assert callable(knows_cube.get_debug_benchmark)
    assert callable(knows_cube.make_debug_agent)


def test_debug_helpers_are_exported_from_package_root() -> None:
    """The cube-registry compliance check introspects the top-level package, not debug.py."""
    import knows_cube

    assert "get_debug_benchmark" in knows_cube.__all__
    assert "make_debug_agent" in knows_cube.__all__


def test_entry_point_resolves() -> None:
    eps = {e.name: e for e in entry_points(group="cube.benchmarks")}
    assert "knows-cube" in eps
    assert eps["knows-cube"].load().__name__ == "KnowsBenchmarkConfig"


def test_entry_point_name_matches_benchmark_metadata_name() -> None:
    """cube test derives the debug module from the entry-point name by string surgery."""
    from knows_cube.benchmark import KnowsBenchmarkConfig

    assert KnowsBenchmarkConfig.benchmark_metadata.name == "knows-cube"


def test_missing_credentials_reports_both_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # missing_credentials() probes Path("auth-data")/"service-account.json" relative
    # to the process cwd, so without chdir this asserts a property of the developer's
    # working directory rather than of the code.
    monkeypatch.chdir(tmp_path)
    assert len(missing_credentials()) == 2  # google-auth + vlm


def test_debug_benchmark_needs_no_credentials() -> None:
    from knows_cube.debug import get_debug_benchmark

    config = get_debug_benchmark()
    assert config.num_tasks == 2
    with config.make() as benchmark:
        assert benchmark is not None


def test_no_leaked_env_from_knows_checkout() -> None:
    """Upstream's _load_evaluator scrapes .env files into os.environ; ours must not.

    Runs in a subprocess and diffs os.environ across the import. Asserting
    ``"S2_API_KEY" not in os.environ`` in-process would instead assert that the
    developer has no S2_API_KEY in their shell — and the cube's own README lists
    it as a requirement for docs_5, so that check fails on a credentialed machine.
    """
    probe = (
        "import os, json;"
        "before = dict(os.environ);"
        "import knows_cube.evaluator;"
        "print(json.dumps(sorted(set(os.environ) - set(before))))"
    )
    env = {**os.environ, "S2_API_KEY": "sentinel-must-survive-unchanged"}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env).stdout
    assert json.loads(out.splitlines()[-1]) == [], "importing knows_cube.evaluator injected env vars"
