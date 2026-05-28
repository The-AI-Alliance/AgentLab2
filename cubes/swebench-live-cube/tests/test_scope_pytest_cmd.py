"""Unit tests for the scoped-eval pytest-command rewriter.

``_scope_pytest_cmd`` is the parser that ``scoped_eval`` uses to swap broad
suite-wide ``test_cmds`` (``pytest keras -rA``) for a scoped invocation
(``xargs -a … pytest -rA``) that reads its test selectors (the filtered
f2p/p2p node IDs) from a file. The cases below are the real-world shapes
seen across the SWE-bench-Live lite split.

``_is_truncated_id`` is the heuristic that drops dataset-truncated parametrized
IDs before scoping (pytest would abort the whole run on them).
"""

from __future__ import annotations

import pytest

from swebench_live_cube.task import _is_truncated_id, _scope_pytest_cmd


_NODE_IDS_FILE = "/tmp/cube_scoped_tests.txt"


def _xargs_head() -> str:
    return f"xargs -a {_NODE_IDS_FILE} -d '\\n' --no-run-if-empty -s 2000000"


# ── Pytest commands we expect to scope ────────────────────────────────────


@pytest.mark.parametrize(
    "cmd, expected_tail",
    [
        # Bare pytest — most common in the dataset (164/337 lite cmds).
        ("pytest -rA", "pytest -rA"),
        # Single positional path — must be stripped.
        ("pytest keras -rA", "pytest -rA"),
        ("pytest -rA .", "pytest -rA"),
        ("pytest -rA tests/", "pytest -rA"),
        # Positional at end (after flags).
        ("pytest -n 8 -rA -v sdk/python/tests", "pytest -n 8 -rA -v"),
        # Env-var prefix preserved — env vars stay BEFORE xargs, never between
        # xargs and pytest (otherwise xargs would treat `KEY=VAL` as the command).
        ("SKIP_APPLICATIONS_TESTS=True pytest keras -rA", "pytest -rA"),
        ("SKIP_APPLICATIONS_TESTS=True pytest -rA keras", "pytest -rA"),
        # Value-flag with non-positional next token (--ignore <path>).
        ("pytest keras --ignore keras/src/applications -rA", "pytest --ignore keras/src/applications -rA"),
        # ``--key=value`` form is one shlex token; passes through untouched.
        ("pytest -rA tests/ --ignore=tests/reliability", "pytest -rA --ignore=tests/reliability"),
        # Value-flag value contains spaces (shlex-quoted in output).
        ("pytest -m unit -rA", "pytest -m unit -rA"),
        ('pytest -k "test_foo or test_bar" tests/', "pytest -k 'test_foo or test_bar'"),
        # Wrappers preserved (poetry / uv / python -m).
        ("poetry run pytest -p no:cov -rA", "poetry run pytest -p no:cov -rA"),
        ("poetry run pytest -rA", "poetry run pytest -rA"),
        ("uv run pytest -rA", "uv run pytest -rA"),
        ("python -m pytest -rA", "python -m pytest -rA"),
        ("python3 -m pytest -rA", "python3 -m pytest -rA"),
        # Optional-value flag (``--cov``) must not eat the next flag —
        # mirrors argparse's nargs="?" rule (DVC's real cmd shape).
        (
            "pytest -rA -n=auto --dist=worksteal --timeout=300 --cov --cov-report=xml",
            "pytest -rA -n=auto --dist=worksteal --timeout=300 --cov --cov-report=xml",
        ),
        # Heavy cmd (xarray): no positional, all value-flags either glued or known.
        (
            "pytest -n 4 --timeout 180 --cov=xarray --cov-report=xml --junitxml=pytest.xml -rA",
            "pytest -n 4 --timeout 180 --cov=xarray --cov-report=xml --junitxml=pytest.xml -rA",
        ),
        # Multiple positionals all dropped.
        ("pytest tests/ test_other/ -rA", "pytest -rA"),
    ],
)
def test_scopes_pytest_cmds(cmd: str, expected_tail: str) -> None:
    out = _scope_pytest_cmd(cmd, _NODE_IDS_FILE)
    # The env-prefix from the input cmd, if any, goes BEFORE the xargs head
    # (so xargs inherits it and execs pytest with it). Everything else is in
    # the expected tail after ``xargs … pytest``.
    import re

    env_match = re.match(r"^((?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+)", cmd)
    env_prefix = env_match.group(1).rstrip() if env_match else ""
    expected = (env_prefix + " " if env_prefix else "") + f"{_xargs_head()} {expected_tail}"
    assert out == expected, f"unexpected scope for {cmd!r}: {out!r}"


# ── Truncated-ID detector ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "nid",
    [
        # Trailing comma inside brackets (CSV-cut mid-list).
        'test/plugins/test_discogs.py::test_get_media_and_albumtype[formats1-Vinyl-7",',
        "test/foo.py::test_bar[a,",
        # Unbalanced double-quote inside brackets (cut mid-string).
        "tests/test_x.py::test_y[Literal['one',",
        # Unbalanced paren inside brackets (cut mid-tuple).
        "xarray/tests/test_cftime_offsets.py::test_add_month_begin[365_day-(1,",
        # Unbalanced ``[`` / ``]`` over the whole ID — cut mid-word, no closing
        # bracket at all. Real kedro pattern (extra_persistent_ds-…-No).
        "tests/runner/test_thread_runner.py::TestSuggestResumeScenario::test_x[extra_persistent_ds-failing_node_names0-No",
        # No brackets at all — paren-style parametrize cut mid-tuple. Real
        # keras pattern (DatasetUtilsTest::test_split_dataset_tensorflow_(100,).
        "keras/src/utils/dataset_utils_test.py::DatasetUtilsTest::test_split_dataset_tensorflow_(100,",
    ],
)
def test_is_truncated_id_flags_malformed(nid: str) -> None:
    assert _is_truncated_id(nid), f"expected to flag truncated: {nid!r}"


@pytest.mark.parametrize(
    "nid",
    [
        # No brackets — plain test ID.
        "test/foo.py::test_bar",
        "tests/test_module.py::TestClass::test_method",
        # Clean parametrize — balanced brackets, no trailing comma.
        "test_x.py::test_param[None-None-None]",
        "test_x.py::test_param[formats0-CD-12]",
        # Double quotes balanced.
        'test_x.py::test_param[Literal["a","b"]]',
        # Parens balanced.
        "test_x.py::test_param[365_day-(1,2)-3]",
    ],
)
def test_is_truncated_id_passes_clean(nid: str) -> None:
    assert not _is_truncated_id(nid), f"falsely flagged clean: {nid!r}"


# ── Non-pytest commands (fall back) ──────────────────────────────────────


@pytest.mark.parametrize(
    "cmd",
    [
        "hatch run test:unit -rA -vv",
        "python3 devscripts/run_tests.py --pytest-args -rA",
        "poe test -- -rA",
        "pdm run test -rA",
        "cat ./scripts/test.sh",
        # Looks like an env-prefix but no pytest follows.
        "FOO=bar make test",
        # Empty.
        "",
    ],
)
def test_falls_back_when_not_pytest(cmd: str) -> None:
    assert _scope_pytest_cmd(cmd, _NODE_IDS_FILE) is None


def test_unterminated_quote_falls_back_cleanly() -> None:
    # A malformed shell cmd shouldn't blow up — shlex.split raises ValueError,
    # we return None and the caller runs unscoped.
    assert _scope_pytest_cmd('pytest -k "unterminated', _NODE_IDS_FILE) is None
