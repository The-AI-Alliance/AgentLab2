"""Loading and invoking upstream KNOWS evaluators.

IMPORTANT — global state (EX-002 exception, disclosed deliberately):
importing ``browsergym.knows.task`` unconditionally runs
``_install_eval_import_shim()``, which inserts a finder into ``sys.meta_path``
and creates fake ``src`` / ``src.browsergym`` namespace packages. This is
required — all 110 upstream evaluators import via ``src.browsergym.knows.eval``.
The shim is idempotent and process-global, and is never uninstalled. The correct
fix site is upstream, not this cube.

Why we do not call upstream's ``KnowsWorkspaceTask._load_evaluator``: it needs a
live task instance (impossible without BrowserGym's ABC), it scrapes ``.env``
files from two directories *above* the KNOWS repo root into ``os.environ``, and
it never caches — every call re-executes the evaluator file and re-authenticates
against Google.
"""

import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from browsergym.knows.task import EVAL_TASKS_DIR, KnowsWorkspaceTask

logger = logging.getLogger(__name__)

CREDENTIAL_ENV_GROUPS: tuple[tuple[str, ...], ...] = (
    ("SERVICE_ACCOUNT_PATH",),
    ("TOKEN_PATH", "CLIENT_SECRETS_PATH"),
    ("GCP_PROJECT_ID", "DRIVE_SA_SECRET_ID"),
)
"""Google auth. Any one satisfied group is enough."""

VLM_ENV_GROUPS: tuple[tuple[str, ...], ...] = (
    ("GOOGLE_AI_API_KEY",),
    ("GOOGLE_CLOUD_PROJECT",),
)
"""Gemini vision judge. Every one of the 22 families routes >=1 scoring step
through it. GOOGLE_CLOUD_LOCATION is deliberately not required — upstream
defaults it to "us-central1" (eval_utils/models.py)."""

_DEFAULT_SERVICE_ACCOUNT = Path("auth-data") / "service-account.json"
"""Upstream's cwd-relative fallback when SERVICE_ACCOUNT_PATH is unset."""


def missing_credentials() -> list[str]:
    """Return unsatisfied credential requirements as readable strings (empty if OK).

    A best-effort warning heuristic, not a gate. It can still report a false
    positive: upstream's Vertex path falls back to ambient application-default
    credentials, which are invisible from the environment. Treat a non-empty
    result as "grading will probably return silent zeros", not as proof.
    """
    missing: list[str] = []
    # Upstream reads $SERVICE_ACCOUNT_PATH, else $CWD/auth-data/service-account.json,
    # and uses it `if os.path.exists(...)` — so the file alone satisfies auth.
    if not _DEFAULT_SERVICE_ACCOUNT.exists() and not any(
        all(os.environ.get(var) for var in group) for group in CREDENTIAL_ENV_GROUPS
    ):
        missing.append(
            "google-auth: need one of "
            + " | ".join("+".join(g) for g in CREDENTIAL_ENV_GROUPS)
            + f", or {_DEFAULT_SERVICE_ACCOUNT} relative to the working directory"
        )
    if not any(all(os.environ.get(var) for var in group) for group in VLM_ENV_GROUPS):
        missing.append("vlm: need one of " + " | ".join("+".join(g) for g in VLM_ENV_GROUPS))
    return missing


def instance_dir(task_family_folder: str, instance_number: int) -> Path:
    """Absolute path to ``<EVAL_TASKS_DIR>/<family>/instance_<n>/``."""
    return EVAL_TASKS_DIR / task_family_folder / f"instance_{instance_number}"


def read_task_markdown(task_family_folder: str, instance_number: int) -> str:
    """Read the instance's ``task.md`` — the authoritative goal text."""
    path = instance_dir(task_family_folder, instance_number) / "task.md"
    if not path.exists():
        raise FileNotFoundError(f"KNOWS task.md not found at {path}")
    return path.read_text()


def load_evaluator(task_family_folder: str, instance_number: int) -> ModuleType:
    """Exec the instance's ``evaluator.py`` and return the module.

    WARNING: 105 of the 110 upstream evaluators call ``initialize_google_services()``
    at module top level, so this function AUTHENTICATES AGAINST GOOGLE as a side
    effect of importing. Callers must cache the result; this function does not.
    """
    idir = instance_dir(task_family_folder, instance_number)
    evaluator_path = idir / "evaluator.py"
    if not evaluator_path.exists():
        raise FileNotFoundError(f"KNOWS evaluator not found at {evaluator_path}")

    module_name = f"knows_cube._evaluators.{task_family_folder}.instance_{instance_number}"
    spec = importlib.util.spec_from_file_location(module_name, str(evaluator_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build a module spec for {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Upstream's paths are correct IN UPSTREAM'S LAYOUT: get_base_path() returns
    # os.getcwd(), so TASK_DIR = <cwd>/src/browsergym/knows/eval/tasks/<family>/
    # instance_<n>/ resolves natively when you run from the Agent-Benchmark repo
    # root. This cube instead consumes browsergym-knows as an installed wheel,
    # which has no src/ prefix and no control over cwd, so the paths have to be
    # repointed at the instance dir. Upstream's own _load_evaluator does the same
    # rewrite, also after exec_module.
    #
    # Pre-seeding before exec_module is NOT an option: every evaluator assigns
    # TASK_DIR unconditionally, so a seeded value is overwritten during import.
    stale_task_dir = getattr(module, "TASK_DIR", None)
    data = idir / "data"
    overrides = {
        "TASK_DIR": f"{idir}{os.sep}",
        "DATA_DIR": f"{data}{os.sep}",
        "DOC_IMAGES_DIR": f"{data / 'images'}{os.sep}",
        "DOC_IMAGES_CROPPED_DIR": f"{data / 'cropped_images'}{os.sep}",
        "PDF_IMAGES_DIR": f"{data / 'pdf_images'}{os.sep}",
        "GOLD_IMAGES_DIR": f"{data / 'gold_images'}{os.sep}",
        "GOLDS_DIR": f"{data / 'golds'}{os.sep}",
        "GOLD_DESCRIPTIONS_CSV": str(data / "gold_descriptions.csv"),
        "ORIGINAL_LOCATIONS_JSON": str(data / "original_image_locations.json"),
        "ORIGINAL_TEXTBOX_LOCATIONS_JSON": str(data / "original_textbox_locations.json"),
    }
    for attr, value in overrides.items():
        if hasattr(module, attr):
            setattr(module, attr, value)

    # Constants DERIVED from the paths above at import time (GOLD_IMAGE_ORIGINAL =
    # GOLDS_DIR + "original_image.png", IMAGES_DIR, TASK_MD_PATH, ...) keep their
    # pre-override values, so rewriting only the names above leaves them pointing
    # into a tree that does not exist in the installed wheel. Evaluators guard
    # those reads with os.path.exists(), so a stale path SILENTLY skips a scoring
    # step -- reward drops with no evaluation_error to distinguish it from an
    # agent failure. 21 of 110 instances across 5 families are affected; all 47
    # such constants live under TASK_DIR, so one prefix rewrite covers them.
    if stale_task_dir:
        for attr, value in list(vars(module).items()):
            if attr not in overrides and isinstance(value, str) and value.startswith(stale_task_dir):
                setattr(module, attr, f"{idir}{os.sep}{value[len(stale_task_dir) :]}")
    return module


def accepted_kwargs(module: ModuleType) -> set[str]:
    """Kwargs ``module.grade_checkpoints`` accepts (upstream's introspection, reused)."""
    return set(KnowsWorkspaceTask._accepted_kwargs(module.grade_checkpoints))


def grade(module: ModuleType, doc_id: str, browsing_history: list[str]) -> Any:
    """Call ``grade_checkpoints`` with only the kwargs it accepts; return a KNOWS ``Result``.

    Signatures vary across the 110 evaluators (uniform within a family), so the
    call is built by introspection rather than assumption. Two declared params
    are never supplied — ``browsing_history_doc_id`` (docs_31) and
    ``client_doc_id`` (slides_30); both have defaults, so nothing crashes, but
    those 10 instances grade on degraded input. Recorded in metadata
    ``known_issues``.
    """
    accepted = accepted_kwargs(module)
    call_kwargs: dict[str, Any] = {}
    if "workspace_doc_id" in accepted:
        call_kwargs["workspace_doc_id"] = doc_id
    if "browsing_history" in accepted:
        call_kwargs["browsing_history"] = browsing_history
    if "browsing_history_list" in accepted:
        call_kwargs["browsing_history_list"] = browsing_history
    if "cached_models" in accepted:
        call_kwargs["cached_models"] = None
    return module.grade_checkpoints(**call_kwargs)
