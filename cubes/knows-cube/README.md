# knows-cube

KNOWS — 110 Google Workspace authoring tasks (22 families × 5 instances) across
Google Docs (25), Sheets (45) and Slides (40). The agent drives a real browser
against real `docs.google.com`; grading calls the upstream KNOWS evaluators,
which hit the Drive/Docs/Sheets/Slides APIs and a Gemini vision model.

Ported following the `webarena-verified` pattern: this cube depends on the KNOWS
eval/task layer **directly**. It deliberately does *not* depend on the
BrowserGym-Knows fork — no forked `browsergym-core 0.14.3.dev0`, no `PYTHONPATH`
injection, no `AgentLab-Knows` submodule. Stock `browsergym-core` from PyPI is
sufficient, because KNOWS needs only `register_task` and `AbstractBrowserTask`
from it.

## Upstream prerequisites

The KNOWS package (`browsergym-knows`) **does not build as published**. Its
`pyproject.toml` reads its version from a path outside the repository:

```toml
[tool.hatch.version]
path = "../core/src/browsergym/core/__init__.py"   # only resolves inside a BrowserGym monorepo
```

`uv build` fails with `OSError: file does not exist`. The fix is two lines —
drop the `[tool.hatch.version]` block and set a static `version` — after which
the wheel builds and imports cleanly against stock `browsergym-core==0.14.3`.
**Until that lands upstream, this cube's `browsergym-knows` dependency will not
resolve for anyone else.**

The upstream repository is also currently private.

## Honest prerequisites for running tasks

**Running a real KNOWS task requires live third-party credentials and costs money.**
Without them **every task returns reward 0.0** — the evaluator fails to import,
the error is captured as `info["evaluation_error"]`, and the score is a silent
zero indistinguishable from an agent failure. Check `evaluation_error` before
reading any 0.0 as a model result.

| Requirement | Env var(s) | Needed for |
|---|---|---|
| Google service account JSON (Drive + Docs scopes) | `SERVICE_ACCOUNT_PATH` | doc creation, sharing, all grading |
| — or OAuth token | `TOKEN_PATH`, `CLIENT_SECRETS_PATH` | same, degraded |
| — or Secret Manager | `GCP_PROJECT_ID`, `DRIVE_SA_SECRET_ID` | same |
| Gemini (paid) | `GOOGLE_AI_API_KEY`, or `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` | **every** family routes ≥1 scoring step through a VLM |
| Google-authenticated browser session | **not yet supportable — see below** | doc creation drives the real Docs UI |
| Semantic Scholar | `S2_API_KEY` | `docs_5` |
| Google Maps | `GOOGLE_MAPS_API_KEY` | `sheets_28` |
| Oxylabs | `OXYLABS_USERNAME`, `OXYLABS_PASSWORD` | `sheets_38` |

`KnowsBenchmark._setup()` logs a warning listing whatever is missing. That check
is a heuristic — upstream's Vertex path can also authenticate via ambient
application-default credentials, which are invisible from the environment.

### Blocking gap: no way to supply an authenticated browser session

**Real episodes cannot run yet.** `create_task_workspace` drives the live Google
Docs UI, which requires a signed-in browser, and `cube-browser-playwright`
currently offers no way to provide one:

* `PlaywrightSessionConfig` exposes no `user_data_dir` and no `storage_state`
  field, and hardcodes `user_data_dir = tempfile.mkdtemp(...)` — every episode
  gets a fresh, logged-out profile.
* `pw_extra_kwargs` is forwarded to `launch_persistent_context`, whose signature
  has no `storage_state` parameter — passing one raises `TypeError` before the
  first episode starts.

Closing this needs an upstream change in `cube-browser-playwright` (a
`user_data_dir` knob, or `storage_state` via `browser.new_context()`), after
which this cube should grow a matching field on `KnowsBenchmarkConfig`. Until
then the cube is exercised only by `cube test` (hermetic) and by
`scripts/smoke/knows_gold_eval.py`, which grades a pre-existing document via
`existing_doc_id` and therefore needs no browser at all.

This cube **never** reads a `.env` from the KNOWS checkout. Upstream's
`_load_evaluator` scrapes `.env` files from two directories *above* its repo root
into `os.environ`; evaluator loading is reimplemented here specifically to avoid
that. **The KNOWS repositories ship committed live API keys and, in the public
BrowserGym-Knows fork, a Google session cookie jar and plaintext account
password — treat all of them as leaked and rotate them.** Do not use them as a
config source.

## What `cube test knows-cube` does and does not prove

`cube test` runs a **synthetic, hermetic debug benchmark** — no Google, no
Gemini, no browser, no network. It exercises this cube's own plumbing: goal
construction, submit/infeasible detection, idempotent terminal evaluation
including the truncation path, `Result` → reward normalisation, and the flat
`eval.cp{i}_*` info schema.

It proves **nothing** about the real evaluators. `cube test` passing does not
mean a KNOWS task will grade.

The debug benchmark is synthetic rather than a real task with a scripted cheat
because **reward 1.0 is unreachable deterministically on any real KNOWS task**:
upstream's `cheat()` raises `NotImplementedError`, all 22 families route at
least one scoring step through a Gemini VLM, and the cheapest task (`docs_1`,
10 points) has a deterministic ceiling of 8/10 because `verify_image_in_region`
has no non-VLM tier.

For real verification run `scripts/smoke/knows_gold_eval.py` (repo root). It
points a task at a committed gold-instance doc, requires full credentials,
`SKIP`s cleanly without them, and asserts `reward >= 0.8` rather than `== 1.0` —
the VLM steps flake, and a test that pretends otherwise is one everyone learns
to ignore.

## Known-bad tasks

* `knows.docs_37_reference_list.4` — the upstream `evaluator.py` does not parse.
  Flagged `is_gradeable=False` and excluded from `KNOWS_CONFIGS["default"]`.
* `knows.docs_31_*` and `knows.slides_30_*` — their evaluators declare
  `browsing_history_doc_id` / `client_doc_id`, which no caller (upstream
  included) supplies. They grade on degraded input; recorded in `known_issues`.
* `knows.sheets_10_paper_sorting.*` — upstream runs a `setup_run.py` subprocess
  during setup. This cube does not, so those instances may under-grade.

## Cleanup

Every episode creates a Google Doc/Sheet/Slides deck in the authenticated
account's Drive, and **nothing deletes it**. A 110-task sweep leaves 110 files.
Budget for periodic Drive cleanup.

## Usage

```python
from knows_cube import KNOWS_CONFIGS

benchmark = KNOWS_CONFIGS["default"]   # 109 gradeable tasks
benchmark = KNOWS_CONFIGS["docs"]      # 25 Docs tasks
benchmark = KNOWS_CONFIGS["default"].subset_from_glob("task_family_folder", "docs_1_*")
```

```bash
make install && make metadata && make test
```
