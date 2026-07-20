# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "knows-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# knows-cube = { path = "../cubes/knows-cube", editable = true }
# ///
"""Reference recipe: Genny on KNOWS (Google Workspace authoring).

This file IS the config — copy it and edit the values. It is not a CLI.

NOT RUNNABLE YET — see cubes/knows-cube/README.md "Blocking gap". KNOWS creates
its workspace file by driving the live Google Docs UI, and cube-browser-playwright
has no way to supply an authenticated browser session (no `user_data_dir`, no
`storage_state`; it always launches a fresh logged-out profile). Closing that
needs an upstream change. This recipe is the intended shape once it lands.

PREREQUISITES beyond that — this recipe costs money:
  * a Google service-account JSON at $SERVICE_ACCOUNT_PATH (Drive + Docs scopes)
  * $GOOGLE_AI_API_KEY (paid Gemini) — every family's grading uses a VLM

Without these, EVERY task returns reward 0.0 with info["evaluation_error"] set.
A 0.0 here is not an agent failure until you have checked that key.

Every episode leaves a new Google Doc/Sheet/Slides deck in the account's Drive;
nothing deletes them. Budget for cleanup.

To run a single family:
    KNOWS_CONFIGS["default"].subset_from_glob("task_family_folder", "docs_1_*")
"""

from knows_cube import KNOWS_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

agent = GENNY_CONFIGS["default"]
agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)

# "default" excludes knows.docs_37_reference_list.4, whose upstream evaluator
# does not parse and therefore scores 0.0 for every agent. An explicit
# subset_name keeps the provenance readable — chaining named_subset() would
# overwrite the recorded "gradeable" label with "docs".
benchmark = KNOWS_CONFIGS["default"].subset_from_glob("workspace_kind", "docs", subset_name="gradeable-docs")

exp = Experiment(
    name="knows-docs",
    agent_config=agent,
    benchmark_config=benchmark,
    max_steps=40,
)

if __name__ == "__main__":
    run(exp)
