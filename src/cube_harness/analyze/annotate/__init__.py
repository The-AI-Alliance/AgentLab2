"""Shared human-annotation portal for the trajectory-judge validation study.

`store` is the concurrency layer (SQLite pool of episodes, leases and labels),
`panel` renders one episode through the real XRay viewer so humans and the judge
read the same evidence, `app` is the Gradio portal, and `cli` is `ch-annotate`.
"""

from cube_harness.analyze.annotate.store import AnnotationStore, Episode, Progress

__all__ = ["AnnotationStore", "Episode", "Progress"]
