"""The annotation portal: one shared link, one episode at a time.

Replaces the per-annotator HTML packet. A single Gradio app (optionally behind a
`share=True` tunnel) hands each arriving annotator an episode from the shared
SQLite pool, shows it through the real XRay event viewer, takes an
outcome/blame/confidence label, and immediately serves the next one.

Two properties the packet did not have:

* **Evidence parity.** The annotator sees what the judge saw — screenshots, the
  event rail, per-step LLM calls — not a transcript truncated to 60k chars.
* **Collective progress.** Everyone draws from one pool, so κ pairs accumulate
  and the header shows the study's coverage, not just the individual's count.

State discipline (EX-002): the store and the viewer are created by the caller
and closed over by the handlers; per-user state (who you are, which episode,
which event) lives in `gr.State` / `gr.BrowserState`, so two annotators in the
same process never see each other's selection.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from cube_harness.analyze import xray_utils
from cube_harness.analyze.annotate.panel import EpisodeViewer, ViewerRender
from cube_harness.analyze.annotate.store import AnnotationStore, Episode
from cube_harness.eval_log import BlameCategory, Outcome

BLAMES: list[str] = [b.value for b in BlameCategory]
OUTCOMES: list[str] = [o.value for o in Outcome]

_INSTRUCTIONS = """\
You are shown one agent episode at a time, with the judge's verdict withheld.

1. Walk the trajectory on the left — click a card, or use ← / → (↑ / ↓) to step.
   Check the screenshots and the LLM calls, not just the actions.
2. Decide what happened (**outcome**) and, if the episode failed, **what is to
   blame**. Pick `none` for blame when the evidence supports no attribution —
   that is a real answer, not a cop-out.
3. Submit. The next episode loads straight away. **Skip** returns an episode to
   the pool for someone else.

Judge on the evidence in the trajectory alone.
"""

_ANNOTATE_CSS = (
    xray_utils.EVENT_VIEW_CSS
    + """
#ann_rail_nav {
    justify-content: center !important;
    gap: 6px !important;
    margin-bottom: 2px !important;
    min-height: 0 !important;
}
#ann_first_btn, #ann_prev_btn, #ann_next_btn, #ann_last_btn {
    min-width: 28px !important;
    max-width: 34px;
    padding: 2px 6px !important;
    flex: 0 0 auto;
}
/* The label form is the thing the annotator returns to after every episode —
   keep it visually distinct from the (grey) evidence panes. */
#ann_form {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 10px 14px;
}
"""
)

# Arrow-key event navigation, mirroring XRay's shortcuts so anyone who has used
# the viewer already knows them. The input/textarea guard keeps arrows working
# normally inside the notes box.
_INIT_JS = """
() => {
    document.body.classList.remove('dark');
    if (window.__annotateInit) return;
    window.__annotateInit = true;
    document.addEventListener('keydown', (e) => {
        const t = e.target, tag = (t.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select' || t.isContentEditable) return;
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        let sel = null;
        if (e.shiftKey) {
            if (e.key === 'ArrowUp') sel = '#ann_first_btn';
            else if (e.key === 'ArrowDown') sel = '#ann_last_btn';
        } else if (e.key === 'Home') sel = '#ann_first_btn';
        else if (e.key === 'End') sel = '#ann_last_btn';
        else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') sel = '#ann_prev_btn';
        else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') sel = '#ann_next_btn';
        if (!sel) return;
        const b = document.querySelector(sel);
        if (b) { e.preventDefault(); b.click(); }
    }, true);
}
"""


def progress_line(store: AnnotationStore, annotator: str) -> str:
    """The header line: personal count, then the pool's collective coverage.

    Renders as: you: 12 · pool: 43/60 at ≥2 labels · 4 annotators
    """
    p = store.progress()
    mine = p.per_annotator.get(annotator, 0)
    people = len(p.per_annotator)
    return (
        f"**you: {mine}** &nbsp;·&nbsp; pool: **{p.at_target}/{p.total}** at ≥{p.target_coverage} labels"
        f" &nbsp;·&nbsp; {people} annotator{'s' if people != 1 else ''}"
    )


def _view_values(render: ViewerRender) -> list[Any]:
    """`ViewerRender` → the ten viewer component values, in declaration order."""
    return [
        render.header,
        render.stats,
        render.goal,
        render.rail,
        render.reasoning,
        render.action,
        gr.update(value=render.images, visible=bool(render.images)),
        render.observation,
        render.chat,
        render.evaluation,
    ]


def _blank_view() -> list[Any]:
    return _view_values(ViewerRender())


def build_app(store: AnnotationStore, viewer: EpisodeViewer, *, passphrase: str | None = None) -> gr.Blocks:
    """Build the portal. `passphrase`, if set, gates the landing screen."""

    def _page(
        *,
        annotator: str,
        uid: str,
        selected: int,
        status: str,
        view: list[Any],
        name: str | None = None,
        landing: bool = False,
        done: str = "",
        reset_form: bool = False,
    ) -> list[Any]:
        """Assemble one full page update.

        Every whole-page transition goes through here, so the order of
        `page_outputs` is defined exactly once. The alternative -- a literal list
        per branch -- makes a silently mis-ordered element far too easy, and
        Gradio only checks the count.
        """
        # The name box keeps whatever the annotator typed or a remembered name,
        # even when no identity was established (bad passphrase, empty name).
        shown_name = annotator if name is None else name
        form = (
            [gr.update(value=None), gr.update(value=None), gr.update(value=3), gr.update(value="")]
            if reset_form
            else [gr.update()] * 4
        )
        return [
            annotator,
            uid,
            selected,
            annotator,  # BrowserState — only a real identity is remembered
            gr.update(value=shown_name),
            gr.update(visible=landing),
            gr.update(visible=not landing and not done),
            gr.update(visible=bool(done), value=done),
            progress_line(store, annotator) if annotator else "",
            *form,
            status,
            *view,
        ]

    def _serve(annotator: str, *, status: str = "") -> list[Any]:
        """Claim the next episode for `annotator` and render the full page."""
        episode = store.claim_next(annotator)
        if episode is None:
            return _page(
                annotator=annotator,
                uid="",
                selected=0,
                status=status,
                reset_form=True,
                done=(
                    "### 🎉 Nothing left to label\n\nEvery episode in the pool has been "
                    f"labelled by everyone available, including you.\n\n{progress_line(store, annotator)}"
                ),
                view=_blank_view(),
            )
        selected = viewer.initial_index(episode)
        return _page(
            annotator=annotator,
            uid=episode.uid,
            selected=selected,
            status=status,
            reset_form=True,
            view=_view_values(viewer.render(episode, selected)),
        )

    def _landing(name: str, message: str) -> list[Any]:
        return _page(
            annotator="",
            uid="",
            selected=0,
            name=name,
            landing=True,
            status=message,
            view=_blank_view(),
        )

    def _episode_or_none(uid: str) -> Episode | None:
        return store.get(uid) if uid else None

    # --- handlers ---------------------------------------------------------

    def on_load(saved_name: str | None, request: gr.Request) -> list[Any]:
        """Resolve the annotator from `?annotator=` or browser storage.

        A personal link (`...?annotator=alice`) skips the prompt entirely. When a
        passphrase is configured the landing gate always runs — the share tunnel
        is unauthenticated, so a remembered name is not evidence of access.
        """
        from_query = (request.query_params.get("annotator") or "") if request else ""
        name = (from_query or saved_name or "").strip()
        if not name or passphrase:
            return _landing(name, "")
        return _serve(name)

    def on_start(name: str, entered: str, saved_name: str) -> list[Any]:
        name = (name or "").strip()
        if not name:
            return _landing(saved_name, "⚠️ Enter a name first — it labels your answers in the study.")
        if passphrase and entered != passphrase:
            return _landing(name, "⚠️ Wrong passphrase.")
        return _serve(name)

    def on_submit(
        annotator: str,
        uid: str,
        selected: int,
        outcome: str | None,
        blame: str | None,
        confidence: int,
        notes: str,
    ) -> list[Any]:
        if not annotator or not uid:
            return _landing("", "⚠️ Session lost — enter your name again.")
        if not outcome or not blame:
            # Nothing is written and the lease is untouched, so the annotator
            # keeps the trajectory they were reading and their partial answers.
            return _page(
                annotator=annotator,
                uid=uid,
                selected=selected,
                status="⚠️ Pick both an outcome and a primary blame.",
                view=[gr.update()] * len(_blank_view()),
            )
        store.submit(
            uid,
            annotator,
            outcome=outcome,
            blame=blame,
            confidence=int(confidence),
            notes=notes or "",
        )
        return _serve(annotator, status=f"✅ {uid} recorded.")

    def on_skip(annotator: str, uid: str) -> list[Any]:
        if not annotator:
            return _landing("", "⚠️ Session lost — enter your name again.")
        if uid:
            store.release(uid, annotator)
        return _serve(annotator, status=f"↷ {uid} returned to the pool." if uid else "")

    def _navigate(uid: str, selected: int, where: str) -> list[Any]:
        episode = _episode_or_none(uid)
        if episode is None:
            return [selected, *_blank_view()]
        new_selected = viewer.step(episode, int(selected or 0), where)
        return [new_selected, *_view_values(viewer.render(episode, new_selected))]

    def on_card_click(clicked: float | None, uid: str, selected: int) -> list[Any]:
        episode = _episode_or_none(uid)
        if episode is None or clicked is None:
            return [selected, *_blank_view()]
        index = max(0, min(int(clicked), max(0, viewer.n_events(episode) - 1)))
        return [index, *_view_values(viewer.render(episode, index))]

    # --- layout -----------------------------------------------------------

    with gr.Blocks(title="CUBE judge validation", theme=gr.themes.Soft(), css=_ANNOTATE_CSS, js=_INIT_JS) as demo:
        annotator_state = gr.State("")
        uid_state = gr.State("")
        sel_state = gr.State(0)
        saved_name = gr.BrowserState("", storage_key="cube-annotate-annotator")

        gr.Markdown("## CUBE — human validation of the trajectory judge")
        progress_md = gr.Markdown("", elem_id="ann_progress")
        status_md = gr.Markdown("", elem_id="ann_status")

        with gr.Column(visible=True, elem_id="ann_landing") as landing_col:
            gr.Markdown(_INSTRUCTIONS)
            name_box = gr.Textbox(label="Your name", placeholder="e.g. alice", max_lines=1, elem_id="ann_name")
            pass_box = gr.Textbox(
                label="Passphrase", type="password", max_lines=1, visible=passphrase is not None, max_length=200
            )
            start_btn = gr.Button("Start labelling", variant="primary")

        done_md = gr.Markdown("", visible=False, elem_id="ann_done")

        with gr.Row(visible=False, equal_height=False, elem_id="ann_work") as work_row:
            # Left: the event-card rail (navigation through the episode).
            with gr.Column(scale=1, min_width=240):
                with gr.Row(elem_id="ann_rail_nav"):
                    first_btn = gr.Button("⤒", size="sm", elem_id="ann_first_btn", min_width=0, scale=0)
                    prev_btn = gr.Button("◀", size="sm", elem_id="ann_prev_btn", min_width=0, scale=0)
                    next_btn = gr.Button("▶", size="sm", elem_id="ann_next_btn", min_width=0, scale=0)
                    last_btn = gr.Button("⤓", size="sm", elem_id="ann_last_btn", min_width=0, scale=0)
                rail_html = gr.HTML(elem_id="xray_rail")

            # Middle: the evidence — identical to what the judge read.
            with gr.Column(scale=3):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=260, variant="panel", elem_classes="compact-header"):
                        header_md = gr.Markdown("")
                        stats_md = gr.Markdown("")
                    with gr.Column(scale=3):
                        goal_html = gr.HTML("")
                with gr.Row(equal_height=True):
                    reasoning_html = gr.HTML("")
                    action_html = gr.HTML("")
                with gr.Tabs():
                    with gr.Tab("Observation"):
                        observation_gallery = gr.Gallery(
                            label="Screenshots",
                            show_label=True,
                            columns=2,
                            height=420,
                            object_fit="contain",
                            visible=False,
                        )
                        observation_html = gr.HTML("")
                    with gr.Tab("Chat"):
                        chat_html = gr.HTML("")
                    with gr.Tab("Evaluation"):
                        evaluation_md = gr.Markdown("")

            # Right: the instrument.
            with gr.Column(scale=2, min_width=280, elem_id="ann_form"):
                gr.Markdown("### Your assessment")
                outcome_radio = gr.Radio(choices=OUTCOMES, label="Outcome", value=None)
                blame_radio = gr.Radio(choices=BLAMES, label="Primary blame", value=None)
                confidence_slider = gr.Slider(1, 5, value=3, step=1, label="Confidence")
                notes_box = gr.Textbox(label="Notes (optional)", lines=3, max_length=4000)
                submit_btn = gr.Button("Submit & next", variant="primary")
                skip_btn = gr.Button("Skip (return to pool)")
                with gr.Accordion("Instructions", open=False):
                    gr.Markdown(_INSTRUCTIONS)

        # Hidden Number the rail's per-card onclick JS writes its index into.
        # `xray_utils._card_onclick` targets this elem_id by name.
        with gr.Row(elem_id="timeline_click_input"):
            card_click = gr.Number(show_label=False, container=False)

        # --- wiring -------------------------------------------------------

        view_outputs = [
            header_md,
            stats_md,
            goal_html,
            rail_html,
            reasoning_html,
            action_html,
            observation_gallery,
            observation_html,
            chat_html,
            evaluation_md,
        ]
        # Order must match `_page`'s return list exactly.
        page_outputs = [
            annotator_state,
            uid_state,
            sel_state,
            saved_name,
            name_box,
            landing_col,
            work_row,
            done_md,
            progress_md,
            outcome_radio,
            blame_radio,
            confidence_slider,
            notes_box,
            status_md,
            *view_outputs,
        ]
        nav_outputs = [sel_state, *view_outputs]

        demo.load(fn=on_load, inputs=[saved_name], outputs=page_outputs)
        start_btn.click(fn=on_start, inputs=[name_box, pass_box, saved_name], outputs=page_outputs)
        name_box.submit(fn=on_start, inputs=[name_box, pass_box, saved_name], outputs=page_outputs)
        submit_btn.click(
            fn=on_submit,
            inputs=[annotator_state, uid_state, sel_state, outcome_radio, blame_radio, confidence_slider, notes_box],
            outputs=page_outputs,
        )
        skip_btn.click(fn=on_skip, inputs=[annotator_state, uid_state], outputs=page_outputs)

        for button, where in ((first_btn, "first"), (prev_btn, "prev"), (next_btn, "next"), (last_btn, "last")):
            button.click(
                fn=lambda uid, selected, where=where: _navigate(uid, selected, where),
                inputs=[uid_state, sel_state],
                outputs=nav_outputs,
            )
        card_click.change(fn=on_card_click, inputs=[card_click, uid_state, sel_state], outputs=nav_outputs)

    return demo
