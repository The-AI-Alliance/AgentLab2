"""Event-stream view model for the XRay viewer.

XRay consumes a trajectory as a **flat, ordered stream** of events
(`LLMCallEvent` / `ToolCallEvent` / `EvaluationEvent` / `AgentErrorEvent`).
There is no "turn" grouping: events stand alone and are linked only by
`ToolCallEvent.parent_event_id` — the id of the `LLMCallEvent` that produced
the call. Selecting one event surfaces its *accompanying* event(s) through
that link:

    observation (ToolCallEvent) -> the LLM response that produced it (parent)
    LLM response (LLMCallEvent) -> the observation(s) it produced (children)

Legacy V1/V2 trajectories are adapted into this same event stream by the
storage loader (`TrajectoryView._step_to_event`), so XRay never sees the old
`EnvironmentOutput | AgentOutput` step shape — backward compatibility lives
entirely in the loader, not here.

This module is Gradio-free and import-light so it stays unit-testable.
"""

from dataclasses import dataclass, field

from cube.core import Action, Observation, StepError

from cube_harness.core import (
    AgentErrorEvent,
    EvaluationEvent,
    LLMCallEvent,
    ToolCallEvent,
    TrajectoryEvent,
)
from cube_harness.llm import LLMCall
from cube_harness.storage import TrajectoryView

# Sentinel `parent_event_id` used by the loader for the initial observation
# (the reset / first env step), which has no originating LLM call.
RESET_PARENT = "__reset__"

# --- Event kinds + card colours -------------------------------------------
# A "kind" is the canonical card category, independent of on-disk layout.

KIND_LLM = "llm"
KIND_OBSERVATION = "observation"
KIND_EVALUATION = "evaluation"
KIND_ERROR = "error"

# Card accent colours, keyed by kind. The *active* card uses the solid accent;
# the *accompanying* card uses the muted accent (see `xray.py` card CSS).
KIND_COLORS: dict[str, str] = {
    KIND_LLM: "#3b82f6",  # blue — agent reasoning / LLM call
    KIND_OBSERVATION: "#10b981",  # green — environment observation
    KIND_EVALUATION: "#a855f7",  # purple — reward / evaluation
    KIND_ERROR: "#ef4444",  # red — agent / framework error
}

KIND_ICONS: dict[str, str] = {
    KIND_LLM: "🧠",
    KIND_OBSERVATION: "🖥️",
    KIND_EVALUATION: "🏁",
    KIND_ERROR: "⚠️",
}


def event_kind(event: TrajectoryEvent) -> str:
    """Canonical card kind for one trajectory event."""
    out = event.output
    if isinstance(out, LLMCallEvent):
        return KIND_ERROR if out.error is not None else KIND_LLM
    if isinstance(out, ToolCallEvent):
        return KIND_ERROR if out.error is not None else KIND_OBSERVATION
    if isinstance(out, EvaluationEvent):
        return KIND_EVALUATION
    if isinstance(out, AgentErrorEvent):
        return KIND_ERROR
    return KIND_OBSERVATION


@dataclass
class EventCard:
    """Display metadata for one event card in the timeline rail.

    `index` is the event's position in the flat stream — the stable id the UI
    uses to select it. `accompanying` are the indices the parent-link pairing
    lights up when this card is selected.
    """

    index: int
    kind: str
    icon: str
    color: str
    title: str
    subtitle: str
    is_error: bool
    accompanying: list[int] = field(default_factory=list)


class EpisodeEvents:
    """Flat, ordered event stream for one episode, with parent-link pairing.

    Wraps a `TrajectoryView` and decodes its events once into a list (XRay
    needs random access for card selection, so lazy per-index decoding buys
    nothing here). All the navigation XRay performs — select a card, find its
    accompanying event(s), render a panel — goes through this object.
    """

    def __init__(self, events: list[TrajectoryEvent]) -> None:
        self.events = events
        # id -> index, for resolving `parent_event_id` to a position.
        self._id_to_index: dict[str, int] = {}
        # parent id -> child indices, for LLM-call -> observation pairing.
        self._children: dict[str, list[int]] = {}
        for i, ev in enumerate(events):
            out = ev.output
            ev_id = getattr(out, "id", None)
            if ev_id is not None:
                self._id_to_index[ev_id] = i
            if isinstance(out, ToolCallEvent):
                self._children.setdefault(out.parent_event_id, []).append(i)

    @classmethod
    def from_view(cls, view: TrajectoryView) -> "EpisodeEvents":
        """Materialize the flat event list from a lazy `TrajectoryView`."""
        return cls(list(view))

    # --- basic access -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, i: int) -> TrajectoryEvent:
        return self.events[i]

    def output(self, i: int):  # -> TrajectoryEventOutput
        """The event payload at index `i` (LLMCallEvent / ToolCallEvent / …)."""
        return self.events[i].output

    # --- parent-link pairing ----------------------------------------------

    def parent_index(self, i: int) -> int | None:
        """Index of the LLM call that produced observation `i`, or None.

        Defined for `ToolCallEvent`s whose `parent_event_id` resolves to an
        LLM call in the stream. Returns None for the reset observation and for
        non-observation events.
        """
        out = self.events[i].output
        if not isinstance(out, ToolCallEvent):
            return None
        if out.parent_event_id == RESET_PARENT:
            return None
        return self._id_to_index.get(out.parent_event_id)

    def child_indices(self, i: int) -> list[int]:
        """Indices of observations produced by the LLM call at `i`.

        These are the parallel siblings of one LLM turn — all `ToolCallEvent`s
        whose `parent_event_id` equals this LLM call's id, in stream order.
        """
        out = self.events[i].output
        if not isinstance(out, LLMCallEvent):
            return []
        return list(self._children.get(out.id, []))

    def accompanying_indices(self, i: int) -> list[int]:
        """The event(s) paired with `i` for joint display.

        Observation -> its parent LLM call. LLM call -> its child
        observation(s). Everything else -> nothing.
        """
        out = self.events[i].output
        if isinstance(out, ToolCallEvent):
            parent = self.parent_index(i)
            return [parent] if parent is not None else []
        if isinstance(out, LLMCallEvent):
            return self.child_indices(i)
        return []

    def resolve_pair(self, i: int) -> tuple[int | None, list[int]]:
        """Return `(llm_index, observation_indices)` for the selection at `i`.

        Normalizes either side of a selection into the same shape so the UI
        can always render an "LLM response" panel beside an "observation"
        panel regardless of which card the user clicked:

          - select an LLM call  -> (i, [its children])
          - select an observation -> (its parent, [i])
          - select anything else  -> (None, [])  (eval / error render alone)
        """
        out = self.events[i].output
        if isinstance(out, LLMCallEvent):
            return i, self.child_indices(i)
        if isinstance(out, ToolCallEvent):
            return self.parent_index(i), [i]
        return None, []

    # --- typed payload extractors (None-safe) -----------------------------

    def llm_call(self, i: int | None) -> LLMCall | None:
        """The `LLMCall` carried by event `i`, if it is an LLM call with one."""
        if i is None:
            return None
        out = self.events[i].output
        return out.call if isinstance(out, LLMCallEvent) else None

    def observation(self, i: int | None) -> Observation | None:
        """The `Observation` carried by event `i`, if it is a tool call."""
        if i is None:
            return None
        out = self.events[i].output
        return out.obs if isinstance(out, ToolCallEvent) else None

    def action(self, i: int | None) -> Action | None:
        """The `Action` dispatched by tool-call event `i`, if any."""
        if i is None:
            return None
        out = self.events[i].output
        return out.action if isinstance(out, ToolCallEvent) else None

    def error(self, i: int | None) -> StepError | None:
        """The `StepError` on event `i`, from whichever field carries it."""
        if i is None:
            return None
        out = self.events[i].output
        return getattr(out, "error", None)

    # --- card rail --------------------------------------------------------

    def cards(self) -> list[EventCard]:
        """One display card per event, in stream order."""
        return [self._card(i) for i in range(len(self.events))]

    def _card(self, i: int) -> EventCard:
        ev = self.events[i]
        out = ev.output
        kind = event_kind(ev)
        is_error = kind == KIND_ERROR
        title, subtitle = _card_text(out, i)
        return EventCard(
            index=i,
            kind=kind,
            icon=KIND_ICONS[kind],
            color=KIND_COLORS[kind],
            title=title,
            subtitle=subtitle,
            is_error=is_error,
            accompanying=self.accompanying_indices(i),
        )


def _card_text(out, index: int) -> tuple[str, str]:
    """`(title, subtitle)` for a card, derived from the event payload."""
    if isinstance(out, LLMCallEvent):
        tag = (out.call.tag if out.call else "") or "LLM call"
        if out.error is not None:
            return "LLM error", _error_preview(out.error)
        sub = ""
        if out.call is not None:
            n_tools = len(out.call.prompt.tools)
            sub = out.call.llm_config.model_name
            if n_tools:
                sub += f" · {n_tools} tools"
        return tag, sub
    if isinstance(out, ToolCallEvent):
        if out.parent_event_id == RESET_PARENT:
            return "Initial observation", "task reset"
        if out.error is not None:
            name = out.action.name if out.action else "tool"
            return f"{name} → error", _error_preview(out.error)
        name = out.action.name if out.action else "observation"
        return name, _action_preview(out.action)
    if isinstance(out, EvaluationEvent):
        scope = "final" if out.is_terminal else "step"
        return f"Evaluation ({scope})", f"reward={out.reward:g}"
    if isinstance(out, AgentErrorEvent):
        return "Agent error", _error_preview(out.error)
    return out.__class__.__name__, ""


def _action_preview(action: Action | None) -> str:
    """Compact one-line `name(arg=…)` preview of a dispatched action."""
    if action is None:
        return ""
    args = action.arguments or {}
    inner = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
    return f"{action.name}({inner})" if inner else action.name


def _error_preview(error: StepError | None) -> str:
    if error is None:
        return ""
    return _short(getattr(error, "error_type", "") or getattr(error, "exception_str", ""), 60)


def _short(value: object, max_len: int = 24) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"
