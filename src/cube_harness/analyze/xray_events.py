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


@dataclass
class EventGroup:
    """One logical group — the connected component of the dependency graph
    rooted at an LLM call (or the reset observation), with its events bucketed
    by role so the detail panes can render the whole step at once.

    `selected` is the card the user clicked (active); `members` minus it are
    the accompanying cards. The role buckets are stream-ordered indices.
    """

    root: int
    selected: int
    members: list[int]
    llm_index: int | None
    observation_indices: list[int]
    evaluation_indices: list[int]
    error_indices: list[int]


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
        # Logical grouping: the connected component of the parent_event_id
        # dependency graph each event belongs to. A group is rooted at an LLM
        # call (or the reset observation) and gathers the tool call(s) it
        # produced, their step-wise evaluations, and any error in the chain —
        # so selecting any card surfaces the whole "why did the agent do this,
        # what did it observe, what reward, any error" story.
        self._root_of = self._compute_roots()
        self._group_members: dict[int, list[int]] = {}
        for i, root in enumerate(self._root_of):
            self._group_members.setdefault(root, []).append(i)

    def _compute_roots(self) -> list[int]:
        """Map each event index to its group-root index.

        Roots are LLM calls and the reset observation. Tool calls and step-wise
        evaluations resolve to their parent's root by *following parent_event_id
        links* (via the precomputed id→index map), so grouping is independent of
        stream order — a child decoded before its parent still resolves
        correctly. Terminal evaluations and agent errors — which currently lack
        a parent link upstream — attach to the most recent root by stream
        position (a heuristic that becomes exact once the producers stamp
        `parent_event_id` on them).
        """
        root_of: list[int | None] = [None] * len(self.events)
        # Cycle guard: indices currently on the active DFS stack.
        # Using a mutable set + explicit pop (O(N) total) rather than
        # rebuilding a frozenset per frame (O(N²)).
        visiting: set[int] = set()

        def resolve(i: int) -> int:
            """Root of the parent chain at `i` (order-independent, memoized)."""
            if root_of[i] is not None:
                return root_of[i]  # type: ignore[return-value]
            if i in visiting:  # malformed self/cyclic parent link
                root_of[i] = i
                return i
            visiting.add(i)
            out = self.events[i].output
            parent_id: str | None = None
            if isinstance(out, ToolCallEvent) and out.parent_event_id != RESET_PARENT:
                parent_id = out.parent_event_id
            elif isinstance(out, EvaluationEvent) and out.parent_event_id:
                parent_id = out.parent_event_id
            p = self._id_to_index.get(parent_id) if parent_id else None
            root = resolve(p) if p is not None else i
            visiting.discard(i)
            root_of[i] = root
            return root

        # Forward pass: parent-linked events resolve through `resolve`;
        # parent-less terminal evals / agent errors attach to the most recent
        # group root by stream position (becomes exact once producers stamp
        # `parent_event_id` on them).
        last_root: int | None = None
        for i, ev in enumerate(self.events):
            out = ev.output
            if isinstance(out, (EvaluationEvent, AgentErrorEvent)) and not getattr(out, "parent_event_id", None):
                root_of[i] = last_root if last_root is not None else i
            else:
                resolve(i)
            if root_of[i] == i:  # this event opens a new group
                last_root = i
        return [r if r is not None else i for i, r in enumerate(root_of)]

    @classmethod
    def from_view(cls, view: TrajectoryView) -> "EpisodeEvents":
        """Materialize the flat event list from a lazy `TrajectoryView`.

        If the stream carries no terminal `EvaluationEvent` (most trajectories
        record the final reward in `reward_info` metadata rather than as an
        event), synthesize one so the final reward shows as a card at the end.
        """
        events = list(view)
        has_terminal_eval = any(isinstance(ev.output, EvaluationEvent) and ev.output.is_terminal for ev in events)
        reward_info = view.reward_info or {}
        if not has_terminal_eval and reward_info.get("reward") is not None:
            info = {k: v for k, v in reward_info.items() if k != "reward"}
            events.append(
                TrajectoryEvent(
                    output=EvaluationEvent(reward=float(reward_info["reward"]), info=info, is_terminal=True)
                )
            )
        return cls(events)

    # --- basic access -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, i: int) -> TrajectoryEvent:
        return self.events[i]

    def output(self, i: int):  # -> TrajectoryEventOutput
        """The event payload at index `i` (LLMCallEvent / ToolCallEvent / …)."""
        return self.events[i].output

    # --- group navigation (logical step = one group) ----------------------

    def group_roots(self) -> list[int]:
        """Ordered root index of each logical group (one entry per group)."""
        return sorted(set(self._root_of))

    def next_group_root(self, i: int) -> int:
        """Root of the group after the one containing `i` (clamped to the last)."""
        roots = self.group_roots()
        pos = roots.index(self._root_of[i])
        return roots[min(pos + 1, len(roots) - 1)]

    def prev_group_root(self, i: int) -> int:
        """Root of the group before the one containing `i` (clamped to the first)."""
        roots = self.group_roots()
        pos = roots.index(self._root_of[i])
        return roots[max(pos - 1, 0)]

    def first_group_root(self) -> int:
        """Root of the first *step* — the group right after the initial-observation
        group (group_roots()[0] is the task-reset observation). Falls back to the
        very first group when the reset is the only group."""
        roots = self.group_roots()
        return roots[1] if len(roots) > 1 else roots[0]

    def last_group_root(self) -> int:
        """Root of the last group (the end of the episode)."""
        return self.group_roots()[-1]

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

    # --- dependency-graph grouping ----------------------------------------

    def group_members(self, i: int) -> list[int]:
        """All event indices in the logical group containing `i`, in order."""
        return list(self._group_members.get(self._root_of[i], [i]))

    def accompanying_indices(self, i: int) -> list[int]:
        """Other members of `i`'s group — the cards the UI highlights as
        accompanying (muted) when `i` is the active selection."""
        return [m for m in self.group_members(i) if m != i]

    def group_for(self, i: int) -> "EventGroup":
        """The logical group containing event `i`, bucketed by role.

        Gives the detail panes everything to render at once: the LLM
        interaction, the observation(s) it produced, the reward(s), and any
        error — regardless of which card in the group the user selected.
        """
        members = self.group_members(i)
        llm_index: int | None = None
        observations: list[int] = []
        evaluations: list[int] = []
        errors: list[int] = []
        for m in members:
            out = self.events[m].output
            if isinstance(out, LLMCallEvent):
                llm_index = m
                if out.error is not None:
                    errors.append(m)
            elif isinstance(out, ToolCallEvent):
                observations.append(m)
                if out.error is not None:
                    errors.append(m)
            elif isinstance(out, EvaluationEvent):
                evaluations.append(m)
            elif isinstance(out, AgentErrorEvent):
                errors.append(m)
        return EventGroup(
            root=self._root_of[i],
            selected=i,
            members=members,
            llm_index=llm_index,
            observation_indices=observations,
            evaluation_indices=evaluations,
            error_indices=errors,
        )

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
        # Title clearly marks this as the LLM turn; the call's tag (e.g. "act",
        # "summary") is a secondary detail in the subtitle so it isn't mistaken
        # for an action.
        tag = out.call.tag if out.call else ""
        title = f"LLM call · {tag}" if tag else "LLM call"
        if out.error is not None:
            return "LLM error", _error_preview(out.error)
        sub = ""
        if out.call is not None:
            n_tools = len(out.call.prompt.tools)
            sub = out.call.llm_config.model_name
            if n_tools:
                sub += f" · {n_tools} tools"
        return title, sub
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
