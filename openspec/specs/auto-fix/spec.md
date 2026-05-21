# Auto-CUBE fix methodology & auto-fix provenance

Authoritative spec for fixes produced by the **Auto-CUBE** autonomous loop
(the orchestrator that runs cube benchmarks, analyzes trajectories via the
**Investigator**, and ships fixes as PRs). Goal: let a reviewer trust most
patches *without* a deep contextual review, surface design rot instead of
accreting band-aids, and keep many in-flight PRs co-existing as the loop's
output rate grows.

Two distinct agent roles to keep straight:

- **Auto-CUBE** — outer loop: runs experiments, invokes Investigator per
  trajectory, proposes fixes, opens PRs. This spec governs its outputs.
- **Investigator** — inner sub-agent: takes one trajectory and produces a
  diagnostic report. Lives at `analyze/investigator/` (`ch-investigate`
  CLI). Distinct from the fix-authoring step.

Auto-CUBE has no live feedback channel to other users. A fix it ships can't
be validated by "does this help other cubes?" — so it must carry its own
generalization proof, an adversarial audit of that proof, and a durable,
context-stamped provenance record so a future agent (or different-context
run) can see what was patched, why, and where it may not generalize.

## 1. Depth taxonomy (classify every fix)

| Class | Meaning | Policy |
|---|---|---|
| **L0** local-correct | restores an invariant/symmetry that already exists elsewhere; no contract change | trust-by-default |
| **L1** layer fix | correct but touches a shared layer/call site | trust if fix-audit clears blast-radius |
| **L2** symptom-of-design | the bug is a symptom; the patch is a band-aid | ship temp PR **and** open a kept-open `design-debt` issue + openspec stub |
| **L3** Auto-CUBE / Investigator defect | the defect is in the methodology or analysis layer itself, not the cube | ship + flag; do not patch the cube |

**Nothing blocks the loop.** L2/L3 still ship a temporary PR. The human's
queue is the `design-debt` backlog, not individual patches.

### When NOT to mark (proportionality)

Markers exist for code that can silently rot. **Skip in-code markers** when
**all** hold: class L0; pure text/comment/docs; low rot-risk (not logic a
later edit could silently break); and already pinned by an
invariant-asserting regression test. Marking such a fix is pollution with
~zero rot/accretion value — the over-application this methodology exists
to prevent, turned on itself.

## 2. The Fix Report (every fix PR carries its own proof)

Template: `.claude/skills/auto-cube/templates/fix_report.md`. It is the
PR body.

**Structure: named subsections + bullets within.** A skimmer reads
section headers + the invariant + the empirical witness in ~30 s and
gets the gist; a careful reviewer reads root-cause + adversarial in
~2 min. Aim for **one tight page total**.

| Section | Style | Why |
|---|---|---|
| **Invariant violated** | 1 plain-English sentence | Anchors the whole report; reader needs no other context to grasp it |
| **Root cause** | 1 short paragraph or 3–4 bullets explaining the **mechanism** | The one section that earns its words — explain *under the hood*, not "added X" |
| **Why this fix / why this layer** | Bullets, 1 sentence each | Easy to skim; flags layering arguments |
| **Blast radius** | Telegraphic — file:line list, grep run on the PR's target branch | Reference material; no prose needed |
| **Adversarial "opposite user"** | 1 short paragraph or 2 bullets | Judgment, not facts → prose helps |
| **Registry lookup** | Bullet list of nearby `auto-fix` footnotes + their `ctx=` fingerprints | Loop's substitute for cross-user feedback |
| **Class** + (for L2/L3) **design-debt issue link** | Single line | Lifecycle dispatch (§5) |
| **Test** | 1 bullet per assertion + a one-line concrete witness (e.g. `off=(0,0) once=(254,0) always=(253,18)`) | Skimmable + tangible — assertions pin the *invariant*, not the reproduction |

The mandate is **structure + pedagogical depth where it earns its
keep**, not exhaustiveness. Jargon-dense bullet lists are an
anti-pattern: the reader can't onboard without external context. The
root-cause and adversarial sections in particular should explain
*mechanism* in plain language. Example + anti-example live in
`.claude/skills/auto-cube/templates/fix_report.md`.

A PR may bundle several independent fixes; the body then carries **one
Fix Report section per marker**.

## 3. The fix-audit

An independent agent gets the diff + Fix Report and is instructed to
**break the generalization claims** (find an unchecked call site,
construct the hurt downstream user, argue the fix is shallow). The
reviewer reads the audit verdict, not the raw diff. Recipe:
`analyze/investigator/use_cases/fix_audit/` (mirrors the audit pass).

## 4. In-code provenance

### Marker (one line each, minimal pollution)

```python
# auto-fix(345)↓
<changed code>
# /auto-fix(345)
```

`345` is a **durable GitHub number** anchoring the fix:
- **L0/L1**: the **PR number** itself (closed/merged PRs persist on
  GitHub; that's anchor enough; no separate tracking issue needed — see §5).
- **L2/L3**: the **design-debt issue number** (issue outlives the PR;
  it's the standing backlog item).

GitHub issues and PRs share a number namespace, so `repo#N` resolves
uniquely in either form — Tier-1 lint can verify `N` exists without
caring which kind.

A marker delimits **one code region = one fix**, not one finding: a single
consolidated change that resolves several findings is *one* `auto-fix(N)`.

### Footnote (module bottom — machine substrate, one line per fix)

```python
# === auto-fix notes ===
# auto-fix-note(345) {class=L1 anchor=PR#345 hash=ab12cd34 ctx=colima/arm64/gpt-5.4-mini/cube@285e0cc}
```

The footnote is **machine-readable provenance only** — what Tier-1 lint
needs and what a different-context run scans to find related fixes.
Prose rationale (symptoms, invariant, why, tested) lives in the PR or
design-debt issue body; duplicating it in code creates a second source
of truth that silently rots.

Fields:
- `anchor=PR#N` (L0/L1) or `anchor=issue#N` (L2/L3) — explicit so
  readers don't have to infer the kind.
- `hash=` — content hash of the marked region; rewritten by lint on
  acknowledged drift, never hand-edited.
- `ctx=` — deterministic context fingerprint
  (infra/arch/model/cube-commit). Contextual to the bug: a docker-socket
  fix names backend/OS; a parsing fix names task/input-shape; a
  model-behaviour fix names model/task. Enough for a different-context
  run to assess whether the fix generalises — not an exhaustive
  environment dump.

## 5. Lifecycle (PR-only by default; issue only for design-debt)

**L0/L1 do not file a tracking issue.** The PR *is* the tracking item —
its number is durable, its body holds the Fix Report, the merged-PR
record on GitHub *is* the archive.

**L0/L1 (the common case):**
1. Open PR with placeholder marker `auto-fix(PENDING)` and the Fix Report
   as the body. Push.
2. Read back the PR number. Amend marker → `auto-fix(N)` (where N is the
   PR number), force-push. (One-time chicken-egg; ~10 s of automation.)
3. On merge: nothing else to do — the PR's `Closed/Merged` state is the
   archive. No issue to close.

Degraded mode: if `gh` is unavailable when writing the marker, leave
`auto-fix(LOCAL-<uuid>)`; a reconciliation pass upgrades it to a real
`auto-fix(N)` when the PR is opened.

**L2/L3 (band-aid + design-debt):**
1. Open `design-debt` issue → get `N`. Body = full Fix Report + the
   long-form *design* context (what makes this L2/L3; the openspec stub
   link).
2. Write `auto-fix(N)` markers (N = issue number); open PR referencing
   `N`. PR body has the Fix Report (same shape as L0/L1).
3. On merge: **issue stays open** (`design-debt` label). It's the backlog
   item the human reviews. Only the PR closes.
4. **Consolidation:** when a refactor promotes band-aids to permanent
   design, it deletes the marked regions + footnotes and the refactor
   PR/openspec cites `supersedes auto-fix N, M, …`; the design-debt
   issues become the audit trail of what the refactor subsumed.

**`Closes` keyword — per-issue line.** A bare `Closes #a #b #c` only
auto-closes the *first* number — GitHub limitation, not Auto-CUBE's.
Use one closing line per issue: `Closes #174\nCloses #175\n…`. (L0/L1
PR-only flow doesn't need closing keywords at all.)

## 6. Multi-PR working substrate (integration worktree)

As the open-PR queue grows, Auto-CUBE needs a place where **all
in-flight changes co-exist** — so a new fix is developed against the
realistic future state of `dev`, not against a stale snapshot. The
methodology mandates the substrate; the agent maintains it.

**Rule: every PR branches from `dev` directly (no stacked PRs).** Each
PR stays independently reviewable, shippable, and mergeable in any
order. Stacking couples review order to dependency order and
multiplies rebase churn on every base-branch merge — we hit this with
small PR queues already; it does not scale.

**Exception (declared): stack when B's *committed source* needs A's
new public surface.** The integration worktree gives B's test
environment A's code, but B's branch still won't compile against `dev`
if it imports A's new symbols. In that case B branches from A and
writes `Stacked-on: PR#A` in its body. On A's merge, Auto-CUBE rebases
B onto `dev`. On A's close-without-merge, B is orphaned: rebase onto
`dev` (may break) or close with a note on A. The `Stacked-on:`
declaration is friction-by-design — when in doubt, don't stack;
duplicate the small utility into B instead.

**Integration worktree** (local-only, never pushed, regenerable):

```bash
# pseudo-code: rebuild on demand and after every base-branch merge
git fetch origin
git checkout -B local/integration origin/dev
for pr in $(open_prs_by_auto_cube):
    git merge --no-ff origin/<pr_head_ref>     # if conflicts: abort,
                                                # flag the pair, continue
```

- **New Auto-CUBE work** happens on a fresh branch off `origin/dev`
  (clean PR-shaped diff). `make test`, `make lint`, and smokes run in
  **integration** to validate that the new change co-exists with the
  rest of the in-flight bag.
- **Conflict in integration = early warning.** If PR-A and PR-B can't
  both apply, Auto-CUBE either (a) auto-rebases the offender for
  mechanical conflicts (footnote merges, import-order diffs — same
  decision table the `pr-cleanup` skill uses today), or (b) surfaces
  the incompatibility to the human with concrete file ranges + an
  `Incompatible-with: #N` note added to both PRs' Fix Reports.
- **One venv per integration worktree.** The worktree owns its own
  `.venv` (`make install` at session start). Sharing a venv across
  worktrees creates a race on the editable-install pointer; per-worktree
  isolation is cheap and rules out the whole class.
- **Cross-repo** (cube-standard ↔ cube-harness): the same pattern applies
  end-to-end via the existing `Depends-on:` declaration in PR bodies.

### Parallel Auto-CUBE sessions

The integration worktree is **per session**, not machine-wide: multiple
Auto-CUBE sessions can run concurrently on one machine, each owning its
worktree + `.venv` + journal subdir
(`~/cube_auto_cube_journal/<session-slug>/`). Cross-session isolation is
preserved by picking **orthogonal scopes** — different cubes by default
(one session on tbench2, one on swe-bench, etc.). Sessions may still file
fixes against shared layers (infra, tool, LLM wrapper); those PRs land in
the same queue and conflicts are resolved **at merge time**, not
prevented in real-time. The `Incompatible-with: #N` note on each PR is
the cross-session signal the human (or the merge step) uses to sequence
them.

## 7. Rot detection — two tiers

**Tier 1 — deterministic lint (`scripts/auto_fix_lint.py`, CI):**

- markers balanced; no nesting/boundary mismatch; close-after-open
- every inline `N` has a footnote stanza and vice-versa (no orphans)
- `N` unique within the repo; footnote machine-line schema valid;
  `repo#N` resolves on GitHub with the right kind+state (PR vs issue,
  open vs closed) consistent with the marker's anchor
- **content-hash drift**: hash of the code **between** the markers —
  *excluding* the `# auto-fix(N)` / `# /auto-fix(N)` lines themselves,
  whitespace-normalized — vs the footnote's `hash=`; mismatch ⇒
  protected code moved. Re-indenting/formatting a marker never trips
  drift.

**Tier 2 — semantic (fix-audit / `/cube-review`, advisory):** is the
changed block still correct / still needed / now subsumed? Lint flags;
an agent decides. Lint never rules on semantics.

### Acknowledgement, not prevention

Hash drift is **not** a hard CI fail — that makes auto-fix blocks
radioactive and people delete markers to go green. The failure condition
is *unacknowledged* drift. Resolve by either: (a) editor updates the
footnote + re-hash (acknowledges they preserved the fix), (b) route to
fix-audit, or (c) remove the marker **with a closing note on the
anchor** (PR or design-debt issue). CI flags; a human/agent must
acknowledge.

## 8. Accretion → refactor

Density is exact: `grep -c 'auto-fix(' <area>`. When an area crosses
the band-aid threshold (≥3 distinct anchors), Auto-CUBE **must** emit an
openspec refactor proposal instead of patch N+1. Patch accretion is a
design signal, not a steady state.

## 9. Human-authorized merge

Auto-CUBE's responsibility ends at **green PR + Fix Report**. Merging
is a human action. The spec does not authorize auto-merge; a future
Auto-CUBE run must not infer authorization from a clean Fix Report
alone. The integration worktree (§6) gives the human a fully-resolved
local substrate before approval; that's the substrate, not the trigger.
