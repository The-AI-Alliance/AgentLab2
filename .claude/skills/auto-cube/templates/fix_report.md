# Fix Report — auto-fix(<N>)

> This is the fix PR body. Spec: `openspec/specs/auto-fix/spec.md` §2.
> Aim for one tight page; structure + pedagogical depth where it earns
> its keep. The fix-audit will try to break the claims below.

**Class**: <L0 | L1 | L2 | L3>  ·  **Anchor**: `PR#<N>` (L0/L1) or `issue#<N>` (L2/L3)
**Context fingerprint**: `<infra>/<arch>/<model>/<cube@commit>`
**Design-debt issue / openspec** (L2/L3 only): <link>

## Invariant violated
<One plain-English sentence: the rule the code must uphold. A reader needs
no other context to grasp it. NOT "task X failed".>

## Root cause
<One short paragraph or 3–4 bullets explaining the **mechanism** — what's
happening under the hood, not "added X". This is the one section that
earns pedagogical depth. State the triggering context in prose: the
task/benchmark that surfaced the bug + the env specifics (OS, infra,
model, input shape) you assess as relevant to *this* bug.>

## Why this fix / why this layer
<Bullets, 1 sentence each. Why this change resolves the invariant. Why
this is the right layer (cube vs cube-standard vs scaffold vs eval).
Defect's contract in a different layer than the fix → wrong layer →
escalate.>

## Blast radius
<Telegraphic — file:line list, paste the grep run on the PR's TARGET
branch (`git grep … origin/dev`), not the local checkout. No prose.>

## Adversarial: the opposite user
<One short paragraph or 2 bullets. Construct the downstream context
(cube / infra / model) this fix could HURT. Rule it out with evidence,
or escalate the class.>

## Registry lookup
<Bullets: existing `auto-fix` footnotes in/near this area + their `ctx=`
fingerprints. Is this the Nth patch here? If ≥3 → refactor proposal,
not patch N+1.>

## Test
<One bullet per assertion + a one-line concrete witness. Assertions pin
the **invariant**, not the reproduction.>

---

## Example (good)

```markdown
# Fix Report — auto-fix(413)

**Class**: L1  ·  **Anchor**: `PR#413`
**Context fingerprint**: `anthropic-beta/interleaved-thinking-2025-05-14/claude-sonnet-4-6`

## Invariant violated
Once a turn opens an extended-thinking block, the assistant must not re-open
thinking on subsequent tool-result continuations when the
interleaved-thinking beta is `off`.

## Root cause
The cadence probe appended a trailing user message after each tool_result.
Anthropic's API treats any user-content suffix as a fresh turn, which makes
thinking eligible to reopen regardless of the beta. With the trailing
message present, `off` and `once` produced identical token counts — the
probe was measuring fresh-turn behavior, not the beta's actual effect.

The hidden contract is positional: the beta gates only the *continuation*
case (tool_result followed by no user content). Add anything after the
tool_result and the gating is bypassed.

## Why this fix / why this layer
- Drop the trailing user message; let the probe issue pure tool-result
  continuations — the only call shape that exercises the beta.
- Lives in the probe (smoke), not the SDK wrapper — the wrapper is
  generic; the probe is what asserts the beta's semantics.

## Blast radius
```
$ git grep -n 'add_user_message' origin/dev -- scripts/smoke/
scripts/smoke/probe_interleaved_thinking.py:84
```
One call site; no other smoke calls the probe.

## Adversarial: the opposite user
Downstream users issuing multi-turn chats where each turn ends with a user
message (the normal chat shape) are unaffected — the probe is specifically
a no-trailing-user shape; ordinary chats keep their semantics.

## Registry lookup
- `auto-fix-note(408)` in `scripts/smoke/cadence_probe.py`
  (`ctx=anthropic-beta/cache/sonnet-4-6`) — sibling beta-cadence probe,
  same call-shape pitfall, different gating. Two related auto-fixes in
  this area; one more would trip the refactor threshold.

## Test
- `tests/test_thinking_probe.py::test_off_no_reopen` asserts
  `(thinking_tokens, thinking_blocks) == (0, 0)` for `off`.
- Witness from a green run: `off=(0,0) once=(254,0) always=(253,18)` —
  off has zero; once has the initial system-anchored 254 tokens only (no
  reopen); always has the initial plus a 253-token reopen on tool_result.
```

## Anti-example (what NOT to write)

```markdown
# Fix — interleaved thinking probe

- Bug: probe broken
- Fix: removed user msg
- Repro: see PR
- Class: L1
- Test: added test_thinking_probe

Was reading wrong fields. Now works.
```

Why this anti-example fails:
- "Bug: probe broken" tells the reader nothing about the invariant.
- "Removed user msg" describes the *change*, not the *mechanism* — the
  reader can't tell why removing a message fixed anything.
- No blast radius, no adversarial, no registry lookup.
- "Now works" is unfalsifiable; the fix-audit can't break a claim that
  was never made.

The Fix Report's job is to *anticipate* the fix-audit and the reviewer's
skepticism. A jargon-dense bullet list makes that work impossible — the
reader can't onboard without external context.
