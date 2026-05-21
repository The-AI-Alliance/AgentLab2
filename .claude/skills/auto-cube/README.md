# Using Auto-CUBE

**Auto-CUBE** is the iterate-and-fix outer loop: you start a session
with a target (a cube + the thing to understand or fix), Auto-CUBE runs
experiments under varying scaffolds and models, dispatches the
**Investigator** sub-agent per trajectory, classifies failures, and
ships Fix Report PRs against the auto-fix methodology. The agent-facing
description of the loop is in [SKILL.md](SKILL.md); this README is the
human entry point.

![Auto-CUBE outer loop: dispatch → per-experiment Investigator on each trajectory → analysis → interventions](diagram.png)

Companion to [`/new-cube`](https://github.com/The-AI-Alliance/cube-standard/tree/main/.claude/skills/new-cube)
(scaffold a cube from scratch) and [`/review-cube`](https://github.com/The-AI-Alliance/cube-standard/tree/main/.claude/skills/review-cube)
(audit against invariants before submission). The workflow is
**scaffold → audit → iterate**. Once a cube passes `cube test`,
Auto-CUBE is what finds everything that only surfaces when real LLMs
touch the benchmark.

## When to use Auto-CUBE

- A cube passes `cube test` (debug suite green) but fails for real LLMs;
  you need to know whether it's the scaffold, the infra, the model, or
  the benchmark itself.
- You're hardening an existing cube against a new infra backend (toolkit,
  Daytona, AWS, Modal, …).
- You're hunting design rot in `cube_harness`: the same bug keeps
  reappearing in different cubes' fix queues.
- You want a multi-day diagnostic run with paper-trail provenance:
  `REPORT.md` + Fix Report PRs + design-debt issues.

Don't use Auto-CUBE for a single one-shot fix, a quick test of one
model on one task, or anything that finishes in under an hour of focused
human work — the methodology overhead doesn't pay off at that scale.

## Setup

In a fresh Claude Code session with cube-harness as the working
directory:

1. **cube-harness on `dev`**, latest pulled.
2. **Env vars** (export in shell or `.env`):
   - `ANTHROPIC_API_KEY` — Investigator + Genny when using `claude-*` models
   - `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` — for `azure/gpt-*`
   - Cube-specific infra: `DAYTONA_API_KEY`, `EAI_PROFILE`, AWS, etc.
3. **Journal path**: `~/cube_auto_cube_journal/` is created on first run;
   sessions live in `<journal>/<session-slug>/`.
4. **Running parallel sessions on the same machine**: each session needs
   its own integration worktree + `.venv` + journal subdir. See
   [§6 of the methodology spec](../../../openspec/specs/auto-fix/spec.md#6-multi-pr-working-substrate-integration-worktree).

## Prompt template

Drop this into a fresh Claude Code session (cube-harness as cwd; the
Auto-CUBE skill auto-loads from this directory). Fill the angle-bracket
slots:

> Use the Auto-CUBE skill to **<objective>**: get `<cube-name>` working
> end-to-end on **<infra-A>** and **<infra-B>**. Start a session in
> `~/cube_auto_cube_journal/` with slug `<cube>-<focus>-r0`. Run small
> rounds (3–5 tasks per round) sweeping `<model-A>` × `<model-B>`
> against a couple of `GENNY_CONFIGS` (`swe`, plus one other).
> Dispatch the Investigator each round; classify failures (infra vs
> scaffold vs model vs **benchmark**); file Fix Reports per
> `openspec/specs/auto-fix/spec.md`. Per-round budget ≈ `$<X>`.
> Stop when every (infra × model × config) cell has a green episode or
> independent failure modes are exhausted. Final deliverable:
> `REPORT.md`.

Adjust the prompt to your scope: a wider sweep (more models, more
configs) or a narrower one (one infra, one model, focus on a specific
failure mode). The agent reads SKILL.md and follows the loop:
hypothesis → experiment → Investigator → conclude → next hypothesis.

## What you get back

| Artefact | Where | Purpose |
|---|---|---|
| `REPORT.md` | `~/cube_auto_cube_journal/<slug>/REPORT.md` | Single human-readable rollup: scope, arc, findings ledger, shipped/open PRs, design signals, cost |
| `session.md` | same dir | Live scope + tracker (lighter than REPORT.md) |
| `round_<N>/notes.md` | same dir | Per-round hypothesis → result trail |
| Fix Report PRs | cube-harness (or cube-standard) | One PR per fix; body matches `templates/fix_report.md` |
| `design-debt` issues | cube-harness | L2/L3 fixes that need refactoring; stay open across PRs |
| `meta_analysis.{json,md}` | each experiment dir + journal mirror | Investigator's structured per-batch synthesis |

## Cost & runtime

Rough order, adjust per model and infra.

- One round = 3–5 episodes × Genny (LLM) + 1× Investigator per
  trajectory + 1× Fix Report per fix shipped.
- With `claude-haiku-4-5` + cheap infra: ~$3–10 per round.
- With `azure/gpt-5.4-mini`: ~$5–15 per round.
- Session = 3–10 rounds typically, 1–3 days wall clock with reviews.

Set `cost_limit` in the experiment recipe before running. The
Investigator + fix-audit dominate per-trajectory cost (~$0.05–0.20
each); cap them if you're sweeping wide.

## Parallel sessions

You can run two or more Auto-CUBE sessions on one machine. Pick
**orthogonal cubes** (one tbench2, one swe-bench, etc.) so PRs naturally
land in different paths. Each session = its own worktree + `.venv` +
journal subdir. Cross-session conflicts on shared layers (infra, tool,
LLM wrapper) are resolved **at merge time** via the
`Incompatible-with: #N` note in the PR body — not prevented in
real-time. Full pattern in
[§6 of the methodology spec](../../../openspec/specs/auto-fix/spec.md#6-multi-pr-working-substrate-integration-worktree).

## Related

- [auto-fix methodology spec](../../../openspec/specs/auto-fix/spec.md)
  — what fixes look like, depth taxonomy (L0–L3), provenance
- [SKILL.md](SKILL.md) — the agent's description of the loop
- [Investigator use cases](../../../src/cube_harness/analyze/investigator/use_cases/)
  — the per-trajectory recipes Auto-CUBE dispatches
- [`/new-cube`](https://github.com/The-AI-Alliance/cube-standard/tree/main/.claude/skills/new-cube)
  — scaffold a new cube
- [`/review-cube`](https://github.com/The-AI-Alliance/cube-standard/tree/main/.claude/skills/review-cube)
  — self-audit before registry submission
