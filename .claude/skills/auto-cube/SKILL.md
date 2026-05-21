# Auto-CUBE

The **Auto-CUBE** outer loop runs an iterative experiment program to
understand and fix failures in the agent/cube stack. It owns the
methodology; the per-use-case **Investigator** recipes
(`analyze/investigator/use_cases/*` — `general_blame`, `profiling`,
`agent_scaffolding`, `hinter`, …) are the *investigate* step it dispatches
on each trajectory.

Inner loop = one episode's root-cause analysis (an Investigator recipe).
Outer loop = sessions of rounds: hypothesis → experiment → investigate →
synthesize → conclude → next hypothesis.

## The loop

1. **Session start.** Pick an objective (a benchmark + a thing to understand
   or fix). Create `<journal>/<session-slug>/session.md` from the template:
   branch, base commit, objective, round index. `<journal>` is
   `~/cube_auto_cube_journal/` (machine-local, never committed).

2. **Round.** For round `N`, create `<session-slug>/round_<N>/`:
   - `exp_config.py` — a copied recipe. Explicit task list via
     `.subset_from_list([...])`, chosen LLM, tool config. No `# ///` header
     (run it with the repo venv: `.venv/bin/python <path>/exp_config.py`).
   - `notes.md` — **before the run**: the hypothesis and what this round
     tests. **After**: results, what the investigation found, conclusion,
     and the code delta since the previous round (commit hash + one line).

3. **Run** the experiment, then the Investigator on its output dir
   (`ch-investigate <exp_dir> --recipe <use_case>`). Read `meta_analysis.md`
   plus the per-episode `findings.json` / `audit.json`.

4. **Conclude & iterate.** Write the round's conclusion. If a root cause is
   confirmed, decide the fix. Start round `N+1` with the next hypothesis.

5. **Session end** (most bugs in scope found, or strong diminishing
   returns). Write **`<session-slug>/REPORT.md`** from
   `templates/report.md` — the single human-readable rollup someone reads
   top-to-bottom to understand the whole session: scope, the arc across
   rounds, findings ledger with dispositions, shipped vs open PRs, the
   consolidated design signal, key learnings, cost. `session.md` stays
   the live scope+tracker; `REPORT.md` is the durable, compaction-proof
   final summary.

A session = `session.md` (scope/target + round index, set at start) ·
`round_<N>/notes.md` (one per round) · `REPORT.md` (final summary).
(Old-era single-`.md` sessions are already their own report — no
separate file.)

## Intervention discipline

While **finding the root cause**, anything goes: hacky one-line patches,
print statements, a throwaway side experiment, a blunt hint that masks a
bug just to confirm the hypothesis. Speed of understanding wins here.

Once the **root cause is confirmed**, the committed fix follows the
**auto-fix methodology** — full spec:
[`openspec/specs/auto-fix/spec.md`](../../../openspec/specs/auto-fix/spec.md).
In brief:

- **Classify** L0–L3 (local-correct → layer → symptom-of-design → Auto-CUBE /
  Investigator defect). Nothing blocks the loop: L2/L3 still ship a temp PR
  **and** open a kept-open `design-debt` issue + openspec stub.
- **Fix Report** is the PR body (`templates/fix_report.md`): invariant in
  one sentence, root-cause mechanism in plain language, blast-radius grep
  on the *target* branch, the adversarial "opposite user", registry
  lookup, class, regression test that asserts the *invariant* not the
  reproduction.
- **fix-audit** independently tries to break the Fix Report's generalization
  claims; the reviewer reads its verdict, not the diff.
- **Provenance**: `# auto-fix(N)↓ … # /auto-fix(N)` markers + a one-line
  machine-readable footnote at module bottom. `N` is the PR number for
  L0/L1, the design-debt issue number for L2/L3. Accretion is exact
  (`grep -c`); ≥3 in an area ⇒ refactor proposal, not patch N+1.
- **Multi-PR**: every PR branches from `dev` directly (no stacked PRs by
  default); the integration worktree (§6 of the spec) is where all
  in-flight changes co-exist locally for testing.
- The diagnostic hack is reverted; only the principled fix is committed.

A hack that confirmed a hypothesis is a successful experiment, not a
deliverable. The deliverable is the right fix, its Fix Report, and the
journal entry that explains *why*.

## Templates

`templates/session.md`, `templates/notes.md`, `templates/exp_config.py`,
`templates/fix_report.md`, `templates/report.md`. Copy, fill, evolve.
Refine these templates as the methodology matures — this skill is
expected to change.
