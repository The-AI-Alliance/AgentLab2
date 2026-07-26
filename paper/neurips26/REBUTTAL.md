# CUBE — rebuttal drafts (NeurIPS 2026 D&B)

Status legend for the author team:
**[DONE]** measured, in the revised PDF · **[RUNNING]** launched, lands before the deadline ·
**[NEEDS YOU]** requires cluster time or annotator time that only you can supply.

---

## Global response (post once, then reference from each reply)

We thank all three reviewers. The reviews converge on one substantive charge, and we
agree with it: the submission argued wrapper faithfulness with the wrapper's own judge,
which is circular, and validated that judge only for self-consistency. We have replaced
both arguments with direct measurements, and we summarise them here because they answer
the Metareview's two "critical for publication" points.

**1. Wrapper faithfulness is now measured directly, not inferred (Metareview 1; R1; R3-W2/Q1).**
The revised §7.2 reports a four-layer parity protocol, all of it reproducible from
`scripts/paper/`:

- *Task-set identity.* Every shipped task id and instruction string is compared against
  the upstream index at the pinned commit. OSWorld: **368/368** ids, **368/368**
  instructions byte-identical. Windows Agent Arena: **152/152** and **152/152**.
- *Evaluation provenance.* Three CUBEs delegate scoring to the original authors' own
  published package (`browsergym-workarena`, `miniwob`, `webarena-verified`), so the
  evaluator is upstream's by construction. Three run the benchmark's own container image
  and test commands (official `swebench/sweb.eval.*` images; Terminal-Bench per-task
  images). The two CUA CUBEs vendor upstream evaluator source, which we now audit
  function-by-function at the AST level, normalising edits that cannot change a score
  (docstrings, logging, annotations, lint renames): **90.3 %** of 236 OSWorld evaluator
  functions and **74.5 %** of 274 WAA functions are semantically identical, with 13 and 23
  respectively flagged as possibly score-affecting and every residual diff released.
- *Oracle parity.* Each SWE CUBE ships a scripted, LLM-free gold-patch agent;
  Terminal-Bench exposes the same check via upstream's own `solve.sh`. On SWE-bench Live's
  300-task lite split, **275** resolve under the CUBE evaluator; an earlier run on
  non-root infrastructure resolved 223, so ~62 tasks turn on container permissions rather
  than evaluation semantics. This is exactly the sort of finding the check exists for, and
  it is why we evaluate agents on the gold-solvable subset.
- *Cross-harness verdict agreement.* Episodes now record the agent's model patch, so the
  identical patches can be re-graded by the upstream SWE-bench harness and the verdicts
  compared task by task. We also now state plainly the three deliberate relaxations our
  SWE evaluator makes against upstream (pre-existing `PASS_TO_PASS` failures are not
  charged to the agent; a fixed list of network-dependent tests is skipped offline;
  pytest's "no tests collected" is not a failure where upstream's task data carries
  untokenisable test ids). Each avoids scoring a correct fix as 0 for reasons outside the
  agent's control, and each makes our SWE numbers a possible upper bound relative to the
  official harness. We would rather publish that than assume it away.

We also adopt R3's implicit standard: following Harbor, the registry contract now carries
a **parity record** (upstream commit, provenance class, oracle result), and the compliance
pipeline withholds a badge when an oracle run regresses.

**2. The judge is now validated against humans and across model families (Metareview 2; R2-Q1/Q2; R3-W3/W4/Q2/Q3).**
We accept that inter-judge agreement measures consistency, not correctness. The revised
Appendix G adds two checks and ships the tooling for both. For human validation we sample
failed episodes stratified over (modality × judge blame) with a floor per cell so that the
rare benchmark-side categories are estimable, and annotators label them blind, from the
same transcript, on the same closed taxonomy, without sight of the judge's verdict. We
report Cohen's κ against human consensus on the 10-way taxonomy and on the
agent/tool/benchmark grouping, plus inter-annotator κ as the achievable ceiling. For
backbone independence we implemented a third investigator driver (`CodexDriver`) that runs
the identical recipe and schema on a non-Anthropic model, so the same episodes can be
re-judged and both per-episode agreement and the aggregate distribution shift reported.

**3. Scaffold sensitivity (Metareview 3; R3-W1/Q4). [DONE]** We ran a second, structurally
different public scaffold (ReAct, a think/act transcript loop) against Genny
(rolling-summary context management) over all 125 MiniWoB tasks, same model
(GPT-5.4-mini), same BrowserGym action set. Aggregates are close, **66.1 %** (Genny) vs
**69.4 %** (ReAct) on the 124 tasks both completed, a **+3.2 pp** difference that a paired
exact McNemar test does not distinguish from zero (*p* = 0.42). The task-level view is more
informative: the scaffolds reach the **same** outcome on **88.7 %** of tasks (Spearman
ρ = 0.74), with 14 discordant tasks split 5 to 9. So roughly one MiniWoB task in nine is
decided by the scaffold rather than the model, while the benchmark-level reading is stable.
We read this as evidence that holding the tool stack constant is what keeps scaffold
variance small (the loops differ in context management but act through an identical action
space), and we have rewritten the results text to say that our numbers are platform
measurements of one configuration and should not be used to support conclusions resting on
gaps of a few points. New §7.5; comparison tooling in `scripts/paper/compare_runs.py`.

**4. Quantitative framework comparison (Metareview 4; R2-Q3; R3-W6).** New Table 1
compares CUBE against Harbor, NeMo Gym, OpenEnv, AgentBeats, Inspect AI, the METR Task
Standard, BrowserGym, and AgentGym on the axes that decide which benchmarks a framework
can host, plus the wrapping unit each asks an author to produce. We also quantify our own
wrapping cost: median **508 SLOC** of authored wrapper per CUBE (range 253–798) plus a
33–144 SLOC debug suite, measured across the corpus.

---

## Reviewer 1 (rating 3, confidence 5)

> My main concern is that I did not see results to validate the pipeline. Terminal-Bench
> GPT-5.4 achieves 75 % in the official report but 11.4 % with their agent.

This is the most important thing for us to answer, and we should have answered it in the
submission. Two separate claims are involved and we now separate them.

*Is the wrapper faithful?* That is now measured directly rather than inferred; see the
global response, point 1. For Terminal-Bench specifically, the CUBE task set is upstream's
89 tasks, run in upstream's own per-task container images with upstream's own test
commands, and upstream's `solve.sh` reference solutions are runnable through
`oracle_mode` as an end-to-end check of provisioning, tooling and scoring.

*Why is the number low?* Because it measures a different thing, and we have made that
explicit rather than leaving it to a caveat sentence. Table 2 reports one generalist agent
under a fixed configuration: a single prompt, one tool stack per modality, no
per-benchmark tuning, a 150-step cap, and a hard budget of **$1 per task** for small models
and **$3** for large ones, with budget exhaustion counted as failure. Published
Terminal-Bench numbers come from benchmark-specific scaffolds (Terminus, Claude Code) with
no comparable budget ceiling; on long-horizon terminal tasks that ceiling binds hard, and
the resulting number is a statement about the generalist-under-budget configuration, not
about the model's ceiling or about the wrapper. The value of the table is that every cell
is comparable to every other cell, which per-benchmark leaderboards are not. We have
rewritten the surrounding text so a reader cannot mistake it for a SOTA claim, and we
report the budget-exhaustion rate per benchmark so the ceiling's effect is visible.
**[NEEDS YOU]** A budget-lift ablation on Terminal-Bench (same agent, cap raised to $10)
is the cleanest single demonstration that the gap is the configuration, not the wrapper.

> Why is the agent abstraction the right one? The JSON-RPC interface is a differentiating
> factor and deserves more space.

We agree, and we have moved the interface material out of the appendix. The design claim
is not that our agent is better; it is that the *action space is a first-class,
replaceable object*, which is what makes tools an independent variable rather than a
property of whichever scaffold ran. That is what lets one browser tool drive WebArena,
WorkArena and MiniWoB, and it is what produced the MiniWoB finding in §7.4: roughly 65 %
of MiniWoB failures are `action_space_limited` or `insufficient_observation` because the
BrowserGym `bid` action set cannot express the pixel-level interactions ~30 % of MiniWoB
tasks need. That is a *tool* result, obtainable only because the tool is swappable while
the environment is held constant.

> Q: What is the right abstraction for tool calling across agents, MCP vs bash?

We do not think there is a single right answer, and the standard is deliberately neutral:
a tool is a typed object whose schema is derived from the Python signature, and the same
tool is exposed both as a native action set and, through the auto-generated JSON-RPC
server, as MCP `tools/list` / `tools/call`. Empirically the two are suited to different
regimes. A bash tool is the better abstraction where the environment is a shell and the
action space is effectively unbounded (SWE, terminal): our SWE agent uses growing history
because tool results carry the state. Structured, schema-constrained actions win where the
observation is a UI and the action must reference a specific element, since an unbounded
string action space makes grounding errors unrecoverable. The interesting consequence of
making this an experimental variable rather than a design commitment is that the same
benchmark can be run under both, which is what the OSWorld PyAutoGUI vs Computer13 rows in
Table 2 already do: the tool choice moves the score by up to 29 points on the same tasks
and the same model.

> Q: Do the authors see reward hacking as a common failure?

The judge schema has carried a `success_lucky` outcome from the start, precisely for
"reward obtained without solving the task", alongside `should_have_been_rewarded` for the
converse. We had not reported the rate, which was an omission; the revised Appendix reports
`success_lucky` per benchmark. **[NEEDS YOU]** aggregate the rate from the existing judge
records. Our qualitative reading is that in this corpus outright reward hacking is rare and
concentrated where evaluators check a proxy for the goal rather than the goal, but we will
report the measured number rather than an impression.

> Q: Are the LLM judges for failure analysis calibrated per modality?

They were not, and the submission reported agreement only for SWE, which R3 also flags.
The revised Appendix G reports agreement and human-vs-judge κ broken out per modality, and
the sampling design floors each (modality × blame) cell so web and CUA are estimable rather
than swamped by SWE volume.

---

## Reviewer 2 (rating 4, confidence 3)

Thank you for the constructive framing. All three weaknesses are addressed.

> 1. The LLM judge has no human ground-truth validation.

Agreed, and this is now a study rather than an argument: blind stratified annotation by
multiple annotators over the same transcripts and the same closed taxonomy, reported as
Cohen's κ against human consensus on both the 10-way taxonomy and the
agent/tool/benchmark grouping, with inter-annotator κ as the ceiling. See global response
point 2; the instrument and the scoring pipeline ship in
`scripts/paper/judge_validation.py` so the protocol is reusable rather than a one-off.

> 2. The judge could be biased towards Claude models. Would be nice to re-run with GPT.

We implemented this rather than argued about it. The investigator's transport was already
a Protocol, so we added `CodexDriver`, a third driver that runs the identical recipe,
prompts and output schema through `codex exec` on a non-Anthropic backbone
(`ch-investigate --driver codex`). Re-judging the same episodes on both backbones gives
per-episode κ and the shift in the aggregate blame distribution; we report both, and we
regard the distribution-level stability as the load-bearing one, since that is what the
82 % claim rests on.

**Pilot result [DONE].** On six MiniWoB episodes judged by both backbones (Sonnet-class vs
GPT-5.4, identical recipe and schema), the two judges chose the **same outcome and the same
primary blame on every episode** (κ = 1.0), including two episodes both attributed to
`action_space_limited` rather than to the model. We are explicit in the paper that six
short episodes from one benchmark cannot bound cross-backbone disagreement; MiniWoB
trajectories are the legible end of the judging task. What it does establish is that the
comparison runs end to end and that the closed taxonomy transfers across model families
without prompt surgery. **[NEEDS YOU]** the same swap over the SWE and CUA episodes, which
is the number that belongs in the appendix.

Building this surfaced a defect worth reporting: the existing terminal driver assumed the
single-object JSON envelope of older Claude CLIs, while current releases emit a message
*array*, so that judging path had been failing outright (and reporting zero cost). Both
envelope shapes are now handled, with regression tests. It is an uncomfortable thing to
find while answering a review about judge reliability, and it is exactly the kind of thing
that having a second backbone makes visible.

> 3. More models across providers.

The evaluation covers two providers by design (a small/large pair each from Anthropic and
OpenAI) so that provider and capability are not confounded. We agree the open-weight gap
matters most, and not only for coverage: it is the class of model the RL post-training
motivation actually needs (R3-W5). We have added an open-weight configuration to the
released recipes so the comparison holds the benchmark and tool stack constant.
**[NEEDS YOU]** the OpenRouter key in the repo `.env` returns 401; with a working key the
open-weight cells run unattended.

> 4. Comparison table against METR, Inspect AI, BrowserGym, AgentGym, OpenEnv.

Added as Table 1; see global response point 4.

---

## Reviewer 3 (rating 2, confidence 5)

We appreciate the precision of this review: every weakness names a specific missing
measurement, which made it actionable. We have addressed all six.

**W2/Q1 (parity), W3/Q2 (human validation of the judge), W4/Q3 (judge coverage beyond
SWE), W6 (quantitative framework comparison)** are answered in the global response,
points 1, 2 and 4. Two things there are worth surfacing here because they respond
directly to your framing:

- You wrote that the faithfulness claim rests on the authors' own judge. That is correct
  and we no longer make it that way. The judge's benchmark-side blame rate is demoted in
  the paper to what it actually is: a *monitoring* signal that surfaces candidate wrapper
  defects at corpus scale (it found real ones, including an escape-sequence bug in an
  evaluator path that suppressed five WAA scores). Faithfulness is now argued from
  task-set identity, evaluation provenance, oracle parity and cross-harness agreement.
- You cite Harbor's parity standard. We agree it is the right bar and have adopted it as a
  registry requirement rather than a one-off experiment, so third-party CUBEs inherit it.

**W1/Q4 (single scaffold).** Done, and the result is worth stating here because it also
answers "do the rankings change a lot?". ReAct vs Genny over all 125 MiniWoB tasks, same
model and same BrowserGym action set: **66.1 %** vs **69.4 %** (+3.2 pp, McNemar
*p* = 0.42), with the two scaffolds reaching the **same outcome on 88.7 %** of tasks
(Spearman ρ = 0.74) and 14 discordant tasks. The benchmark-level reading is stable; about
one task in nine is scaffold-decided. This does not refute your concern so much as bound
it: holding the tool stack constant is what keeps the variance this small, which is an
argument for the standard's design rather than against the measurement. We have also
rewritten the results text so the numbers are presented as platform measurements of one
configuration, and we say explicitly that conclusions resting on gaps of a few points
should not be drawn from a single-scaffold table, ours included.

**W5/Q5 (open weights and the RL claim).** This is a fair hit and we have changed the
paper rather than defended it. Two separate things were conflated: what CUBE *enables*
(a common interface, resource lifecycle and trajectory export that a training stack can
sample from) and what we *demonstrated* (evaluation of four closed API models). We now
state in the Limitations that no RL training run is reported and that the post-training
motivation is untested end-to-end, which the submission should have said. Concretely we
add an open-weight configuration to the released recipes that holds benchmark and tool
stack constant. On throughput we can give one honest datapoint now: the episode loop that
produces the trajectories an RL stack would consume sustains **≈1,200 complete MiniWoB
rollouts per hour on a single 8-core laptop** with four Ray workers (≈2.4 environment
steps/s), measured from the runs above. That is a property of the rollout path, not
evidence of training, and we present it as such.
**[NEEDS YOU]** an open-weight API key (the one in `.env` returns 401), a cluster-scale
throughput run through the NeMo Gym connector, and, if at all possible, a short RL run on
one CUBE: even a single learning curve would convert this limitation into a contribution.

**Limitations.** All three gaps you list are now stated explicitly in §8: that judge
accuracy bounds the faithfulness argument (and what we did about it), that wrapper
validation previously rested on our own judge (and what replaced it), and that no RL
post-training run is reported.

---

## What exists right now, and where

| Artefact | Status | Path |
|---|---|---|
| Static parity audit (task-set identity, AST evaluator provenance) | measured | `scripts/paper/parity_audit.py`, results in `paper/neurips26/data/parity/` |
| Residual evaluator diffs (OSWorld, WAA), released with the paper | measured | `data/parity/*-residual-diffs.txt` |
| Wrapping-effort measurement (median 508 SLOC) | measured | `scripts/paper/wrapping_effort.py` |
| Scaffold sensitivity, ReAct vs Genny on MiniWoB | measured | `recipes/paper/miniwob_scaffold_model_matrix.py`, `scripts/paper/compare_runs.py`, `data/scaffold_miniwob.json` |
| Model-patch capture, so SWE episodes are re-gradeable upstream | implemented | `cubes/swebench-verified-cube/.../task.py` |
| Cross-harness regrade tool | implemented, needs a run | `scripts/paper/swebench_cross_harness.py` |
| Oracle-parity reporting | implemented, needs a run | `scripts/paper/oracle_parity.py` |
| Human-validation kit (sample → blind packet → κ) | verified end-to-end; blinding checked; needs annotators | `scripts/paper/judge_validation.py` |
| Non-Anthropic judge backbone | verified; pilot κ=1.0 on n=6 MiniWoB | `CodexDriver`, `--driver codex` |
| Backbone-swap comparison | run, output in LaTeX | `judge_validation.py swap`, `data/judge_swap.tex` |
| Claude 2.x envelope fix (judge path was broken) | fixed, tested | `agent_driver.py`, `tests/test_investigator.py` |
| Revised paper | compiles, 10 content pages | `paper/neurips26/main.tex` → `main.pdf` |

**Page budget:** the revision is 10 content pages against the 9-page submission limit
(camera-ready allows 10). Cutting the framework comparison table to the appendix, or
trimming §3's two code listings, recovers the page if the revision must fit 9.

## Author checklist before posting

1. **[NEEDS YOU — annotators]** Run the human study: `judge_validation.py sample
   <results> --n 100`, two annotators, then `score`. Paste κ into Appendix G and
   into the R2/R3 replies. Nothing else substitutes for this one.
2. **[CLUSTER]** Jobs J1–J5 are specified for the cluster agent in
   [`cluster/RUNBOOK.md`](cluster/RUNBOOK.md), with `cluster/cluster-bundle.patch`
   carrying the code that is not yet on `dev` (crucially, `model_patch` capture,
   without which the cross-harness regrade cannot run):
   - **J1** SWE-bench Verified gold-patch oracle → §7.2 oracle-parity row.
   - **J2** cross-harness regrade of agent patches → §7.2 agreement row.
   - **J3** open-weight policy served on-cluster, driven through the rollout
     service → answers R3's "RL is never demonstrated" and the closed-weights
     complaint; report episodes/hour and token-record counts.
   - **J4** open-weight MiniWoB cells (replaces the dead OpenRouter route).
   - **J5** Terminal-Bench budget lift, if time remains.
3. Backbone swap at real n: the local batch judges all 81 MiniWoB failures with
   the API-key driver; re-run the same set under `--driver codex` and `swap`.
   Replaces the n=6 pilot, which is currently reported as a pilot precisely
   because six episodes cannot bound cross-backbone disagreement.
4. Replace every bracketed placeholder in the revised PDF before upload.
