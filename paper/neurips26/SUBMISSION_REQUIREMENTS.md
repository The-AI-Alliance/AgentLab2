# NeurIPS 2026 Submission Requirements — CUBE Paper

**Track: Evaluations & Datasets (E&D)**
Safer fit than Main Track. E&D reviewers focus on whether the contribution meaningfully
advances evaluation practice — they will not penalize for not outperforming SOTA models.

Sources: [E&D Call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets) · [Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) · [Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) · [Reviewer Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines)

---

## Deadlines (same as Main Track)

| Milestone | Date (AOE) |
|---|---|
| Abstract submission | **May 4, 2026** ✓ |
| Full paper + supplementary | **May 6, 2026** ✓ |
| Author notification | September 24, 2026 |
| Camera-ready | TBD |

**Portal:** Separate OpenReview portal from Main Track. Do not submit to the Main Track portal.

---

## CRITICAL: Desk-Rejection Risks

- [ ] **Checklist not in PDF** — must appear after references; does not count toward page limit
- [ ] **Not anonymized** — no author names, affiliations, or identifying links in submitted PDF
- [ ] **Identifying self-citations** — use "Smith et al." not "our previous work"
- [ ] **Anonymous repo links violated** — all linked repos must be anonymized (anonymous.4open.science); non-compliance = desk rejection
- [ ] **Code/data not accessible at submission** — for a benchmark suite, code is mandatory and must be accessible to reviewers without a personal request
- [ ] **Over 9 content pages** — references + checklist do not count
- [ ] **Dual submission** — cannot be under review elsewhere simultaneously (ICML position paper is a preprint, not archival — OK, but do not cite it)

---

## Formatting

- **Template**: LaTeX only (`neurips_2026.sty`). Anonymous submission mode (no `[final]`).
- **Length**: 9 content pages max for submission; 10 for camera-ready.
- **Single PDF**: paper + references + appendices + checklist.
- **Supplementary ZIP** (separate, optional): ≤100MB — extended tables, code samples.
- **File size**: PDF ≤50MB.
- **OpenReview profiles**: all authors must have one.

---

## Contribution Acknowledgement (E&D-specific directive)

> "The main paper must clearly articulate the **evaluative role** of the contribution: what claims it supports, under which assumptions, and its limitations. For benchmark suites, describe how the suite is intended to be used meaningfully in evaluative practice rather than as an endpoint."

This is not a standalone section. It is satisfied through the paper's existing structure:

**Claims the paper makes → Abstract + §1 Introduction (already written):**
- CUBE reduces cross-benchmark integration cost from N×M to N+M wrappers.
- Tools held constant per modality makes cross-benchmark scores genuinely comparable.
- `CompositeBenchmark` enables composite evaluations without per-project engineering.
- Cross-modality evaluation exposes capability profiles invisible to per-benchmark reports.

**Assumptions → must be explicit in §7 setup and/or §8 Discussion:**
- "Tools held constant" requires researchers to use the same `ToolConfig`; scores are only comparable when enforced by the harness.
- Capability profile claims are limited to the \NUMEXP benchmarks evaluated; generalization to unseen benchmark families is untested.
- Compliance badge establishes a floor (debug tasks pass), not a correctness guarantee.
- Harness-specific claims (parallelism, cost per task) depend on the reference harness, not the standard alone.

**Limitations → §8 Discussion, final paragraph (already planned):**
- Streaming actions/observations not yet supported.
- Multi-agent benchmarks not yet supported.
- Dataset-only benchmarks (no interactive environment) are out of scope.
- Composite difficulty calibration is a snapshot; saturation dynamics over time not studied.

**"Meaningful evaluative practice" framing** — E&D reviewers will ask: *is this a useful tool for the community, or a one-shot result?* Emphasize in §1 and §8:
- Any researcher can wrap their benchmark once and reach all CUBE-compliant platforms.
- The registry and compliance CI make the corpus self-maintaining.
- `CompositeBenchmark` lets the community recalibrate difficulty as agents improve, without re-engineering.

---

## E&D Track — Extra Obligations vs. Main Track

### Croissant Metadata — CUBE is EXEMPT (confirmed with track chair, 2026-05-04)

**FAQ ruling**: "If your submission is an environment for evaluation only, you do not need to
follow data-hosting guidelines. The dataset-hosting and Croissant requirements apply only to
submissions that introduce new datasets."

CUBE is an interop standard and evaluation harness — no new dataset. Clean exemption applies.

**Action required**: none for the standard itself or the registry.

**Exception**: if any wrapped sub-benchmark in the corpus ships genuinely new task data
(not just wrapping existing benchmarks), that sub-benchmark's data would need Croissant +
RAI metadata. Check at submission time. Current corpus (WorkArena, WebArena, SWE-bench,
OSWorld, MiniWoB, TerminalBench, WAA) — all wrap existing data, none introduce new tasks.

**Soft expectation**: the FAQ notes that even exempt submissions "should still file something
Croissant-shaped" (dataset-level metadata, no RecordSets). Low priority; do only if time
permits before May 6.

Note: 2026 also added RAI fields as a required extension to Croissant for submissions that
*do* need it — not relevant for CUBE but good to know for future dataset papers.

### Code and Data Accessibility

- [ ] Code must be **accessible to all reviewers, ACs, and SACs at submission time, without a personal request**
- [ ] Anonymous repo must be browsable without login (anonymous.4open.science satisfies this)
- [ ] No datasets >4GB (CUBE ships no data; environments are pulled at install time)

---

## Paper Checklist (16 Questions — append LaTeX after `\bibliography`)

Copy the official checklist LaTeX from the NeurIPS 2026 sample file. Answers visible to reviewers and ACs.

| # | Question | Answer | Notes |
|---|---|---|---|
| 1 | Claims accurately reflect contributions? | Yes | §1 contributions list matches §3–7 |
| 2 | Limitations section included? | Yes | §8 Discussion, last paragraph |
| 3 | Full assumptions + proofs for theoretical results? | N/A | No theoretical results |
| 4 | Steps to make results reproducible? | Yes | Registry + anon repo + locked task IDs in §6 |
| 5 | Code, data, instructions to reproduce? | Yes | Anonymous repo; compliance CI |
| 6 | All eval details specified? | Yes | §7: models, tool configs, agent variants |
| 7 | Error bars / statistical significance? | TBD | Add confidence intervals once runs complete |
| 8 | Compute resources documented? | Yes | §7: wall-clock, $/task, provisioning overhead |
| 9 | NeurIPS Code of Ethics conformed to? | Yes | — |
| 10 | Negative societal impacts discussed? | Yes | §8 Discussion |
| 11 | Safeguards for high-risk model release? | N/A | No model released |
| 12 | Existing assets cited, licenses respected? | Yes | Each benchmark cited; licenses in §5 table |
| 13 | New assets documented? | Yes | Registry, anonymous repo, corpus in §5 |
| 14 | Crowdsourcing / human subjects details? | N/A | No human subjects |
| 15 | IRB approvals? | N/A | — |
| 16 | LLM usage described? | Yes | `/new-cube`+`/review-cube` in §3; Genny agent in §7 |

---

## Anonymity Checklist

- [ ] `neurips_2026.sty` loaded without `[final]` option
- [ ] Acknowledgements section removed (restore for camera-ready)
- [ ] All GitHub URLs replaced with anonymous.4open.science links
- [ ] `\ICML position paper` not cited anywhere
- [ ] §3 footnote anonymous repo URL updated from placeholder `XXXX`
- [ ] Org names (ServiceNow, IBM, NVIDIA, The AI Alliance) absent from main PDF
- [ ] Supplementary ZIP also anonymized

---

## E&D Reviewer Mindset (how to write defensively)

E&D reviewers evaluate: *does this meaningfully advance evaluation practice?*

**What they will look for:**
- Clear scope: what benchmarks fit CUBE, what don't (state explicitly)
- Honest accounting: does the wrapper add overhead? how much?
- Evaluative use: how does a researcher actually use this? what decisions does it enable?
- Portability evidence: does the same code genuinely run on multiple backends?

**What they will not penalize:**
- Not beating SOTA — E&D explicitly states this
- Using existing methods (gym API, JSON-RPC) — novel *combination* and *framing* is sufficient

**Common adversarial angles to pre-empt in the paper:**
- *"This is just an API wrapper"* → pre-empt in §1: the integration tax is concrete (cite specific N×M costs), the resource lifecycle (3 infra models through 1 interface) is non-trivial
- *"Results could be tool-confounded"* → pre-empt in §7: tool configs held constant per modality; explicit table showing which ToolConfig each benchmark uses
- *"Compliance CI doesn't guarantee correctness"* → acknowledge directly in §8 limitations
- *"Only N benchmarks wrapped"* → frame as a growing registry, cite compliance CI as the mechanism for community growth

---

## Before Uploading

- [ ] Submit to **E&D track portal** (not Main Track)
- [ ] `pdflatex` run twice (cross-references converge)
- [ ] Content ≤ 9 pages before references
- [ ] Checklist LaTeX appended after `\bibliography`
- [ ] All `[?]` placeholders (`\NUMCUBES`, `\NUMMODELS`, `\NUMTASKS`, `\TOPRESULT`) filled
- [ ] §3 footnote: anonymous repo URL placeholder replaced
- [ ] Croissant metadata file created and linked
- [ ] Abstract/intro claims cross-checked against actual experimental results
- [ ] All co-authors have OpenReview profiles
- [ ] Conflict of interest declarations complete (last 3 years)
- [ ] At least one author registered for in-person attendance (required for accepted papers)
