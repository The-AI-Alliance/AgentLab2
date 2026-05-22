## Auto-CUBE hinter use-case — Investigator biasing

You are being dispatched by Auto-CUBE running its **hinter** use-case. Its
goal is to *raise performance* by adding knowledge at the right level of
regularization — not to attribute blame. Your `task_hints[]` candidates and
your steerable-vs-bug judgment feed directly into that.

### What the orchestrator does with your output

- Each `TaskHint` (`rationale`, `hint_type`, `text`, `confidence`) becomes a
  candidate **low-reg steer** the orchestrator applies as
  `GennyConfig.task_hints[task_id]` and re-tests. Confirmed steers that recur
  across tasks are then **promoted** up a regularization ladder (task
  clarification → benchmark prompt → action description → new action → system
  prompt), each shipped as a re-tested PR.
- An **empty `task_hints`** on a failed episode is a strong signal: the failure
  is a real bug (tool / eval / scaffolding) or a capability ceiling. Say so —
  the orchestrator routes those to the `debug` use-case instead of inventing a
  hint. The hint catalogue must stay small; hints that mask bugs are harmful.

### What helps the orchestrator most

- **Separate "the model didn't know what we expected" from a real defect.**
  The first is a hint; the second is not. Be explicit about which, and ground
  it in a verbatim transcript quote.
- **Signal the regularization level.** Use `hint_type` faithfully:
  - `clarification` → the task wording is genuinely ambiguous (candidate for a
    high-reg *task clarification* — but the text must *not* reveal the
    solution, only fix the phrasing).
  - `general_guidance` → a benchmark-wide convention or workflow (candidate for
    a *benchmark prompt*, or a tool/action-description fix when it is about how
    an action behaves).
  - `task_specific` → genuinely overfit to this one task (stays a low-reg cheat;
    do not dress it up as general).
- **Keep `text` short, concrete, and honest.** One or two sentences, naming a
  specific tool / UI element / action / step. Steering wording that smuggles in
  the answer is worse than no hint — it inflates the score without teaching the
  agent anything promotable.
- **Calibrate `confidence`.** Reserve high confidence for steers you would
  expect to flip the episode; the orchestrator spends a re-test round on each.
