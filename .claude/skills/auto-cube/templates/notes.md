# Round <N> — <short label>

**Code delta since round <N-1>**: <commit hash> — <one line> (or "none")

## Hypothesis (before run)

<What we believe is happening and why. What this round is designed to test.
Be specific enough that the result can confirm or refute it.>

## Experiment

- **exp_config**: `exp_config.py` (this dir)
- **Tasks**: <subset list>
- **Agent / model**: <e.g. Genny[swe] + gpt-5.4-mini>
- **Infra**: <local docker | …>
- **Cost cap**: <$/task>

## Results (after run)

- **Outcomes**: <X/Y solved; reward distribution>
- **Experiment dir**: <path>
- **Investigator recipe**: <use_case> → `meta_analysis.md` says: <key patterns>

## Findings

<Root causes — confirmed or refuted. Cite specific episodes / evidence.
Distinguish "the agent failed because X" from "the investigator mis-attributed Y".>

## Conclusion & next

<What this round established. The fix decision (and whether it's a confirmed
root cause yet). The next hypothesis.>
