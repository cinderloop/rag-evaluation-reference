# 0010: How the judges are validated

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

A judge's pass rate means nothing until it is known how often the judge agrees with a
careful reader. Each judge (ADR 0008) is therefore compared with Pass/Fail labels on
real traces before any of its numbers are reported.

## Decisions

### Labels on dev traces, judge by judge

The labels are made on dev traces only. They come from the three error-analysis runs
(ADR 0006) and two more deliberately weak configs (`configs/judge-validation/`), added
because the first three produced too few failures for some judges. One forbids saying
the answer is missing, so the system guesses; the other refuses unless every part of the
question is answered, so it holds back answers it has. The labelling
page (`rag-eval review judges-export`, `data/review/judge-labels.html`) shows each trace
once, with one Pass/Fail/Skip question per judge that applies to it:

| Judge | Traces offered |
|---|---|
| groundedness | 217 |
| correctness | 177 |
| refusal | 177 |
| abstention | 37 |

These are the traces left after two exclusions:

- **Few-shot traces.** A trace whose content is an example in a judge's prompt is not
  offered for that judge. Otherwise the judge would be scored on its own examples.
- **Rule-decided traces.** Deterministic refusals are decided by a rule, so they are not
  offered at all.

**What the page hides.** Judge verdicts are never shown, so labels are not anchored on
them. Traces are shuffled in a fixed order with the config hidden. The expected evidence
is highlighted in the retrieved passages, as on the open-coding page, because three of
the four questions are about what the passages contain.

"Skip" is allowed for cases a reader cannot decide; skipped marks are not used.

**Fewer than 50 of a kind.** The target is roughly 50 Pass and 50 Fail per judge. A mode
that is rare in these traces gets fewer, and the number is reported with the result.
Abstention has only 37 unanswerable traces (eight questions), and refusal and abstention
have few failures. The results are in `docs/judges.md`.

### Split by question into judge-dev and judge-test

Labels are split by a hash of the case ID: about half the questions go to judge-dev and
half to judge-test, with all traces of a question in the same half. Judge-dev is used to
improve judge prompts. Judge-test is used once, for the numbers in `docs/judges.md`.
Splitting by question, not by trace, stops the same question (answered by five configs)
from appearing on both sides.

### TPR and TNR with Wilson intervals

For each judge, `rag-eval review judges-validate` reports:

- **TPR:** the share of human-Pass cases the judge also passes.
- **TNR:** the share of human-Fail cases the judge also fails.

Each comes with a 95% Wilson interval. Raw accuracy is not reported alone, because it
hides a judge that passes everything when most cases pass. Judge replies that cannot be
read are counted separately, not as either verdict.

## Consequences

- Validated numbers hold for one judge model and one prompt version. Every validation
  result records the judge config, model and prompt hashes. Changing any of them
  requires validating again.
- With about 50 labels per side, intervals are wide (roughly ±10–14 points). They are
  reported as they are.
