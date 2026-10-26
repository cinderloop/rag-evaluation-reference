# Judges

The four binary LLM judges (ADR 0008), how well each agrees with Pass/Fail labels, and
what that means for reading their scores. The validation method is in ADR 0010.

## Setup

| | |
|---|---|
| Judge model | Claude Haiku 5.5 (`claude-haiku-5-5`) via Claude Code headless (`configs/judges/claude-code-haiku.yaml`, ADR 0007) |
| System under test | `qwen3-nothink:8b` (a different model family) |
| Prompts | [`src/rag_eval/evaluators/judges/prompts/`](../src/rag_eval/evaluators/judges/prompts/) |
| Prompt hashes (first 12 characters) | groundedness `70979614658c`, correctness `1921299340a6`, abstention `e87235ef167d`, refusal `a859e5685960` |
| Labels | [`data/labels/judge-validation/judge-labels-20261026-c4544fa9.json`](../data/labels/judge-validation/judge-labels-20261026-c4544fa9.json) |
| Final result | [`results/judge-validation/agreement/20261026T222019Z-judge-test.json`](../results/judge-validation/agreement/20261026T222019Z-judge-test.json) |

These numbers hold only for this model and these prompt hashes. A change to either
requires validating again.

## Labels

The labels cover 217 dev traces from five configs: the three error-analysis runs (ADR 0006)
and two runs added to produce more failures (`configs/judge-validation/`). One config
forbids saying the answer is missing; the other refuses unless every part of the
question is answered.

There are 608 marks in total. They are split by question into judge-dev (used to improve
the prompts) and judge-test (used once, for the numbers below):

| Judge | judge-dev Pass / Fail | judge-test Pass / Fail |
|---|---|---|
| groundedness | 80 / 25 | 94 / 18 |
| correctness | 33 / 41 | 59 / 44 |
| refusal | 67 / 8 | 96 / 6 |
| abstention | 25 / 2 | 9 / 1 |

Two failure modes stay rare even with the extra configs:

- **Wrongly declining.** Answers that decline a part the passages contain are uncommon.
- **Answering trick questions.** The dev set has only eight unanswerable questions, and
  the system declines most of them.

These judges therefore have few Fail labels, and their TNR is barely measured.

## Iterating on judge-dev

The first judge-dev run showed two prompt problems:

- **Correctness failed answers for leaving out background.** It failed answers that
  answered the question but omitted details that only the expected answer gives.
- **Refusal counted loosely related facts.** It counted any related fact in the passages
  as something the answer should have given.

One round of changes clarified both prompts:

- **Correctness:** judge against what the question asks, but fail a "why" answer that
  only restates the conclusion.
- **Refusal:** define a "part" as something the question asks for on its own.

| Judge | judge-dev before (TPR / TNR) | judge-dev after (TPR / TNR) |
|---|---|---|
| correctness | 24/33 = 73% / 37/41 = 90% | 22/33 = 67% / 39/41 = 95% |
| refusal | 63/67 = 94% / 8/8 = 100% | 65/67 = 97% / 8/8 = 100% |

**Correctness disagreements.** The remaining correctness disagreements on judge-dev come
from three questions, each answered by several configs. In all three, the judge and the
labels read "what the question asks for" differently:

- what the tug Steel Skipper was doing;
- why the Three Girls captain had no immersion suit;
- which tow was faster.

Changing the prompt further to match these questions would have tuned the judge to
judge-dev, so iteration stopped after one round.

## Results on judge-test

TPR is the share of labelled-Pass traces the judge passes. TNR is the share of
labelled-Fail traces it fails. Each has a 95% Wilson interval. No judge reply was
unreadable.

| Judge | TPR | TNR |
|---|---|---|
| groundedness | 92/94 = 98% (93–99%) | 18/18 = 100% (82–100%) |
| correctness | 56/59 = 95% (86–98%) | 43/44 = 98% (88–100%) |
| refusal | 93/96 = 97% (91–99%) | 6/6 = 100% (61–100%) |
| abstention | 7/9 = 78% (45–94%) | 1/1 = 100% (21–100%) |

**How to read them:**

- **Groundedness and correctness** are measured well on both sides. Their scores can be
  used, with the intervals above.
- **Correctness scored better on judge-test than on judge-dev** (95% against 67% TPR).
  That is not a sign of tuning: the split put the three ambiguous questions in judge-dev.
  The gap shows how much one question can move a rate when labels cluster by question.
- **Refusal** agrees with passes, but its TNR rests on six failures. It should be read as
  "rarely flags a correct answer", not as a proven detector of unnecessary refusals.
- **Abstention** is not validated. It has one Fail label in judge-test, and it failed two
  of nine passes. Both were answers that did not correct the false premise outright,
  which the judge read as accepting it. Its scores should not be relied on. The dev set
  needs more unanswerable questions before this judge can be measured.

**The 11 judge-test disagreements.** Most are borderline cases rather than clear errors:

- an answer that hedges a guess ("it is likely that…");
- an answer that answers the question but leaves out the mechanism the expected answer
  explains.

No pass-rate correction for judge error is applied. With TNR this close to 100% and
intervals this wide, the correction would add little, and for the rare modes it would
rest on one to six labels.

## Known judge biases

- **Self-preference.** A judge tends to rate answers in its own style higher. The judge
  (Claude) and the system under test (Qwen) come from different model families, so the
  answers are not in the judge's own style.
- **Verbosity.** Judges can favour long, confident answers. The prompts ask one narrow
  question and treat unsupported detail as a failure, so length works against an answer
  for groundedness. The labelled set includes long padded answers from the permissive
  configs, and groundedness failed all 18 labelled failures in judge-test.
- **Strictness drift on omissions.** Where an expected answer contains more detail than
  the question asks for, the correctness judge leans towards failing the answer. This is
  the main source of disagreement and is visible in the judge-dev numbers.
- **Non-determinism.** Claude Code does not expose temperature, so a re-run on an
  uncached input could differ. Every verdict is cached and stored with its reason, so
  reported numbers are reproducible from the stored replies.
