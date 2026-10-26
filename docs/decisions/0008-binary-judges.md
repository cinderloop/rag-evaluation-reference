# 0008: Binary judges as DeepEval custom metrics, rules first

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

The failure taxonomy (`docs/failure-taxonomy.md`) found six failure modes. The
evaluators have to detect them. DeepEval offers three ways to build an LLM-judged
metric: `GEval` (a 1–5 score from criteria or evaluation steps), `DAGMetric` (a
decision tree of judge calls) and custom metrics (a subclass of `BaseMetric` with its
own `measure`).

## Decisions

### One binary judge per failure mode

| Judge | Failure modes | Cases | Inputs |
|---|---|---|---|
| groundedness | B guesses, E borrowed facts, F extra claims | all | question, passages, answer |
| correctness | D misreadings, and any wrong answer | answerable | question, expected answer, answer |
| abstention | E trick questions | unanswerable | question, why it is unanswerable, answer |
| refusal | C half the answer declined | answerable | question, passages, answer |

Modes A ("answer not retrieved") and the retrieval half of B and C are measured
without a judge, by the retrieval metrics (ADR 0009).

Each judge answers **Pass or Fail**, never a score. A 1–5 scale invites the judge to
pick the middle, makes thresholds arbitrary, and cannot be validated as cleanly: a
binary verdict can be compared with a label to get true-positive and true-negative
rates, which is what Phase 5 reports. Each judge sees only the inputs it needs; the
correctness judge does not see the passages and the groundedness judge does not see the
expected answer, so neither can answer a different question than the one asked.

### Custom metrics, not GEval or DAGMetric

Each judge is a `BinaryJudgeMetric` (a `BaseMetric` subclass) with its own prompt, which
returns 1 (Pass) or 0 (Fail) with a reason. `GEval` is excluded because it scores 1–5 and
`strict_mode` only rounds that score. `DAGMetric` fits decisions with several branches;
every judge here is one question, so a tree would add structure without adding a
decision. A custom metric also keeps the prompt as a plain file that can be read,
hashed and versioned (`src/rag_eval/evaluators/judges/prompts/`), and lets any model be
the judge (ADR 0007) without going through DeepEval's model wrappers.

The metrics work with DeepEval's own tooling (`LLMTestCase`, `deepeval test run`);
`rag-eval evaluate` calls them directly so that it controls caching, the cost estimate
and the `--max-cost` guard.

### Rules before judges

A rule decides wherever one can. When AnythingLLM refuses without calling the LLM (no
passage cleared the similarity threshold), the verdict follows from that alone:
groundedness Pass (no claims), correctness Fail (no answer), abstention Pass, refusal
Pass (nothing was retrieved to hold back). No judge call is made. Detecting a refusal
by matching phrases in the answer was rejected: the weak-grounding config declines in
its own words, so a phrase list would be brittle.

### Prompts with three labelled examples

Each prompt defines the one failure mode, gives explicit Pass and Fail conditions, and
shows three examples taken from the error-analysis labels: a clear pass, a clear fail
and a borderline case with its decision. The examples are shortened excerpts of real
traces, and their trace keys are listed in `FEW_SHOT_TRACES`, so judge validation can
leave them out instead of testing a judge on its own examples. Replies are a JSON object
`{"result": "Pass"|"Fail", "reason": "..."}`; anything else is recorded as an error, not
guessed.

## Consequences

- Four judge calls at most per trace, fewer for unanswerable cases (two) and none for
  deterministic refusals.
- Pass rates are reported per judge. Without validation (Phase 5) they are a judge's
  opinion, not a measurement; the summaries say which judge model and prompt produced
  them.
- A prompt change produces a new prompt hash, which misses the cache and requires
  re-validation.
