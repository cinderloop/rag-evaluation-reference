# 0012: Eval-gate thresholds

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

The eval gate (`rag-eval gate`) runs the evaluation and fails when quality drops. Its
thresholds decide two error rates:

- **False alarm:** the gate fails although nothing changed, just run-to-run noise.
- **Missed regression:** the gate passes although the system got worse.

Fixed round numbers ("correctness must be at least 50%") ignore both: they are not tied
to what the system actually scores, or to how much one run of 66 questions can vary.

## Decision

**Gated metrics.** Only metrics whose measurement is trusted fail the gate:

| Metric | Direction | Why gated |
|---|---|---|
| passage hit | must not drop | deterministic; detected every retrieval mutation (docs/mutations.md) |
| groundedness | must not drop | judge validated (TPR 98%, TNR 100%; docs/judges.md); detected both generation mutations |
| correctness | must not drop | judge validated (TPR 95%, TNR 98%) |
| no-context refusals | must not rise | deterministic; detected the threshold mutation |

**Report-only metrics.** These are reported, but never fail the gate:

- **passage recall:** it overlaps with passage hit.
- **refusal:** its judge was validated on only six failures.
- **abstention:** its judge is not validated.

**Threshold rule.** Each threshold is the baseline rate moved, in the bad direction, by a
margin that a single run of this size would rarely cross by chance (`configs/gate.yaml`,
written by `rag-eval gate derive`):

    margin = z × SE,   z = Φ⁻¹(1 − α / k)

- α = 0.05 is the false-alarm rate for one gate run as a whole.
- k = 4 is the number of gated metrics, so each metric gets a one-sided α / 4 (a
  Bonferroni split; z = 2.24).
- SE is the baseline rate's standard error: binomial with (x + 0.5) / (n + 1), so a 0%
  rate still gets a margin. With repeated runs it is the cluster-robust SE over
  questions (ADR 0011).

**Current thresholds.** From the Phase 7 baseline run on the 66 test questions
(`20261027T002255Z-baseline`):

| Metric | Baseline | Gate passes if |
|---|---|---|
| passage hit | 68.5% | ≥ 54.3% |
| groundedness | 80.3% | ≥ 69.2% |
| correctness | 40.7% | ≥ 25.7% |
| no-context refusals | 0.0% | ≤ 2.4% |

**Check against the mutation runs.** `rag-eval gate check` on each Phase 7 run gives:

| Run | Gate | Metrics that failed |
|---|---|---|
| baseline | pass | — |
| weaker-llm | **fail** | groundedness 66.7%, correctness 24.1% |
| top-n-2 | **fail** | passage hit 53.7% |
| threshold-078 | **fail** | passage hit 48.1%, no-context refusals 24.2% |
| weak-prompt | **fail** | groundedness 56.1% |
| remove-document | pass | — |
| knowledge-edit | pass | — |

The gate fails all four mutations the evaluation detected, and passes the two that are
too small to detect in aggregate (docs/mutations.md).

## Consequences

- **The validation is in-sample.** The thresholds come from the same baseline run they
  are checked against. A fresh baseline run is the first real test of the false-alarm
  rate.
- **Some catches are narrow.** top-n-2 failed passage hit by 0.6 points and weaker-llm
  failed correctness by 1.6 points. A run-to-run swing of one or two questions could
  flip either result. Margins this tight come from gating one run of 66 questions: a
  regression smaller than about 11–15 points can pass (the minimum detectable effects
  in ADR 0011).
- **Stronger thresholds need more data.** Deriving them from three baseline repeats
  (the gate's SE then becomes the clustered one) and running the gate with repeats
  would narrow the margins. More test questions would too.
- **When to re-derive.** Thresholds must be re-derived when the baseline config, the
  dataset, the judge model or a judge prompt changes. The gate config records the run
  it came from.
- **What the gate does not catch.** It catches broad regressions, not damage limited to
  a few questions (the removed-document case). Those need targeted tests.
