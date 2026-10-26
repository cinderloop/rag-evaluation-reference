# 0011: Statistics choices

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

The evaluation produces pass rates and scores on small sets: 66 held-out questions,
each run three times in the mutation experiments (Phase 7). Two configs are compared on
the same questions. The statistics have to stay honest at this size: intervals that do
not go below 0% or above 100%, repeated runs not counted as independent evidence, and a
"no difference found" that says how small a difference could have been missed.

## Decisions

The module is `src/rag_eval/stats/`, standard library only (no NumPy or SciPy). Every
function is checked in `tests/test_stats.py` against values computed by hand, with the
working shown in comments.

| Question | Method | Why |
|---|---|---|
| Interval for a pass rate | Wilson score interval | Stays inside 0–100% and keeps sensible coverage near 0% and 100% and at small n, where the normal-approximation interval fails |
| Interval for any other statistic | Percentile bootstrap (10,000 resamples, fixed seed) | No distribution assumption; reproducible |
| Repeated runs of a question | Questions as clusters: cluster-robust standard error, and a bootstrap that resamples whole questions | Runs of one question succeed or fail together, so 66 questions × 3 runs is not 198 independent observations |
| Two configs, pass/fail, same items | Exact McNemar test (two-sided binomial test on the discordant items) | Only items where the configs disagree carry information; exact, so valid with few discordant items |
| Two configs, continuous scores, same items | Paired bootstrap of the per-item differences | Keeps each item's pair together |
| Two configs, repeated runs, same questions | Paired bootstrap that resamples whole questions (all runs under both configs) | A difference in pass rate with an interval that does not count repeats as independent items |
| Repeated-run stability | Share of items that pass on every run, fail on every run, or are mixed | Shows flakiness that a pooled rate hides |
| What a test of this size can detect | Minimum detectable effect (MDE) at α = 0.05 and 80% power | Turns "not significant" into "no change larger than the MDE" |

**Clustered standard error.** This is the cluster-robust (CR1) estimator for a mean:
the square root of G/(G−1) × Σ(sum of residuals per question)², divided by the number
of observations.

**Minimum detectable effect.** The formulas are normal approximations, used for
planning, not for testing:

- **Paired pass rates:** (z₁₋α/₂ + z_power) × √(d / n), where d is the share of items
  expected to flip between the configs.
- **Two independent rates:** (z₁₋α/₂ + z_power) × √(2p(1−p) / n).

**Raw accuracy on its own.** It is not reported anywhere: judges get TPR and TNR (ADR
0010), and config comparisons get the size of the change with its interval and a p-value.

## Consequences

- With 66 held-out questions and one run each, a paired comparison can detect changes of
  about 11, 15 or 19 points when 10%, 20% or 30% of questions flip between the configs.
  Smaller mutation effects can go undetected. Those are reported as blind spots with
  their MDE, not as passes.
- Three repeats narrow this only partly. Treating 198 runs as independent would suggest
  9 points at 20% flips. Because runs of a question move together, the real figure lies
  between that and the single-run 15 points, so reports use the clustered methods.
- The McNemar test answers whether two configs differ on these questions. The effect
  size with its interval answers by how much, and reports give both.
- Bootstrap results depend on the seed; a fixed default seed keeps reported numbers
  reproducible.
