# Mutations: do the evaluations notice when the system gets worse?

An evaluation is only useful if it moves when the system breaks. This page applies the
idea of **mutation testing** to the RAG system: make a small, deliberate change that
should make it worse in a known way, run the evaluation, and check whether it notices.
A mutation the evaluation does not catch is a **blind spot**, and it is reported as one.

## Background

- **Mutation testing** comes from software testing. Deliberate faults (mutants) are
  inserted into a program, and a test suite is judged by how many it detects (DeMillo,
  Lipton and Sayward, "Hints on Test Data Selection: Help for the Practicing Programmer",
  *IEEE Computer*, 1978).
- **Metamorphic testing** checks how outputs should change when inputs change in a known
  way, for systems that have no exact expected output (Chen, Cheung and Yiu,
  "Metamorphic Testing: A New Approach for Generating Next Test Cases", HKUST technical
  report, 1998).
- **For RAG specifically**, "When Knowledge Changes: Metamorphic Testing of RAG Systems
  with Mutations" (ASE 2026,
  [conference listing](https://conf.researchr.org/details/ase-2026/ase-2026-research-track/247/When-Knowledge-Changes-Metamorphic-Testing-of-RAG-Systems-with-Mutations))
  defines mutation operators on the corpus, retrieval settings and retrieved context. It
  reports that snapshot metrics such as those in RAGAS can miss faults that changing the
  knowledge base introduces.

Here the mutations are config or corpus diffs from the baseline
(`configs/mutations/`). The thing under test is the evaluation suite: its retrieval
metrics and judges (ADRs 0008, 0009).

## Setup

**Runs.** The baseline and each mutation run once on the held-out test set: 66
questions, 12 of them unanswerable. The test set was not used to build or tune anything.

One run per config keeps the cost down. Three repeats per config would add data on how
consistent repeated runs are, and narrower intervals; the comparison tooling supports
repeats without changes.

**Scoring.** Every run is scored with `rag-eval evaluate`, which gives:

- the retrieval metrics: passage hit and passage recall;
- AnythingLLM's no-context refusals;
- the four judges (groundedness, correctness, refusal, abstention).

**Comparison.** `rag-eval mutations compare` compares each mutation with the baseline,
question by question (ADR 0011):

- **Effect:** the change in the pooled rate, with a 95% interval from a bootstrap that
  resamples whole questions.
- **Test:** an exact McNemar test on each question's outcome (its majority outcome, when
  there are repeats).
- **Detected:** significant (p < 0.05) and in the predicted direction.
- **Minimum detectable effect:** the smallest change the comparison could reliably have
  seen.

**Judge caveats** (`docs/judges.md`):

- Groundedness and correctness are validated.
- Refusal has few failure labels.
- Abstention is not validated, so its results are reported but not relied on.

## Pre-registered expectations

These expectations were written before any mutation was run, and committed before the
runs. They are mirrored in `data/mutations/expectations.yaml`, which the comparison
reads.

| # | Mutation | Change | Expected to drop | Expected not to change |
|---|---|---|---|---|
| 1 | weaker-llm | qwen3 1.7B instead of 8B (thinking off) | correctness, groundedness | passage hit, passage recall, no-context refusals |
| 2 | top-n-2 | 2 retrieved passages instead of 4 | passage recall, passage hit, correctness | no-context refusals |
| 3 | threshold-078 | similarity threshold 0.78 instead of 0.25 | passage hit, passage recall, correctness; no-context refusals **rise** | — |
| 4 | weak-prompt | system prompt without the grounding and refusal rules | groundedness, abstention | passage hit, passage recall, no-context refusals |
| 5 | remove-document | report MIR-26-03 left out | — (aggregate) | correctness, passage hit, passage recall (aggregate) |
| 6 | knowledge-edit | one fact changed in report MIR-25-19 | — (aggregate) | correctness, passage hit, passage recall (aggregate) |

**Why these predictions:**

1. **Weaker model.** Retrieval does not involve the LLM, so it must not move. A smaller
   model reading the same passages should answer worse and add more unsupported claims.
2. **Fewer passages.** Multi-span and two-report questions lose evidence. With the
   grounded prompt, the model should decline more rather than guess, so groundedness is
   not predicted.
3. **Higher threshold.** Baseline passages score 0.67–0.88 (median 0.77 on the dev
   runs), so about half fall below 0.78. Questions with no passage above it get
   AnythingLLM's no-context refusal.
4. **Weaker prompt.** Retrieval is unchanged. Without the rules, the model should add
   outside knowledge and answer trick questions.
5. **Removed document.** Only two of the 66 questions use MIR-26-03 (q-b005-1, q-b005-2),
   far below the minimum detectable effect. The aggregate metrics are **expected not to
   detect** this mutation. The two targeted questions are expected to lose their evidence
   and fail correctness. The prediction is meant to show where aggregate
   metrics are blind, and why targeted checks are needed.
6. **Knowledge edit.** The report is changed to say the City of Louisville was towed to
   a repair yard in Cape Girardeau, Missouri, instead of Paducah, Kentucky
   (`data/mutations/knowledge-edit.yaml`). The edited text is uploaded as a new PDF under
   the original document ID. The run is scored against the edited expected answer. The
   targeted question (q-b029-1) is expected to name Cape Girardeau,
   showing that answers follow the documents rather than the model's memory; the
   closed-book baseline found no memorised answers. One question cannot move an
   aggregate metric, so this is a targeted check.

**What can and cannot be detected.** With 66 questions, a paired comparison detects
changes of roughly 11–19 percentage points, depending on how many questions flip (ADR
0011). A smaller real effect may be reported as "not detected"; the result then gives
the minimum detectable effect, not a claim that nothing changed.

## Results

**Runs.** One run per config on the 66 test questions, judged with Claude Haiku 5.5. The
runs are in [`results/mutations/`](../results/mutations/) and the comparisons in
[`results/mutations/comparisons/`](../results/mutations/comparisons/). All seven runs
are from commit `a714fbc`, which holds these expectations. The manifests record the
working tree as modified, for two reasons:

- the weaker-LLM config and its Modelfile were added after that commit, and are
  committed with these results;
- a local, uncommitted runner change paused between questions to keep the host cool;
  it does not change answers or recorded latencies.

No run had an error. Judge cost was about $0.29 at list-price equivalent.

**Baseline** (`20261027T002255Z-baseline`):

| Metric | Value |
|---|---|
| Passage hit | 68.5% |
| Passage recall | 48.8% |
| Groundedness | 80.3% |
| Correctness | 40.7% |
| Refusal | 83.3% |
| Abstention (12 questions) | 91.7% |
| No-context refusals | 0% |

### Pre-registered predictions

Each difference is mutation minus baseline, with its 95% interval and the exact
McNemar p (passage recall has no p; its interval decides).

| Mutation | Metric | Predicted | Baseline → mutation | Difference (95% CI) | p | Result |
|---|---|---|---|---|---|---|
| weaker-llm | correctness | down | 40.7% → 24.1% | −16.7 (−27.8 to −7.4) | 0.004 | **detected** |
| weaker-llm | groundedness | down | 80.3% → 66.7% | −13.6 (−24.2 to −3.0) | 0.035 | **detected** |
| top-n-2 | passage hit | down | 68.5% → 53.7% | −14.8 (−24.1 to −5.6) | 0.008 | **detected** |
| top-n-2 | passage recall | down | 48.8% → 34.0% | −14.8 (−22.8 to −7.7) | — | **detected** |
| top-n-2 | correctness | down | 40.7% → 37.0% | −3.7 (−13.0 to +5.6) | 0.69 | not detected (MDE 12.7) |
| threshold-078 | passage hit | down | 68.5% → 48.1% | −20.4 (−33.3 to −9.3) | 0.003 | **detected** |
| threshold-078 | passage recall | down | 48.8% → 32.1% | −16.7 (−25.9 to −8.0) | — | **detected** |
| threshold-078 | no-context refusals | up | 0% → 24.2% | +24.2 (+15.2 to +34.8) | <0.001 | **detected** |
| threshold-078 | correctness | down | 40.7% → 31.5% | −9.3 (−18.5 to 0.0) | 0.125 | not detected (MDE 13.7) |
| weak-prompt | groundedness | down | 80.3% → 56.1% | −24.2 (−37.9 to −10.6) | 0.002 | **detected** |
| weak-prompt | abstention | down | 91.7% → 75.0% | −16.7 (−41.7 to 0.0) | 0.5 | not detected (12 questions; judge not validated) |

**Summary.** 8 of the 11 predicted changes were detected. All 13 "no change"
predictions held, including:

- retrieval unchanged under the weaker model and the weaker prompt;
- no-context refusals unchanged where nothing touched the threshold;
- no aggregate change for the removed document and the edited fact.

### What the evaluation caught, and what it missed

- **Retrieval metrics.** They respond to every retrieval mutation and stay still under
  the others. Passage-level metrics carry the signal: under threshold-078 a quarter of
  the questions got no context at all.
- **Groundedness.** It catches the two generation mutations. The weaker model and the
  prompt without grounding rules both add unsupported claims.
- **Correctness under retrieval mutations.** It did not detect top-n-2 or
  threshold-078. Correctness is already 41% at baseline, mostly because the evidence
  passage is often not retrieved even with four passages (see `docs/failure-taxonomy.md`).
  Removing passages therefore turns few correct answers into wrong ones, and the drops
  (3.7 and 9.3 points) are below what 66 questions can detect. The retrieval metrics
  are the sensitive measure for these mutations.
- **Abstention.** It moved in the predicted direction under weak-prompt, but 12
  questions cannot show a 17-point drop to be significant, and this judge is not
  validated. This is a blind spot, as `docs/judges.md` warned.
- **Not pre-registered: refusal under weak-prompt.** The refusal judge rose by 14.8
  points (p = 0.008). Without the instruction to decline, the model gives whatever the
  passages contain, so it holds back less. This is the same change in behaviour that
  groundedness detects as harm.

### Targeted checks

- **remove-document.** As predicted, the aggregate metrics did not move. The targeted
  check is weaker than planned: both affected questions (q-b005-1, q-b005-2) already
  failed correctness at baseline, so the mutation could not make them fail. The removal
  does show in passage recall for q-b005-1 (1.0 → 0.5), the evidence from MIR-26-03 that
  is no longer there.
- **knowledge-edit.** The edited report was ingested: AnythingLLM's stored copy says
  "repair yard in Cape Girardeau, Missouri". But retrieval never returned that sentence
  for q-b029-1, in the baseline or the mutation, and both answers said the documents do
  not contain the answer. The check is **inconclusive**: it cannot show whether answers
  follow an edited fact the system never retrieves.

**Lesson for both checks.** A targeted mutation only tests something if the baseline
gets the targeted questions right first. The questions and the edited fact should be
chosen from those the baseline answers correctly.

### Measurement noise

Every config re-embeds the corpus in its own workspace. The embeddings are not
bit-identical between workspaces: passage scores differ by about ±0.0005, which
occasionally reorders two near-tied passages. This explains the +1.9-point passage hit
change, from a single question, under mutations that do not touch retrieval. It is far
below the size of any detected effect.
