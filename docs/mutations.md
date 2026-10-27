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

Not yet run. Results will be added from `results/mutations/comparisons/`, each linked
to its run manifests.
