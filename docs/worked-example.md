# Worked example: from a failing score to its root cause

One real failure from the baseline run on the test set (`20261027T002255Z-baseline`,
[manifest](../results/mutations/20261027T002255Z-baseline/manifest.json)), followed
from the metric that flagged it to the reason it happened. The point is the order of the
questions: the evaluation is built so that each step narrows down where the failure
started.

## 1. A metric fails

**The question** (q-b029-1):

> After the towing vessel City of Louisville ran aground near Thebes, Illinois, to which
> city's repair yard was it brought, and on what date was it hauled out of the water
> there?

**The expected answer:** it was refloated and towed to a repair yard in Paducah,
Kentucky, where it was dry docked on August 10.

**The system's answer:** "The documents do not contain the answer."

The scores for this trace (`results/mutations/20261027T002255Z-baseline/scores.jsonl`):

| Evaluator | Result | What it says |
|---|---|---|
| correctness | **Fail** | declines, while the expected answer gives Paducah and August 10 |
| groundedness | Pass | saying the passages do not contain the answer is supported |
| refusal | Pass | none of the retrieved passages mention the repair yard, so declining is right |
| document hit | 1 | the right report (MIR-25-19) was retrieved |
| passage hit | **0** | no retrieved passage holds any part of the evidence quote |

## 2. Retrieval or generation?

Each evaluator answers a different question, and together they say where the failure
started:

- **Groundedness and refusal pass.** The model made nothing up and held nothing back.
  Given the four passages it saw, declining was the right answer, so generation
  behaved correctly.
- **Document hit = 1, passage hit = 0.** The retriever found the right report but not
  the part of it that answers the question. The failure is in retrieval.

This is mode A in the failure taxonomy ("answer not retrieved, model says so"), the most
common failure on the dev set. Looking at correctness alone would suggest the model got
it wrong; the other evaluators show the model was never given the answer.

## 3. Why retrieval missed it

The trace records the four passages the model saw and their similarity scores. All four
come from the right report:

| Rank | Score | Passage begins |
|---|---|---|
| 1 | 0.8136 | Grounding of Towing Vessel City of Louisville MIR-25-19 … On July 29, 2023, about 1449 local time, the towing vessel City of Louisville was upbound … |
| 2 | 0.7740 | Grounding of Towing Vessel City of Louisville MIR-25-19 … Casualty Summary … near Thebes, Illinois … |
| 3 | 0.7693 | Grounding of Towing Vessel City of Louisville MIR-25-19 … door of the wheelhouse; he estimated that he returned to the helm at 1447 … |
| 4 | 0.7634 | Grounding of Towing Vessel City of Louisville MIR-25-19 … 1 Factual Information 1.1 Background … |

AnythingLLM's vector search (`POST /api/v1/workspace/{slug}/vector-search` on the same
workspace, top 20) shows where the evidence passage ("…towed to a repair yard in Paducah,
Kentucky, where it was dry docked on August 10") actually ranks: **6th, at 0.7565**. That
is 0.007 below the fourth passage. The twelve best matches all come from MIR-25-19.

**The cause** is visible in the passages themselves. Every chunk of this report starts
with the PDF's running header, "Grounding of Towing Vessel City of Louisville
MIR-25-19". That header shares most of its words with the question ("towing vessel City
of Louisville ran aground"), so every chunk of the report scores well. Chunks that also
mention Thebes or the grounding score best. The chunk with the answer uses different
words from the question ("towed to a repair yard … dry docked" against "brought to …
hauled out of the water"), so it lands just outside the top four.

## 4. What would fix it, and how to check

Each fix is a config change that can be tested the same way as a mutation: run it, score
it, and compare with the baseline (`rag-eval mutations compare`).

| Candidate fix | Expected effect |
|---|---|
| Retrieve more passages (top N 6 instead of 4) | this question passes; more text per answer for every question |
| Rerank (AnythingLLM's `vectorSearchMode: rerank`) | a reranker scores the question against each passage's content, not the shared header |
| Strip running headers before embedding | chunks of one report stop looking alike; needs a preprocessing step |

A fix is accepted only if it improves passage hit and correctness without lowering
groundedness, measured on the dev set. The test set and the eval gate then confirm it.

## 5. Why this case matters beyond one question

The same pattern explains the inconclusive knowledge-edit mutation (`docs/mutations.md`).
That mutation edited this exact sentence (Paducah to Cape Girardeau), but because
retrieval never returns it, the edit could not reach the answer. A targeted test is only
as good as the baseline's ability to answer the targeted question.
