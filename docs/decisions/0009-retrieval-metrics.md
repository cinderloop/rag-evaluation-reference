# 0009: Retrieval metrics at passage and document level

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

Most failures in the taxonomy start at retrieval (modes A, B and C: 55 of 66). Retrieval
can be measured without a judge, because every answerable case has labelled evidence:
verbatim quotes per document (ADR 0004). The labels are deliberately not tied to chunk
IDs, since chunking changes under mutation, so retrieved passages have to be matched to
the quotes by text.

## Decision

**Matching rule.** Texts are normalized (whitespace, quote characters) and compared in
overlapping 6-word shingles (`src/rag_eval/evaluators/matching.py`):

- A retrieved passage is **relevant** if it holds at least one shingle of an evidence
  quote from its own document.
- A quote is **recalled** if the retrieved passages of its document contain the whole
  quote: in one passage, or, for a quote cut by a chunk boundary, with every word of the
  quote covered by a shingle found in one of the passages. A piece shorter than six
  words on one side of a cut cannot be matched, so that quote is not counted.

**Metrics,** for answerable cases only, at cutoff k (default: all retrieved passages,
i.e. the config's top N):

| Metric | Passage level | Document level |
|---|---|---|
| Hit rate | share of cases with a relevant passage in the top k | share with a passage from an evidence document |
| Recall@k | share of evidence quotes recalled | share of evidence documents retrieved |
| MRR | mean of 1 / rank of the first relevant passage (0 if none) | same, by document |

Both levels are reported because they answer different questions. Document level asks
whether the right report was found; passage level asks whether the part of it that
holds the answer was found. On the error-analysis runs the difference is large: the
baseline finds an evidence document for 97% of answerable dev questions but recalls
only 48% of the evidence quotes, and cutting top N from 4 to 1 lowers passage recall
to 19% while document hit rate stays at 97%. A document-level metric alone would
miss most of what goes wrong.

## Consequences

- Unanswerable cases have no retrieval scores; how the system handles them is judged
  (abstention).
- The 6-word shingle can, in principle, match a common phrase that is not evidence;
  quotes are long and specific (they come from the dataset checks), so this is rare,
  and it can only make retrieval look slightly better than it is, never worse.
- Recall counts a quote only when it is retrieved in full, so a case that retrieved
  half of a two-sentence quote counts it as missed, while its hit still counts.
