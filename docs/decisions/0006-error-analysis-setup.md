# 0006: How traces are produced and coded for error analysis

- **Status:** Accepted
- **Date:** 2026-10-23

## Context

Error analysis comes before any evaluator: the failure modes found here decide what the
judges in Phase 4 measure. It needs enough traces with enough different failures to
find the modes worth measuring, and notes that describe what actually went wrong rather
than what the coder expected.

## Decisions

### Three configs on the dev set: 132 traces

The baseline alone gives 44 dev traces, too few and too uniform. Two deliberately weak
configs add failures of different kinds, each extending the baseline and changing one
thing:

| Config | Change | Failures it should surface |
|---|---|---|
| `configs/baseline.yaml` | none | whatever the real system gets wrong |
| `configs/error-analysis/weak-retrieval.yaml` | `top_n` 4 → 1 | missing context, especially for multi-span and two-report questions |
| `configs/error-analysis/weak-grounding.yaml` | chat mode, permissive prompt | answers not grounded in the documents, no refusals on unanswerable questions |

Neither needs a new model download. These are not the Phase 7 mutations: they exist to
produce failures to study, not to test whether the evaluators catch them, and they run on
the dev set only. The test set is not used.

### Blind, shuffled coding with the expected evidence shown

The coding page (`rag-eval review traces-export`) shows the 132 traces in a fixed
shuffled order, with no run or config name, so a note is not coloured by knowing that a
trace came from a weak config. (The number of retrieved passages can still give the
weak-retrieval config away; hiding that would hide the evidence needed for the note.)

Each trace shows whether each expected evidence quote was retrieved (fully, partly or
not at all) and highlights it in the retrieved passages. This is a deterministic text
match, not a judgement, and it makes the "first upstream failure" visible: if the
evidence was not retrieved, the note should say so and not also blame the answer.

### Notes are free text, one per trace

Open coding records the coder's own words, or "nothing went wrong". There are no
predefined categories: those come from axial coding afterwards, grouping the notes into
failure modes (`docs/failure-taxonomy.md`). The notes file is stored unmodified under
`data/labels/error-analysis/`.

## Consequences

- Each config has its own workspace (ADR 0002), so each embeds the corpus once
  (about 1.5 minutes); the three runs take about 10 minutes in total.
- The baseline retrieved all expected evidence for only 11 of 36 answerable dev
  questions (fully or partly for 30). Some "partly" results come from evidence that spans
  a chunk boundary, which the dataset includes on purpose.
