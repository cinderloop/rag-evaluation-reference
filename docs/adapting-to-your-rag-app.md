# Adapting this to your own RAG application

The method (error analysis, binary judges validated against labels, mutation testing,
honest statistics, an eval gate) does not depend on AnythingLLM or on marine accident
reports. This page lists what is specific to this repository and what to replace.

## What stays the same

These parts read only the trace and dataset formats, so they work unchanged for any RAG
system:

- the trace record format (`src/rag_eval/runner/records.py`);
- the evaluators (`src/rag_eval/evaluators/`);
- judge validation (`src/rag_eval/review/`);
- statistics (`src/rag_eval/stats/`);
- mutation comparison (`src/rag_eval/mutations/compare.py`);
- the gate (`src/rag_eval/gate/`).

## What to replace

| Part | In this repository | What to do for your system |
|---|---|---|
| **Client** | `src/rag_eval/client/`: AnythingLLM's developer API | A client that sends one question and returns the answer **and the passages the model actually saw**, with document IDs and scores. If your system cannot return its retrieved context, add that first: without it, retrieval and generation failures cannot be told apart. |
| **Provisioning** | `src/rag_eval/provision/`: workspace, documents and settings from a config | Whatever puts your system into a known state from a config file. Keep it idempotent and record a config hash in every run. |
| **Runner** | `src/rag_eval/runner/`: one fresh thread per case | Same rule: no state shared between cases, one `TraceRecord` per case, and a manifest per run (versions, config hash, dataset hash). |
| **Config schema** | `src/rag_eval/config.py`, `configs/baseline.yaml` | Your system's settings: model, retrieval depth, thresholds, prompt, chunking. Mutations are configs that extend the baseline and change one setting. |
| **Corpus** | 77 NTSB reports, `corpus/sources.yaml` | Your documents, pinned by checksum. Check licences before committing anything derived from them. |
| **Dataset** | `data/dev.jsonl`, `data/test.jsonl` | Questions with **verbatim evidence quotes** per document. Keep a held-out test split that nothing is tuned on. Include unanswerable questions, enough to validate an abstention judge (see below). |
| **Judge prompts** | `src/rag_eval/evaluators/judges/prompts/` | Rewrite the domain wording and **replace the three examples** with ones from your own labelled traces. Keep one failure mode per judge and a Pass/Fail output. |
| **Judge model** | `configs/judges/` | Any model from a different family than your system under test. Validate it before trusting it. |
| **Gate thresholds** | `configs/gate.yaml` | Re-derive from your own baseline (`rag-eval gate derive`), never copy these numbers. |

## In order, for a new system

1. **Client and runner first.** Run a handful of questions and check that every trace
   shows the passages the model saw.
2. **A small dataset** (dev only): enough questions to see failures. Check the evidence
   quotes against your documents' text.
3. **Error analysis** on dev traces: open coding, then grouping into failure modes. Write
   your own `failure-taxonomy.md`; your system will fail differently from this one.
4. **One evaluator per failure mode.** Use a deterministic check where a rule can decide
   and a binary judge otherwise. Skip modes that do not occur.
5. **Validate each judge** against labels on dev traces, about 50 Pass and 50 Fail each.
   Plan the dataset for this from the start: a judge for a rare failure needs enough
   examples of that failure, which may mean more questions of that kind or weak configs
   that provoke it.
6. **Test set and mutations.** Write expected effects down before running, then run each
   mutation and compare. Fix the blind spots the mutations reveal, or document them.
7. **Gate.** Derive thresholds from a baseline run, preferably with three repeats, and
   run the gate before every change to the system.

## Pitfalls seen in this project

- **Repeated headers mislead retrieval.** A running header on every chunk makes all
  chunks of one document look alike to the retriever (`docs/worked-example.md`).
- **Low baselines hide regressions.** Correctness at 41% barely moved when retrieval got
  worse; retrieval metrics were the sensitive measure.
- **Targeted tests need answered questions.** A mutation aimed at a question the
  baseline already fails cannot show anything.
- **Chat history leaks.** Shared history changes what the model sees (and in AnythingLLM
  it is not visible in the returned sources). Use a fresh session per case.
