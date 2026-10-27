# 0014: Query mode, not chat mode, for the system under test

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

AnythingLLM answers in one of two modes, set per workspace or per request:

- **chat:** the model answers with or without retrieved context.
- **query:** the model answers only from retrieved context.

The spike (`docs/spike-findings.md`, section 5) compared them on synthetic documents.

**When nothing clears the similarity threshold:**

- **chat mode** still calls the model. In one case the model made up a conversation
  history and confidently answered "3,000 psi"; the document says 412 kPa.
- **query mode** returns the workspace's refusal text with no LLM call (56 ms, no
  sources).

**When context is retrieved but does not hold the answer,** both modes let the model
decide, and it declined.

## Decision

The baseline uses **query mode**, with a distinctive refusal text (`NO_RELEVANT_CONTEXT`),
and every request sends the mode explicitly. The reasons:

- **It matches the task.** This is document question answering: an answer not taken
  from the documents is a failure, not a feature.
- **It avoids confident fabrication.** Chat mode turns every retrieval miss into a
  chance for that.
- **It splits refusals into two kinds:**
  - **deterministic:** nothing retrieved. Recognised by exact match on the refusal text,
    recorded in the trace as `refusal_path: deterministic`, and decided by rule, not by
    a judge (ADR 0008).
  - **model decided:** context retrieved but judged insufficient. Judged by the refusal
    and abstention judges.
- **It keeps the trace honest.** The `mode` field is always sent, because the default
  (`automatic`) routes to AnythingLLM's agent pipeline, which the trace does not
  describe.

## Consequences

- **Chat mode is a weak config, not the baseline.** It is used only where a weak config
  is wanted: error analysis (`configs/error-analysis/weak-grounding.yaml`) and the
  closed-book baseline, which has no documents to retrieve.
- **The threshold has a visible effect.** A raised similarity threshold produces
  no-context refusals instead of guesses. The threshold mutation showed exactly that:
  24% no-context refusals (`docs/mutations.md`). The eval gate watches that rate
  (ADR 0012).
- **Query mode does not stop guessing from irrelevant context.** The model can still
  guess when the retrieved passages are irrelevant, which is what the groundedness judge
  measures.
