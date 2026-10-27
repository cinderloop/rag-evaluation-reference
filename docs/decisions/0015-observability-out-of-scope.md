# 0015: Observability and tracing platforms are out of scope

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

LLM observability platforms (Langfuse, and similar tools such as LangSmith, Arize
Phoenix or Confident AI's hosted platform for DeepEval) record traces from a live
application, show them in dashboards, collect user feedback, and run evaluators on
production traffic. A production RAG system usually needs one.

## Decision

This repository uses none. Results are plain files:

- `results/<run-id>/` holds `traces.jsonl`, `manifest.json`, `scores.jsonl` and
  `summary.json`;
- the review steps are self-contained HTML pages.

DeepEval's telemetry is turned off, and its results are not sent to a hosted platform.

**Why:**

- **Reproducibility from the repository alone.** Every reported number can be traced to
  a committed file and recomputed without an account or a running service.
- **The subject is the evaluation method.** Error analysis, validated judges, mutation
  testing and statistics work the same whichever tool stores the traces.
- **No extra infrastructure.** A guide that needed a tracing server would be harder to
  run and to adapt.

## Consequences

**A production extension.** An adopter of these methods would usually add an
observability platform and connect it in three places:

1. **Collect traces from live traffic** in the format of `TraceRecord` (question,
   retrieved passages with scores, answer, model and latency), which is what the
   evaluators read.
2. **Sample production traces** for error analysis and for new judge labels, which keeps
   the failure taxonomy and the judges' validation current.
3. **Run the validated judges on a sample of production traffic**, and alert on the same
   gated metrics as the eval gate (ADR 0012).

**What the files cannot do.** Without a platform there is no live dashboard and no
user-feedback signal. Monitoring between releases is the adopter's responsibility.
