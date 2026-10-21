# 0004: Dataset size, corpus size and how candidates are generated

- **Status:** Accepted
- **Date:** 2026-10-21

## Context

This repository is a guide to evaluating a RAG system, not a benchmark. It needs enough questions to show the methods working (error analysis, judge validation, mutation detection with honest confidence intervals), and it should keep the cost of reproducing it, and the owner's review effort, modest.

Three quantities are easy to conflate, so they are worth separating:

| Quantity | What it drives |
|---|---|
| **Number of reports in the corpus** | How hard retrieval is: how many similar-looking distractors the retriever must get past. Hardly affects generation cost or review effort, because each question is written from, and checked against, only the one or two reports it is about. |
| **Number of candidate questions** | Generation cost and the owner's review effort, almost linearly. |
| **Number of approved questions** | Statistical power: confidence-interval width and the smallest effect a mutation must have to be detected. |

## Decisions

### Keep all 77 reports

The corpus stays at all 77 NTSB Marine Investigation Reports from 2025–2026 (ADR 0003). Fewer reports would not reduce generation calls. It would make retrieval easier and less realistic, since the corpus has dozens of structurally identical reports (groundings, engine-room fires, bridge contacts). Finding the right one among look-alikes is the realistic test a RAG evaluation guide should show. The extra cost is one embedding pass of about two minutes on a local embedder.

We considered and rejected restricting the corpus to the 30 reports from 2026. That would remove the two best-known casualties (Dali / Francis Scott Key Bridge and Titan, both 2025 reports) and so the biggest memorisation risk, but it makes retrieval easier and concentrates questions on fewer reports. Memorisation is instead measured by the closed-book baseline, and those reports can be excluded later with `corpus.exclude` if the measurement calls for it.

### Generate 110 candidates, aiming for about 70–85 approved

The plan's starting point was 140 candidates for 80–100 approved items. We reduced it to 110 to cut generation cost and review effort by about a fifth. The cost is somewhat less statistical power. For example, an 80% pass rate measured on 45 held-out items has a 95% Wilson interval of 66–89% (±11.5 points), against 68–88% (±10.4) on 55 items. The statistics module reports minimum detectable effects for the actual held-out size, so this trade-off is visible in the results rather than hidden.

### Batch generation per report, with prompt caching

Each generation call covers one report (or one pair of reports for comparison and multi-document questions) and writes all the questions assigned to it, typically 2–3. The report text is the cached prefix. This needs roughly 60–70 calls instead of one call per question (~180 including retries), which cuts the input-token cost by more than half without changing what the generator sees. The cost: items from the same call can be more alike, so near-duplicate detection runs across all candidates, and the per-report cap in `data/dimensions.yaml` limits how much any one report contributes.

### Spend controls

Before any hosted call, generation prints an estimate of calls, tokens and cost and waits for confirmation. `--max-cost` aborts once the running cost would exceed the cap, and results are cached by (model, prompt hash, inputs), so re-running never pays twice for the same call.

## Generator model

**Claude Sonnet 5.5** (`claude-sonnet-5-5`), chosen by the owner to keep cost as low as possible. Estimated ~$3–5 for 110 candidates with batching, against ~$6–10 on Claude Opus 5.5. The expected price is somewhat more candidates rejected in review, which the 110 → 70–85 margin absorbs.

The generator must not be the system under test (`qwen3:8b`), or the dataset would be biased towards what that model finds easy to phrase and answer. Sonnet is a different model family, which meets this. A local Ollama model was rejected: it needs a multi-gigabyte download of a different-family model and would give noticeably weaker questions.

## Pairing reports for multi-document questions

Multi-document questions need report pairs that are genuinely linked. A first attempt
detected links from shared capitalised words in report titles, which paired unrelated
reports on ordinary words (two vessels both named "...Rose", "Cape", "Offshore",
"Terminal"). Links are therefore curated in `data/report_links.yaml`: eight pairs that
share a named bridge, vessel or operator, found in the report texts. Planning checks
that each entity really occurs in both reports. Remaining pairs are reports about the
same kind of casualty (two fires, two groundings), which suit comparison questions.
Long reports are the last choice as pair partners.

## Reading AnythingLLM's parsed text

Evidence quotes must match the text exactly as AnythingLLM parsed it from each PDF,
and chunk-boundary questions need AnythingLLM's actual chunks. Neither is exposed by
the developer API, so `rag-eval dataset extract` reads them read-only from the
container's storage (see `src/rag_eval/client/storage.py`). This is used only to
build the dataset, never during evaluation runs.

## Where money is (and is not) spent

- **Pull-request CI never calls an LLM.** It runs lint, types and unit tests only, so it costs nothing however often it runs.
- **Question generation is a one-time cost.** The reviewed dataset is frozen and committed, and generation calls are cached, so re-running with unchanged prompts is free.
- **The system under test runs locally** on Ollama, at no API cost.
- **Recurring cost comes only from runs the owner starts:** LLM-judge scoring (`rag-eval evaluate`, cached per case) and the manual eval-gate workflow. Each run shows an estimate and asks for confirmation first.

## Consequences

- The held-out (test) set will hold about 40–50 items. Small mutation effects may go undetected; when they do, they are reported as blind spots with the minimum detectable effect, not as passes.
- Every candidate is hand-reviewed by the owner. Generation only proposes; nothing generated is treated as a human label.
