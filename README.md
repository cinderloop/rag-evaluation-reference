# rag-evaluation-reference

A reference implementation of RAG evaluation: error analysis first, binary LLM judges validated against labels, mutation testing to prove the evaluation catches regressions, and statistics that stay honest at small sample sizes. The system under test is [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm), answering questions over 77 NTSB marine accident reports; the judges are [DeepEval](https://deepeval.com) custom metrics.

> **Companion series:** [The Tester's Guide to RAG Evaluation](https://ashenasserts.hashnode.dev/series/testers-guide-rag-evaluation) explains the concepts behind this project from the ground up, for testers new to AI evaluation. This repository is a finished example that applies every stage of it.

## What is evaluated, and why

A RAG answer can fail in two places: retrieval can miss the passage that holds the answer, or generation can mishandle what it was given (make things up, misread it, hold it back). A single "answer quality" score cannot tell these apart, and an LLM judge's score means nothing until you know how often the judge is right. So this repository:

1. **Looks at real failures before measuring anything** ([failure taxonomy](docs/failure-taxonomy.md)), and builds one evaluator per failure mode that actually occurs.
2. **Measures retrieval deterministically**, against verbatim evidence quotes, at passage level.
3. **Judges generation with binary Pass/Fail judges**, each validated against labels before its numbers are used ([judges](docs/judges.md)).
4. **Breaks the system on purpose** to check the evaluation notices ([mutations](docs/mutations.md)), with expectations written down before any run.
5. **Gates changes** on the metrics that proved trustworthy ([eval gate](docs/decisions/0012-gate-thresholds.md)).

The full walk-through is in [`docs/methodology.md`](docs/methodology.md); one failure traced end to end is in [`docs/worked-example.md`](docs/worked-example.md); what to change for another RAG system is in [`docs/adapting-to-your-rag-app.md`](docs/adapting-to-your-rag-app.md).

## The evaluation pyramid

Cheap and certain at the bottom, expensive and uncertain at the top; use the lowest level that can answer the question.

| Level | What | Cost | Trust |
|---|---|---|---|
| Judged | Groundedness, correctness, abstention, unnecessary refusal: one binary LLM judge per failure mode | judge calls | only after validation (TPR/TNR against labels) |
| Retrieval | Hit rate, recall@k, MRR at passage and document level, matched to evidence quotes | free | exact, given the matching rule |
| Deterministic | No-context refusals (exact match), config and pin checks, unit-tested logic | free | exact |

## Findings

All numbers come from committed runs; each links to its source.

- **Correct answers come from the documents, not from memory.** With no documents, the model answered 3 of 90 questions correctly, all guessable ([closed-book baseline](docs/results/closed-book-baseline.md); [run manifest](results/closed-book/20261022T204917Z-closed-book/manifest.json)).
- **Most failures start at retrieval.** In error analysis on dev, 55 of 66 failures began with the answer's passage not being retrieved ([taxonomy](docs/failure-taxonomy.md); [baseline manifest](results/error-analysis/20261022T222121Z-baseline/manifest.json)).
- **Finding the right report is not finding the answer.** On the 66 test questions the baseline retrieved the right report for 96.3% of answerable questions but a passage holding the evidence for only 68.5%; groundedness 80.3%, correctness 40.7% ([manifest](results/mutations/20261027T002255Z-baseline/manifest.json); [summary](results/mutations/20261027T002255Z-baseline/summary.json)).
- **Two judges are validated, one is not.** Groundedness agrees with labels on 98% of passes and 100% of failures, correctness on 95% and 98%; abstention could not be validated (one failure to test on) ([judges](docs/judges.md); [judge-test result](results/judge-validation/agreement/20261026T222019Z-judge-test.json)).
- **The evaluation catches most deliberate regressions.** Of 11 pre-registered changes across six mutations, 8 were detected; all 13 "no change" predictions held. Correctness alone missed the retrieval mutations; the retrieval metrics caught them ([mutations](docs/mutations.md); [comparisons](results/mutations/comparisons/)).
- **The gate fails every detected regression and passes the baseline** ([thresholds](configs/gate.yaml); per-run results in each `results/mutations/*/gate.json`).

## Quick start

Requirements: Docker, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) running natively on the host (on Apple Silicon this gives the models GPU acceleration; AnythingLLM in Docker reaches it via `host.docker.internal`), and, for the default judge, the [Claude Code](https://claude.com/claude-code) CLI logged in.

```bash
# Setup (once)
ollama pull qwen3:8b && ollama pull nomic-embed-text
ollama create qwen3-nothink:8b -f models/qwen3-nothink.Modelfile   # thinking off (ADR 0005)
docker compose up -d                                                # pinned AnythingLLM, localhost only
cp .env.example .env   # set ANYTHINGLLM_API_KEY as described in the file
set -a; source .env; set +a
uv sync
uv run rag-eval fetch-corpus                                        # ~110 MB from ntsb.gov, checksums verified

# 1. Provision: workspace, documents and settings from the config (idempotent)
uv run rag-eval provision -c configs/baseline.yaml

# 2. Run the test set, then score it (the judge step shows an estimate and asks first)
uv run rag-eval run -c configs/baseline.yaml -d data/test.jsonl
uv run rag-eval evaluate results/<run-id>
```

`rag-eval run` provisions too, so any reported run is reproduced with its config and dataset: `rag-eval run -c <config> -d <dataset>`. A two-document smoke test is in `tests/smoke/`.

## How to read the results

Each run directory under `results/` holds:

| File | Contents |
|---|---|
| `manifest.json` | git commit, config file and hash, AnythingLLM image digest, model digests, dataset and corpus hashes, timestamps |
| `traces.jsonl` | one record per question: the answer, the passages the model saw with scores, latency, token usage |
| `scores.jsonl` | per question: retrieval scores and each judge's verdict with its reason |
| `summary.json` | rates per metric, the judge model and prompt hashes, judge cost |
| `gate.json` | gate result per gated metric (after `rag-eval gate check`) |

Read a failure from the bottom of the pyramid up: did retrieval find the evidence (`scores.jsonl` → `retrieval`)? If not, the answer could not be right, whatever the judges say. If it did, the judges' reasons show what generation did with it. [`docs/worked-example.md`](docs/worked-example.md) does this for one case.

## Corpus

The system is evaluated on the **77 NTSB Marine Investigation Reports published in 2025 and 2026**: public-domain US government reports, each on one maritime casualty, chosen because they postdate the model under test (see [ADR 0003](docs/decisions/0003-corpus-choice.md)). Courtesy: National Transportation Safety Board; see [`corpus/ATTRIBUTION.md`](corpus/ATTRIBUTION.md).

The PDFs are not committed. `uv run rag-eval fetch-corpus` downloads them from ntsb.gov into `corpus/documents/` and verifies each against the SHA-256 pinned in [`corpus/sources.yaml`](corpus/sources.yaml).

The full list, with a link to each report, is in [`corpus/README.md`](corpus/README.md).

## Dataset

110 questions about the reports (44 dev, 66 test), each with verbatim evidence quotes, built with `rag-eval dataset ...` and explained in [ADR 0004](docs/decisions/0004-dataset-size-and-generation.md). Known weak items are listed in [`data/candidates/review-notes.md`](data/candidates/review-notes.md).

Self-contained pages show the review steps; download and open them in a browser:

- [`data/review/review.html`](data/review/review.html): reviewing generated candidates (approve, edit or reject, with each quote shown in its report context).
- [`data/review/traces.html`](data/review/traces.html): open coding for error analysis, with each trace's expected evidence highlighted in the retrieved passages (traces shuffled, config hidden).
- [`data/review/judge-labels.html`](data/review/judge-labels.html): marking Pass or Fail per judge on real traces, to measure how well each judge agrees ([ADR 0010](docs/decisions/0010-judge-validation.md)).
- [`data/review/closed-book.html`](data/review/closed-book.html): marking the model's answers with no documents. The marks made with it are in [`data/labels/closed-book/`](data/labels/closed-book/) and the result in [`docs/results/closed-book-baseline.md`](docs/results/closed-book-baseline.md): 3 of 90 questions answered correctly from memory, all of them guessable.

## Evaluators

`rag-eval evaluate results/<run-id>` scores a run and writes `scores.jsonl` (one record per trace) and `summary.json` next to its traces. Each evaluator targets failure modes from the [taxonomy](docs/failure-taxonomy.md):

| Evaluator | Kind | Catches |
|---|---|---|
| Retrieval: hit rate, recall@k, MRR, at passage and document level | Deterministic ([ADR 0009](docs/decisions/0009-retrieval-metrics.md)) | Answer not retrieved (A), and the retrieval side of B and C |
| Groundedness | Binary LLM judge | Guesses, borrowed facts, extra claims (B, E, F) |
| Correctness against the expected answer | Binary LLM judge | Misreadings and wrong answers (D) |
| Abstention on unanswerable questions | Binary LLM judge | Trick questions answered (E) |
| Unnecessary refusal on answerable questions | Binary LLM judge | Answers held back (C) |

Each judge is a DeepEval custom metric that returns Pass or Fail with a reason; its prompt, with three labelled examples, is a plain file in [`src/rag_eval/evaluators/judges/prompts/`](src/rag_eval/evaluators/judges/prompts/) ([ADR 0008](docs/decisions/0008-binary-judges.md)). Rules decide instead of a judge where they can.

**Choosing the judge model.** The judge is set by a config file (`--judge-config`). The default runs Claude Haiku 5.5 through Claude Code's headless mode, which uses a logged-in Claude subscription instead of an API key: the cheapest option for this guide ([ADR 0007](docs/decisions/0007-judge-model.md)). A template switches to the Claude API:

| Config | Judge | Needs |
|---|---|---|
| [`claude-code-haiku.yaml`](configs/judges/claude-code-haiku.yaml) (default) | Claude Haiku 5.5 via `claude -p` | Claude Code, logged in |
| [`anthropic-api.yaml`](configs/judges/anthropic-api.yaml) | Claude via the API | `ANTHROPIC_API_KEY` |

Other models plug in the same way: a config with `provider: openai_compatible`, a `base_url`, an `api_key_env` and a pinned `model` uses any Chat Completions endpoint, and a new kind of provider is one small class in [`backends.py`](src/rag_eval/evaluators/judges/backends.py). The judge must come from a different model family than the system under test. Judge replies are cached by provider, model and full prompt, every run shows an estimate first, and `--max-cost` stops a run before it exceeds a cap. Judges are validated against Pass/Fail labels before their scores are used ([results](docs/judges.md)) (`rag-eval review judges-export`, `judges-import`, `judges-validate`; [ADR 0010](docs/decisions/0010-judge-validation.md)).

## Eval gate

`rag-eval gate run` runs the baseline config on the test set, scores it, and exits with an error if a gated metric (passage hit, groundedness, correctness, no-context refusals) moves past its threshold in [`configs/gate.yaml`](configs/gate.yaml). The thresholds are derived from the baseline with `rag-eval gate derive`, at a 5% false-alarm rate for the gate as a whole ([ADR 0012](docs/decisions/0012-gate-thresholds.md)). The gate fails all four mutations the evaluation detects and passes the baseline. `rag-eval gate check <run>` checks an already scored run.

Pull-request CI stays fast and makes no LLM calls; the gate runs manually through [`eval-gate.yml`](.github/workflows/eval-gate.yml) on a self-hosted runner that can serve the models ([ADR 0013](docs/decisions/0013-ci-and-eval-gate.md)).

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

Design notes: [`docs/spike-findings.md`](docs/spike-findings.md) and the decision records in [`docs/decisions/`](docs/decisions/).

## License

Code: [MIT](LICENSE). The NTSB reports are US government works in the public domain; see [`corpus/ATTRIBUTION.md`](corpus/ATTRIBUTION.md).
