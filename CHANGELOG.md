# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Spike findings for the AnythingLLM 1.18.0 developer API (`docs/spike-findings.md`).
- Project scaffold: uv, ruff, pyright (strict), pytest, and PR CI with no LLM calls.
- `docker-compose.yml` for AnythingLLM 1.18.0, pinned by digest and bound to localhost.
- Typed AnythingLLM client and idempotent config-as-code provisioning (`rag-eval provision`).
- Runner with one new thread per case, trace records and run manifests (`rag-eval run`).
- Baseline configuration (`configs/baseline.yaml`) and ADRs 0001–0003.
- Corpus: 77 NTSB Marine Investigation Reports (2025–2026), pinned by URL and SHA-256 in
  `corpus/sources.yaml` and fetched with `rag-eval fetch-corpus`.
- Dataset tooling (`rag-eval dataset ...`): approved question dimensions
  (`data/dimensions.yaml`), curated report links, a deterministic batched generation plan,
  generation with Claude Sonnet 5.5 (cost estimate, cap and cache), verbatim-evidence
  validation, automated checks and coverage report, a local review page with import
  of decisions, and a stratified dev/test split. ADR 0004.
- Session generation path (`rag-eval dataset brief`, `check-batch`, `collect`): candidates
  written by Claude Code helper agents from per-batch briefs, under the same checks.
- Dataset: 110 approved items, stratified into `data/dev.jsonl` (44) and `data/test.jsonl` (66).
- System under test runs with thinking off: `qwen3-nothink:8b`, built from
  `models/qwen3-nothink.Modelfile` (same weights as qwen3:8b). ADR 0005.
- Closed-book baseline: `configs/closed-book.yaml` (`corpus.closed_book`, no documents)
  and a marking page with import (`rag-eval dataset closed-book-export` / `-import`).
  Result: 3/90 answered correctly from memory (95% CI 1–9%), all on questions a guess
  could get right (`docs/results/closed-book-baseline.md`).
- Error analysis setup: two weak configs (`configs/error-analysis/`), 132 dev traces
  (baseline plus both), and a blind, shuffled open-coding page with evidence
  highlighting (`rag-eval review traces-export` / `traces-import`). ADR 0006.
- Failure taxonomy (`docs/failure-taxonomy.md`): open coding of the 132 dev traces
  grouped into six failure modes.
- Evaluators (`rag-eval evaluate`): deterministic retrieval metrics (hit rate, recall@k,
  MRR at passage and document level, ADR 0009) and four binary LLM judges as DeepEval
  custom metrics (groundedness, correctness, abstention, unnecessary refusal; ADR 0008),
  with cached replies, cost estimates and a `--max-cost` guard.
- Pluggable judge models (ADR 0007): Claude Haiku 5.5 via Claude Code headless by
  default, with a Claude API template in `configs/judges/` and a backend for any
  OpenAI-compatible endpoint.
- Retrieval scores for the three error-analysis runs.
- Judge validation tooling (`rag-eval review judges-export` / `judges-import` /
  `judges-validate`): a labelling page with Pass/Fail per judge, a judge-dev/judge-test
  split by question, and TPR/TNR with Wilson intervals. ADR 0010.
- Judge validation results (`docs/judges.md`): 608 Pass/Fail labels on 217 dev traces
  from five configs (two added in `configs/judge-validation/`), one round of prompt
  clarification on judge-dev, and TPR/TNR on judge-test.
- Statistics module (`src/rag_eval/stats/`, standard library only): Wilson and bootstrap
  intervals, cluster-robust standard errors and cluster bootstrap for repeated runs, exact
  McNemar and paired (and paired-cluster) bootstrap for config comparisons, run consistency, and minimum
  detectable effect; unit-tested against hand-computed values. ADR 0011.
- Mutation experiments (`docs/mutations.md`): six mutations (weaker LLM, top-N 2, threshold
  0.78, weakened prompt, removed document, knowledge edit), expectations pre-registered
  before any run, one run per config on the test set, and `rag-eval mutations compare`.
  8 of 11 predicted changes detected; all 13 no-change predictions held.
