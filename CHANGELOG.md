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
- Dataset: 110 approved items (blanket approval by the owner, recorded as such),
  stratified into `data/dev.jsonl` (44) and `data/test.jsonl` (66).
- System under test runs with thinking off: `qwen3-nothink:8b`, built from
  `models/qwen3-nothink.Modelfile` (same weights as qwen3:8b). ADR 0005.
- Closed-book baseline: `configs/closed-book.yaml` (`corpus.closed_book`, no documents)
  and a marking page with import (`rag-eval dataset closed-book-export` / `-import`).
  Result: 3/90 answered correctly from memory (95% CI 1–9%), all on questions a guess
  could get right (`docs/results/closed-book-baseline.md`; marked by Claude at the
  owner's request).
