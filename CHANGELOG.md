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
