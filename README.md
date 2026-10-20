# rag-evaluation-reference

Reference implementation of RAG evaluation: error analysis, validated binary LLM judges, mutation testing and statistically sound results. System under test: AnythingLLM, with DeepEval.

> **Status:** work in progress. The foundation (provisioning and runner) is in place; the dataset, evaluators and results are not yet.

## Quick start

Requirements: Docker, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) running natively on the host (on Apple Silicon this gives the models GPU acceleration; AnythingLLM in Docker reaches it via `host.docker.internal`).

```bash
# 1. Models (digests are pinned in configs/baseline.yaml and verified on every run)
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 2. AnythingLLM (pinned image, localhost only)
docker compose up -d

# 3. API key and environment (see .env.example)
cp .env.example .env   # then set ANYTHINGLLM_API_KEY as described in the file
set -a; source .env; set +a

# 4. Provision and run (smoke test on two synthetic documents)
uv sync
uv run rag-eval provision -c tests/smoke/smoke.yaml
uv run rag-eval run -c tests/smoke/smoke.yaml -d tests/smoke/cases.jsonl
```

Each run writes `results/<run-id>/traces.jsonl` (one record per case) and `manifest.json` (git commit, config hash, pinned image and model digests, dataset and corpus hashes).

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

Design notes: [`docs/spike-findings.md`](docs/spike-findings.md) and [`docs/decisions/`](docs/decisions/).
