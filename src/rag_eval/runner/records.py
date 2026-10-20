"""Trace records (results/<run-id>/traces.jsonl) and the run manifest (manifest.json)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_eval.client.models import ChatResponse
from rag_eval.client.text import split_reasoning, strip_metadata_header
from rag_eval.dataset.schema import Case


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievedPassage(_Record):
    doc_id: str | None
    chunk_id: str | None
    text: str
    score: float | None


class TokenUsage(_Record):
    prompt: int | None
    completion: int | None
    total: int | None


class TraceRecord(_Record):
    run_id: str
    case_id: str
    repeat: int
    split: str
    question: str
    dimensions: dict[str, str]
    mode: Literal["chat", "query"]
    workspace_slug: str
    thread_slug: str | None
    answer: str | None
    reasoning: str | None
    response_type: str | None
    refusal_path: Literal["deterministic"] | None
    retrieved: list[RetrievedPassage]
    latency_s: float | None
    tokens: TokenUsage | None
    model: str | None
    error: str | None
    started_at: str


def build_trace(
    *,
    run_id: str,
    case: Case,
    repeat: int,
    mode: Literal["chat", "query"],
    refusal_response: str,
    workspace_slug: str,
    thread_slug: str,
    response: ChatResponse,
    latency_s: float,
    started_at: str,
) -> TraceRecord:
    reasoning, answer = split_reasoning(response.text_response or "")
    retrieved = [
        RetrievedPassage(
            doc_id=s.title, chunk_id=s.id, text=strip_metadata_header(s.text), score=s.score
        )
        for s in response.sources
    ]
    # In query mode, AnythingLLM answers with the refusal text without calling the LLM
    # when no chunk clears the similarity threshold.
    deterministic_refusal = mode == "query" and not retrieved and answer == refusal_response.strip()
    m = response.metrics
    return TraceRecord(
        run_id=run_id,
        case_id=case.id,
        repeat=repeat,
        split=case.split,
        question=case.question,
        dimensions=case.dimensions,
        mode=mode,
        workspace_slug=workspace_slug,
        thread_slug=thread_slug,
        answer=answer if response.text_response is not None else None,
        reasoning=reasoning,
        response_type=response.type,
        refusal_path="deterministic" if deterministic_refusal else None,
        retrieved=retrieved,
        latency_s=latency_s,
        tokens=TokenUsage(
            prompt=m.prompt_tokens, completion=m.completion_tokens, total=m.total_tokens
        ),
        model=m.model,
        error=response.error,
        started_at=started_at,
    )


def error_trace(
    *,
    run_id: str,
    case: Case,
    repeat: int,
    mode: Literal["chat", "query"],
    workspace_slug: str,
    thread_slug: str | None,
    error: str,
    started_at: str,
) -> TraceRecord:
    return TraceRecord(
        run_id=run_id,
        case_id=case.id,
        repeat=repeat,
        split=case.split,
        question=case.question,
        dimensions=case.dimensions,
        mode=mode,
        workspace_slug=workspace_slug,
        thread_slug=thread_slug,
        answer=None,
        reasoning=None,
        response_type=None,
        refusal_path=None,
        retrieved=[],
        latency_s=None,
        tokens=None,
        model=None,
        error=error,
        started_at=started_at,
    )


class PinnedModel(_Record):
    name: str
    digest: str


class FileRef(_Record):
    path: str
    sha256: str


class CorpusDocumentRef(_Record):
    sha256: str
    anythingllm_location: str


class Manifest(_Record):
    run_id: str
    started_at: str
    finished_at: str | None
    git_commit: str | None
    git_dirty: bool | None
    package_version: str
    python_version: str
    config: FileRef
    config_name: str
    config_hash: str
    resolved_config: dict[str, object]
    anythingllm_image: str
    anythingllm_image_verified: bool
    workspace_slug: str
    llm: PinnedModel
    embedder: PinnedModel
    temperature: float
    seed: None  # AnythingLLM does not forward a seed to Ollama
    judge_model: str | None
    judge_prompt_hashes: dict[str, str]
    dataset: FileRef
    n_cases: int
    repeats: int
    corpus_hash: str
    corpus: dict[str, CorpusDocumentRef]
    n_records: int
    n_errors: int
