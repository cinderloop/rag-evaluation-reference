"""Executes a dataset against a provisioned config and writes traces plus a manifest.

Every case (and every repeat) runs in a new AnythingLLM thread, so no chat history
carries over: shared history would leak answers and would also let AnythingLLM
backfill context from earlier turns that never shows up in ``sources``. Cases run
sequentially; temperature-0 determinism was only observed for sequential calls.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx

from rag_eval.client import AnythingLLMClient, AnythingLLMError
from rag_eval.config import RunConfig
from rag_eval.dataset.schema import Case
from rag_eval.provision import ProvisionResult
from rag_eval.runner.records import (
    CorpusDocumentRef,
    FileRef,
    Manifest,
    PinnedModel,
    TraceRecord,
    build_trace,
    error_trace,
)

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id(config: RunConfig) -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{config.name}"


def file_ref(path: Path) -> FileRef:
    return FileRef(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except OSError, subprocess.CalledProcessError:
        return None
    return done.stdout.strip()


def _run_case(
    *,
    client: AnythingLLMClient,
    config: RunConfig,
    workspace_slug: str,
    run_id: str,
    case: Case,
    repeat: int,
) -> TraceRecord:
    mode = config.generation.mode
    started_at = _now()
    thread_slug: str | None = None
    try:
        thread_slug = client.new_thread(workspace_slug, f"{run_id} {case.id} r{repeat}")
        t0 = time.perf_counter()
        response = client.thread_chat(workspace_slug, thread_slug, case.question, mode)
        latency_s = time.perf_counter() - t0
    except (AnythingLLMError, httpx.HTTPError) as error:
        return error_trace(
            run_id=run_id,
            case=case,
            repeat=repeat,
            mode=mode,
            workspace_slug=workspace_slug,
            thread_slug=thread_slug,
            error=f"{type(error).__name__}: {error}",
            started_at=started_at,
        )
    return build_trace(
        run_id=run_id,
        case=case,
        repeat=repeat,
        mode=mode,
        refusal_response=config.generation.refusal_response,
        workspace_slug=workspace_slug,
        thread_slug=thread_slug,
        response=response,
        latency_s=latency_s,
        started_at=started_at,
    )


def run(
    *,
    client: AnythingLLMClient,
    config: RunConfig,
    config_path: Path,
    provisioned: ProvisionResult,
    image_verified: bool,
    dataset_path: Path,
    cases: list[Case],
    repeats: int,
    out_dir: Path,
    run_id: str,
) -> Path:
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    git_status = _git("status", "--porcelain")
    manifest = Manifest(
        run_id=run_id,
        started_at=_now(),
        finished_at=None,
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=None if git_status is None else bool(git_status),
        package_version=version("rag-eval"),
        python_version=platform.python_version(),
        config=file_ref(config_path),
        config_name=config.name,
        config_hash=config.config_hash(),
        resolved_config=config.model_dump(mode="json"),
        anythingllm_image=config.anythingllm_image,
        anythingllm_image_verified=image_verified,
        workspace_slug=provisioned.workspace_slug,
        llm=PinnedModel(name=config.llm.model.name, digest=config.llm.model.digest),
        embedder=PinnedModel(name=config.embedder.model.name, digest=config.embedder.model.digest),
        temperature=config.llm.temperature,
        seed=None,
        judge_model=None,
        judge_prompt_hashes={},
        dataset=file_ref(dataset_path),
        n_cases=len(cases),
        repeats=repeats,
        corpus_hash=provisioned.corpus.content_hash,
        corpus={
            d.doc_id: CorpusDocumentRef(
                sha256=d.sha256, anythingllm_location=provisioned.locations[d.doc_id]
            )
            for d in provisioned.corpus.documents
        },
        n_records=0,
        n_errors=0,
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")

    n_records = n_errors = 0
    with (run_dir / "traces.jsonl").open("w", encoding="utf-8") as traces:
        for repeat in range(repeats):
            for case in cases:
                record = _run_case(
                    client=client,
                    config=config,
                    workspace_slug=provisioned.workspace_slug,
                    run_id=run_id,
                    case=case,
                    repeat=repeat,
                )
                traces.write(record.model_dump_json() + "\n")
                traces.flush()
                n_records += 1
                n_errors += record.error is not None
                log.info(
                    "[%d/%d] %s r%d %s",
                    n_records,
                    len(cases) * repeats,
                    case.id,
                    repeat,
                    "ERROR " + record.error if record.error else "ok",
                )

    final = manifest.model_copy(
        update={"finished_at": _now(), "n_records": n_records, "n_errors": n_errors}
    )
    manifest_path.write_text(final.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run_dir
