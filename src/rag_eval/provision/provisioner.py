"""Config-as-code provisioning: make AnythingLLM match a RunConfig, idempotently.

Each config gets its own workspace, named after a hash of the config and the corpus
content. A workspace with that name therefore always holds documents embedded with
that config's chunking and embedder. Re-running provisioning only fills in what is
missing and re-applies (and verifies) every setting.

Chunking and the embedder are instance-wide in AnythingLLM. They are applied before
any document is embedded, so configs that differ in them must be provisioned and run
one after another, never side by side. Changing the embedder makes AnythingLLM delete
every document from every workspace; the next provisioning of any workspace re-ingests.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from rag_eval.client import AnythingLLMClient, AnythingLLMError
from rag_eval.client.models import Workspace
from rag_eval.config import RunConfig
from rag_eval.provision.corpus import Corpus, load_corpus

log = logging.getLogger(__name__)

_EMBEDDER_KEYS = ("EmbeddingEngine", "EmbeddingModelPref")


@dataclass(frozen=True)
class ProvisionResult:
    workspace_slug: str
    workspace_name: str
    corpus: Corpus
    locations: dict[str, str]
    uploaded: list[str] = field(default_factory=list[str])
    removed: list[str] = field(default_factory=list[str])
    instance_changes: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class DocumentSync:
    to_upload: list[str]
    to_remove: list[str]  # AnythingLLM docpaths
    kept: dict[str, str]  # doc_id -> docpath


def workspace_name(config: RunConfig, corpus: Corpus) -> str:
    key = hashlib.sha256(f"{config.config_hash()}:{corpus.content_hash}".encode()).hexdigest()
    return f"{config.name}-{key[:10]}"


def plan_document_sync(
    present: Iterable[tuple[str | None, str]], wanted: Iterable[str]
) -> DocumentSync:
    """Work out which documents to upload and remove.

    ``present`` is (title, docpath) for each document in the workspace. Documents
    that are unknown, untitled or duplicated are removed.
    """
    wanted_ids = list(wanted)
    kept: dict[str, str] = {}
    to_remove: list[str] = []
    for title, docpath in present:
        if title in wanted_ids and title not in kept:
            kept[title] = docpath
        else:
            to_remove.append(docpath)
    to_upload = [doc_id for doc_id in wanted_ids if doc_id not in kept]
    return DocumentSync(to_upload=to_upload, to_remove=to_remove, kept=kept)


def workspace_settings(config: RunConfig) -> dict[str, Any]:
    return {
        "chatProvider": config.llm.provider,
        "chatModel": config.llm.model.name,
        "openAiTemp": config.llm.temperature,
        "topN": config.retrieval.top_n,
        "similarityThreshold": config.retrieval.similarity_threshold,
        "vectorSearchMode": config.retrieval.search_mode,
        "chatMode": config.generation.mode,
        "openAiPrompt": config.generation.system_prompt,
        "queryRefusalResponse": config.generation.refusal_response,
    }


def settings_mismatches(workspace: Workspace, config: RunConfig) -> list[str]:
    expected: dict[str, object] = {
        "chat_provider": config.llm.provider,
        "chat_model": config.llm.model.name,
        "temperature": config.llm.temperature,
        "top_n": config.retrieval.top_n,
        "similarity_threshold": config.retrieval.similarity_threshold,
        "search_mode": config.retrieval.search_mode,
        "chat_mode": config.generation.mode,
        "system_prompt": config.generation.system_prompt,
        "refusal_response": config.generation.refusal_response,
    }
    mismatches: list[str] = []
    for attr, want in expected.items():
        got: object = getattr(workspace, attr)
        same = (
            isinstance(got, (int, float))
            and isinstance(want, (int, float))
            and abs(got - want) < 1e-9
        ) or got == want
        if not same:
            mismatches.append(f"{attr}: expected {want!r}, got {got!r}")
    return mismatches


def instance_env(config: RunConfig) -> dict[str, str]:
    return {
        "EmbeddingEngine": config.embedder.engine,
        "EmbeddingModelPref": config.embedder.model.name,
        "EmbeddingModelMaxChunkLength": str(config.embedder.max_chunk_length),
        "OllamaLLMTokenLimit": str(config.llm.context_window),
    }


def _apply_instance_settings(client: AnythingLLMClient, config: RunConfig) -> list[str]:
    settings = client.system_settings()
    if settings.get("MemoryEnabled") is not False:
        raise AnythingLLMError(
            "AnythingLLM memories are enabled; they inject content across chats. Disable them."
        )

    wanted = instance_env(config)
    changes = {k: v for k, v in wanted.items() if str(settings.get(k)) != v}
    if any(k in changes for k in _EMBEDDER_KEYS):
        log.warning(
            "Embedder changes (%s): AnythingLLM will delete the documents of EVERY workspace; "
            "they are re-ingested when each config is next provisioned.",
            {k: changes[k] for k in _EMBEDDER_KEYS if k in changes},
        )
    if changes:
        client.update_env(changes)
        settings = client.system_settings()
        wrong = {k: settings.get(k) for k, v in wanted.items() if str(settings.get(k)) != v}
        if wrong:
            raise AnythingLLMError(f"instance settings did not apply: {wrong}")

    applied = [f"{k}={v}" for k, v in changes.items()]
    size, overlap = config.chunking.size, config.chunking.overlap
    if client.text_splitter_settings() != (size, overlap):
        client.set_text_splitter(size, overlap)
        if client.text_splitter_settings() != (size, overlap):
            raise AnythingLLMError("text splitter settings did not apply")
        applied.append(f"chunking={size}/{overlap}")
    return applied


def _find_or_create_workspace(client: AnythingLLMClient, name: str) -> Workspace:
    matches = [w for w in client.list_workspaces() if w.name == name]
    if len(matches) > 1:
        raise AnythingLLMError(f"more than one workspace named {name}; delete the duplicates")
    if matches:
        return client.get_workspace(matches[0].slug)
    log.info("Creating workspace %s", name)
    return client.create_workspace(name)


def provision(client: AnythingLLMClient, config: RunConfig) -> ProvisionResult:
    """Bring AnythingLLM in line with ``config``. Safe to run repeatedly."""
    corpus = load_corpus(config.corpus)
    client.check_auth()
    instance_changes = _apply_instance_settings(client, config)

    name = workspace_name(config, corpus)
    slug = _find_or_create_workspace(client, name).slug
    client.update_workspace(slug, workspace_settings(config))
    # Read back rather than trusting the update response, which omits documents.
    workspace = client.get_workspace(slug)
    mismatches = settings_mismatches(workspace, config)
    if mismatches:
        raise AnythingLLMError(f"workspace settings did not apply: {mismatches}")

    sync = plan_document_sync(
        ((d.title, d.docpath) for d in workspace.documents),
        (d.doc_id for d in corpus.documents),
    )
    if sync.to_remove:
        log.info("Removing %d unexpected document(s)", len(sync.to_remove))
        client.remove_documents(workspace.slug, sync.to_remove)

    locations = dict(sync.kept)
    documents = corpus.by_id()
    for doc_id in sync.to_upload:
        log.info("Uploading and embedding %s", doc_id)
        uploaded = client.upload_document(
            doc_id, documents[doc_id].path.read_bytes(), workspace_slug=workspace.slug
        )
        if uploaded.title != doc_id:
            raise AnythingLLMError(f"{doc_id} was stored with title {uploaded.title!r}")
        locations[doc_id] = uploaded.location

    final = client.get_workspace(workspace.slug)
    present = sorted(d.title or "" for d in final.documents)
    if present != sorted(documents):
        raise AnythingLLMError(
            f"workspace documents {present} do not match corpus {sorted(documents)}"
        )

    return ProvisionResult(
        workspace_slug=workspace.slug,
        workspace_name=name,
        corpus=corpus,
        locations={doc_id: locations[doc_id] for doc_id in sorted(locations)},
        uploaded=sync.to_upload,
        removed=sync.to_remove,
        instance_changes=instance_changes,
    )
