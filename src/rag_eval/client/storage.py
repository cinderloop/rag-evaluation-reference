"""Read-only access to AnythingLLM's storage volume, for building the dataset only.

The developer API does not expose a document's parsed text or its chunks. Both are
needed when building the dataset: evidence quotes must match the text exactly as
AnythingLLM parsed it from the PDF, and chunk-boundary questions need the real
chunks. They are read from the container with ``docker exec``:

- ``documents/<docpath>``: the parsed document (``pageContent``)
- ``vector-cache/<uuid5(URL, docpath)>.json``: the chunks as embedded

These are internal storage formats of AnythingLLM 1.18.0 (see
docs/spike-findings.md). Evaluation runs never read them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from typing import Any, cast

_STORAGE = "/app/server/storage"


class StorageError(RuntimeError):
    pass


def _cat(container: str, path: str) -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise StorageError("the docker CLI is required to read AnythingLLM's storage")
    done = subprocess.run(
        [docker, "exec", container, "cat", path], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise StorageError(f"cannot read {path} in {container}: {done.stderr.strip()}")
    return done.stdout


def document_text(container: str, docpath: str) -> str:
    # Parsed PDFs can contain raw control characters, which strict JSON rejects.
    parsed: object = json.loads(_cat(container, f"{_STORAGE}/documents/{docpath}"), strict=False)
    if not isinstance(parsed, dict):
        raise StorageError(f"unexpected document format for {docpath}")
    text = cast(dict[str, Any], parsed).get("pageContent")
    if not isinstance(text, str):
        raise StorageError(f"no pageContent in {docpath}")
    return text


def document_chunks(container: str, docpath: str) -> list[str]:
    """Chunk texts in embedding order, including AnythingLLM's metadata header."""
    key = uuid.uuid5(uuid.NAMESPACE_URL, docpath)
    try:
        raw = _cat(container, f"{_STORAGE}/vector-cache/{key}.json")
    except StorageError as error:
        raise StorageError(
            f"no vector cache for {docpath}. AnythingLLM purges it when chunking settings "
            "change; re-embed the workspace (delete it and provision again)."
        ) from error
    batches = cast(list[list[dict[str, Any]]], json.loads(raw, strict=False))
    return [str(vector["metadata"]["text"]) for batch in batches for vector in batch]
