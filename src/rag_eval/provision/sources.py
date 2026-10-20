"""Fetching the corpus from its published sources, verified by checksum.

The corpus PDFs are not committed (they total ~110 MB). ``corpus/sources.yaml`` pins
every document's URL and SHA-256; fetching downloads what is missing and fails on any
mismatch, so the corpus on disk is always exactly the one the checksums describe.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    report: str
    title: str
    url: str = Field(pattern=r"^https://")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)


class ChecksumMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchResult:
    downloaded: list[str]
    already_present: list[str]
    unexpected: list[str]


def load_sources(path: Path) -> list[Source]:
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or "documents" not in loaded:
        raise ValueError(f"{path}: expected a mapping with a 'documents' list")
    documents = cast(dict[str, Any], loaded)["documents"]
    sources = [Source.model_validate(d) for d in cast(list[object], documents)]
    ids = [s.doc_id for s in sources]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate doc_ids: {duplicates}")
    return sources


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def http_download(url: str) -> bytes:
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    return response.content


def fetch_sources(
    sources: list[Source],
    dest: Path,
    download: Callable[[str], bytes] = http_download,
) -> FetchResult:
    """Download missing documents into ``dest`` and verify every document's checksum.

    Existing files with the wrong checksum are never overwritten: that would hide a
    changed source. Delete the file to re-download it.
    """
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    present: list[str] = []
    for source in sources:
        target = dest / source.doc_id
        if target.exists():
            actual = _sha256(target.read_bytes())
            if actual != source.sha256:
                raise ChecksumMismatchError(
                    f"{target} has sha256 {actual}, sources.yaml pins {source.sha256}. "
                    "Delete it to re-download, or investigate why it changed."
                )
            present.append(source.doc_id)
            continue
        log.info("Downloading %s (%s)", source.doc_id, source.report)
        data = download(source.url)
        actual = _sha256(data)
        if actual != source.sha256:
            raise ChecksumMismatchError(
                f"{source.url} now has sha256 {actual}, sources.yaml pins {source.sha256}. "
                "The published report may have been reissued; review it before updating the pin."
            )
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(data)
        partial.replace(target)
        downloaded.append(source.doc_id)

    known = {s.doc_id for s in sources}
    unexpected = sorted(
        p.name for p in dest.iterdir() if not p.name.startswith(".") and p.name not in known
    )
    return FetchResult(downloaded=downloaded, already_present=present, unexpected=unexpected)
