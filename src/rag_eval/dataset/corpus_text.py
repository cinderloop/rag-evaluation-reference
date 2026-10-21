"""Each report's text and baseline chunks as AnythingLLM parsed them, and quote matching.

Evidence quotes are matched after collapsing all whitespace to single spaces, because
PDF extraction puts line breaks mid-sentence. Everything else (case, punctuation,
PDF artefacts such as split words) must match exactly. This is the same matching
rule the retrieval metrics use, so a quote that is "found" here can be found in a
retrieved passage too.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rag_eval.client import AnythingLLMClient
from rag_eval.client.storage import document_chunks, document_text
from rag_eval.client.text import strip_metadata_header

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class ReportText:
    doc_id: str
    text: str
    chunks: tuple[str, ...]  # metadata header stripped

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def contains(self, quote: str) -> bool:
        return normalize(quote) in normalize(self.text)

    def within_one_chunk(self, quote: str) -> bool:
        needle = normalize(quote)
        return any(needle in normalize(chunk) for chunk in self.chunks)

    def context(self, quote: str, chars: int = 300) -> tuple[str, str, str] | None:
        """(before, quote, after) around the quote's first occurrence, normalized."""
        haystack, needle = normalize(self.text), normalize(quote)
        start = haystack.find(needle)
        if start < 0:
            return None
        end = start + len(needle)
        return haystack[max(0, start - chars) : start], needle, haystack[end : end + chars]

    def chunk_boundaries(self, chars: int = 160) -> list[str]:
        """Text around each boundary between consecutive chunks (to target boundary spans)."""
        excerpts: list[str] = []
        for left, right in zip(self.chunks, self.chunks[1:], strict=False):
            excerpts.append(normalize(left[-chars:]) + " ‖ " + normalize(right[:chars]))
        return excerpts


def extract(client: AnythingLLMClient, container: str, workspace_slug: str, out_dir: Path) -> int:
    """Write data/corpus_text/<doc_id>.json for every document in the workspace."""
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = client.get_workspace(workspace_slug)
    for doc in workspace.documents:
        if doc.title is None:
            raise ValueError(f"document without title in {workspace_slug}: {doc.docpath}")
        chunks = [strip_metadata_header(c) for c in document_chunks(container, doc.docpath)]
        record = {
            "doc_id": doc.title,
            "docpath": doc.docpath,
            "text": document_text(container, doc.docpath),
            "chunks": chunks,
        }
        (out_dir / f"{doc.title}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
    return len(workspace.documents)


def load_report_texts(directory: Path) -> dict[str, ReportText]:
    reports: dict[str, ReportText] = {}
    for path in sorted(directory.glob("*.json")):
        raw: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        chunks = raw["chunks"]
        if not isinstance(chunks, list):
            raise ValueError(f"{path}: chunks must be a list")
        reports[str(raw["doc_id"])] = ReportText(
            doc_id=str(raw["doc_id"]),
            text=str(raw["text"]),
            chunks=tuple(str(c) for c in cast(list[object], chunks)),
        )
    if not reports:
        raise FileNotFoundError(f"no report texts in {directory}; run `rag-eval dataset extract`")
    return reports
