"""The corpus as a set of documents with stable IDs.

A document's ID is its filename. AnythingLLM keeps the uploaded filename as the
document ``title`` and returns it on every retrieved chunk, which is how retrieved
passages are mapped back to labelled evidence. AnythingLLM's own document locations
change on every upload and are never used as IDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag_eval.config import CorpusConfig

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".html", ".pdf"})


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class Corpus:
    documents: tuple[CorpusDocument, ...]

    @property
    def content_hash(self) -> str:
        """Hash of every document ID and its content; changes if any document changes."""
        lines = "".join(f"{d.doc_id}\t{d.sha256}\n" for d in self.documents)
        return hashlib.sha256(lines.encode()).hexdigest()

    def by_id(self) -> dict[str, CorpusDocument]:
        return {d.doc_id: d for d in self.documents}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(config: CorpusConfig) -> Corpus:
    """List the documents a config puts in its workspace, after exclusions and replacements."""
    if not config.dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {config.dir}")
    entries = sorted(p for p in config.dir.iterdir() if not p.name.startswith("."))
    nested = [p.name for p in entries if p.is_dir()]
    if nested:
        raise ValueError(f"corpus directory must be flat (document ID = filename): {nested}")
    unsupported = [p.name for p in entries if p.suffix.lower() not in SUPPORTED_SUFFIXES]
    if unsupported:
        raise ValueError(f"unsupported corpus files: {unsupported}")

    ids = {p.name for p in entries}
    unknown = (set(config.exclude) | set(config.replace)) - ids
    if unknown:
        raise ValueError(f"exclude/replace refer to unknown document IDs: {sorted(unknown)}")
    both = set(config.exclude) & set(config.replace)
    if both:
        raise ValueError(f"documents both excluded and replaced: {sorted(both)}")

    documents: list[CorpusDocument] = []
    for entry in entries:
        if entry.name in config.exclude:
            continue
        source = config.replace.get(entry.name, entry)
        if not source.is_file():
            raise FileNotFoundError(f"replacement for {entry.name} not found: {source}")
        documents.append(CorpusDocument(doc_id=entry.name, path=source, sha256=_sha256(source)))
    if not documents:
        raise ValueError("corpus is empty after exclusions")
    return Corpus(documents=tuple(documents))
