"""The knowledge-edit mutation: change one fact in one document.

The edit is declared in ``data/mutations/knowledge-edit.yaml``: the document, the exact
text to change, its replacement, and the dataset cases whose expected answer changes
with it. ``build_edited_document`` applies the edit to the text AnythingLLM parsed from
the original PDF and writes it as a new, plain PDF, which the mutation config uploads
under the original document ID (``corpus.replace``). Evaluation of that run uses the
edited cases, so a correct answer must follow the edited document, not the model's
memory of the original report.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rag_eval.dataset.schema import Case, Evidence

LINE_CHARS = 95
LINES_PER_PAGE = 60


class CaseEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    expected_answer: str
    evidence: tuple[Evidence, ...]


class KnowledgeEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    find: str = Field(min_length=1, description="Text to change; must occur exactly once")
    replace: str = Field(min_length=1)
    cases: tuple[CaseEdit, ...]


def load_edit(path: Path) -> KnowledgeEdit:
    return KnowledgeEdit.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def apply_edit(text: str, edit: KnowledgeEdit) -> str:
    """Replace the one occurrence of ``find``; whitespace in the source text may differ."""
    pattern = r"\s+".join(re.escape(word) for word in edit.find.split())
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"'{edit.find}' occurs {len(matches)} times in {edit.doc_id}; need 1")
    match = matches[0]
    return text[: match.start()] + edit.replace + text[match.end() :]


def edited_cases(cases: dict[str, Case], edit: KnowledgeEdit) -> dict[str, Case]:
    """The dataset with the edited cases' expected answers and evidence replaced."""
    out = dict(cases)
    for change in edit.cases:
        if change.id not in out:
            raise ValueError(f"unknown case {change.id}")
        out[change.id] = out[change.id].model_copy(
            update={"expected_answer": change.expected_answer, "evidence": change.evidence}
        )
    return out


def _pdf_string(line: str) -> bytes:
    raw = line.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def text_pdf(text: str) -> bytes:
    """A minimal, deterministic PDF (Helvetica, WinAnsi encoding) holding the text."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(" ".join(paragraph.split()), LINE_CHARS) or [""])
    pages = [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    pages = pages or [[""]]

    objects: list[bytes] = []
    n_pages = len(pages)
    first_page = 4
    kids = " ".join(f"{first_page + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    for i, page in enumerate(pages):
        content = (
            b"BT /F1 10 Tf 12 TL 50 750 Td "
            + b" ".join(b"(" + _pdf_string(line) + b") Tj T*" for line in page)
            + b" ET"
        )
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {first_page + 2 * i + 1} 0 R >>"
            ).encode()
        )
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def build_edited_document(edit: KnowledgeEdit, corpus_text: Path, out: Path) -> Path:
    """Apply the edit to the parsed text of the original and write it as a PDF."""
    parsed = json.loads((corpus_text / f"{edit.doc_id}.json").read_text(encoding="utf-8"))
    edited = apply_edit(str(parsed["text"]), edit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text_pdf(edited))
    return out
