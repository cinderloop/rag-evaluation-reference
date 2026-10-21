"""Shared builders for dataset tests."""

from rag_eval.dataset.candidates import Candidate, GeneratorInfo
from rag_eval.dataset.corpus_text import ReportText
from rag_eval.dataset.schema import Evidence

TEXT_A = (
    "Grounding of Vessel Alpha \nOn June 19, 2024, about 1517 local time, the vessel Alpha \n"
    "grounded near the inlet. Damage to the vessel was estimated at $1 million. "
    "The captain maneuvered the vessel into deeper water."
)
TEXT_B = (
    "Fire aboard Vessel Bravo \nA fire started in the engine room at 0300. Damage was $2 million."
)

REPORTS = {
    "A.pdf": ReportText(
        doc_id="A.pdf",
        text=TEXT_A,
        chunks=(TEXT_A[:110], TEXT_A[90:]),
    ),
    "B.pdf": ReportText(doc_id="B.pdf", text=TEXT_B, chunks=(TEXT_B,)),
}

INFO = GeneratorInfo(
    model_requested="m", model_served="m", prompt_sha256="p", response_id="r", attempt=1
)


def candidate(
    cid: str = "b001-1",
    question: str = "How much damage did the grounding of Alpha cause?",
    evidence: tuple[tuple[str, str], ...] = (
        ("A.pdf", "Damage to the vessel was estimated at $1 million."),
    ),
    answerable: bool = True,
    dimensions: dict[str, str] | None = None,
) -> Candidate:
    return Candidate(
        candidate_id=cid,
        batch_id=cid.split("-", maxsplit=1)[0],
        doc_ids=sorted({d for d, _ in evidence}) or ["A.pdf"],
        answerable=answerable,
        dimensions=dimensions or {"question_type": "factual", "phrasing": "paraphrased"},
        question=question,
        expected_answer="About $1 million.",
        evidence=[Evidence(doc_id=d, quote=q) for d, q in evidence],
        generator=INFO,
    )
