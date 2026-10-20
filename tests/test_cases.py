import json
from pathlib import Path

import pytest

from rag_eval.dataset.schema import load_cases

ANSWERABLE: dict[str, object] = {
    "id": "c1",
    "split": "dev",
    "question": "Q?",
    "answerable": True,
    "expected_answer": "A.",
    "evidence": [{"doc_id": "a.txt", "quote": "A."}],
}
UNANSWERABLE: dict[str, object] = {
    "id": "c2",
    "split": "test",
    "question": "Q2?",
    "answerable": False,
}


def _write(tmp_path: Path, *rows: dict[str, object]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_smoke_cases_load() -> None:
    assert len(load_cases(Path("tests/smoke/cases.jsonl"))) == 3


def test_loads_answerable_and_unanswerable(tmp_path: Path) -> None:
    cases = load_cases(_write(tmp_path, ANSWERABLE, UNANSWERABLE))
    assert [c.answerable for c in cases] == [True, False]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({**ANSWERABLE, "evidence": []}, "need expected_answer and evidence"),
        ({**ANSWERABLE, "expected_answer": None}, "need expected_answer and evidence"),
        (
            {**UNANSWERABLE, "evidence": [{"doc_id": "a.txt", "quote": "x"}]},
            "must not have evidence",
        ),
        ({**ANSWERABLE, "split": "train"}, "split"),
        ({**ANSWERABLE, "chunk_id": "123"}, "Extra inputs"),
    ],
)
def test_invalid_cases_are_rejected_with_line_number(
    tmp_path: Path, row: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=rf"(?s)cases.jsonl:1: .*{message}"):
        load_cases(_write(tmp_path, row))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate case IDs"):
        load_cases(_write(tmp_path, ANSWERABLE, ANSWERABLE))
