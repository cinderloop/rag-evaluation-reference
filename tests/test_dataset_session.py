import json
from pathlib import Path

from dataset_fixtures import REPORTS

from rag_eval.dataset.dimensions import load_dimensions
from rag_eval.dataset.plan import Batch, Plan, Slot
from rag_eval.dataset.session import check_batch, collect, write_briefs

SPEC = load_dimensions(Path("data/dimensions.yaml"))
TITLES = {"A.pdf": "Grounding of Vessel Alpha", "B.pdf": "Fire aboard Vessel Bravo"}
GOOD = "Damage to the vessel was estimated at $1 million."
BATCH = Batch(
    batch_id="b001",
    kind="single",
    doc_ids=["A.pdf"],
    slots=[
        Slot(slot_id="b001-1", question_type="factual"),
        Slot(slot_id="b001-2", question_type="factual"),
    ],
)
PLAN = Plan(seed=0, n_slots=2, batches=[BATCH])


def _write(path: Path, quotes: list[str]) -> None:
    items = [
        {
            "slot_id": f"b001-{i}",
            "question": "How much damage was there?",
            "expected_answer": "About $1 million.",
            "evidence": [{"doc_id": "A.pdf", "quote": q}],
            "notes": "",
        }
        for i, q in enumerate(quotes, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}), encoding="utf-8")


def test_briefs_carry_instructions_slots_and_report_text(tmp_path: Path) -> None:
    [path] = write_briefs(
        PLAN,
        spec=SPEC,
        reports=REPORTS,
        titles=TITLES,
        briefs_dir=tmp_path / "briefs",
        outputs_dir=tmp_path / "out",
    )
    brief = path.read_text(encoding="utf-8")
    assert "Write exactly one question per slot" in brief
    assert '"slot_id": "b001-2"' in brief
    assert "check-batch b001" in brief
    assert (path.parent / "A.pdf.txt").read_text(encoding="utf-8") == REPORTS["A.pdf"].text


def test_check_batch_reports_missing_file_bad_quotes_and_missing_slots(tmp_path: Path) -> None:
    output = tmp_path / "b001.json"
    assert "no output file" in check_batch(BATCH, output, REPORTS)[0]
    _write(output, ["Damage was one million dollars, roughly."])
    problems = check_batch(BATCH, output, REPORTS)
    assert any(p.startswith("b001-1: quote not found verbatim") for p in problems)
    assert "b001-2: no item for this slot" in problems
    _write(output, [GOOD, GOOD])
    assert check_batch(BATCH, output, REPORTS) == []


def test_collect_keeps_valid_slots_and_records_the_rest(tmp_path: Path) -> None:
    _write(tmp_path / "b001.json", [GOOD, "invented text that is not in the report"])
    outcome = collect(
        PLAN, spec=SPEC, reports=REPORTS, titles=TITLES, outputs_dir=tmp_path, model="m"
    )
    assert [c.candidate_id for c in outcome.candidates] == ["b001-1"]
    assert outcome.candidates[0].generator.response_id == "session:b001"
    assert list(outcome.failures) == ["b001-2"]
