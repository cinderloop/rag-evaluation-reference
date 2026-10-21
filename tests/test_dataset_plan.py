from collections import Counter
from pathlib import Path

import pytest

from rag_eval.dataset.dimensions import load_dimensions
from rag_eval.dataset.plan import (
    MAX_LONG_REPORT_SLOTS,
    MAX_SLOTS_PER_REPORT,
    MIN_REPORTS_USED,
    ReportInfo,
    ReportLink,
    build_plan,
    category,
    largest_remainder,
    load_links,
)
from rag_eval.provision.sources import load_sources

SPEC = load_dimensions(Path("data/dimensions.yaml"))
SOURCES = load_sources(Path("corpus/sources.yaml"))
LONG = {"MIR2540.pdf", "MIR2516.pdf", "MIR2624.pdf", "MIR2536.pdf"}
REPORTS = [
    ReportInfo(doc_id=s.doc_id, title=s.title, text_length=200_000 if s.doc_id in LONG else 20_000)
    for s in SOURCES
]


def test_largest_remainder_sums_exactly() -> None:
    counts = largest_remainder({"a": 0.26, "b": 0.14, "c": 0.6}, 110)
    assert sum(counts.values()) == 110
    assert counts == {"a": 29, "b": 15, "c": 66}


def test_committed_dimensions_load_and_targets_sum_to_one() -> None:
    assert SPEC.sizes.candidates == 110
    assert set(SPEC.dimension("question_type").values) >= {"factual", "unanswerable"}


LINKS = [
    ReportLink(reports=("MIR2525.pdf", "MIR2614.pdf"), entity="Pelican Island Bridge"),
    ReportLink(reports=("MIR2617.pdf", "MIR2620.pdf"), entity="Vicksburg"),
]


def test_categories_from_titles() -> None:
    assert category("Engine Room Fire aboard Towing Vessel Thor") == "fire"
    assert category("Contact of Duke Tow with Pelican Island Bridge") == "contact"


def test_links_must_name_an_entity_found_in_both_reports(tmp_path: Path) -> None:
    path = tmp_path / "links.yaml"
    path.write_text(
        "links:\n  - { reports: [a.pdf, b.pdf], entity: Pelican Island Bridge }\n",
        encoding="utf-8",
    )
    texts = {"a.pdf": "struck the Pelican Island\nBridge", "b.pdf": "the Pelican Island Bridge"}
    assert load_links(path, texts)[0].entity == "Pelican Island Bridge"
    with pytest.raises(ValueError, match="does not occur"):
        load_links(path, {**texts, "b.pdf": "another bridge"})


def test_curated_links_are_used_as_pairs() -> None:
    pairs = [
        b.doc_ids for b in build_plan(SPEC, REPORTS, LINKS, seed=1).batches if b.kind == "pair"
    ]
    assert ["MIR2525.pdf", "MIR2614.pdf"] in pairs
    assert ["MIR2617.pdf", "MIR2620.pdf"] in pairs


def test_plan_meets_targets_and_constraints() -> None:
    plan = build_plan(SPEC, REPORTS, LINKS, seed=1)
    slots = [s for b in plan.batches for s in b.slots]
    assert len(slots) == plan.n_slots == 110
    assert len({s.slot_id for s in slots}) == 110

    expected = largest_remainder(SPEC.dimension("question_type").targets(), 110)
    assert Counter(s.question_type for s in slots) == Counter(expected)

    usage = Counter(d for b in plan.batches for _ in b.slots for d in b.doc_ids)
    assert max(usage.values()) <= MAX_SLOTS_PER_REPORT
    assert sum(usage[d] for d in LONG) <= MAX_LONG_REPORT_SLOTS
    assert len(usage) >= MIN_REPORTS_USED

    for batch in plan.batches:
        for slot in batch.slots:
            if slot.question_type in ("comparison", "multi_document"):
                assert batch.kind == "pair"
                assert slot.complexity in ("multi_span", "inference")
            if slot.question_type == "unanswerable":
                assert slot.complexity is None
                assert slot.unanswerable_kind is not None
            if slot.unanswerable_kind == "out_of_corpus":
                assert batch.kind == "out_of_corpus"
            if slot.target_cross_chunk:
                assert batch.kind == "single"


def test_plan_is_deterministic_per_seed() -> None:
    assert build_plan(SPEC, REPORTS, LINKS, seed=7) == build_plan(SPEC, REPORTS, LINKS, seed=7)
    assert build_plan(SPEC, REPORTS, LINKS, seed=7) != build_plan(SPEC, REPORTS, LINKS, seed=8)


def test_plan_fails_loudly_when_caps_cannot_be_met() -> None:
    with pytest.raises(ValueError, match="report"):
        build_plan(SPEC, REPORTS[:10], LINKS, seed=1)
