from dataset_fixtures import REPORTS, candidate

from rag_eval.dataset.checks import content_words, evidence_location, lexical_overlap, run_checks
from rag_eval.dataset.corpus_text import ReportText, normalize


def test_normalize_collapses_whitespace_only() -> None:
    assert normalize("  The vessel \n grounded\t near.  ") == "The vessel grounded near."


def test_quotes_match_across_pdf_line_breaks_but_not_other_changes() -> None:
    report = REPORTS["A.pdf"]
    assert report.contains("the vessel Alpha grounded near the inlet.")
    assert not report.contains("the vessel alpha grounded near the inlet.")  # case matters
    assert not report.contains("the ship Alpha grounded near the inlet.")


def test_within_one_chunk_detects_boundary_spans() -> None:
    report = REPORTS["A.pdf"]
    assert report.within_one_chunk("On June 19, 2024")
    assert not report.within_one_chunk(
        "grounded near the inlet. Damage to the vessel was estimated at $1 million. "
        "The captain maneuvered the vessel into deeper water."
    )


def test_context_returns_text_around_the_quote() -> None:
    ctx = REPORTS["A.pdf"].context("estimated at $1 million.", chars=10)
    assert ctx is not None
    before, match, after = ctx
    assert match == "estimated at $1 million."
    assert len(before) == 10
    assert after.startswith(" The capta")


def test_content_words_drop_stopwords_and_plural_s() -> None:
    assert content_words("What were the damages to the vessels?") == {"damage", "vessel"}


def test_lexical_overlap() -> None:
    assert lexical_overlap("vessel damage", ["Damage to the vessel"]) == 1.0
    assert lexical_overlap("How costly was the stranding?", ["Damage was $1 million"]) == 0.0
    assert lexical_overlap("anything", []) is None


def test_checks_flag_duplicates_easy_items_and_boundaries() -> None:
    items = [
        candidate("b001-1", "How much damage did the grounding of Alpha cause?"),
        candidate("b001-2", "How much damage did the grounding of Alpha cause in total?"),
        candidate(
            "b002-1",
            "Vessel damage estimated?",
            evidence=(("A.pdf", "Damage to the vessel was estimated at $1 million."),),
        ),
        candidate(
            "b003-1",
            "What happened after Alpha ran aground?",
            evidence=(
                (
                    "A.pdf",
                    "grounded near the inlet. Damage to the vessel was estimated at $1 million. "
                    "The captain maneuvered the vessel into deeper water.",
                ),
            ),
        ),
    ]
    checked = {c.candidate_id: c.checks for c in run_checks(items, REPORTS)}
    assert checked["b001-1"].duplicate_of is None
    assert checked["b001-2"].duplicate_of == "b001-1"
    assert checked["b002-1"].too_easy
    assert checked["b002-1"].chunk_boundary == "within_chunk"
    assert checked["b003-1"].chunk_boundary == "cross_chunk"


SECTIONED = ReportText(
    doc_id="S.pdf",
    text=(
        "On May 1 the tow struck the bridge.\nCasualty Summary \nDamage $1 million est.\n"
        "1 Factual Information \nThe pilot slowed the tow.\n2 Analysis \nThe pilot was tired.\n"
        "Vessel Particulars \nLength 60 ft\n"
    ),
    chunks=(),
)


def test_section_of_maps_report_headings_to_evidence_locations() -> None:
    assert SECTIONED.section_of("On May 1 the tow struck") == "narrative"
    assert SECTIONED.section_of("Damage $1 million est.") == "summary_table"
    assert SECTIONED.section_of("The pilot slowed the tow.") == "narrative"
    assert SECTIONED.section_of("The pilot was tired.") == "analysis"
    assert SECTIONED.section_of("Length 60 ft") == "summary_table"
    assert REPORTS["A.pdf"].section_of("Damage to the vessel") is None  # no headings


def test_evidence_location_keeps_plan_if_any_quote_matches_else_uses_first() -> None:
    reports = {"S.pdf": SECTIONED}
    dims = {"question_type": "factual", "evidence_location": "summary_table"}
    both = candidate(
        evidence=(("S.pdf", "The pilot was tired."), ("S.pdf", "Damage $1 million est.")),
        dimensions=dims,
    )
    assert evidence_location(both, reports) == "summary_table"
    elsewhere = candidate(evidence=(("S.pdf", "The pilot was tired."),), dimensions=dims)
    assert evidence_location(elsewhere, reports) == "analysis"
