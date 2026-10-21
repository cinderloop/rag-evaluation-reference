"""Automated checks on candidates, run before human review.

- Near-duplicates: questions too similar to an earlier one are marked and dropped.
- Lexical overlap: share of a question's content words that appear in its evidence.
  High overlap means retrieval can succeed on copied wording alone, so the item is
  flagged as too easy (the reviewer decides).
- Chunk boundary: whether any evidence quote is not contained in a single baseline
  chunk (measured on AnythingLLM's real chunks).
- Evidence location: the report section the evidence is in. The planned location is
  kept when any quote is in it, otherwise the section of the first quote is used.
- Coverage: counts per dimension value against the targets, plus the constraints in
  data/dimensions.yaml.
"""

from __future__ import annotations

import re
from collections import Counter

from rag_eval.dataset.candidates import Candidate, Checks
from rag_eval.dataset.corpus_text import ReportText
from rag_eval.dataset.dimensions import DimensionSpec
from rag_eval.dataset.plan import MAX_LONG_REPORT_SLOTS, MAX_SLOTS_PER_REPORT, MIN_REPORTS_USED

DUPLICATE_THRESHOLD = 0.6
TOO_EASY_THRESHOLD = 0.7

_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "during",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "many",
        "more",
        "most",
        "much",
        "of",
        "on",
        "or",
        "other",
        "over",
        "same",
        "should",
        "since",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "under",
        "up",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
    ]
)
_TOKEN = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    words = {t.removesuffix("s") if len(t) > 3 else t for t in _TOKEN.findall(text.lower())}
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def lexical_overlap(question: str, evidence_texts: list[str]) -> float | None:
    q = content_words(question)
    if not q or not evidence_texts:
        return None
    e: set[str] = set()
    for text in evidence_texts:
        e |= content_words(text)
    return len(q & e) / len(q)


def chunk_boundary(candidate: Candidate, reports: dict[str, ReportText]) -> str | None:
    if not candidate.evidence:
        return None
    crosses = any(not reports[e.doc_id].within_one_chunk(e.quote) for e in candidate.evidence)
    return "cross_chunk" if crosses else "within_chunk"


def evidence_location(candidate: Candidate, reports: dict[str, ReportText]) -> str | None:
    planned = candidate.dimensions.get("evidence_location")
    found = [reports[e.doc_id].section_of(e.quote) for e in candidate.evidence]
    if not found or planned in found:
        return planned
    return next((f for f in found if f is not None), planned)


def run_checks(candidates: list[Candidate], reports: dict[str, ReportText]) -> list[Candidate]:
    checked: list[Candidate] = []
    kept_words: list[tuple[str, set[str]]] = []
    for candidate in candidates:
        words = content_words(candidate.question)
        duplicate_of = next(
            (cid for cid, other in kept_words if jaccard(words, other) >= DUPLICATE_THRESHOLD),
            None,
        )
        if duplicate_of is None:
            kept_words.append((candidate.candidate_id, words))
        quotes = [e.quote for e in candidate.evidence]
        overlap = lexical_overlap(candidate.question, quotes)
        checked.append(
            candidate.model_copy(
                update={
                    "checks": Checks(
                        lexical_overlap=None if overlap is None else round(overlap, 3),
                        too_easy=overlap is not None and overlap >= TOO_EASY_THRESHOLD,
                        chunk_boundary=chunk_boundary(candidate, reports),
                        evidence_location=evidence_location(candidate, reports),
                        duplicate_of=duplicate_of,
                    )
                }
            )
        )
    return checked


def coverage_report(
    candidates: list[Candidate],
    spec: DimensionSpec,
    long_report_ids: set[str],
    title: str = "Coverage report",
) -> str:
    items = [c for c in candidates if c.checks.duplicate_of is None]
    lines = [f"# {title}", "", f"{len(items)} items (duplicates excluded).", ""]
    for name, dimension in spec.dimensions.items():
        if name == "chunk_boundary":
            values = [c.checks.chunk_boundary for c in items]
        elif name == "evidence_location":
            values = [c.checks.evidence_location for c in items]
        else:
            values = [c.dimensions.get(name) for c in items]
        applicable = [v for v in values if v is not None]
        counts = Counter(applicable)
        lines += [f"## {name}", "", "| value | count | share | target |", "|---|---|---|---|"]
        for value, info in dimension.values.items():
            n = counts.get(value, 0)
            share = n / len(applicable) if applicable else 0.0
            lines.append(f"| {value} | {n} | {share:.0%} | {info.target:.0%} |")
        lines.append("")

    per_report = Counter(d for c in items for d in c.doc_ids)
    over_cap = {d: n for d, n in per_report.items() if n > MAX_SLOTS_PER_REPORT}
    long_total = sum(per_report[d] for d in long_report_ids)
    overlaps = [c.checks.lexical_overlap for c in items if c.checks.lexical_overlap is not None]
    by_phrasing: dict[str, list[float]] = {}
    for c in items:
        if c.checks.lexical_overlap is not None:
            by_phrasing.setdefault(c.dimensions.get("phrasing", "-"), []).append(
                c.checks.lexical_overlap
            )
    lines += [
        "## Constraints",
        "",
        f"- Reports used: {len(per_report)} (minimum {MIN_REPORTS_USED})",
        f"- Reports over the {MAX_SLOTS_PER_REPORT}-question cap: {over_cap or 'none'}",
        f"- Questions on the long reports: {long_total} (maximum {MAX_LONG_REPORT_SLOTS})",
        "",
        "## Lexical overlap (question vs evidence)",
        "",
    ]
    if overlaps:
        lines.append(f"- Mean overall: {sum(overlaps) / len(overlaps):.2f}")
    for phrasing, values in sorted(by_phrasing.items()):
        lines.append(
            f"- Mean for {phrasing}: {sum(values) / len(values):.2f} ({len(values)} items)"
        )
    n_easy = sum(c.checks.too_easy for c in items)
    lines.append(f"- Flagged too easy (overlap >= {TOO_EASY_THRESHOLD}): {n_easy}")
    duplicates = [c for c in candidates if c.checks.duplicate_of is not None]
    lines += ["", f"## Near-duplicates removed: {len(duplicates)}", ""]
    lines += [
        f"- {c.candidate_id} duplicates {c.checks.duplicate_of}: {c.question}" for c in duplicates
    ]
    return "\n".join(lines) + "\n"
