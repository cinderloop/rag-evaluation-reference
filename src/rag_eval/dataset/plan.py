"""Turning dimension targets into concrete question slots, batched by report.

Deterministic for a given seed and inputs. Each slot is one dimension tuple; each
batch is one generation call covering one report (or a pair of reports for
comparison and multi-document questions) and the slots assigned to it. Batching and
the sizes used are explained in docs/decisions/0004-dataset-size-and-generation.md.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.dimensions import DimensionSpec

PAIR_TYPES = ("comparison", "multi_document")
SINGLE_TYPES = ("factual", "causal", "procedural")
MULTI_EVIDENCE = ("multi_span", "inference")
SLOTS_PER_SINGLE_BATCH = 2
MAX_SLOTS_PER_REPORT = 3
MAX_LONG_REPORT_SLOTS = 8
MIN_REPORTS_USED = 40


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    question_type: str
    unanswerable_kind: str | None = None
    complexity: str | None = None
    phrasing: str | None = None
    evidence_location: str | None = None
    target_cross_chunk: bool = False

    def dimensions(self) -> dict[str, str]:
        dims = {
            "question_type": self.question_type,
            "unanswerable_kind": self.unanswerable_kind,
            "complexity": self.complexity,
            "phrasing": self.phrasing,
            "evidence_location": self.evidence_location,
        }
        return {k: v for k, v in dims.items() if v is not None}


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    kind: Literal["single", "pair", "out_of_corpus"]
    doc_ids: list[str]
    slots: list[Slot]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    n_slots: int
    batches: list[Batch]


@dataclass(frozen=True)
class ReportInfo:
    doc_id: str
    title: str
    text_length: int


def largest_remainder(targets: dict[str, float], total: int) -> dict[str, int]:
    """Integer counts proportional to ``targets`` that sum exactly to ``total``."""
    raw = {k: v * total for k, v in targets.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    leftover = total - sum(counts.values())
    by_remainder = sorted(raw, key=lambda k: (-(raw[k] - counts[k]), k))
    for key in by_remainder[:leftover]:
        counts[key] += 1
    return counts


def _quota(targets: dict[str, float], n: int, rng: random.Random) -> list[str]:
    values = [k for k, c in largest_remainder(targets, n).items() for _ in range(c)]
    rng.shuffle(values)
    return values


def category(title: str) -> str:
    t = title.lower()
    for name, keywords in (
        ("fire", ("fire", "explosion")),
        ("grounding", ("grounding",)),
        ("collision", ("collision",)),
        ("contact", ("contact", "breakaway", "collapse")),
        ("flooding_sinking", ("flooding", "sinking", "capsizing", "listing", "implosion")),
        ("machinery", ("engine failure", "wire failure")),
    ):
        if any(k in t for k in keywords):
            return name
    return "other"


class ReportLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reports: tuple[str, str]
    entity: str


def load_links(path: Path, texts: dict[str, str]) -> list[ReportLink]:
    """Curated report pairs; each entity must occur in both reports' text."""
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw: object = cast(dict[str, Any], loaded).get("links") if isinstance(loaded, dict) else None
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a 'links' list")
    links = [ReportLink.model_validate(item) for item in cast(list[object], raw)]
    for link in links:
        for doc_id in link.reports:
            if doc_id not in texts:
                raise ValueError(f"{path}: unknown report {doc_id}")
            if link.entity.lower() not in " ".join(texts[doc_id].split()).lower():
                raise ValueError(f"{path}: {link.entity!r} does not occur in {doc_id}")
    return links


def _make_slots(spec: DimensionSpec, n: int, rng: random.Random) -> list[Slot]:
    type_counts = largest_remainder(spec.dimension("question_type").targets(), n)
    slots = [
        Slot(slot_id="", question_type=qtype)
        for qtype, count in type_counts.items()
        for _ in range(count)
    ]
    unanswerable = [s for s in slots if s.question_type == "unanswerable"]
    answerable = [s for s in slots if s.question_type != "unanswerable"]

    for slot, kind in zip(
        unanswerable,
        _quota(spec.dimension("unanswerable_kind").targets(), len(unanswerable), rng),
        strict=True,
    ):
        slot.unanswerable_kind = kind

    # Complexity: comparison and multi-document slots choose first, from multi-evidence values.
    remaining = Counter(largest_remainder(spec.dimension("complexity").targets(), len(answerable)))
    constrained_first = sorted(answerable, key=lambda s: s.question_type not in PAIR_TYPES)
    for slot in constrained_first:
        allowed = MULTI_EVIDENCE if slot.question_type in PAIR_TYPES else tuple(remaining)
        choice = max(allowed, key=lambda v: (remaining[v], rng.random()))
        slot.complexity = choice
        remaining[choice] -= 1

    for name in ("phrasing", "evidence_location"):
        for slot, value in zip(
            answerable,
            _quota(spec.dimension(name).targets(), len(answerable), rng),
            strict=True,
        ):
            setattr(slot, name, value)

    cross_share = spec.dimension("chunk_boundary").targets()["cross_chunk"]
    eligible = [s for s in answerable if s.question_type in SINGLE_TYPES]
    for slot in rng.sample(eligible, min(len(eligible), round(cross_share * len(answerable)))):
        slot.target_cross_chunk = True
    return slots


@dataclass
class _Usage:
    """Questions assigned per report, enforcing the per-report and long-report caps."""

    long_ids: set[str]
    counts: Counter[str] = field(default_factory=Counter[str])

    def can_use(self, doc_id: str, n: int) -> bool:
        long_used = sum(self.counts[d] for d in self.long_ids)
        if doc_id in self.long_ids and long_used + n > MAX_LONG_REPORT_SLOTS:
            return False
        return self.counts[doc_id] + n <= MAX_SLOTS_PER_REPORT

    def add(self, doc_id: str, n: int) -> None:
        self.counts[doc_id] += n


def _pair_batches(
    slots: list[Slot],
    reports: list[ReportInfo],
    links: list[ReportLink],
    usage: _Usage,
    rng: random.Random,
) -> list[Batch]:
    """Curated entity-linked pairs first (natural multi-document questions), then pairs
    of reports about the same kind of casualty (natural comparisons)."""
    pair_slots = [s for s in slots if s.question_type in PAIR_TYPES]
    n_pairs = math.ceil(len(pair_slots) / 2)
    linked = [link.reports for link in links]
    rng.shuffle(linked)
    same_category = [
        (a.doc_id, b.doc_id)
        for i, a in enumerate(reports)
        for b in reports[i + 1 :]
        if category(a.title) == category(b.title) != "other"
    ]
    rng.shuffle(same_category)
    # Long reports are poor pair partners (and include the most memorable casualties).
    same_category.sort(key=lambda p: p[0] in usage.long_ids or p[1] in usage.long_ids)
    chosen: list[tuple[str, str]] = []
    in_pairs: set[str] = set()
    for a, b in linked + same_category:
        if len(chosen) == n_pairs:
            break
        if a in in_pairs or b in in_pairs or not (usage.can_use(a, 2) and usage.can_use(b, 2)):
            continue
        chosen.append((a, b))
        in_pairs.update((a, b))
    if len(chosen) < n_pairs:
        raise ValueError(f"could only form {len(chosen)} of {n_pairs} report pairs")
    # Interleave so most pairs get one multi-document and one comparison slot.
    multi = [s for s in pair_slots if s.question_type == "multi_document"]
    comparison = [s for s in pair_slots if s.question_type == "comparison"]
    rng.shuffle(multi)
    rng.shuffle(comparison)
    ordered = [s for pair in zip(multi, comparison, strict=False) for s in pair]
    ordered += multi[len(comparison) :] + comparison[len(multi) :]
    batches: list[Batch] = []
    for i, (a, b) in enumerate(chosen):
        batch_slots = ordered[2 * i : 2 * i + 2]
        usage.add(a, len(batch_slots))
        usage.add(b, len(batch_slots))
        batches.append(Batch(batch_id="", kind="pair", doc_ids=[a, b], slots=batch_slots))
    return batches


def _single_batches(
    slots: list[Slot],
    reports: list[ReportInfo],
    usage: _Usage,
    avoid: set[str],
    rng: random.Random,
) -> list[Batch]:
    """Single-report batches, preferring reports not already used by a pair."""
    single = [
        s
        for s in slots
        if s.question_type in SINGLE_TYPES
        or s.unanswerable_kind in ("missing_detail", "false_premise")
    ]
    rng.shuffle(single)
    order = [r.doc_id for r in reports]
    rng.shuffle(order)
    order.sort(key=lambda d: (d in avoid, d in usage.long_ids))
    batches: list[Batch] = []
    position = 0
    for doc_id in order:
        if position >= len(single):
            break
        take = min(SLOTS_PER_SINGLE_BATCH, len(single) - position)
        if not usage.can_use(doc_id, take):
            continue
        batch_slots = single[position : position + take]
        batches.append(Batch(batch_id="", kind="single", doc_ids=[doc_id], slots=batch_slots))
        usage.add(doc_id, take)
        position += take
    if position < len(single):
        raise ValueError("not enough reports for the single-report slots under the caps")
    return batches


def build_plan(
    spec: DimensionSpec, reports: list[ReportInfo], links: list[ReportLink], seed: int
) -> Plan:
    rng = random.Random(seed)
    slots = _make_slots(spec, spec.sizes.candidates, rng)
    typical = median(r.text_length for r in reports)
    usage = _Usage(long_ids={r.doc_id for r in reports if r.text_length > 4 * typical})

    batches = _pair_batches(slots, reports, links, usage, rng)
    in_pairs = {d for b in batches for d in b.doc_ids}
    batches += _single_batches(slots, reports, usage, in_pairs, rng)
    out_of_corpus = [s for s in slots if s.unanswerable_kind == "out_of_corpus"]
    if out_of_corpus:
        batches.append(Batch(batch_id="", kind="out_of_corpus", doc_ids=[], slots=out_of_corpus))

    for b_index, batch in enumerate(batches, start=1):
        batch.batch_id = f"b{b_index:03d}"
        for s_index, slot in enumerate(batch.slots, start=1):
            slot.slot_id = f"{batch.batch_id}-{s_index}"

    used = {d for b in batches for d in b.doc_ids}
    if len(used) < MIN_REPORTS_USED:
        raise ValueError(f"plan uses only {len(used)} reports (< {MIN_REPORTS_USED})")
    unknown = used - {r.doc_id for r in reports}
    if unknown:
        raise ValueError(f"plan refers to unknown reports {sorted(unknown)}")
    return Plan(seed=seed, n_slots=len(slots), batches=batches)
