"""Generating candidate questions from the plan with a hosted model.

One call per batch (one report, or a pair, plus the slots assigned to it). Output is
schema-constrained JSON. Every evidence quote is checked verbatim against the
report text; slots that fail are retried once with the errors fed back, then
dropped and recorded. Calls are cached by request hash, so re-running never pays
twice. Spend is estimated up front and capped (docs/decisions/0004).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.candidates import Candidate, GeneratorInfo
from rag_eval.dataset.corpus_text import ReportText, normalize
from rag_eval.dataset.dimensions import DimensionSpec
from rag_eval.dataset.plan import PAIR_TYPES, Batch, Plan, Slot
from rag_eval.dataset.schema import Evidence

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
# USD per million tokens for claude-sonnet-5-5 (cache write = 1.25x input, 5-minute TTL).
PRICE_INPUT = 2.00
PRICE_CACHE_WRITE = 2.50
PRICE_CACHE_READ = 0.20
PRICE_OUTPUT = 10.00
# Output budget per slot for the estimate: thinking at medium effort plus the JSON.
EST_OUTPUT_TOKENS_PER_SLOT = 1800
EST_RETRY_SHARE = 0.25
MIN_QUOTE_CHARS = 20
BOUNDARY_EXCERPTS_PER_SLOT = 8

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string"},
                    "question": {"type": "string"},
                    "expected_answer": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["doc_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": ["slot_id", "question", "expected_answer", "evidence", "notes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


class GeneratedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    question: str
    expected_answer: str
    evidence: list[Evidence]
    notes: str


class GeneratedBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[GeneratedItem]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_INPUT
            + self.cache_creation_input_tokens * PRICE_CACHE_WRITE
            + self.cache_read_input_tokens * PRICE_CACHE_READ
            + self.output_tokens * PRICE_OUTPUT
        ) / 1_000_000


@dataclass(frozen=True)
class CallResult:
    text: str
    model_served: str
    response_id: str
    usage: Usage
    cached: bool


class Caller(Protocol):
    def __call__(self, request: dict[str, Any]) -> CallResult: ...


# --- prompts -------------------------------------------------------------------------


def system_prompt(spec: DimensionSpec) -> str:
    lines: list[str] = []
    for name in (
        "question_type",
        "unanswerable_kind",
        "complexity",
        "phrasing",
        "evidence_location",
    ):
        dimension = spec.dimension(name)
        lines.append(f"{name}: {' '.join(dimension.description.split())}")
        for value, info in dimension.values.items():
            text = " ".join(info.description.split())
            example = f' Example: "{info.example}"' if info.example else ""
            lines.append(f"  - {value}: {text}{example}")
    definitions = "\n".join(lines)
    return f"""You write evaluation questions for a retrieval-augmented question-answering \
system over NTSB Marine Investigation Reports (2025-2026). A human reviews every question \
before it is used.

You receive one or two reports and a list of question slots. Write exactly one question per \
slot, following the slot's dimensions.

Dimensions:
{definitions}

Rules for every question:
- It must make sense to someone who has not seen the report: name the vessel or casualty \
explicitly so the right report can be found, never "this report" or "the vessel".
- Do not ask about report metadata (report numbers, page or figure numbers, footnotes, \
publication dates).
- expected_answer: a concise, complete answer (one or two sentences) that a careful reader \
of the report would give.
- evidence: verbatim quotes copied character for character from the report text provided, \
each at least one full clause long. Line breaks inside a quote may be written as single \
spaces; nothing else may change. Each quote gives the doc_id it comes from.
- phrasing "paraphrased": avoid the distinctive words and phrases of the evidence; use \
synonyms, lay terms or a different sentence structure. Names of vessels, places and \
organisations may stay. phrasing "close": reuse the report's key terms.
- complexity "inference": the answer needs a small step beyond the text, such as arithmetic, \
a duration between two times, or a unit conversion; quote every span the step uses.
- evidence_location: take the evidence from that part of the report (summary_table: the \
casualty summary or vessel particulars tables; narrative: the account of the casualty and \
the response; analysis: the analysis, probable cause and lessons learned).
- comparison and multi_document slots: the question needs both reports, and the evidence \
includes at least one quote from each.
- Slots with target_cross_chunk: base the question on one passage that spans one of the \
boundary_excerpts given for the slot (the boundary is marked with ‖); quote the passage from \
the report text, without the marker.

Unanswerable slots:
- missing_detail: ask about the named casualty for a plausible detail the report does not \
state anywhere. evidence is empty. expected_answer says the reports do not give this detail.
- false_premise: the question presupposes something the report contradicts (wrong vessel \
type, cause, outcome or location). evidence: the quote(s) that contradict the premise. \
expected_answer corrects the premise.
- out_of_corpus: ask about a plausible marine casualty that none of the listed reports covers \
(a vessel name and casualty that do not appear in the list). evidence is empty. \
expected_answer says the reports do not cover it.

notes: one sentence for the human reviewer on what the question tests."""


def _slot_spec(
    slot: Slot, batch: Batch, reports: dict[str, ReportText], rng_key: str
) -> dict[str, Any]:
    spec: dict[str, Any] = {"slot_id": slot.slot_id, **slot.dimensions()}
    if slot.target_cross_chunk:
        excerpts = reports[batch.doc_ids[0]].chunk_boundaries()
        # Deterministic spread of excerpts per slot.
        step = max(1, len(excerpts) // BOUNDARY_EXCERPTS_PER_SLOT)
        offset = int(hashlib.sha256(rng_key.encode()).hexdigest(), 16) % step
        spec["target_cross_chunk"] = True
        spec["boundary_excerpts"] = excerpts[offset::step][:BOUNDARY_EXCERPTS_PER_SLOT]
    return spec


def build_request(
    batch: Batch,
    slots: list[Slot],
    *,
    spec: DimensionSpec,
    reports: dict[str, ReportText],
    titles: dict[str, str],
    feedback: str | None = None,
) -> dict[str, Any]:
    if batch.kind == "out_of_corpus":
        listing = "\n".join(f"- {titles[d]}" for d in sorted(titles))
        context = (
            "<corpus_titles>\nThe corpus contains exactly these reports:\n"
            f"{listing}\n</corpus_titles>"
        )
    else:
        context = "\n\n".join(
            f'<report doc_id="{d}" title="{titles[d]}">\n{reports[d].text}\n</report>'
            for d in batch.doc_ids
        )
    slot_specs = [_slot_spec(s, batch, reports, s.slot_id) for s in slots]
    task = "Question slots:\n" + json.dumps(slot_specs, indent=1, ensure_ascii=False)
    if feedback:
        task += f"\n\nYour previous attempt for these slots had problems; fix them:\n{feedback}"
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [
            {"type": "text", "text": system_prompt(spec), "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": task},
                ],
            }
        ],
        "output_config": {
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }


def request_key(request: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# --- validation ----------------------------------------------------------------------


def validate_item(
    slot: Slot, item: GeneratedItem, batch: Batch, reports: dict[str, ReportText]
) -> list[str]:
    errors: list[str] = []
    if not item.question.strip().endswith("?"):
        errors.append("question must be a single question ending with '?'")
    if not item.expected_answer.strip():
        errors.append("expected_answer is empty")
    kind = slot.unanswerable_kind
    if kind in ("missing_detail", "out_of_corpus"):
        if item.evidence:
            errors.append(f"{kind} questions must have no evidence")
        return errors
    if not item.evidence:
        errors.append("evidence is required")
    for evidence in item.evidence:
        if evidence.doc_id not in batch.doc_ids:
            errors.append(f"evidence doc_id {evidence.doc_id!r} is not one of {batch.doc_ids}")
            continue
        if len(normalize(evidence.quote)) < MIN_QUOTE_CHARS:
            errors.append(f"quote too short: {evidence.quote!r}")
        elif not reports[evidence.doc_id].contains(evidence.quote):
            errors.append(
                f"quote not found verbatim in {evidence.doc_id}: {evidence.quote[:120]!r}"
            )
    if slot.question_type in PAIR_TYPES and {e.doc_id for e in item.evidence} != set(batch.doc_ids):
        errors.append("comparison/multi_document evidence must quote both reports")
    return errors


def to_candidate(slot: Slot, item: GeneratedItem, batch: Batch, info: GeneratorInfo) -> Candidate:
    false_premise = slot.unanswerable_kind == "false_premise"
    return Candidate(
        candidate_id=slot.slot_id,
        batch_id=batch.batch_id,
        doc_ids=batch.doc_ids,
        answerable=slot.question_type != "unanswerable",
        dimensions=slot.dimensions(),
        question=item.question.strip(),
        expected_answer=item.expected_answer.strip(),
        evidence=[] if false_premise else item.evidence,
        premise_evidence=item.evidence if false_premise else [],
        notes=item.notes.strip(),
        generator=info,
    )


# --- orchestration -------------------------------------------------------------------


@dataclass
class GenerationOutcome:
    candidates: list[Candidate] = field(default_factory=list[Candidate])
    failures: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    usage: list[Usage] = field(default_factory=list[Usage])
    calls: int = 0
    cached_calls: int = 0

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage)


class BudgetExceededError(RuntimeError):
    pass


def _parse(text: str) -> GeneratedBatch:
    return GeneratedBatch.model_validate_json(text)


def run_generation(
    plan: Plan,
    spec: DimensionSpec,
    reports: dict[str, ReportText],
    titles: dict[str, str],
    call: Caller,
    *,
    max_cost_usd: float,
    estimate_per_batch: Callable[[Batch], float],
) -> GenerationOutcome:
    outcome = GenerationOutcome()

    def guarded(request: dict[str, Any], batch: Batch) -> CallResult:
        if outcome.cost_usd + estimate_per_batch(batch) > max_cost_usd:
            raise BudgetExceededError(
                f"stopping before {batch.batch_id}: spent ${outcome.cost_usd:.2f}, "
                f"cap ${max_cost_usd:.2f}"
            )
        result = call(request)
        outcome.calls += 1
        outcome.cached_calls += result.cached
        if not result.cached:
            outcome.usage.append(result.usage)
        return result

    for batch in plan.batches:
        pending = list(batch.slots)
        feedback: str | None = None
        for attempt in (1, 2):
            request = build_request(
                batch, pending, spec=spec, reports=reports, titles=titles, feedback=feedback
            )
            result = guarded(request, batch)
            info = GeneratorInfo(
                model_requested=MODEL,
                model_served=result.model_served,
                prompt_sha256=request_key(request),
                response_id=result.response_id,
                attempt=attempt,
            )
            try:
                items = {i.slot_id: i for i in _parse(result.text).items}
            except ValueError as error:
                items = {}
                log.warning("%s: unparseable output (%s)", batch.batch_id, error)
            failed: list[Slot] = []
            problems: list[str] = []
            for slot in pending:
                item = items.get(slot.slot_id)
                errors = (
                    ["no item returned for this slot"]
                    if item is None
                    else validate_item(slot, item, batch, reports)
                )
                if item is not None and not errors:
                    outcome.candidates.append(to_candidate(slot, item, batch, info))
                else:
                    failed.append(slot)
                    problems.extend(f"{slot.slot_id}: {e}" for e in errors)
                    outcome.failures[slot.slot_id] = errors
            for slot in pending:
                if slot not in failed:
                    outcome.failures.pop(slot.slot_id, None)
            if not failed:
                break
            pending, feedback = failed, "\n".join(problems)
        log.info(
            "%s done (%d candidates so far, $%.2f)",
            batch.batch_id,
            len(outcome.candidates),
            outcome.cost_usd,
        )
    return outcome
