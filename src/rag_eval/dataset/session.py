"""Generating candidates in a Claude Code session instead of through the API.

The same plan, instructions and checks as the API path (``generate.py``), with the
model call replaced by an agent working from files: ``write_briefs`` renders each
batch's request as a brief plus the report texts; the agent writes its items to
``<batch_id>.json`` in the ``GeneratedBatch`` format and self-checks them with
``check_batch`` (the retry loop); ``collect`` validates every batch file again and
builds the candidates. Why this path exists: docs/decisions/0004.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from rag_eval.dataset.candidates import Candidate, GeneratorInfo
from rag_eval.dataset.corpus_text import ReportText
from rag_eval.dataset.dimensions import DimensionSpec
from rag_eval.dataset.generate import (
    GeneratedBatch,
    build_request,
    system_prompt,
    to_candidate,
    validate_item,
)
from rag_eval.dataset.plan import Batch, Plan

OUTPUT_FORMAT = """Write your answer as JSON to {output} (and nothing else in that file):

{{"items": [{{"slot_id": "...", "question": "...", "expected_answer": "...",
  "evidence": [{{"doc_id": "...", "quote": "..."}}], "notes": "..."}}]}}

One item per slot. Then run `uv run rag-eval dataset check-batch {batch_id}` and fix
every reported problem until it prints OK. Quotes are matched against the report
files given above after collapsing whitespace; everything else must match exactly."""


def _report_file(doc_id: str) -> str:
    return f"{doc_id}.txt"


def brief_text(
    batch: Batch,
    *,
    spec: DimensionSpec,
    reports: dict[str, ReportText],
    titles: dict[str, str],
    output: Path,
) -> str:
    """The batch's API request, rendered for an agent that reads files."""
    request = build_request(batch, batch.slots, spec=spec, reports=reports, titles=titles)
    context, task = (block["text"] for block in request["messages"][0]["content"])
    if batch.kind != "out_of_corpus":
        context = "\n".join(
            f"- {d} ({titles[d]}): read the full text in {_report_file(d)} next to this brief"
            for d in batch.doc_ids
        )
    return (
        f"# Brief {batch.batch_id}\n\n## Instructions\n\n{system_prompt(spec)}\n\n"
        f"## Reports\n\n{context}\n\n## Task\n\n{task}\n\n## Output\n\n"
        + OUTPUT_FORMAT.format(output=output, batch_id=batch.batch_id)
        + "\n"
    )


def write_briefs(
    plan: Plan,
    *,
    spec: DimensionSpec,
    reports: dict[str, ReportText],
    titles: dict[str, str],
    briefs_dir: Path,
    outputs_dir: Path,
) -> list[Path]:
    """Write ``briefs_dir/<batch_id>/brief.md`` and the report texts each brief needs."""
    paths: list[Path] = []
    for batch in plan.batches:
        directory = briefs_dir / batch.batch_id
        directory.mkdir(parents=True, exist_ok=True)
        for doc_id in batch.doc_ids:
            (directory / _report_file(doc_id)).write_text(reports[doc_id].text, encoding="utf-8")
        path = directory / "brief.md"
        output = outputs_dir / f"{batch.batch_id}.json"
        path.write_text(
            brief_text(batch, spec=spec, reports=reports, titles=titles, output=output),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def brief_sha256(brief: str, batch: Batch, reports: dict[str, ReportText]) -> str:
    """Hash of what the agent was given: the brief and the report texts it points to."""
    payload = {"brief": brief, "reports": {d: reports[d].sha256 for d in batch.doc_ids}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def check_batch(batch: Batch, output: Path, reports: dict[str, ReportText]) -> list[str]:
    """Every problem with an agent's batch file, as ``<slot_id>: <problem>`` lines."""
    if not output.exists():
        return [f"{batch.batch_id}: no output file {output}"]
    try:
        generated = GeneratedBatch.model_validate_json(output.read_text(encoding="utf-8"))
    except ValueError as error:
        return [f"{batch.batch_id}: invalid JSON for the output format: {error}"]
    items = {i.slot_id: i for i in generated.items}
    expected = {s.slot_id for s in batch.slots}
    problems = [f"{sid}: not a slot of {batch.batch_id}" for sid in items if sid not in expected]
    for slot in batch.slots:
        item = items.get(slot.slot_id)
        if item is None:
            problems.append(f"{slot.slot_id}: no item for this slot")
            continue
        problems.extend(f"{slot.slot_id}: {e}" for e in validate_item(slot, item, batch, reports))
    return problems


@dataclass
class CollectOutcome:
    candidates: list[Candidate] = field(default_factory=list[Candidate])
    failures: dict[str, list[str]] = field(default_factory=dict[str, list[str]])


def collect(
    plan: Plan,
    *,
    spec: DimensionSpec,
    reports: dict[str, ReportText],
    titles: dict[str, str],
    outputs_dir: Path,
    model: str,
) -> CollectOutcome:
    """Validate every batch file; valid items become candidates, the rest are recorded."""
    outcome = CollectOutcome()
    for batch in plan.batches:
        output = outputs_dir / f"{batch.batch_id}.json"
        problems = check_batch(batch, output, reports)
        failed = {p.split(":")[0] for p in problems}
        if batch.batch_id in failed:  # whole file missing or unreadable
            for slot in batch.slots:
                outcome.failures[slot.slot_id] = problems
            continue
        items = {
            i.slot_id: i
            for i in GeneratedBatch.model_validate_json(output.read_text(encoding="utf-8")).items
        }
        info = GeneratorInfo(
            model_requested=model,
            model_served=model,
            prompt_sha256=brief_sha256(
                brief_text(batch, spec=spec, reports=reports, titles=titles, output=output),
                batch,
                reports,
            ),
            response_id=f"session:{batch.batch_id}",
            attempt=1,
        )
        for slot in batch.slots:
            if slot.slot_id in failed:
                outcome.failures[slot.slot_id] = [
                    p for p in problems if p.startswith(f"{slot.slot_id}:")
                ]
            else:
                outcome.candidates.append(to_candidate(slot, items[slot.slot_id], batch, info))
    return outcome
