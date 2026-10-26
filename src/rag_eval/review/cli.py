"""``rag-eval review ...``: human review steps of error analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_eval.dataset.schema import Case, load_cases
from rag_eval.review.traces import export_coding_page, load_open_coding, store_open_coding
from rag_eval.runner.records import TraceRecord

app = typer.Typer(no_args_is_help=True, help="Review traces for error analysis.")

DATA = Path("data")
TRACES_PAGE = DATA / "review" / "traces.html"
OPEN_CODING_LABELS = DATA / "labels" / "error-analysis"
DEFAULT_SEED = 20261023

RunsOption = Annotated[
    list[Path], typer.Option("--run", help="Run directory to include (repeatable)")
]


def _inputs(runs: list[Path]) -> tuple[list[TraceRecord], dict[str, Case]]:
    traces = [
        TraceRecord.model_validate_json(line)
        for run in runs
        for line in (run / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = {c.id: c for split in ("dev", "test") for c in load_cases(DATA / f"{split}.jsonl")}
    return traces, cases


@app.command("traces-export")
def traces_export_command(runs: RunsOption, seed: int = DEFAULT_SEED) -> None:
    """Write the open-coding page for the given runs (data/review/traces.html)."""
    traces, cases = _inputs(runs)
    n = export_coding_page(traces, cases, TRACES_PAGE, seed)
    typer.echo(f"{n} traces to code: open {TRACES_PAGE} in a browser.")


@app.command("traces-import")
def traces_import_command(
    notes: Annotated[Path, typer.Argument(exists=True, dir_okay=False)], runs: RunsOption
) -> None:
    """Validate the open-coding notes and store them under data/labels/error-analysis/."""
    traces, _ = _inputs(runs)
    coding = load_open_coding(notes, traces)
    stored = store_open_coding(notes, OPEN_CODING_LABELS)
    flagged = sum(not n.ok for n in coding.notes.values())
    typer.echo(f"{len(coding.notes)} traces, {flagged} with a failure noted. Labels: {stored}")
