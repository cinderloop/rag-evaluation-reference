"""``rag-eval mutations ...``: build the knowledge edit and compare runs with the baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rag_eval.evaluators.evaluate import TraceScores, load_traces
from rag_eval.mutations.compare import (
    EXTRACTORS,
    MetricName,
    by_question,
    compare_metric,
    load_expectations,
    targeted,
)
from rag_eval.mutations.knowledge_edit import build_edited_document, load_edit

app = typer.Typer(no_args_is_help=True, help="Mutation testing: does the evaluation notice?")

KNOWLEDGE_EDIT = Path("data/mutations/knowledge-edit.yaml")
EDITED_DIR = Path("corpus/edits")
EXPECTATIONS = Path("data/mutations/expectations.yaml")
COMPARISONS = Path("results/mutations/comparisons")


@app.command("build-edit")
def build_edit_command() -> None:
    """Write the edited document for the knowledge-edit mutation (corpus/edits/)."""
    edit = load_edit(KNOWLEDGE_EDIT)
    out = build_edited_document(edit, Path("data/corpus_text"), EDITED_DIR / edit.doc_id)
    typer.echo(f"Wrote {out}: '{edit.find}' -> '{edit.replace}'")


def _scores(run_dir: Path) -> list[TraceScores]:
    path = run_dir / "scores.jsonl"
    if not path.exists():
        raise typer.BadParameter(f"{run_dir} has no scores.jsonl; run rag-eval evaluate first")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [TraceScores.model_validate_json(line) for line in lines if line.strip()]


@app.command("compare")
def compare_command(
    baseline: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    mutation: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    name: Annotated[str, typer.Option(help="Mutation name in expectations.yaml")],
) -> None:
    """Compare a scored mutation run with the scored baseline run."""
    expectation = next(
        (m for m in load_expectations(EXPECTATIONS).mutations if m.name == name), None
    )
    if expectation is None:
        raise typer.BadParameter(f"no mutation named {name} in {EXPECTATIONS}")
    base_traces, mut_traces = load_traces(baseline), load_traces(mutation)
    base_scores, mut_scores = _scores(baseline), _scores(mutation)
    metrics: list[MetricName] = list(EXTRACTORS)
    base: dict[MetricName, dict[str, list[float]]] = {
        m: by_question(base_scores, base_traces, m) for m in metrics
    }
    mut: dict[MetricName, dict[str, list[float]]] = {
        m: by_question(mut_scores, mut_traces, m) for m in metrics
    }
    rows = [
        compare_metric(m, expectation.expect.get(m, "any"), base[m], mut[m])
        for m in metrics
        if base[m] and mut[m]
    ]
    focus = targeted(expectation.targeted_cases, base, mut)
    COMPARISONS.mkdir(parents=True, exist_ok=True)
    out = COMPARISONS / f"{name}.json"
    out.write_text(
        json.dumps(
            {
                "mutation": name,
                "baseline_run": baseline.name,
                "mutation_run": mutation.name,
                "metrics": [r.model_dump() for r in rows],
                "targeted": [t.model_dump() for t in focus],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    for r in rows:
        p = "" if r.p_value is None else f", p={r.p_value:.3g}"
        typer.echo(
            f"{r.metric:22} {r.baseline:6.1%} -> {r.mutation:6.1%} "
            f"({r.difference:+.1%}, CI {r.interval[0]:+.1%} to {r.interval[1]:+.1%}{p}) "
            f"expected {r.expected}: {r.outcome}"
        )
    for t in focus:
        typer.echo(f"  {t.case_id} {t.metric}: baseline {t.baseline} -> mutation {t.mutation}")
    typer.echo(f"Wrote {out}")
