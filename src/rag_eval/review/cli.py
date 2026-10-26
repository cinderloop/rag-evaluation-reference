"""``rag-eval review ...``: review steps of error analysis and judge validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from rag_eval.dataset.schema import Case, load_cases
from rag_eval.evaluators.evaluate import Job, Spend, estimate, run_judges
from rag_eval.evaluators.judges.backends import CachedBackend, load_judge_config, make_backend
from rag_eval.evaluators.judges.judges import JUDGES, JudgeName, Verdict
from rag_eval.review.judge_labels import (
    JudgeSplit,
    agreement,
    export_label_page,
    labelled_cases,
    load_judge_labels,
    store_judge_labels,
)
from rag_eval.review.traces import (
    export_coding_page,
    load_open_coding,
    store_open_coding,
    trace_key,
)
from rag_eval.runner.records import TraceRecord

app = typer.Typer(no_args_is_help=True, help="Review traces and label judges.")

DATA = Path("data")
TRACES_PAGE = DATA / "review" / "traces.html"
OPEN_CODING_LABELS = DATA / "labels" / "error-analysis"
DEFAULT_SEED = 20261023
DEFAULT_JUDGE_CONFIG = Path("configs/judges/claude-code-haiku.yaml")
JUDGE_CACHE = DATA / "cache" / "judges"

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


JUDGE_LABELS_PAGE = DATA / "review" / "judge-labels.html"
JUDGE_LABELS = DATA / "labels" / "judge-validation"
JUDGE_VALIDATION_RESULTS = Path("results/judge-validation/agreement")
JUDGE_SEED = 20261028


@app.command("judges-export")
def judges_export_command(runs: RunsOption, seed: int = JUDGE_SEED) -> None:
    """Write the judge-labelling page for the given runs (data/review/judge-labels.html)."""
    traces, cases = _inputs(runs)
    counts = export_label_page(traces, cases, JUDGE_LABELS_PAGE, seed)
    total = sum(counts.values())
    typer.echo(f"{total} marks to make: " + ", ".join(f"{j} {n}" for j, n in counts.items()))
    typer.echo(f"Open {JUDGE_LABELS_PAGE} in a browser.")


@app.command("judges-import")
def judges_import_command(
    labels: Annotated[Path, typer.Argument(exists=True, dir_okay=False)], runs: RunsOption
) -> None:
    """Validate the judge labels and store them under data/labels/judge-validation/."""
    traces, cases = _inputs(runs)
    loaded = load_judge_labels(labels, traces, cases)
    stored = store_judge_labels(labels, JUDGE_LABELS)
    rows = labelled_cases(loaded, traces)
    for judge in JUDGES:
        mine = [r for r in rows if r.judge == judge]
        by_split = {
            s: (
                sum(r.human == "Pass" for r in mine if r.split == s),
                sum(r.human == "Fail" for r in mine if r.split == s),
            )
            for s in ("judge-dev", "judge-test")
        }
        typer.echo(
            f"{judge}: " + "; ".join(f"{s} {p} Pass / {f} Fail" for s, (p, f) in by_split.items())
        )
    typer.echo(f"Labels: {stored}")


@app.command("judges-validate")
def judges_validate_command(
    labels: Annotated[Path, typer.Option("--labels", exists=True, dir_okay=False)],
    runs: RunsOption,
    *,
    split: Annotated[str, typer.Option(help="judge-dev (iterate) or judge-test (final)")],
    judge_config: Annotated[
        Path, typer.Option("--judge-config", exists=True, dir_okay=False)
    ] = DEFAULT_JUDGE_CONFIG,
    max_cost: Annotated[float, typer.Option()] = 2.0,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    """Run the judges on labelled traces of one split and report TPR and TNR."""
    if split not in ("judge-dev", "judge-test"):
        raise typer.BadParameter("--split must be judge-dev or judge-test")
    chosen: JudgeSplit = split
    traces, cases = _inputs(runs)
    loaded = load_judge_labels(labels, traces, cases)
    rows = labelled_cases(loaded, traces, chosen)
    by_key = {trace_key(t): t for t in traces}
    keys = sorted({r.key for r in rows})
    index = {k: i for i, k in enumerate(keys)}
    jobs = [
        Job(index[r.key], JUDGES[r.judge], by_key[r.key], cases[by_key[r.key].case_id])
        for r in rows
    ]
    config = load_judge_config(judge_config)
    backend = CachedBackend(make_backend(config), JUDGE_CACHE)
    est = estimate(jobs, backend)
    typer.echo(
        f"{chosen}: {len(jobs)} labelled verdicts; {est.calls} judge calls, {est.cached} cached. "
        f"Estimated cost ~${(est.cost_usd or 0.0):.4f} (cap ${max_cost:.2f})."
    )
    if est.calls and not yes and not typer.confirm("Proceed?"):
        raise typer.Exit(code=1)
    per_call = (est.cost_usd or 0.0) / est.calls if est.calls else 0.0
    verdicts = run_judges(jobs, backend, Spend(cap_usd=max_cost, per_call_usd=per_call))
    by_label: dict[tuple[str, JudgeName], Verdict] = {
        (keys[i], name): v for (i, name), v in verdicts.items()
    }
    results = [
        agreement(j, chosen, rows, by_label) for j in JUDGES if any(r.judge == j for r in rows)
    ]
    JUDGE_VALIDATION_RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = JUDGE_VALIDATION_RESULTS / f"{stamp}-{chosen}.json"
    out.write_text(
        json.dumps(
            {
                "split": chosen,
                "labels": str(labels),
                "judge_config": str(judge_config),
                "judge_model": config.model,
                "prompt_sha256": {j: JUDGES[j].prompt_sha256 for j in JUDGES},
                "agreement": [a.model_dump() for a in results],
                "verdicts": [
                    {"key": k, "judge": j, **v.model_dump()}
                    for (k, j), v in sorted(by_label.items())
                ],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    for a in results:
        tpr = "n/a" if a.tpr is None else f"{a.tpr:.0%} ({a.tpr_ci[0]:.0%}-{a.tpr_ci[1]:.0%})"
        tnr = "n/a" if a.tnr is None else f"{a.tnr:.0%} ({a.tnr_ci[0]:.0%}-{a.tnr_ci[1]:.0%})"
        typer.echo(
            f"{a.judge}: TPR {tpr} on {a.human_pass} Pass, TNR {tnr} on {a.human_fail} Fail"
            + (f", {a.errors} errors" if a.errors else "")
        )
    typer.echo(f"Wrote {out}")
