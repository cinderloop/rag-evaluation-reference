"""``rag-eval gate ...``: derive thresholds from a baseline, check a run against them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rag_eval.evaluators.evaluate import TraceScores, load_traces
from rag_eval.gate.thresholds import (
    GATED,
    REPORT_ONLY,
    check,
    derive,
    load_gate_config,
    write_gate_config,
)
from rag_eval.mutations.compare import MetricName, by_question

app = typer.Typer(no_args_is_help=True, help="Eval gate: fail when a gated metric drops.")

GATE_CONFIG = Path("configs/gate.yaml")
HEADER = """\
# Eval-gate thresholds (docs/decisions/0012-gate-thresholds.md), derived from a scored
# baseline run with `rag-eval gate derive`. A gated metric fails the gate when it moves
# past its threshold in the bad direction; report-only metrics never fail it.
"""


def run_rates(run_dir: Path) -> dict[MetricName, dict[str, list[float]]]:
    """Per-question outcomes of every gated and report-only metric in a scored run."""
    path = run_dir / "scores.jsonl"
    if not path.exists():
        raise typer.BadParameter(f"{run_dir} has no scores.jsonl; run rag-eval evaluate first")
    lines = path.read_text(encoding="utf-8").splitlines()
    scores = [TraceScores.model_validate_json(line) for line in lines if line.strip()]
    traces = load_traces(run_dir)
    metrics: list[MetricName] = [*GATED, *REPORT_ONLY]
    return {m: by_question(scores, traces, m) for m in metrics}


@app.command("derive")
def derive_command(
    baseline: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    family_alpha: Annotated[float, typer.Option(min=0.001, max=0.5)] = 0.05,
    out: Annotated[Path, typer.Option("--out")] = GATE_CONFIG,
) -> None:
    """Write gate thresholds derived from a scored baseline run."""
    config = derive(run_rates(baseline), baseline.name, family_alpha)
    write_gate_config(config, out, HEADER)
    for metric, t in config.metrics.items():
        bound = ">=" if t.direction == "min" else "<="
        typer.echo(f"{metric:22} baseline {t.baseline:6.1%}  pass if {bound} {t.threshold:.1%}")
    typer.echo(f"Wrote {out}")


def run_check(run_dir: Path, gate_config: Path) -> bool:
    """Print the gate result for a scored run and save it next to the run; True if it passes."""
    results = check(run_rates(run_dir), load_gate_config(gate_config))
    for r in results:
        if r.passed is None:
            typer.echo(f"{r.metric:22} {r.value:6.1%}  (report only)")
        else:
            bound = ">=" if r.direction == "min" else "<="
            status = "PASS" if r.passed else "FAIL"
            typer.echo(f"{r.metric:22} {r.value:6.1%}  {status} (needs {bound} {r.threshold:.1%})")
    passed = all(r.passed is not False for r in results)
    (run_dir / "gate.json").write_text(
        json.dumps(
            {
                "gate_config": str(gate_config),
                "passed": passed,
                "checks": [r.model_dump() for r in results],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo("Gate PASSED" if passed else "Gate FAILED")
    return passed


@app.command("check")
def check_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    gate_config: Annotated[Path, typer.Option("--gate-config", exists=True)] = GATE_CONFIG,
) -> None:
    """Check a scored run against the gate thresholds; exit code 1 if any gated metric fails."""
    if not run_check(run_dir, gate_config):
        raise typer.Exit(code=1)
