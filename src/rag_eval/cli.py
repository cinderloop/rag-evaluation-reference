"""Command-line entry point: ``rag-eval provision | run | evaluate | report | gate``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, cast

import typer

from rag_eval.client import AnythingLLMClient
from rag_eval.client.ollama import installed_models
from rag_eval.config import RunConfig, load_config
from rag_eval.dataset.cli import app as dataset_app
from rag_eval.dataset.schema import load_cases
from rag_eval.env import Env
from rag_eval.evaluators.evaluate import (
    CostCapReachedError,
    Spend,
    estimate,
    load_traces,
    plan_jobs,
    run_judges,
    score_traces,
    summarize,
    write_scores,
)
from rag_eval.evaluators.judges.backends import CachedBackend, load_judge_config, make_backend
from rag_eval.evaluators.judges.judges import JUDGES, JudgeName, Verdict
from rag_eval.gate.cli import GATE_CONFIG, run_check
from rag_eval.gate.cli import app as gate_app
from rag_eval.mutations.cli import app as mutations_app
from rag_eval.mutations.knowledge_edit import edited_cases, load_edit
from rag_eval.provision import ProvisionResult, provision
from rag_eval.provision.pins import verify_container_image, verify_ollama_models
from rag_eval.provision.sources import fetch_sources, load_sources
from rag_eval.review.cli import app as review_app
from rag_eval.runner import new_run_id, run

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(dataset_app, name="dataset")
app.add_typer(review_app, name="review")
app.add_typer(mutations_app, name="mutations")
app.add_typer(gate_app, name="gate")

ConfigOption = Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _check_pins(env: Env, config: RunConfig) -> bool:
    verify_ollama_models(
        installed_models(env.ollama_base_url), [config.llm.model, config.embedder.model]
    )
    verified = verify_container_image(env.anythingllm_container, config.anythingllm_image)
    if not verified:
        logging.warning(
            "Could not verify the AnythingLLM image (docker CLI or container %s unavailable); "
            "recorded as unverified in the manifest.",
            env.anythingllm_container,
        )
    return verified


def _provision(
    env: Env, client: AnythingLLMClient, config: RunConfig
) -> tuple[ProvisionResult, bool]:
    verified = _check_pins(env, config)
    result = provision(client, config)
    typer.echo(
        f"Workspace {result.workspace_slug}: {len(result.corpus.documents)} documents "
        f"({len(result.uploaded)} uploaded, {len(result.removed)} removed); "
        f"instance changes: {', '.join(result.instance_changes) or 'none'}"
    )
    return result, verified


@app.command("fetch-corpus")
def fetch_corpus_command(
    sources: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "corpus/sources.yaml"
    ),
    dest: Annotated[Path, typer.Option()] = Path("corpus/documents"),
) -> None:
    """Download the corpus documents listed in sources.yaml and verify their checksums."""
    result = fetch_sources(load_sources(sources), dest)
    typer.echo(
        f"{len(result.downloaded)} downloaded, {len(result.already_present)} already present; "
        "all checksums verified."
    )
    if result.unexpected:
        typer.echo(
            f"Not in sources.yaml (would also be ingested; remove them): {result.unexpected}",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command("provision")
def provision_command(config: ConfigOption) -> None:
    """Create or update the workspace for a config, ingest its corpus and apply its settings."""
    env = Env.from_environ()
    with AnythingLLMClient(env.anythingllm_base_url, env.anythingllm_api_key) as client:
        _provision(env, client, load_config(config))


def _run(config: Path, dataset: Path, repeats: int, out: Path) -> Path:
    env = Env.from_environ()
    run_config = load_config(config)
    cases = load_cases(dataset)
    with AnythingLLMClient(env.anythingllm_base_url, env.anythingllm_api_key) as client:
        provisioned, verified = _provision(env, client, run_config)
        return run(
            client=client,
            config=run_config,
            config_path=config,
            provisioned=provisioned,
            image_verified=verified,
            dataset_path=dataset,
            cases=cases,
            repeats=repeats,
            out_dir=out,
            run_id=new_run_id(run_config),
        )


@app.command("run")
def run_command(
    config: ConfigOption,
    dataset: Annotated[Path, typer.Option("--dataset", "-d", exists=True, dir_okay=False)],
    repeats: Annotated[int, typer.Option(min=1)] = 1,
    out: Annotated[Path, typer.Option("--out")] = Path("results"),
) -> None:
    """Provision (idempotently), then run every case in a fresh thread and write traces."""
    typer.echo(f"Wrote {_run(config, dataset, repeats, out)}")


def _not_yet(command: str) -> None:
    typer.echo(f"'{command}' is not implemented yet.", err=True)
    raise typer.Exit(code=2)


DEFAULT_JUDGE_CONFIG = Path("configs/judges/claude-code-haiku.yaml")
JUDGE_CACHE = Path("data/cache/judges")


@app.command("evaluate")
def evaluate_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    *,
    judge_config: Annotated[
        Path, typer.Option("--judge-config", exists=True, dir_okay=False)
    ] = DEFAULT_JUDGE_CONFIG,
    judges: Annotated[
        str, typer.Option(help="Comma-separated judges, or 'none' for retrieval metrics only")
    ] = ",".join(JUDGES),
    max_cost: Annotated[
        float, typer.Option(help="Stop before judge spend exceeds this (USD)")
    ] = 2.0,
    workers: Annotated[int, typer.Option(min=1, max=16)] = 4,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt")] = False,
    knowledge_edit: Annotated[
        Path | None,
        typer.Option(
            "--knowledge-edit",
            exists=True,
            dir_okay=False,
            help="Score against the edited expected answers of a knowledge-edit mutation",
        ),
    ] = None,
) -> None:
    """Score a run: retrieval metrics, then binary judges (asks before any judge call)."""
    traces = load_traces(run_dir)
    cases = {c.id: c for split in ("dev", "test") for c in load_cases(Path(f"data/{split}.jsonl"))}
    if knowledge_edit is not None:
        cases = edited_cases(cases, load_edit(knowledge_edit))
    names = [] if judges == "none" else [n.strip() for n in judges.split(",") if n.strip()]
    unknown = sorted(set(names) - set(JUDGES))
    if unknown:
        raise typer.BadParameter(f"unknown judges {unknown}; choose from {list(JUDGES)}")
    specs = [JUDGES[cast(JudgeName, n)] for n in names]

    config = load_judge_config(judge_config) if specs else None
    verdicts: dict[tuple[int, JudgeName], Verdict] = {}
    spent = 0.0
    if config is not None:
        backend = CachedBackend(make_backend(config), JUDGE_CACHE)
        jobs = plan_jobs(traces, cases, specs)
        est = estimate(jobs, backend)
        cost = (
            "unknown (no prices in the judge config)"
            if est.cost_usd is None
            else f"~${est.cost_usd:.4f}"
        )
        typer.echo(
            f"Judge {config.provider}:{config.model}, {len(jobs)} verdicts: {est.calls} calls, "
            f"{est.cached} cached, {est.by_rule} decided by rule.\n"
            f"~{est.input_tokens:,} input and ~{est.output_tokens:,} output tokens; "
            f"estimated cost {cost}. Cap: ${max_cost:.2f}."
        )
        if config.provider == "claude_code":
            typer.echo("Claude Code: counts toward the logged-in subscription's usage limits.")
        if est.calls and not yes and not typer.confirm("Proceed?"):
            raise typer.Exit(code=1)
        per_call = (est.cost_usd or 0.0) / est.calls if est.calls else 0.0
        spend = Spend(cap_usd=max_cost, per_call_usd=per_call)

        def progress(done: int, total: int) -> None:
            if done % 25 == 0 or done == total:
                typer.echo(f"  {done}/{total} verdicts")

        try:
            verdicts = run_judges(jobs, backend, spend, workers=workers, progress=progress)
        except CostCapReachedError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error
        spent = spend.spent_usd

    scores = score_traces(traces, cases, verdicts)
    summary = summarize(scores, specs, config, judge_config if config else None, spent)
    write_scores(run_dir, scores, summary)
    typer.echo(f"Wrote {run_dir / 'scores.jsonl'} and {run_dir / 'summary.json'}")
    typer.echo(
        json.dumps({"retrieval": summary["retrieval"], "judges": summary["judges"]}, indent=1)
    )


@app.command("report")
def report_command() -> None:
    """Summarise scored runs as markdown. Not implemented yet."""
    _not_yet("report")


@gate_app.command("run")
def gate_run_command(
    config: ConfigOption = Path("configs/baseline.yaml"),
    dataset: Annotated[Path, typer.Option("--dataset", "-d", exists=True, dir_okay=False)] = Path(
        "data/test.jsonl"
    ),
    *,
    gate_config: Annotated[Path, typer.Option("--gate-config", exists=True)] = GATE_CONFIG,
    max_cost: Annotated[float, typer.Option(help="Judge spend cap (USD)")] = 2.0,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt")] = False,
    out: Annotated[Path, typer.Option("--out")] = Path("results/gate"),
) -> None:
    """Run the config on the dataset, score it, and check it against the gate thresholds."""
    run_dir = _run(config, dataset, 1, out)
    typer.echo(f"Wrote {run_dir}")
    evaluate_command(
        run_dir,
        judge_config=DEFAULT_JUDGE_CONFIG,
        judges=",".join(JUDGES),
        max_cost=max_cost,
        workers=4,
        yes=yes,
        knowledge_edit=None,
    )
    if not run_check(run_dir, gate_config):
        raise typer.Exit(code=1)
