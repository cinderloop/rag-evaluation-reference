"""Command-line entry point: ``rag-eval provision | run | evaluate | report | gate``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from rag_eval.client import AnythingLLMClient
from rag_eval.client.ollama import installed_models
from rag_eval.config import RunConfig, load_config
from rag_eval.dataset.schema import load_cases
from rag_eval.env import Env
from rag_eval.provision import ProvisionResult, provision
from rag_eval.provision.pins import verify_container_image, verify_ollama_models
from rag_eval.runner import new_run_id, run

app = typer.Typer(no_args_is_help=True, add_completion=False)

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


@app.command("provision")
def provision_command(config: ConfigOption) -> None:
    """Create or update the workspace for a config, ingest its corpus and apply its settings."""
    env = Env.from_environ()
    with AnythingLLMClient(env.anythingllm_base_url, env.anythingllm_api_key) as client:
        _provision(env, client, load_config(config))


@app.command("run")
def run_command(
    config: ConfigOption,
    dataset: Annotated[Path, typer.Option("--dataset", "-d", exists=True, dir_okay=False)],
    repeats: Annotated[int, typer.Option(min=1)] = 1,
    out: Annotated[Path, typer.Option("--out")] = Path("results"),
) -> None:
    """Provision (idempotently), then run every case in a fresh thread and write traces."""
    env = Env.from_environ()
    run_config = load_config(config)
    cases = load_cases(dataset)
    with AnythingLLMClient(env.anythingllm_base_url, env.anythingllm_api_key) as client:
        provisioned, verified = _provision(env, client, run_config)
        run_dir = run(
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
    typer.echo(f"Wrote {run_dir}")


def _not_yet(command: str) -> None:
    typer.echo(f"'{command}' is not implemented yet.", err=True)
    raise typer.Exit(code=2)


@app.command("evaluate")
def evaluate_command() -> None:
    """Score a run's traces (retrieval metrics and judges). Not implemented yet."""
    _not_yet("evaluate")


@app.command("report")
def report_command() -> None:
    """Summarise scored runs as markdown. Not implemented yet."""
    _not_yet("report")


@app.command("gate")
def gate_command() -> None:
    """Fail if any metric drops below its threshold. Not implemented yet."""
    _not_yet("gate")
