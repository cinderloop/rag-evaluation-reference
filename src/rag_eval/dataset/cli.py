"""``rag-eval dataset ...``: building the evaluation dataset (see data/dimensions.yaml)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from rag_eval.client import AnythingLLMClient
from rag_eval.config import load_config
from rag_eval.dataset.candidates import load_candidates, write_candidates
from rag_eval.dataset.checks import coverage_report, run_checks
from rag_eval.dataset.corpus_text import extract, load_report_texts
from rag_eval.dataset.dimensions import load_dimensions
from rag_eval.dataset.generate import (
    EFFORT,
    EST_OUTPUT_TOKENS_PER_SLOT,
    EST_RETRY_SHARE,
    MODEL,
    PRICE_INPUT,
    PRICE_OUTPUT,
    build_request,
    run_generation,
    system_prompt,
)
from rag_eval.dataset.plan import Batch, Plan, ReportInfo, build_plan, load_links
from rag_eval.dataset.review import export_review_page, import_decisions, store_labels
from rag_eval.dataset.schema import load_cases
from rag_eval.dataset.session import check_batch, collect, write_briefs
from rag_eval.dataset.split import stratified_split, write_split
from rag_eval.env import Env
from rag_eval.provision import provision
from rag_eval.provision.sources import load_sources

app = typer.Typer(no_args_is_help=True, help="Build the evaluation dataset.")
log = logging.getLogger(__name__)

DIMENSIONS = Path("data/dimensions.yaml")
LINKS = Path("data/report_links.yaml")
SOURCES = Path("corpus/sources.yaml")
CORPUS_TEXT = Path("data/corpus_text")
CANDIDATES_DIR = Path("data/candidates")
PLAN = CANDIDATES_DIR / "plan.json"
CANDIDATES = CANDIDATES_DIR / "candidates.jsonl"
CACHE = Path("data/cache/generation")
BRIEFS = Path("data/cache/briefs")
SESSION_OUTPUTS = CANDIDATES_DIR / "session"
SESSION_MODEL = "claude-opus-5-5 (Claude Code session)"
REVIEW_PAGE = Path("data/review/review.html")
LABELS = Path("data/labels/review")
APPROVED = Path("data/approved.jsonl")
DEFAULT_SEED = 20261021


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _titles() -> dict[str, str]:
    return {s.doc_id: s.title for s in load_sources(SOURCES)}


@app.command("extract")
def extract_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/baseline.yaml"),
) -> None:
    """Save each report's parsed text and baseline chunks from AnythingLLM (data/corpus_text)."""
    env = Env.from_environ()
    with AnythingLLMClient(env.anythingllm_base_url, env.anythingllm_api_key) as client:
        result = provision(client, load_config(config))
        n = extract(client, env.anythingllm_container, result.workspace_slug, CORPUS_TEXT)
    typer.echo(f"Extracted {n} reports to {CORPUS_TEXT}")


@app.command("plan")
def plan_command(seed: int = DEFAULT_SEED) -> None:
    """Turn the dimension targets into question slots, batched by report."""
    spec = load_dimensions(DIMENSIONS)
    reports = load_report_texts(CORPUS_TEXT)
    titles = _titles()
    infos = [
        ReportInfo(doc_id=d, title=titles[d], text_length=len(r.text)) for d, r in reports.items()
    ]
    links = load_links(LINKS, {d: r.text for d, r in reports.items()})
    plan = build_plan(spec, infos, links, seed)
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(plan.model_dump_json(indent=1) + "\n", encoding="utf-8")
    kinds = {k: sum(b.kind == k for b in plan.batches) for k in ("single", "pair", "out_of_corpus")}
    used = {d for b in plan.batches for d in b.doc_ids}
    typer.echo(f"{plan.n_slots} slots in {len(plan.batches)} batches {kinds}")
    typer.echo(f"{len(used)} reports used")
    typer.echo(f"Plan written to {PLAN}")


@app.command("generate")
def generate_command(
    max_cost: Annotated[
        float, typer.Option(help="Abort before spending more than this (USD)")
    ] = 8.0,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt")] = False,
) -> None:
    """Generate candidates for every slot in the plan (hosted model; asks before spending)."""
    from rag_eval.dataset.llm import CachedAnthropicCaller  # noqa: PLC0415 - optional at import

    spec = load_dimensions(DIMENSIONS)
    reports = load_report_texts(CORPUS_TEXT)
    titles = _titles()
    plan = Plan.model_validate_json(PLAN.read_text(encoding="utf-8"))
    caller = CachedAnthropicCaller(CACHE)

    typer.echo(f"Counting input tokens for {len(plan.batches)} batches (free)...")
    per_batch: dict[str, float] = {}
    total_in = total_out = 0
    for batch in plan.batches:
        request = build_request(batch, batch.slots, spec=spec, reports=reports, titles=titles)
        tokens_in = caller.count_input_tokens(request)
        tokens_out = EST_OUTPUT_TOKENS_PER_SLOT * len(batch.slots)
        total_in += tokens_in
        total_out += tokens_out
        per_batch[batch.batch_id] = (tokens_in * PRICE_INPUT + tokens_out * PRICE_OUTPUT) / 1e6
    base = sum(per_batch.values())
    estimate = base * (1 + EST_RETRY_SHARE)
    typer.echo(
        f"Model {MODEL} (effort {EFFORT}): {len(plan.batches)} calls, ~{total_in:,} input and "
        f"~{total_out:,} output tokens.\nEstimated cost ${base:.2f}, up to ${estimate:.2f} with "
        f"retries (cached calls are free). Hard cap: ${max_cost:.2f}."
    )
    if not yes and not typer.confirm("Proceed?"):
        raise typer.Exit(code=1)

    def estimate_for(batch: Batch) -> float:
        return per_batch[batch.batch_id]

    outcome = run_generation(
        plan, spec, reports, titles, caller, max_cost_usd=max_cost, estimate_per_batch=estimate_for
    )
    write_candidates(CANDIDATES, outcome.candidates)
    served = sorted({c.generator.model_served for c in outcome.candidates})
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_requested": MODEL,
        "models_served": served,
        "effort": EFFORT,
        "system_prompt_sha256": hashlib.sha256(system_prompt(spec).encode()).hexdigest(),
        "dimensions_sha256": _sha256(DIMENSIONS),
        "plan_sha256": _sha256(PLAN),
        "corpus_text_sha256": {d: r.sha256 for d, r in sorted(reports.items())},
        "calls": outcome.calls,
        "cached_calls": outcome.cached_calls,
        "cost_usd_this_run": round(outcome.cost_usd, 4),
        "n_candidates": len(outcome.candidates),
        "failed_slots": outcome.failures,
    }
    (CANDIDATES_DIR / "generation.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    typer.echo(
        f"{len(outcome.candidates)} candidates, {len(outcome.failures)} slots failed; "
        f"{outcome.calls} calls ({outcome.cached_calls} cached), ${outcome.cost_usd:.2f} spent."
    )


def _load_plan() -> Plan:
    return Plan.model_validate_json(PLAN.read_text(encoding="utf-8"))


@app.command("brief")
def brief_command() -> None:
    """Write one brief per batch for generating candidates in a Claude Code session."""
    paths = write_briefs(
        _load_plan(),
        spec=load_dimensions(DIMENSIONS),
        reports=load_report_texts(CORPUS_TEXT),
        titles=_titles(),
        briefs_dir=BRIEFS,
        outputs_dir=SESSION_OUTPUTS,
    )
    typer.echo(f"{len(paths)} briefs in {BRIEFS}; outputs go to {SESSION_OUTPUTS}/<batch_id>.json")


@app.command("check-batch")
def check_batch_command(batch_id: str) -> None:
    """Validate one session batch file (verbatim quotes, slot rules). Prints OK or problems."""
    batches = {b.batch_id: b for b in _load_plan().batches}
    if batch_id not in batches:
        raise typer.BadParameter(f"unknown batch {batch_id!r}")
    problems = check_batch(
        batches[batch_id], SESSION_OUTPUTS / f"{batch_id}.json", load_report_texts(CORPUS_TEXT)
    )
    for problem in problems:
        typer.echo(problem)
    if problems:
        raise typer.Exit(code=1)
    typer.echo("OK")


@app.command("collect")
def collect_command() -> None:
    """Build candidates.jsonl from the session batch files (invalid slots are recorded)."""
    spec = load_dimensions(DIMENSIONS)
    reports = load_report_texts(CORPUS_TEXT)
    outcome = collect(
        _load_plan(),
        spec=spec,
        reports=reports,
        titles=_titles(),
        outputs_dir=SESSION_OUTPUTS,
        model=SESSION_MODEL,
    )
    write_candidates(CANDIDATES, outcome.candidates)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": SESSION_MODEL,
        "method": "session briefs (rag-eval dataset brief / check-batch / collect)",
        "system_prompt_sha256": hashlib.sha256(system_prompt(spec).encode()).hexdigest(),
        "dimensions_sha256": _sha256(DIMENSIONS),
        "plan_sha256": _sha256(PLAN),
        "corpus_text_sha256": {d: r.sha256 for d, r in sorted(reports.items())},
        "n_candidates": len(outcome.candidates),
        "failed_slots": outcome.failures,
    }
    (CANDIDATES_DIR / "generation.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    typer.echo(f"{len(outcome.candidates)} candidates, {len(outcome.failures)} slots failed.")


@app.command("check")
def check_command() -> None:
    """Run automated checks and write the coverage report."""
    spec = load_dimensions(DIMENSIONS)
    reports = load_report_texts(CORPUS_TEXT)
    checked = run_checks(load_candidates(CANDIDATES), reports)
    write_candidates(CANDIDATES, checked)
    lengths = sorted(len(r.text) for r in reports.values())
    typical = lengths[len(lengths) // 2]
    long_ids = {d for d, r in reports.items() if len(r.text) > 4 * typical}
    report = coverage_report(checked, spec, long_ids, "Candidate coverage (before review)")
    (CANDIDATES_DIR / "coverage.md").write_text(report, encoding="utf-8")
    flagged = sum(c.checks.too_easy for c in checked)
    dupes = sum(c.checks.duplicate_of is not None for c in checked)
    typer.echo(f"{len(checked)} candidates: {dupes} near-duplicates, {flagged} flagged too easy.")


@app.command("review-export")
def review_export_command() -> None:
    """Write the local review page (data/review/review.html)."""
    sources = {s.doc_id: s for s in load_sources(SOURCES)}
    n = export_review_page(
        load_candidates(CANDIDATES), load_report_texts(CORPUS_TEXT), sources, REVIEW_PAGE
    )
    typer.echo(f"{n} candidates to review: open {REVIEW_PAGE} in a browser.")


@app.command("review-import")
def review_import_command(
    decisions: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate the reviewer's decisions, store them as labels and write data/approved.jsonl."""
    parsed, approved = import_decisions(
        decisions, load_candidates(CANDIDATES), load_report_texts(CORPUS_TEXT)
    )
    stored = store_labels(decisions, parsed, LABELS)
    APPROVED.write_text("".join(c.model_dump_json() + "\n" for c in approved), encoding="utf-8")
    rejected = sum(d.decision == "reject" for d in parsed.decisions.values())
    typer.echo(
        f"{len(approved)} approved, {rejected} rejected. Labels: {stored}; items: {APPROVED}"
    )


@app.command("split")
def split_command(seed: int = DEFAULT_SEED) -> None:
    """Stratified dev/test split of the approved items (data/dev.jsonl, data/test.jsonl)."""
    spec = load_dimensions(DIMENSIONS)
    dev, test = stratified_split(load_cases(APPROVED), spec.sizes.split["dev"], seed)
    path = write_split(dev, test, APPROVED.parent, seed, _sha256(APPROVED))
    typer.echo(f"dev {len(dev)}, test {len(test)} -> {path}")
