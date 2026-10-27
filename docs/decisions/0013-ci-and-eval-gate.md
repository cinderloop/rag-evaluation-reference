# 0013: LLM-free pull-request CI, and a manual eval gate

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

Two kinds of check protect this repository.

**Code checks.** They show that the code is correct and stays that way: lint, format,
types and unit tests.

**Quality checks.** They show that the RAG system is still good: the evaluation itself.
Running the evaluation needs:

- AnythingLLM, with the corpus embedded;
- a local model server holding about 5 GB of models;
- hosted judge calls.

A run takes from tens of minutes to hours, and costs judge usage.

## Decision

**Pull-request CI stays fast, free and deterministic** (`.github/workflows/ci.yml`). It
runs lint, format, strict types and the unit tests on a GitHub-hosted runner. It makes no
LLM or AnythingLLM call. The unit tests cover every piece of logic that decides a result:

- retrieval matching;
- statistics, against hand-computed values;
- judge reply parsing;
- gate thresholds;
- config parsing.

The reasons:

- **Fast feedback.** A pull request gets a result in about a minute.
- **No secrets on pull requests.** Pull requests, including from forks, never need
  model credentials.
- **Deterministic.** A failed check means the code changed, not that a model replied
  differently.
- **Free.** The CI costs nothing however often it runs.

**The eval gate runs manually** (`.github/workflows/eval-gate.yml`, `workflow_dispatch`).
It runs `rag-eval gate run`: the baseline config on the chosen dataset, the judges, then
the thresholds of ADR 0012. A failed gate fails the job. A summary of each gated metric
goes to the job page.

**When to run it.** Before merging a change to the system under test (its config,
prompts, models or corpus), or after changing the judges or the dataset.

**Requirements.** The workflow file documents them:

- **A self-hosted runner** labelled `rag-eval`, with:
  - Docker;
  - Ollama holding the pinned models (including the thinking-off variant built from
    `models/`);
  - the Claude Code CLI.

  GitHub-hosted runners have no GPU and too little memory to serve the models at a
  usable speed.
- **Secrets:**
  - `ANYTHINGLLM_API_KEY`;
  - `CLAUDE_CODE_OAUTH_TOKEN`, a token from `claude setup-token`, so Claude Code can
    authenticate without an interactive login.

  A repository can use the Claude API judge template (`configs/judges/anthropic-api.yaml`)
  with `ANTHROPIC_API_KEY` instead.
- **Inputs** reach the shell only as environment variables, never interpolated into a
  command.
- **One run at a time:** a concurrency group stops two gate runs from sharing one
  AnythingLLM instance.

## Consequences

- **Gate results are not automatic.** A regression is only caught when someone runs the
  gate. That is the price of keeping pull requests free and secret-free.
- **The workflow has not run on a real runner yet.** The gate command it calls has been
  run locally against every Phase 7 run (ADR 0012); the workflow wiring around it is
  untested until a `rag-eval` runner is registered.
- **Long runs.** The gate takes about as long as one baseline run plus judging. The job
  times out after three hours. Judge replies are cached on the runner, so a re-run
  after a timeout resumes rather than starting over.
