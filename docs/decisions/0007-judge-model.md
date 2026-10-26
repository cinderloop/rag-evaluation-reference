# 0007: Judge model and pluggable judge backends

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

The LLM judges need a model to run on. Two constraints apply. The judge must come from
a different model family than the system under test (`qwen3`), so that it does not
share that model's blind spots or prefer its phrasing. And the judge model must be
pinned, because a judge is only valid for the model it was validated on.

This repository is a guide, not a leaderboard. The aim is to show the method (binary
judges, validated against labels, with known error rates), not to reach the highest
possible judge accuracy. Cost matters more than the last few points of agreement.

Options considered:

| Option | Cost | Setup | Notes |
|---|---|---|---|
| Claude Haiku 5.5 via Claude Code headless (`claude -p`) | Counts toward an existing Claude subscription | Claude Code CLI, logged in | No API key; about 2–3 s per call |
| Claude via the API | Pay per call | `ANTHROPIC_API_KEY` | Same models, billed separately |
| OpenAI or xAI models via their APIs | Pay per call | API key | Different family; equally valid |
| A local model via Ollama (e.g. Gemma) | Free | Model download | Weaker grader; runs on the same machine as the system under test |

## Decision

**Claude Haiku 5.5 (`claude-haiku-5-5`), called through Claude Code's headless mode**
(`configs/judges/claude-code-haiku.yaml`). The reason is cost: it uses a Claude
subscription that is already paid for, so judging adds no API bill. Haiku is the
cheapest Claude model and fast. It is a different family from `qwen3`. Since accuracy
is not the goal, there is no fallback to a larger model; Phase 5 measures how well Haiku
agrees with the labels and reports that, whatever it is.

Each headless call runs with the judge prompt as the only system prompt, no tools, no
MCP servers, no settings, in an empty temporary directory and without saving a session,
so nothing but the judge prompt and the case reaches the model.

**Any model can be the judge.** The judge is chosen by a config file:

- `configs/judges/claude-code-haiku.yaml`: the default above.
- `configs/judges/anthropic-api.yaml`: the same model through the Claude API.

Other providers need only a config: the `openai_compatible` backend sends the judge
prompt to any Chat Completions endpoint (OpenAI, xAI and many others) given a
`base_url`, an `api_key_env` and a pinned `model`. Adding a different kind of provider
means writing one class with a `complete(system, user)` method
(`src/rag_eval/evaluators/judges/backends.py`).

## Consequences

- Judging needs no API key, but whoever re-runs the judges needs Claude Code installed
  and logged in, or switches to another template.
- Claude Code does not expose sampling settings such as temperature, so repeat calls on
  the same input may differ slightly. Replies are cached by (provider, model, full
  prompt), so a re-run replays earlier verdicts instead of re-asking.
- Usage counts toward the subscription's limits. A large run may pause until the limits
  reset; the cache means nothing already judged is lost.
- Claude Code reports a list-price-equivalent cost per call, which is recorded and used
  for the `--max-cost` guard even though a subscription is not billed per call.
- The manual eval-gate workflow (Phase 8) will need a Claude Code login token stored as
  a CI secret (`claude setup-token`), or one of the API templates.
- Changing the judge model or its prompt invalidates the judge validation (Phase 5);
  the summary of every scored run records the judge config, model and prompt hashes.
