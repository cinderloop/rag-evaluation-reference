# 0005: Run the system under test with thinking turned off

- **Status:** Accepted
- **Date:** 2026-10-23

## Context

`qwen3:8b` is a thinking model: before each answer it writes a hidden reasoning trace.
On the first closed-book run, the trace was about 2,200 characters against a 370-character
answer, so roughly 80% of the generation time went into thinking (about 8 s per question,
up to 15 s). Every evaluation run, every repeat and every mutation pays this cost, and the
owner's machine ran hot through a full run.

This repository is a guide to evaluation methods (error analysis, validated judges,
mutation testing, honest statistics). Those methods work the same on a model that answers
directly, so thinking adds run time and heat without adding to what the guide teaches.

## Decision

The baseline uses `qwen3-nothink:8b`: the same `qwen3:8b` weights and parameters with a
chat template that always opens the answer with an empty `<think></think>` block. This is
exactly what qwen3's own template does when a client asks for thinking to be off. The
variant is built from `models/qwen3-nothink.Modelfile` and pinned by digest in
`configs/baseline.yaml`; rebuilding it from the Modelfile gives the same digest.

Measured on the same three questions through AnythingLLM: about 1.3 s per answer instead
of about 10 s, and about 10× fewer generated tokens.

## Alternatives considered

- **`/no_think` in the system prompt** (qwen3's soft switch): tried; the model kept
  thinking.
- **AnythingLLM's reasoning setting:** AnythingLLM 1.18 can send Ollama's `think: false`,
  but only for chats started from its own UI. The developer API that the runner uses has
  no way to request it.
- **`/no_think` appended to every question:** would work, but the suffix would also be
  embedded for retrieval and change what is searched for.
- **A non-thinking model of another family** (for example Llama 3.1 8B): a further
  multi-gigabyte download, and it would undo the corpus and memorisation reasoning in
  ADR 0003, which is about Qwen3.

## Consequences

- Setup needs one extra step, `ollama create qwen3-nothink:8b -f models/qwen3-nothink.Modelfile`
  (in the README). Runs fail fast if the variant is missing, because the digest is checked.
- Traces have empty `reasoning`. The code that splits off `<think>` blocks stays, so a
  thinking configuration remains possible as a mutation (with vs without thinking is a
  meaningful comparison for the guide).
- Answer quality may be somewhat lower than with thinking, mainly on inference and
  multi-document questions. Scores describe this configuration, not qwen3 at its best.
- The workspace name includes the config hash, so the baseline gets a new workspace and
  re-embeds the corpus once (a few minutes). Chunking and the embedder are unchanged, so
  the chunks used to build the dataset are the same.
