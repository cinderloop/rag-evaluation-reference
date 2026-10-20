# 0002: One workspace per config, provisioned idempotently from code

- **Status:** Accepted
- **Date:** 2026-10-21

## Context

Every number we report must be reproducible from a config file. The spike (`docs/spike-findings.md`) showed how AnythingLLM 1.18.0 behaves:

- Workspace settings (LLM, temperature, top-N, threshold, prompt, mode, refusal text) are per workspace and settable through `/v1`.
- Chunking, the embedder and the LLM context window are **instance-wide**, and chunking and the embedder only take effect when documents are embedded.
- Changing the embedder makes AnythingLLM **delete every document from every workspace** (`handleVectorStoreReset`). Re-sending the same value is a no-op.
- Document locations change on every upload. The uploaded filename survives as the document `title`, which is also returned on every retrieved chunk.

## Decision

1. **The config file is the source of truth.** `docker-compose.yml` only bootstraps the instance. `rag-eval provision` sets every relevant setting and reads it back, failing if anything did not apply.
2. **One workspace per config,** named `<config name>-<hash>`, where the hash covers the full resolved config and the content of every corpus document. A workspace with that name was therefore built with exactly that chunking, embedder and corpus. Changing any value, or any document, produces a new workspace rather than mutating an old one.
3. **Idempotent sync.** Provisioning uploads missing documents and removes unexpected or duplicate ones, and does nothing when the workspace is already correct. After an embedder change has emptied the workspaces, the next provisioning of each config re-ingests it.
4. **Instance settings are sent only when they differ,** and an embedder change is logged as a warning because of the global wipe.
5. **`run` always provisions first,** then runs every case (and repeat) sequentially in a new thread.
6. **Pins are verified, not just recorded:** Ollama model digests must match the config, and the running container's image digest is checked with `docker inspect`. The manifest records whether the image was verified.
7. **Document ID = corpus filename,** so the corpus directory must be flat with unique names. A knowledge-edit mutation uploads its edited copy under the original filename.

## Consequences

- Configs that differ in chunking or embedder cannot run concurrently. The runner is sequential anyway, because temperature-0 determinism was only observed for sequential calls.
- Each config costs one embedding pass over the corpus. This is cheap with a local embedder and a corpus of this size.
- Old workspaces accumulate on the instance. They are harmless, and they can be deleted in the AnythingLLM UI.
- Chunking is set through an internal (non-`/v1`) endpoint that only works without a password. The instance is bound to `127.0.0.1` for that reason.
