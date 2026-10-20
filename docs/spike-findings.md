# Phase 0 spike findings

**Status:** draft for owner review. Everything below was observed against a live instance unless marked *source-read only*.
**Date:** 2026-10-21
**Raw evidence:** `spike/evidence/*.json` (API responses as returned). The spike's compose file, fixtures and probe script are in `spike/`.

## Environment

| Component | Value |
|---|---|
| AnythingLLM | `mintplexlabs/anythingllm:1.18.0@sha256:5461782e7db41318a516925949a8f8366e258dd576411fca7961fe77dff054cc` (released 2026-10-13) |
| Ollama | 0.40.6, native on host (macOS, Apple Silicon, 64 GB) |
| LLM (spike only) | `qwen3:8b` — digest: `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` (capabilities: completion, tools, thinking) |
| Embedder (spike only) | `nomic-embed-text:latest`, digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` |
| Vector DB | LanceDB (AnythingLLM built-in) |
| Docker | 29.8.2 |

The spike model choices are placeholders for testing the API. Final model and embedder choices are pinned in Phase 1.

Fixtures are two short **synthetic, fictional** documents (`spike/fixtures/`), so any correct answer must come from retrieval rather than from what the model memorised.

## Getting an API key without the UI

A fresh single-user instance with no password has no auth on its internal endpoints, so a developer key can be minted headlessly:

```
POST /api/system/generate-api-key        {"name":"spike"}
→ {"apiKey":{"secret":"…"}}               # use as Authorization: Bearer …
GET  /api/v1/auth                         → {"authenticated":true}
```

This only works while the instance has no password or multi-user mode. That is fine for a local, throwaway eval instance. Never expose port 3001 beyond localhost.

The full OpenAPI spec for 1.18.0 is saved at `spike/openapi-1.18.0.json` (copied from `/app/server/swagger/openapi.json` in the container).

## 1. Workspace creation, upload and embedding via API — ✅ confirmed

```
POST /api/v1/workspace/new
{"name":"spike-ws","similarityThreshold":0.25,"openAiTemp":0,"openAiHistory":20,"chatMode":"query","topN":4}
→ {"workspace":{"id":1,"slug":"spike-ws",…}}

POST /api/v1/document/upload      (multipart: file=@doc.txt, addToWorkspaces=spike-ws)
→ {"success":true,"documents":[{"location":"custom-documents/zorvath-valve.txt-<uuid>.json","title":"zorvath-valve.txt",…}]}
```

- `addToWorkspaces` parses and embeds synchronously: by the time the call returns, the vectors exist (`GET /api/v1/system/vector-count`).
- **The document `location` carries a random UUID per upload.** Re-uploading the same file creates a new document. So AnythingLLM's document IDs are **not stable across provisioning runs**. Our dataset must identify documents by our own ID (filename/title, or a `title` set via `POST /v1/document/raw-text` metadata) and map it to AnythingLLM locations at provisioning time.
- Adding and removing documents in a workspace: `POST /api/v1/workspace/{slug}/update-embeddings {"adds":[location], "deletes":[location]}`. Verified: after a delete, `vector-search` returns 0 results for that document (`evidence/08`). This is the mechanism for **Mutation 5 (key document removed)**.
- **Retrieval without the LLM:** `POST /api/v1/workspace/{slug}/vector-search {"query":…, "topN":…, "scoreThreshold":…}` returns chunk text, metadata and score (`evidence/03`). This lets retrieval metrics be computed deterministically and cheaply, separate from generation.

## 2. Do chat responses include full passage text, document IDs and scores?

**Vector search** (`evidence/03`), observed:

```json
{"id":"6dfe293f-…","text":"<document_metadata>\nsourceDocument: zorvath-valve.txt\npublished: 10/20/2026, 6:51:29 PM\n</document_metadata>\n\nZorvath Model 7 …",
 "metadata":{"title":"zorvath-valve.txt","url":"file:///app/collector/hotdir/zorvath-valve.txt","published":"…","wordCount":97,"tokenCount":141,…},
 "distance":0.429,"score":0.571}
```

- `text` is the **full chunk**, but AnythingLLM **prepends a `<document_metadata>` header** to every chunk at embed time. Evidence matching must strip this header first, or it will inflate lexical overlap.
- `score` = `1 − cosine distance`, floored at 0 (source: `lance/index.js#distanceToSimilarity`).
- The document is identified by `metadata.title` (= our filename). There is no stable document ID (see §1).

**Chat `sources`**: ✅ confirmed (`evidence/10`). The response from `POST /api/v1/workspace/{slug}/thread/{threadSlug}/chat {"message":…,"mode":"query"}` includes:

```json
{"id":"…","type":"textResponse","textResponse":"<think>…</think>The … every **1,850 operating hours** …",
 "sources":[{"id":"<chunk uuid>","title":"zorvath-valve.txt","text":"<document_metadata>…</document_metadata>\n\nZorvath Model 7 …","score":0.7405,
             "_distance":0.259,"url":"…","published":"…","wordCount":97,"token_count_estimate":141,"docSource":"…","docAuthor":"…","chunkSource":"","description":"…"}, …],
 "metrics":{"prompt_tokens":466,"completion_tokens":237,"total_tokens":703,"outputTps":78.95,"duration":3.00,"model":"qwen3:8b","provider":"OllamaAILLM",…}}
```

- `sources[].text` is the **full chunk** (789 chars here, the whole chunk including the metadata header), not a snippet. `score` and the chunk `id` are present, and `title` identifies the document.
- `metrics` gives **token usage, the model that actually answered, and duration**. These go straight into the trace record.
- With a fresh thread and no pinned documents, `sources` equals exactly what the LLM saw (see §3 for why this only holds without history).

*Source-read:* chat `sources` go through `curateSources`, which keeps `text` and `score`. Pinned documents and chat attachments are truncated to 1,000 characters in `sources`. We will not use either.

## 3. Isolating cases

✅ **One new thread per case works.** Probe (`evidence/11*`): plant a codeword in one chat, then ask for it in another.

| Second call made via | Leaked? |
|---|---|
| workspace `/chat`, no `sessionId` (same default history) | **yes**: answered `MAGENTA-4417` |
| workspace `/chat` with a new `sessionId` | no: `NONE` |
| new thread (`POST /v1/workspace/{slug}/thread/new`, then `/thread/{threadSlug}/chat`) | no: `NONE` |

**Decision proposed:** one new thread per case (`thread/new` → `thread/{slug}/chat`), with the thread slug recorded in the trace. Threads are visible in the UI, which helps human review, and can be deleted afterwards (`DELETE /v1/workspace/{slug}/thread/{threadSlug}`). A unique `sessionId` would also isolate, but threads are easier to inspect.

*Source-read facts that drive the design* (`utils/chats/apiChatHandler.js`, `utils/chats/index.js`):

- History is looked up by `(workspace, user, thread, api_session_id)`. A workspace-level `/chat` without `sessionId` shares **one default history** across all such calls.
- **`openAiHistory: 0` does not disable history.** The handler does `messageLimit = workspace.openAiHistory || 20`, so 0 falls back to 20. Isolation cannot rely on this setting.
- **History also changes retrieval.** When the vector search returns fewer than `topN` chunks, `fillSourceWindow` backfills context with chunks cited in earlier turns of the same history. These backfilled chunks reach the LLM but are **not** listed in that response's `sources`. Any shared history would make traces lie about what the model saw. Per-case isolation is therefore needed for trace correctness, not only for independence.
- **Memories:** 1.18 has a "memories" feature that injects stored memories into the system prompt across chats. `MemoryEnabled` is `false` by default (`GET /api/v1/system`), but `MemoryAutoExtraction` defaults to `true`. Provisioning must assert `MemoryEnabled=false` and fail otherwise.

## 4. Which settings are controllable via API

| Setting | Scope | How | Verified |
|---|---|---|---|
| LLM provider/model | per workspace | `POST /v1/workspace/{slug}/update {"chatProvider":"ollama","chatModel":"…"}` | ✅ read back (`evidence/15b`); workspace override is actually used: setting `chatModel` to a non-existent model makes chat abort with `model 'does-not-exist:1b' not found` (`evidence/16`), and `metrics.model` reports the model used |
| LLM default model | instance | `POST /v1/system/update-env {"OllamaLLMModelPref":"…"}` | source-read |
| Embedder | **instance-wide only** | `POST /v1/system/update-env {"EmbeddingEngine","EmbeddingModelPref","EmbeddingBasePath"}` | ✅ (`evidence/15c`) |
| Chunk size / overlap | **instance-wide only, not in `/v1`** | `POST /api/admin/system-preferences {"text_splitter_chunk_size":…, "text_splitter_chunk_overlap":…}` (internal UI endpoint) | ✅ chunks change (`evidence/05`) |
| top-N | per workspace | `topN` | ✅ |
| Similarity threshold | per workspace | `similarityThreshold` | ✅ |
| System prompt | per workspace | `openAiPrompt` | ✅ |
| Temperature | per workspace | `openAiTemp` | ✅ |
| Chat vs query mode | per workspace, **and per request** | `chatMode`, or `"mode"` in the chat body | ✅ |
| Query-mode refusal text | per workspace | `queryRefusalResponse` | ✅ (`evidence/07`) |
| Reranking | per workspace | `vectorSearchMode: "default"\|"rerank"` | read back only |
| Seed | ❌ not exposed | AnythingLLM sends only `temperature` and `num_ctx` to Ollama | source-read |
| Thinking / reasoning effort | ❌ not via the developer chat API | `reasoningEffort` is not passed by `apiChatHandler` | source-read; see §6. qwen3 thinks by default and `<think>…</think>` ends up inside `textResponse` |

Consequences:

- **Chunking and the embedder are instance-level and only apply at embed time.** Workspaces that are already embedded keep their old chunks (observed: `spike-ws` still has 1 chunk per document after the splitter was set to 200). A config that changes chunking or the embedder must re-upload and re-embed. Changing the splitter purges AnythingLLM's whole vector cache (`systemSettings.js`), so re-embedding really does re-chunk (`evidence/09`). Provisioning must re-ingest whenever the chunking/embedder hash differs from what the workspace was built with.
- Chunk size is measured in **characters**, not tokens (200 → chunks of ~200–300 chars incl. header; `evidence/05`).
- **The default system prompt contains `{datetime}`**, which is expanded to the current date and time on every call. That breaks determinism and judge-cache keys. Baseline must set an explicit prompt with no time variables.
- **Default `chatMode` is `automatic`.** In automatic mode, any model that supports native tool calling is routed to the **agent** path (`EphemeralAgentHandler`), a different pipeline. Every request must send `mode` explicitly (`chat` or `query`).

## 5. `chat` vs `query` with no relevant documents

Observed without LLM (`evidence/07`): in `query` mode, if no chunk clears `similarityThreshold`, AnythingLLM returns `queryRefusalResponse` (or a fixed default) **without calling the LLM** (56 ms, `sources: []`). This abstention is deterministic and can be checked by exact string match, with no judge.

Observed with the LLM (fresh thread per call):

| Situation | `chat` mode | `query` mode |
|---|---|---|
| Nothing clears the threshold (`spike-strict`, threshold 0.9; `evidence/12`) | LLM answers anyway. It **made up a conversation history** ("the assistant responded … 1,500 psi to 3,000 psi") and answered **"3,000 psi"**. The true value in the document is 412 kPa. Fresh thread, `sources: []`, workspace history 0. | `queryRefusalResponse` returned, no LLM call |
| Context retrieved but the answer isn't in it (weight of the valve; `evidence/13`) | "not specified in the provided context" | same as chat; the LLM decided to abstain |

Implications:

- `chat` mode turns retrieval misses into confident hallucinations. That is useful as a **weak mutation**, but `query` is the right baseline mode for a document-QA system. To be recorded in the chat-vs-query ADR.
- In `query` mode there are **two different abstention paths**: (a) deterministic refusal when retrieval returns nothing above the threshold, detectable by exact match on `queryRefusalResponse`; and (b) the LLM choosing to abstain when given irrelevant context, which needs a judge. Setting a distinctive `queryRefusalResponse` (e.g. `NO_RELEVANT_CONTEXT`) lets traces tell these apart, and lets the "failure to abstain" metric credit path (a) with no judge call.

## 6. Determinism at temperature 0

✅ at temperature 0, **5/5 identical** `textResponse` (including the thinking block) and identical `sources` (titles + scores) across 5 fresh threads with the same question (`evidence/14-*`; SHA-256 prefix `b774f0f254c6710e` for all 5).

Caveats, to be measured rather than assumed in later phases:
- This is a sample of 5 calls on one question, sequential, on one machine. Concurrent requests (Ollama batching) and different hardware may change the output. The runner should run cases **sequentially** by default, and the stats module's repeated-run consistency metric (Phase 6) should report determinism on the real dataset.
- No seed can be set through AnythingLLM (§4), so determinism relies on greedy decoding at temperature 0.

**Thinking models.** `qwen3:8b` thinks by default. AnythingLLM wraps Ollama's `thinking` field as `<think>…</think>` at the start of `textResponse`, and these tokens count in `completion_tokens` and latency. The qwen3 soft switch `/no_think` in the system prompt **did not** disable it (`evidence/17`), and the developer chat API does not forward a reasoning setting. So either:
1. strip `<think>…</think>` in the client: store it as a separate `reasoning` field in the trace, and judge only the final answer; or
2. choose a non-thinking model for the system under test.

Option 1 is needed anyway as a defensive measure. Which model to use is an owner decision for Phase 1.

## Other observations

- With `DISABLE_TELEMETRY=true`, the server still fetches a remote model map and pricing data at startup. This is harmless but means the container is not offline by default.
- Container timestamps are UTC. The `published` date in chunk headers is the upload time, so chunk text differs between provisioning runs. Hash evidence *after* stripping the header.

## Proposed plan changes

None of these block the plan. Each one refines it:

1. **Document identity.** AnythingLLM document locations change on every upload. The dataset's evidence labels use **our own document ID** (corpus filename, preserved as the AnythingLLM `title`), and provisioning keeps a `{doc_id → location}` map per run in the manifest. *(Fits the plan's "document/passage level, never chunk IDs" rule.)*
2. **Evidence matching strips the `<document_metadata>` header** from chunk text before matching passages or computing lexical overlap.
3. **Chunking and embedder are instance-wide.** They cannot vary per workspace at the same moment. Provisioning applies them, then re-uploads and re-embeds the corpus into a workspace named after the config hash. Configs that differ in chunking or embedder must be provisioned **sequentially**, not side by side. The plan's `baseline.yaml` keeps these fields, but the provisioner treats them as instance-level.
4. **Chunk size/overlap need an internal (non-`/v1`) endpoint** (`/api/admin/system-preferences`), which only works on an instance without auth. This is acceptable for a local eval instance. The client will isolate this one call and document it. If the owner prefers not to depend on it, the alternative is to fix chunking in env vars and drop chunking from the mutation set (no current mutation needs it).
5. **Always send `mode` explicitly** (`automatic` routes to the agent pipeline), **always set `openAiPrompt` explicitly** (the default contains `{datetime}`), set a distinctive `queryRefusalResponse`, and **assert `MemoryEnabled=false`** at provisioning.
6. **Isolation = new thread per case**, run sequentially. `openAiHistory` cannot be set to 0.
7. **Trace record additions:** `reasoning` (stripped `<think>` content), `response_type` (`textResponse`/`abort`), `refusal_path` (`deterministic` when the answer equals `queryRefusalResponse` with zero sources), and `metrics.model` as the model actually used.
8. **Retrieval metrics can use `/vector-search`**, but should be computed from the chat response's own `sources`. In a fresh thread these are exactly the chunks the LLM saw, so retrieval and generation are scored on the same retrieval. `/vector-search` remains useful for retrieval-only checks with no LLM cost.
9. **Owner decisions needed for Phase 1:** the system-under-test LLM (thinking vs non-thinking model; this also affects the judge family choice) and the embedder. `qwen3:8b` / `nomic-embed-text` were used for the spike only.

## Reproducing the spike

```bash
ollama pull qwen3:8b && ollama pull nomic-embed-text      # Ollama running natively on the host
docker compose -f spike/docker-compose.spike.yml up -d
curl -s -X POST localhost:3001/api/system/generate-api-key -H 'Content-Type: application/json' \
  -d '{"name":"spike"}' | jq -r .apiKey.secret > spike/.apikey
# The non-LLM probes (workspace/upload/vector-search/chunking/removal) were run interactively; their
# exact requests are shown above and their responses are in spike/evidence/01–09, 15.
spike/llm_probes.sh    # LLM probes → spike/evidence/10–14 (expects workspaces spike-ws and spike-strict)
```

`spike/` is throwaway. It is superseded by the typed client, provisioning and `docker-compose.yml` in Phase 1.
