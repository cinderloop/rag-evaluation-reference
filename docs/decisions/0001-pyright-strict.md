# 0001: Type-check with pyright in strict mode

- **Status:** Accepted
- **Date:** 2026-10-21

## Context

The suite's correctness depends on data passing intact between stages: API responses, trace records, scores and statistics. We want one strict type checker running in CI. The two realistic options are pyright and mypy, both in strict mode.

## Decision

Use **pyright** (`typeCheckingMode = "strict"`, configured in `pyproject.toml`), installed as a pinned dev dependency through uv.

## Reasons

- Pydantic v2 models type-check in pyright through `dataclass_transform`, with no plugin. mypy needs the pydantic plugin to understand model constructors.
- Strict mode flags `Any` leaking in from untyped JSON (`reportUnknown*`). That is the main risk when parsing AnythingLLM responses, and it is why the client validates every response into a typed model.
- It is fast enough to run on every save and in CI with no cache setup.

## Consequences

- Values parsed from JSON or YAML must be narrowed explicitly (`isinstance` plus `cast`) before use. This is deliberate: it keeps unchecked data at the edges.
- Contributors who use mypy locally may see different diagnostics. CI (pyright) is authoritative.
