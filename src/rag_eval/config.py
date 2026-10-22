"""Run configuration as code.

A config is one YAML file. A mutation is a YAML file that ``extends`` another config
and overrides only what it changes; mappings are deep-merged, everything else is
replaced. ``extends`` is resolved relative to the file that declares it. All other
paths (``corpus.dir``, ``corpus.replace`` values) are relative to the repository
root, which is where every command is run from.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# AnythingLLM expands {variables} in system prompts (e.g. {datetime}); any expansion
# makes prompts vary between calls and breaks determinism and judge-cache keys.
_PROMPT_VARIABLE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.]*\}")
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OllamaModel(_Frozen):
    name: str = Field(description="Ollama tag, e.g. qwen3:8b")
    digest: str = Field(pattern=r"^[0-9a-f]{64}$", description="Verified before every run")


class LLMConfig(_Frozen):
    provider: Literal["ollama"] = "ollama"
    model: OllamaModel
    temperature: float = Field(ge=0)
    context_window: int = Field(gt=0, description="Sent to Ollama as num_ctx (instance-wide)")


class EmbedderConfig(_Frozen):
    engine: Literal["ollama"] = "ollama"
    model: OllamaModel
    max_chunk_length: int = Field(gt=0)


class ChunkingConfig(_Frozen):
    size: int = Field(gt=0, description="Characters, not tokens")
    overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError("chunking.overlap must be smaller than chunking.size")
        return self


class RetrievalConfig(_Frozen):
    top_n: int = Field(gt=0)
    similarity_threshold: float = Field(ge=0, le=1)
    search_mode: Literal["default", "rerank"] = "default"


class GenerationConfig(_Frozen):
    mode: Literal["chat", "query"]
    system_prompt: str = Field(min_length=1)
    refusal_response: str = Field(min_length=1)

    @model_validator(mode="after")
    def _no_prompt_variables(self) -> Self:
        found = _PROMPT_VARIABLE.findall(self.system_prompt)
        if found:
            raise ValueError(f"system_prompt must not contain AnythingLLM variables: {found}")
        return self


class CorpusConfig(_Frozen):
    dir: Path
    closed_book: bool = Field(
        default=False,
        description="Provision the workspace with no documents (closed-book baseline)",
    )
    exclude: tuple[str, ...] = Field(
        default=(), description="Document IDs left out of the workspace"
    )
    replace: dict[str, Path] = Field(
        default_factory=dict[str, Path],
        description="Document ID -> path of an edited copy, uploaded under the same ID",
    )

    @model_validator(mode="after")
    def _closed_book_has_no_edits(self) -> Self:
        if self.closed_book and (self.exclude or self.replace):
            raise ValueError("corpus.closed_book cannot be combined with exclude or replace")
        return self


class RunConfig(_Frozen):
    name: str
    description: str = ""
    anythingllm_image: str = Field(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
    llm: LLMConfig
    embedder: EmbedderConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    corpus: CorpusConfig

    @model_validator(mode="after")
    def _valid_name(self) -> Self:
        if not _NAME.match(self.name):
            raise ValueError("name must be lowercase letters, digits and dashes (max 41 chars)")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``: mappings merge recursively, other values replace."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(cast(dict[str, Any], current), cast(dict[str, Any], value))
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ValueError(f"circular 'extends': {chain}")
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    raw = cast(dict[str, Any], loaded)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ValueError(f"{path}: 'extends' must be a path string")
    base = _load_raw(path.parent / parent, (*seen, path))
    return deep_merge(base, raw)


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(_load_raw(path))
