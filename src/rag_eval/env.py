"""Environment variables. Every variable is documented in .env.example."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Env:
    anythingllm_base_url: str
    anythingllm_api_key: str
    anythingllm_container: str
    ollama_base_url: str

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] = os.environ) -> Env:
        api_key = environ.get("ANYTHINGLLM_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANYTHINGLLM_API_KEY is not set. See .env.example for how to create one."
            )
        return cls(
            anythingllm_base_url=environ.get("ANYTHINGLLM_BASE_URL", "http://localhost:3001"),
            anythingllm_api_key=api_key,
            anythingllm_container=environ.get("ANYTHINGLLM_CONTAINER", "rag-eval-anythingllm"),
            ollama_base_url=environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
