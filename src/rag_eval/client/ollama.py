"""Reads installed model digests from the host's Ollama, to verify pinned models."""

from __future__ import annotations

from typing import Any, cast

import httpx


def installed_models(base_url: str) -> dict[str, str]:
    """Map of installed model tag -> digest."""
    response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=30.0)
    response.raise_for_status()
    body = cast(dict[str, Any], response.json())
    models = cast(list[dict[str, Any]], body.get("models", []))
    return {str(m["name"]): str(m["digest"]) for m in models}
