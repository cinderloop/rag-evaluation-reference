"""The hosted-model caller for generation, with an on-disk cache keyed by request hash."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import anthropic
from anthropic.types.beta import BetaMessage

from rag_eval.dataset.generate import CallResult, Usage, request_key


class GenerationRefusedError(RuntimeError):
    pass


class CachedAnthropicCaller:
    """Calls the Messages API once per distinct request; replays cached responses after."""

    def __init__(self, cache_dir: Path, client: anthropic.Anthropic | None = None) -> None:
        self._cache_dir = cache_dir
        self._client = client or anthropic.Anthropic()
        cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, request: dict[str, Any]) -> CallResult:
        key = request_key(request)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return CallResult(
                text=cached["text"],
                model_served=cached["model_served"],
                response_id=cached["response_id"],
                usage=Usage(**cached["usage"]),
                cached=True,
            )
        # Server-side fallback re-runs a declined request on Anthropic's recommended model
        # for that refusal category; the model that actually answered is recorded.
        # The request is built as a plain dict (it is also the cache key), so the SDK's
        # overloads cannot be resolved statically; the non-streaming call returns a BetaMessage.
        response = cast(
            BetaMessage,
            self._client.beta.messages.create(  # pyright: ignore[reportCallIssue, reportUnknownMemberType]
                **request,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ),
        )
        if response.stop_reason == "refusal":
            raise GenerationRefusedError(f"generation refused ({response.stop_details})")
        if response.stop_reason == "max_tokens":
            raise GenerationRefusedError("generation hit max_tokens; raise MAX_TOKENS")
        text = next((b.text for b in response.content if b.type == "text"), "")
        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
            cache_read_input_tokens=u.cache_read_input_tokens or 0,
        )
        record = {
            "text": text,
            "model_served": response.model,
            "response_id": response.id,
            "usage": usage.__dict__,
        }
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return CallResult(
            text=text,
            model_served=response.model,
            response_id=response.id,
            usage=usage,
            cached=False,
        )

    def count_input_tokens(self, request: dict[str, Any]) -> int:
        counted = self._client.messages.count_tokens(
            model=request["model"], system=request["system"], messages=request["messages"]
        )
        return counted.input_tokens
