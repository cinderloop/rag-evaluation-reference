"""Judge backends: where a judge prompt is sent. Any model can be a judge.

A judge config (``configs/judges/*.yaml``) picks a provider and pins a model:

- ``claude_code``: Claude Code in headless mode (``claude -p``), using the logged-in
  Claude subscription instead of an API key.
- ``anthropic``: the Claude API (Messages API), key from ``api_key_env``.
- ``openai_compatible``: any Chat Completions endpoint, e.g. OpenAI (ChatGPT models) or
  xAI (Grok models), set by ``base_url`` with the key from ``api_key_env``.

Adding a provider means adding one class with a ``complete`` method. Every backend is
wrapped in ``CachedBackend``, so a given (provider, model, prompt) is only sent once.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field

Provider = Literal["claude_code", "anthropic", "openai_compatible"]


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
    model: str = Field(min_length=1, description="Pinned model ID")
    base_url: str | None = None
    api_key_env: str | None = None
    # USD per million tokens, used for cost estimates and the --max-cost guard. Leave
    # unset when unknown: estimates then show tokens only.
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None
    max_output_tokens: int = 512
    timeout_s: float = 180.0


def load_judge_config(path: Path) -> JudgeConfig:
    return JudgeConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class JudgeReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    model_served: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    cached: bool = False


class JudgeBackend(Protocol):
    config: JudgeConfig

    def complete(self, system: str, user: str) -> JudgeReply: ...


class JudgeBackendError(RuntimeError):
    pass


def price(config: JudgeConfig, input_tokens: int, output_tokens: int) -> float | None:
    if config.price_input_per_mtok is None or config.price_output_per_mtok is None:
        return None
    return (
        input_tokens * config.price_input_per_mtok + output_tokens * config.price_output_per_mtok
    ) / 1e6


def _api_key(config: JudgeConfig) -> str:
    if not config.api_key_env:
        raise JudgeBackendError(f"{config.provider} judge needs api_key_env in its config")
    key = os.environ.get(config.api_key_env, "")
    if not key:
        raise JudgeBackendError(f"{config.api_key_env} is not set (see .env.example)")
    return key


class ClaudeCodeBackend:
    """``claude -p`` with the judge prompt as the only system prompt, no tools, no MCP
    servers, no settings or memory (it runs in an empty temporary directory), and no
    saved session. The reported cost is the list-price equivalent; on a subscription it
    counts toward the plan's usage limits rather than being billed."""

    def __init__(self, config: JudgeConfig, executable: str = "claude") -> None:
        self.config = config
        self._executable = executable

    def complete(self, system: str, user: str) -> JudgeReply:
        command = [
            self._executable,
            "-p",
            "--model",
            self.config.model,
            "--system-prompt",
            system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
        ]
        with tempfile.TemporaryDirectory(prefix="rag-eval-judge-") as cwd:
            try:
                done = subprocess.run(
                    command,
                    input=user,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=self.config.timeout_s,
                    check=False,
                )
            except FileNotFoundError as error:
                raise JudgeBackendError(
                    "Claude Code CLI not found; install it and log in (see README)"
                ) from error
        try:
            payload = cast(dict[str, Any], json.loads(done.stdout))
        except json.JSONDecodeError as error:
            raise JudgeBackendError(
                f"claude -p failed (exit {done.returncode}): {done.stderr.strip()[:300]}"
            ) from error
        if payload.get("is_error"):
            raise JudgeBackendError(f"claude -p error: {str(payload.get('result'))[:300]}")
        usage = cast(dict[str, Any], payload.get("usage", {}))
        model_usage = cast(dict[str, Any], payload.get("modelUsage", {}))
        served = next(iter(model_usage), self.config.model)
        return JudgeReply(
            text=str(payload.get("result", "")),
            model_served=served,
            input_tokens=int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cost_usd=float(payload.get("total_cost_usd", 0.0)),
        )


class AnthropicBackend:
    """The Claude API through the official SDK."""

    def __init__(self, config: JudgeConfig) -> None:
        import anthropic  # noqa: PLC0415 - only needed for this provider

        self.config = config
        self._client = anthropic.Anthropic(api_key=_api_key(config), timeout=config.timeout_s)

    def complete(self, system: str, user: str) -> JudgeReply:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        u = response.usage
        return JudgeReply(
            text=text,
            model_served=response.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cost_usd=price(self.config, u.input_tokens, u.output_tokens),
        )


class OpenAICompatibleBackend:
    """Any Chat Completions endpoint (``POST {base_url}/chat/completions``), e.g.
    OpenAI or xAI. Only the fields every such endpoint accepts are sent."""

    def __init__(self, config: JudgeConfig, client: httpx.Client | None = None) -> None:
        if not config.base_url:
            raise JudgeBackendError("openai_compatible judge needs base_url in its config")
        self.config = config
        self._client = client or httpx.Client(timeout=config.timeout_s)
        self._headers = {"Authorization": f"Bearer {_api_key(config)}"}

    def complete(self, system: str, user: str) -> JudgeReply:
        response = self._client.post(
            f"{(self.config.base_url or '').rstrip('/')}/chat/completions",
            headers=self._headers,
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        if response.status_code != 200:
            raise JudgeBackendError(f"{response.status_code}: {response.text[:300]}")
        payload = cast(dict[str, Any], response.json())
        choices = cast(list[dict[str, Any]], payload.get("choices", []))
        text = str(choices[0]["message"]["content"]) if choices else ""
        usage = cast(dict[str, Any], payload.get("usage", {}))
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        return JudgeReply(
            text=text,
            model_served=str(payload.get("model", self.config.model)),
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=price(self.config, tokens_in, tokens_out),
        )


def make_backend(config: JudgeConfig) -> JudgeBackend:
    match config.provider:
        case "claude_code":
            return ClaudeCodeBackend(config)
        case "anthropic":
            return AnthropicBackend(config)
        case "openai_compatible":
            return OpenAICompatibleBackend(config)


def request_key(config: JudgeConfig, system: str, user: str) -> str:
    """Cache key: provider, pinned model and the full prompt (judge prompt plus inputs)."""
    blob = json.dumps(
        {"provider": config.provider, "model": config.model, "system": system, "user": user},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedBackend:
    """Replays a stored reply for any request already sent; stores new ones."""

    def __init__(self, inner: JudgeBackend, cache_dir: Path) -> None:
        self.config = inner.config
        self._inner = inner
        self._dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def is_cached(self, system: str, user: str) -> bool:
        return (self._dir / f"{request_key(self.config, system, user)}.json").exists()

    def complete(self, system: str, user: str) -> JudgeReply:
        path = self._dir / f"{request_key(self.config, system, user)}.json"
        if path.exists():
            stored = JudgeReply.model_validate_json(path.read_text(encoding="utf-8"))
            return stored.model_copy(update={"cached": True})
        reply = self._inner.complete(system, user)
        path.write_text(reply.model_dump_json(), encoding="utf-8")
        return reply
