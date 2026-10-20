"""Normalising text that AnythingLLM decorates."""

from __future__ import annotations

import re

# AnythingLLM prepends this block to every chunk at embed time. It is not document
# content: leaving it in would inflate lexical overlap and break evidence matching.
_METADATA_HEADER = re.compile(r"\A\s*<document_metadata>.*?</document_metadata>\s*", re.DOTALL)

# Thinking models (e.g. qwen3) emit reasoning that AnythingLLM wraps in <think> tags
# at the start of the response text.
_THINK_BLOCK = re.compile(r"\A\s*<think>(.*?)</think>\s*", re.DOTALL)


def strip_metadata_header(chunk_text: str) -> str:
    return _METADATA_HEADER.sub("", chunk_text, count=1)


def split_reasoning(response_text: str) -> tuple[str | None, str]:
    """Return ``(reasoning, answer)``; reasoning is None when there is no think block."""
    match = _THINK_BLOCK.match(response_text)
    if match is None:
        return None, response_text.strip()
    return match.group(1).strip(), response_text[match.end() :].strip()
