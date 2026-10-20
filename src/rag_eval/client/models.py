"""Typed views of the AnythingLLM developer API responses this suite relies on.

Only the fields we use are declared; unknown fields are ignored so that additive API
changes don't break parsing. Shapes were verified against AnythingLLM 1.18.0
(see docs/spike-findings.md).
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


class WorkspaceDocument(_ApiModel):
    docpath: str
    metadata: str = "{}"

    @property
    def title(self) -> str | None:
        parsed: object = json.loads(self.metadata or "{}")
        if not isinstance(parsed, dict):
            return None
        title = cast(dict[str, Any], parsed).get("title")
        return title if isinstance(title, str) else None


class Workspace(_ApiModel):
    id: int
    slug: str
    name: str
    chat_mode: str | None = Field(default=None, alias="chatMode")
    top_n: int | None = Field(default=None, alias="topN")
    similarity_threshold: float | None = Field(default=None, alias="similarityThreshold")
    temperature: float | None = Field(default=None, alias="openAiTemp")
    system_prompt: str | None = Field(default=None, alias="openAiPrompt")
    refusal_response: str | None = Field(default=None, alias="queryRefusalResponse")
    chat_provider: str | None = Field(default=None, alias="chatProvider")
    chat_model: str | None = Field(default=None, alias="chatModel")
    search_mode: str | None = Field(default=None, alias="vectorSearchMode")
    documents: list[WorkspaceDocument] = Field(default_factory=list[WorkspaceDocument])


class UploadedDocument(_ApiModel):
    location: str
    title: str


class ChatSource(_ApiModel):
    id: str | None = None
    title: str | None = None
    text: str = ""
    score: float | None = None


class ChatMetrics(_ApiModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration: float | None = None
    model: str | None = None
    provider: str | None = None


class ChatResponse(_ApiModel):
    id: str
    type: Literal["textResponse", "abort"] | str
    text_response: str | None = Field(default=None, alias="textResponse")
    sources: list[ChatSource] = Field(default_factory=list[ChatSource])
    error: str | None = None
    metrics: ChatMetrics = Field(default_factory=ChatMetrics)
