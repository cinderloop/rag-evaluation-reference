"""Typed client for the parts of the AnythingLLM developer API this suite uses.

Endpoints and payloads are documented with examples in docs/spike-findings.md.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Literal, Self, cast

import httpx

from rag_eval.client.models import ChatResponse, UploadedDocument, Workspace

ChatMode = Literal["chat", "query"]


class AnythingLLMError(RuntimeError):
    pass


def _as_dict(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnythingLLMError(f"unexpected {what} response: {value!r}")
    return cast(dict[str, Any], value)


class AnythingLLMClient:
    def __init__(self, base_url: str, api_key: str, *, timeout_s: float = 300.0) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise AnythingLLMError(
                f"{method} {path} -> HTTP {response.status_code}: {response.text[:500]}"
            )
        return _as_dict(response.json(), f"{method} {path}")

    # --- system ---------------------------------------------------------------

    def check_auth(self) -> None:
        body = self._request("GET", "/api/v1/auth")
        if body.get("authenticated") is not True:
            raise AnythingLLMError("API key rejected by AnythingLLM")

    def system_settings(self) -> dict[str, Any]:
        return _as_dict(self._request("GET", "/api/v1/system").get("settings"), "system settings")

    def update_env(self, values: dict[str, str]) -> None:
        """Instance-wide settings. Changing the embedder wipes every workspace's vectors."""
        body = self._request("POST", "/api/v1/system/update-env", json=values)
        if body.get("error"):
            raise AnythingLLMError(f"update-env failed: {body['error']}")

    # Chunk size/overlap are not exposed by the /v1 developer API. These two calls use
    # the internal endpoints behind the settings UI, which accept requests without a
    # session only while the instance has no password. That is acceptable for a local,
    # never-exposed eval instance; see docs/spike-findings.md.

    def text_splitter_settings(self) -> tuple[int | None, int | None]:
        labels = "text_splitter_chunk_size,text_splitter_chunk_overlap"
        body = self._request("GET", "/api/admin/system-preferences-for", params={"labels": labels})
        settings = _as_dict(body.get("settings"), "system preferences")

        def as_int(value: object) -> int | None:
            return int(value) if isinstance(value, (int, str)) and str(value).isdigit() else None

        return as_int(settings.get("text_splitter_chunk_size")), as_int(
            settings.get("text_splitter_chunk_overlap")
        )

    def set_text_splitter(self, size: int, overlap: int) -> None:
        """Applies to documents embedded afterwards; also purges the vector cache."""
        payload = {"text_splitter_chunk_size": size, "text_splitter_chunk_overlap": overlap}
        body = self._request("POST", "/api/admin/system-preferences", json=payload)
        if body.get("success") is not True:
            raise AnythingLLMError(f"setting the text splitter failed: {body}")

    # --- workspaces -----------------------------------------------------------

    def list_workspaces(self) -> list[Workspace]:
        items: object = self._request("GET", "/api/v1/workspaces").get("workspaces", [])
        if not isinstance(items, list):
            raise AnythingLLMError("unexpected workspaces response")
        return [Workspace.model_validate(item) for item in cast(list[object], items)]

    def get_workspace(self, slug: str) -> Workspace:
        found: object = self._request("GET", f"/api/v1/workspace/{slug}").get("workspace")
        # This endpoint returns a one-element list.
        if isinstance(found, list) and found:
            found = cast(list[object], found)[0]
        if not isinstance(found, dict):
            raise AnythingLLMError(f"workspace not found: {slug}")
        return Workspace.model_validate(found)

    def create_workspace(self, name: str) -> Workspace:
        body = self._request("POST", "/api/v1/workspace/new", json={"name": name})
        return Workspace.model_validate(body.get("workspace"))

    def update_workspace(self, slug: str, settings: dict[str, Any]) -> Workspace:
        body = self._request("POST", f"/api/v1/workspace/{slug}/update", json=settings)
        return Workspace.model_validate(body.get("workspace"))

    # --- documents --------------------------------------------------------------

    def upload_document(
        self, filename: str, content: bytes, *, workspace_slug: str
    ) -> UploadedDocument:
        """Upload, parse and embed one document into a workspace (synchronous)."""
        body = self._request(
            "POST",
            "/api/v1/document/upload",
            files={"file": (filename, content)},
            data={"addToWorkspaces": workspace_slug},
        )
        documents: object = body.get("documents")
        if (
            body.get("success") is not True
            or not isinstance(documents, list)
            or len(cast(list[object], documents)) != 1
        ):
            raise AnythingLLMError(f"upload of {filename} failed: {body.get('error')}")
        return UploadedDocument.model_validate(cast(list[object], documents)[0])

    def remove_documents(self, slug: str, docpaths: list[str]) -> None:
        self._request(
            "POST", f"/api/v1/workspace/{slug}/update-embeddings", json={"deletes": docpaths}
        )

    # --- chat -------------------------------------------------------------------

    def new_thread(self, slug: str, name: str) -> str:
        body = self._request("POST", f"/api/v1/workspace/{slug}/thread/new", json={"name": name})
        thread = _as_dict(body.get("thread"), "thread")
        thread_slug = thread.get("slug")
        if not isinstance(thread_slug, str):
            raise AnythingLLMError(f"thread creation returned no slug: {body}")
        return thread_slug

    def thread_chat(
        self, slug: str, thread_slug: str, message: str, mode: ChatMode
    ) -> ChatResponse:
        body = self._request(
            "POST",
            f"/api/v1/workspace/{slug}/thread/{thread_slug}/chat",
            json={"message": message, "mode": mode},
        )
        return ChatResponse.model_validate(body)
