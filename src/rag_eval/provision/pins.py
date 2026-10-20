"""Checks that what is running matches what the config pins."""

from __future__ import annotations

import json
import shutil
import subprocess

from rag_eval.config import OllamaModel


class PinMismatchError(RuntimeError):
    pass


def verify_ollama_models(installed: dict[str, str], pinned: list[OllamaModel]) -> None:
    for model in pinned:
        digest = installed.get(model.name)
        if digest is None:
            raise PinMismatchError(
                f"Ollama model {model.name} is not installed (ollama pull {model.name})"
            )
        if digest != model.digest:
            raise PinMismatchError(
                f"Ollama model {model.name} has digest {digest}, config pins {model.digest}"
            )


def verify_container_image(container: str, pinned_image: str) -> bool:
    """True if the running container was started from the pinned image digest.

    Returns False (unverified) when the docker CLI is unavailable or the container
    is not found; raises if the container runs a different image.
    """
    docker = shutil.which("docker")
    if docker is None:
        return False
    inspected = subprocess.run(
        [docker, "inspect", "--format", "{{.Image}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode != 0:
        return False
    image_id = inspected.stdout.strip()
    repo_digests = subprocess.run(
        [docker, "image", "inspect", "--format", "{{json .RepoDigests}}", image_id],
        capture_output=True,
        text=True,
        check=True,
    )
    digests: list[str] = json.loads(repo_digests.stdout)
    pinned_digest = pinned_image.split("@", 1)[1]
    if not any(d.endswith(f"@{pinned_digest}") for d in digests):
        raise PinMismatchError(f"container {container} runs {digests}, config pins {pinned_image}")
    return True
