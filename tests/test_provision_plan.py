from pathlib import Path

from rag_eval.client.models import Workspace
from rag_eval.config import load_config
from rag_eval.provision.corpus import load_corpus
from rag_eval.provision.provisioner import (
    plan_document_sync,
    settings_mismatches,
    workspace_name,
    workspace_settings,
)

SMOKE = Path("tests/smoke/smoke.yaml")


def test_sync_uploads_missing_and_keeps_present() -> None:
    sync = plan_document_sync([("a.txt", "cd/a.json")], ["a.txt", "b.txt"])
    assert sync.to_upload == ["b.txt"]
    assert sync.kept == {"a.txt": "cd/a.json"}
    assert sync.to_remove == []


def test_sync_removes_unknown_untitled_and_duplicate_documents() -> None:
    present = [
        ("a.txt", "cd/a1.json"),
        ("a.txt", "cd/a2.json"),
        ("old.txt", "cd/o.json"),
        (None, "cd/n.json"),
    ]
    sync = plan_document_sync(present, ["a.txt"])
    assert sync.kept == {"a.txt": "cd/a1.json"}
    assert sync.to_remove == ["cd/a2.json", "cd/o.json", "cd/n.json"]
    assert sync.to_upload == []


def test_sync_of_an_up_to_date_workspace_is_a_no_op() -> None:
    sync = plan_document_sync([("a.txt", "x"), ("b.txt", "y")], ["a.txt", "b.txt"])
    assert (sync.to_upload, sync.to_remove) == ([], [])


def test_workspace_name_changes_with_config_and_corpus(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    corpus = load_corpus(config.corpus)
    name = workspace_name(config, corpus)
    assert name.startswith("smoke-")
    assert name == workspace_name(config, corpus)

    changed_config = config.model_copy(update={"description": "different"})
    assert workspace_name(changed_config, corpus) != name

    edited = tmp_path / "edited.txt"
    edited.write_text("edited", encoding="utf-8")
    changed_corpus = load_corpus(
        config.corpus.model_copy(update={"replace": {"zorvath-valve.txt": edited}})
    )
    assert workspace_name(config, changed_corpus) != name


def test_settings_read_back_matches_what_was_sent() -> None:
    config = load_config(SMOKE)
    workspace = Workspace.model_validate(
        {"id": 1, "slug": "s", "name": "s", **workspace_settings(config)}
    )
    assert settings_mismatches(workspace, config) == []


def test_settings_mismatches_are_reported() -> None:
    config = load_config(SMOKE)
    sent = workspace_settings(config) | {"topN": 9, "openAiPrompt": "other"}
    workspace = Workspace.model_validate({"id": 1, "slug": "s", "name": "s", **sent})
    mismatches = settings_mismatches(workspace, config)
    assert len(mismatches) == 2
    assert any(m.startswith("top_n") for m in mismatches)
