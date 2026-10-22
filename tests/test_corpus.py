from pathlib import Path

import pytest

from rag_eval.config import CorpusConfig, load_config
from rag_eval.provision.corpus import load_corpus


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "a.txt").write_text("alpha", encoding="utf-8")
    (directory / "b.md").write_text("bravo", encoding="utf-8")
    (directory / ".gitkeep").touch()
    return directory


def test_documents_are_identified_by_filename(corpus_dir: Path) -> None:
    corpus = load_corpus(CorpusConfig(dir=corpus_dir))
    assert [d.doc_id for d in corpus.documents] == ["a.txt", "b.md"]


def test_exclude_removes_a_document_and_changes_the_hash(corpus_dir: Path) -> None:
    full = load_corpus(CorpusConfig(dir=corpus_dir))
    reduced = load_corpus(CorpusConfig(dir=corpus_dir, exclude=("a.txt",)))
    assert [d.doc_id for d in reduced.documents] == ["b.md"]
    assert reduced.content_hash != full.content_hash


def test_replace_keeps_the_id_but_uses_the_edited_content(corpus_dir: Path, tmp_path: Path) -> None:
    edited = tmp_path / "a-edited.txt"
    edited.write_text("alpha, edited", encoding="utf-8")
    full = load_corpus(CorpusConfig(dir=corpus_dir))
    replaced = load_corpus(CorpusConfig(dir=corpus_dir, replace={"a.txt": edited}))
    doc = replaced.by_id()["a.txt"]
    assert doc.path == edited
    assert doc.sha256 != full.by_id()["a.txt"].sha256
    assert replaced.content_hash != full.content_hash


def test_content_hash_depends_on_content(corpus_dir: Path) -> None:
    before = load_corpus(CorpusConfig(dir=corpus_dir)).content_hash
    (corpus_dir / "a.txt").write_text("alpha!", encoding="utf-8")
    assert load_corpus(CorpusConfig(dir=corpus_dir)).content_hash != before


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"exclude": ("missing.txt",)}, "unknown document IDs"),
        ({"replace": {"missing.txt": Path("x")}}, "unknown document IDs"),
        ({"exclude": ("a.txt", "b.md")}, "empty"),
    ],
)
def test_invalid_corpus_configs(
    corpus_dir: Path, config_kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_corpus(CorpusConfig.model_validate({"dir": corpus_dir, **config_kwargs}))


def test_nested_directories_are_rejected(corpus_dir: Path) -> None:
    (corpus_dir / "sub").mkdir()
    with pytest.raises(ValueError, match="flat"):
        load_corpus(CorpusConfig(dir=corpus_dir))


def test_unsupported_files_are_rejected(corpus_dir: Path) -> None:
    (corpus_dir / "data.csv").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_corpus(CorpusConfig(dir=corpus_dir))


def test_closed_book_corpus_is_empty_and_takes_no_edits(tmp_path: Path) -> None:
    corpus = load_corpus(CorpusConfig(dir=tmp_path / "unused", closed_book=True))
    assert corpus.documents == ()
    with pytest.raises(ValueError, match="closed_book"):
        CorpusConfig(dir=tmp_path, closed_book=True, exclude=("a.txt",))


def test_closed_book_config_extends_the_baseline() -> None:
    config = load_config(Path("configs/closed-book.yaml"))
    baseline = load_config(Path("configs/baseline.yaml"))
    assert config.corpus.closed_book
    assert config.generation.mode == "chat"
    assert config.llm == baseline.llm
