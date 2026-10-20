import hashlib
from pathlib import Path

import pytest

from rag_eval.provision.sources import ChecksumMismatchError, Source, fetch_sources, load_sources

SOURCES = Path("corpus/sources.yaml")
DATA = b"%PDF-1.7 report"


def _source(doc_id: str = "MIR2501.pdf", data: bytes = DATA) -> Source:
    return Source(
        doc_id=doc_id,
        report="MIR-25-01",
        title="Title",
        url=f"https://example.org/{doc_id}",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )


def _never_called(url: str) -> bytes:
    raise AssertionError(f"unexpected download of {url}")


def test_committed_sources_file_is_valid() -> None:
    sources = load_sources(SOURCES)
    assert len(sources) == 77
    assert all(s.url.endswith(s.doc_id) for s in sources)


def test_every_source_is_linked_from_the_readme() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    missing = [s.doc_id for s in load_sources(SOURCES) if s.url not in readme]
    assert missing == []


def test_downloads_missing_and_verifies(tmp_path: Path) -> None:
    result = fetch_sources([_source()], tmp_path, download=lambda _url: DATA)
    assert result.downloaded == ["MIR2501.pdf"]
    assert (tmp_path / "MIR2501.pdf").read_bytes() == DATA


def test_present_files_are_verified_not_downloaded(tmp_path: Path) -> None:
    (tmp_path / "MIR2501.pdf").write_bytes(DATA)
    result = fetch_sources([_source()], tmp_path, download=_never_called)
    assert result.already_present == ["MIR2501.pdf"]
    assert result.downloaded == []


def test_changed_download_is_rejected_and_not_written(tmp_path: Path) -> None:
    with pytest.raises(ChecksumMismatchError, match="reissued"):
        fetch_sources([_source()], tmp_path, download=lambda _url: b"different")
    assert list(tmp_path.iterdir()) == []


def test_modified_local_file_is_rejected_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "MIR2501.pdf").write_bytes(b"tampered")
    with pytest.raises(ChecksumMismatchError, match="Delete it"):
        fetch_sources([_source()], tmp_path, download=_never_called)
    assert (tmp_path / "MIR2501.pdf").read_bytes() == b"tampered"


def test_unexpected_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "extra.pdf").write_bytes(b"x")
    (tmp_path / ".gitkeep").touch()
    result = fetch_sources([_source()], tmp_path, download=lambda _url: DATA)
    assert result.unexpected == ["extra.pdf"]


def test_duplicate_doc_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    entry = _source().model_dump()
    path.write_text(
        "documents:\n"
        + "".join(
            "- " + "\n  ".join(f"{k}: {v}" for k, v in entry.items()) + "\n" for _ in range(2)
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_sources(path)
