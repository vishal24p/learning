"""Unit test: Markdown loader hashes paths to stable ids."""
from __future__ import annotations

from pathlib import Path


def test_load_markdown_files(tmp_path: Path):
    from src.loaders.markdown_loader import load_markdown_files

    a = tmp_path / "a.md"
    a.write_text("# Hello", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    b = sub / "b.md"
    b.write_text("## World", encoding="utf-8")

    docs = load_markdown_files(str(tmp_path))
    paths = {d["path"] for d in docs}
    assert str(a) in paths
    assert str(b) in paths
    for d in docs:
        assert "id" in d
        assert len(d["id"]) == 12
