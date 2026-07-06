"""Tests for src.evaluation.dataset_gen."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.evaluation.dataset_gen import build_qa_set


def _write(root: Path, name: str, body: str) -> None:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_build_qa_set_returns_deterministic_rows():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Three short markdown files; only one with a long-enough H2 + paragraph.
        _write(root, "a.md",
               "## Overview\n\nThis is a sufficiently long paragraph about an"
               " overview so the builder will accept it (>80 chars needed).\n")
        _write(root, "b.md", "## Tiny\n\ntoo short\n")
        _write(root, "c.md",
               "## Bigger section name\n\n"
               + ("Sentence one. " * 8) + "Sentence two. " * 8 + "\n")

        # Same seed => same row order across calls.
        a = build_qa_set(str(root), n=5)
        b = build_qa_set(str(root), n=5)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        # Both runs yield the same length.
        assert len(a) == len(b)
        # We asked for 5 rows; corpus only had 2 valid rows, so len <= 5.
        assert len(a) <= 5


def test_build_qa_set_skips_shorts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "short.md", "## Title\n\ntoo short\n")
        rows = build_qa_set(str(root), n=5)
        assert rows == []

