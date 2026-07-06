"""Markdown loader: emits stable {id, path, content} for each .md file.

By default the loader skips folders that produce noise (not useful for
Q&A-style indexing) such as Hugo includes, image directories, partner
listings, blog announcements, careers, etc. Pass ``include_subdirs`` to
restrict the walk further, or ``exclude`` to add custom skip patterns.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


# Default folders that should NOT be indexed when walking the corpus root.
# These typically contain junk includes, marketing pages, job listings,
# one-liner announcements, etc.
DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    "blog",
    "careers",
    "case-studies",
    "community",
    "examples",
    "includes",
    "partners",
    "releases",
    "training",
    "_common-resources",
    "images",
    ".git",
    "node_modules",
)

DEFAULT_EXCLUDE_FILES: tuple[str, ...] = (
    "OWNERS",
    "test.md",
    "search.md",
    "_index.html",
)


def _is_skipped(path: Path, exclude_dirs: tuple[str, ...], exclude_files: tuple[str, ...]) -> bool:
    parts = set(path.parts)
    if parts & set(exclude_dirs):
        return True
    if path.name in exclude_files:
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}-", path.name):  # blog post slug style dates
        return True
    return False


def load_markdown_files(
    root_dir: str,
    *,
    include_subdirs: tuple[str, ...] | None = None,
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
    exclude_files: tuple[str, ...] = DEFAULT_EXCLUDE_FILES,
) -> list[dict]:
    """Walk ``root_dir`` and return one dict per .md file.

    Parameters
    ----------
    root_dir:
        Top-level directory to index. We recurse unless ``include_subdirs``
        restricts the recursion to specific subfolders (relative to root).
    include_subdirs:
        Optional whitelist of subdirectories under root_dir to descend into.
        Example: ``include_subdirs=("docs",)`` will only walk
        ``website-main/content/en/docs`` and the subfolders it contains.
    exclude_dirs:
        Folder names (matched by component) to skip anywhere in the tree.
    exclude_files:
        File names to skip anywhere in the tree.
    """
    root = Path(root_dir)
    docs: list[dict] = []

    if include_subdirs:
        bases = [root / sub for sub in include_subdirs]
    else:
        bases = [root]

    for base in bases:
        if not base.exists():
            continue
        for file in base.rglob("*.md"):
            if _is_skipped(file, exclude_dirs, exclude_files):
                continue
            try:
                content = file.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {file}: {exc}")
                continue
            if not content.strip():
                continue
            doc_id = hashlib.md5(str(file.resolve()).encode()).hexdigest()[:12]
            docs.append({"id": doc_id, "path": str(file), "content": content})
    return docs
