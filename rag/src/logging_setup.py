"""Stdout-only logging for the RAG pipeline.

Goals:
- Pure stdlib (no extra deps).
- Single-line format. No JSON. No colors.
- Configurable level from env LOG_LEVEL (default INFO).
- Safe across CLI + Streamlit (re-configuring won't double-log).
- Third-party libraries (httpx, urllib3, huggingface_hub, etc.) are quieted
  to WARNING so the line timeline is clean.

Format example:
    2026-06-25 14:11:09 [INFO] src.retrieval.hybrid_retriever: rrf pool=17 top_k=20 took=2.81s
"""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Names whose own logging we silence. Our loggers (under src.*) stay at the
# user-requested level so the full timeline is visible.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "filelock",
    "huggingface_hub",
    "huggingface_hub.utils",
    "sentence_transformers",
    "transformers",
    "asyncio",
)


def _level_from_env() -> int:
    raw = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    return logging.getLevelNamesMapping().get(raw, logging.INFO)


@lru_cache(maxsize=1)
def setup_logging() -> logging.Logger:
    """Configure the root logger once. Returns the root logger."""
    root = logging.getLogger()
    if getattr(root, "_rag_configured", False):
        return root

    root.setLevel(_level_from_env())

    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(DEFAULT_FMT, datefmt=DEFAULT_DATEFMT))
    handler.setLevel(root.level)
    root.addHandler(handler)

    # Mute third-party chatter. Their INFO + WARNING records are filtered out;
    # ERROR and above still surface (a real library failure won't be hidden).
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    root._rag_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
