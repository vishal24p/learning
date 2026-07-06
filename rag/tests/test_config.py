"""Smoke test: config builds a DB URL from a synthetic env."""
from __future__ import annotations

import os


def test_settings_db_url(monkeypatch):
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "56432")
    monkeypatch.setenv("DB_NAME", "rag")
    monkeypatch.setenv("DB_USER", "rag")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text-v2-moe")

    # Drop the lazy-loaded singleton
    import importlib
    import src.config as cfg_mod
    importlib.reload(cfg_mod)
    s = cfg_mod.Settings()
    assert s.db_url == "postgresql://rag:secret@localhost:56432/rag"
