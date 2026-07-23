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


def test_settings_rerank_use_softmax_parses_true_and_false(monkeypatch):
    import importlib
    import src.config as cfg_mod

    monkeypatch.setenv("RERANK_USE_SOFTMAX", "true")
    importlib.reload(cfg_mod)
    assert cfg_mod.Settings().rerank_use_softmax is True

    monkeypatch.setenv("RERANK_USE_SOFTMAX", "false")
    importlib.reload(cfg_mod)
    assert cfg_mod.Settings().rerank_use_softmax is False


def test_settings_rerank_use_softmax_defaults_disabled(monkeypatch):
    import importlib
    import src.config as cfg_mod

    monkeypatch.delenv("RERANK_USE_SOFTMAX", raising=False)
    importlib.reload(cfg_mod)
    assert cfg_mod.Settings().rerank_use_softmax is False
