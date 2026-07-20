"""Tests for src.query_rewrite: single-query rewriting.

Behavior contract:
- rewrite_query returns ONE query string (never a list).
- Returns the cleaned rewritten string when the LLM returns a non-empty value.
- Falls back to the original query on any LLM error or empty response.
- Honors QUERY_REWRITE_ENABLED=false: skips the LLM call entirely.
- Skips rewriting for empty input.
"""
from __future__ import annotations

import importlib


class _FakeChatResp:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


class _FakeLLM:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[list] = []

    def chat(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return _FakeChatResp(self.response)


def _reload_module(monkeypatch, *, enabled: bool):
    """Reload src.query_rewrite so QUERY_REWRITE_ENABLED env flips take effect."""
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("QUERY_REWRITE_MODEL", "minimax-m3:cloud")

    import src.config as cfg_mod
    importlib.reload(cfg_mod)
    import src.query_rewrite as qr
    importlib.reload(qr)
    return qr


def test_rewrite_returns_one_query_string(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=True)
    fake = _FakeLLM("kubernetes pod scheduling how it works")
    rw = qr.QueryRewriter(llm=fake)

    out = rw.rewrite("how do pods get scheduled?")

    assert isinstance(out, str)
    assert out == "kubernetes pod scheduling how it works"
    assert len(fake.calls) == 1
    assert len(fake.calls[0]) == 2


def test_rewrite_strips_assistant_role_prefix(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=True)
    fake = _FakeLLM("assistant: What is a Pod in Kubernetes?")
    rw = qr.QueryRewriter(llm=fake)

    out = rw.rewrite("what is the Pods?")

    assert out == "What is a Pod in Kubernetes?"


def test_rewrite_collapses_multiline_output(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=True)
    fake = _FakeLLM("kubernetes pod scheduling\nhow kubernetes schedules pods")
    rw = qr.QueryRewriter(llm=fake)

    out = rw.rewrite("how do pods get scheduled?")

    assert out == "kubernetes pod scheduling how kubernetes schedules pods"


def test_rewrite_via_rewrite_query_function(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=True)
    fake = _FakeLLM("kubernetes pod scheduling")
    out = qr.rewrite_query("how do pods get scheduled?", rewriter=qr.QueryRewriter(llm=fake))
    assert out == "kubernetes pod scheduling"


def test_rewrite_query_function_disabled_skips_llm(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=False)
    fake = _FakeLLM("rewritten text")
    rw = qr.QueryRewriter(llm=fake)

    out = qr.rewrite_query("how do pods work?", rewriter=rw)
    assert out == "how do pods work?"
    assert fake.calls == []


def test_rewrite_skips_when_input_empty(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=True)
    fake = _FakeLLM("never used")
    rw = qr.QueryRewriter(llm=fake)

    assert rw.rewrite("") == ""
    assert rw.rewrite("   ") == "   "
    assert fake.calls == []


def test_rewrite_returns_original_on_llm_exception(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=True)
    fake = _FakeLLM(RuntimeError("ollama down"))
    rw = qr.QueryRewriter(llm=fake)

    out = rw.rewrite("how do pods work?")
    assert out == "how do pods work?"


def test_rewrite_returns_original_on_empty_response(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=True)
    fake = _FakeLLM("")
    rw = qr.QueryRewriter(llm=fake)

    out = rw.rewrite("how do pods work?")
    assert out == "how do pods work?"


def test_rewrite_returns_original_on_whitespace_response(monkeypatch):
    qr = _reload_module(monkeypatch, enabled=True)
    fake = _FakeLLM("   \n  ")
    rw = qr.QueryRewriter(llm=fake)

    out = rw.rewrite("how do pods work?")
    assert out == "how do pods work?"
