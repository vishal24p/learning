"""Regression tests for the Streamlit RAGAS button key derivation.

The implementation must produce a deterministic key from (query, top5 content),
so that calling the helper twice with the same triple yields the same key
(no duplicate-key error in Streamlit across reruns).
"""
from __future__ import annotations

from src.ui.keys import derive_ragas_button_key


def _top5(*ids: int) -> list[dict]:
    return [{"chunk_id": cid, "content": f"ctx-{cid}"} for cid in ids]


def test_deterministic_for_same_query_and_top5():
    q = "How do headings work?"
    k1 = derive_ragas_button_key(q, _top5(1, 2, 3, 4, 5))
    k2 = derive_ragas_button_key(q, _top5(1, 2, 3, 4, 5))
    assert k1 == k2


def test_different_query_yields_different_key():
    k1 = derive_ragas_button_key("First question", _top5(1, 2, 3, 4, 5))
    k2 = derive_ragas_button_key("Second unrelated question", _top5(1, 2, 3, 4, 5))
    assert k1 != k2


def test_different_top5_yields_different_key():
    k1 = derive_ragas_button_key("Same query", _top5(1, 2, 3, 4, 5))
    k2 = derive_ragas_button_key("Same query", _top5(5, 4, 3, 2, 1))
    assert k1 != k2


def test_key_uses_ragas_prefix():
    k = derive_ragas_button_key("any", _top5(1))
    assert k.startswith("ragas-")


def test_key_handles_missing_chunk_id():
    # When chunk_id is missing, fallback to index.
    k = derive_ragas_button_key("q", [{"content": "a"}, {"content": "b"}])
    assert k.startswith("ragas-")
