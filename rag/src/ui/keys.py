"""Streamlit UI utilities (widget key derivation, etc.)."""
from __future__ import annotations


def derive_ragas_button_key(query: str, top5: list[dict]) -> str:
    """Stable, content-derived key, so Streamlit treats repeated widgets with
    the same payload as one widget across reruns.

    Exposed for unit testing; no Streamlit imports needed.
    """
    chunk_ids = ",".join(str(c.get("chunk_id", i)) for i, c in enumerate(top5))
    return f"ragas-{hash((query, chunk_ids))}"
