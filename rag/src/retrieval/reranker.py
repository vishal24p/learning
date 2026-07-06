"""Cross-encoder reranker over retrieved chunks.

Wraps sentence-transformers CrossEncoder. Lazy-loads the model on first call.
If the model can't be imported (no internet / fails), `rerank()` falls back to
identity ordering so the pipeline still completes during local debugging.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from src.config import settings

log = logging.getLogger(__name__)


def filter_positive_chunks(
    reranked: list[dict],
    fallback_top_n: int = 1,
) -> list[dict]:
    """Keep only reranked chunks with a strictly positive score.

    The reranker emits a floating "score" per chunk (cross-encoder logits).
    For generation we only want to pass evidence that the model thinks is
    actually relevant -- so we drop anything <= 0.

    Fallback: if EVERY chunk scores <= 0, we still want to answer the user
    instead of silently refusing. In that case we hand the LLM the single
    top-ranked chunk (the reranker's "best guess", even if weakly negative)
    so it at least has some context to work with.

    The function is a no-op for the identity-fallback path (no scores) and
    for empty input: empty -> [].

    Parameters
    ----------
    reranked : list[dict]
        Output of ``CrossEncoderReranker.rerank()``. Each item has keys
        ``chunk_id``, ``content``, ``score``. Already sorted by score desc.
    fallback_top_n : int
        How many chunks to keep when all scores are non-positive.
        Defaults to 1 (the top chunk only).

    Returns
    -------
    list[dict]
        The filtered (or fallback) chunk list, preserving the original
        descending-score order.
    """
    if not reranked:
        return []

    positive = [c for c in reranked if float(c.get("score", 0.0)) > 0.0]
    if positive:
        dropped = len(reranked) - len(positive)
        if dropped:
            log.info(
                "filter_positive_chunks: kept %d positive chunks, dropped %d non-positive",
                len(positive), dropped,
            )
        return positive

    # All chunks non-positive -- use the top-N so the LLM still sees *something*.
    fallback = reranked[: max(1, fallback_top_n)]
    top_scores = ", ".join(f"{c['score']:.2f}" for c in fallback)
    log.warning(
        "filter_positive_chunks: all %d scores were non-positive; "
        "falling back to top-%d chunk(s) scores=[%s]",
        len(reranked), len(fallback), top_scores,
    )
    return fallback


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str | None = None,
        score_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.model_name = model_name or settings.rerank_model
        self._model = None
        self._score_fn = score_fn  # allows tests to inject a stub

    def _ensure_model(self) -> None:
        if self._model is not None or self._score_fn is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            t = time.perf_counter()
            log.info("loading CrossEncoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            log.info("cross-encoder loaded: %s (%.1f ms)",
                     self.model_name, (time.perf_counter() - t) * 1000.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("cross-encoder unavailable, identity fallback: %s", exc)
            self._model = None

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        if not candidates:
            return []

        t = time.perf_counter()
        self._ensure_model()
        keep = top_n if top_n is not None else settings.rerank_top_n

        # Compute scores
        if self._score_fn is not None:
            scores = [float(self._score_fn(query, c["content"])) for c in candidates]
            log.info("rerank via injected score_fn (fallback/test)")
        elif self._model is not None:
            t_pred = time.perf_counter()
            pairs = [(query, c["content"]) for c in candidates]
            log.info("rerank pairs=%d (truncated text per pair below)",
                     len(pairs))
            for i, (q, c) in enumerate(pairs):
                log.debug("  pair[%d] q=%r c=%r", i, q[:60], c[:60])
            raw = self._model.predict(pairs, convert_to_numpy=True)
            try:
                scores = [float(x) for x in raw.tolist()]
            except AttributeError:
                scores = [float(x) for x in raw]
            log.info("cross-encoder predict scored=%d pairs (%.1f ms)",
                     len(scores), (time.perf_counter() - t_pred) * 1000.0)
        else:
            log.info("rerank kept in RRF order (identity fallback) in=%d out=%d",
                     len(candidates), keep)
            return candidates[:keep]

        scored = [
            {"chunk_id": c["chunk_id"], "content": c["content"], "score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        out = scored[:keep]
        top3 = ", ".join(f"{r['score']:.2f}" for r in out)
        log.info(
            "rerank in=%d out=%d top=%s (total %.1f ms)",
            len(candidates), len(out), top3,
            (time.perf_counter() - t) * 1000.0,
        )
        log.debug("rerank scores (full): %s", [round(s, 4) for s in scores])
        return out
