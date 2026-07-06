"""RAGAS evaluator.

Builds a ragas SingleTurnSample dataset and runs `ragas.evaluate` with the
three cheapest metrics (no labeled doc lists required):

  * faithfulness            was the answer supported by the retrieved context?
  * answer_relevancy        did the answer actually address the question?
  * context_precision       of retrieved chunks, how many were useful?

Returns a dict with `per_row` (list of metric scores per question) and
`mean` (the run summary).
"""
from __future__ import annotations

import logging
import time
from typing import Iterable

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.evaluation.judge import build_judge_embeddings
from src.retrieval.reranked_retriever import RerankedRetriever

log = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


def run_rag_pipeline(query: str, conn_str: str) -> tuple[list[str], str]:
    """Return (context_passages, generated_answer) for one query."""
    rr = RerankedRetriever(conn_str)
    _, top = rr.retrieve(
        query,
        top_k_dense=10,
        top_k_sparse=10,
        candidates=20,
        top_n=5,
    )
    contexts = [c["content"] for c in top]
    from src.generation.generator import Generator
    answer = Generator().generate(query, top)
    return contexts, answer


def score_with_ragas(
    query: str,
    contexts: list[str],
    answer: str,
    judge_llm: LangchainLLMWrapper,
    judge_embeddings,
) -> dict:
    """Score one (q, retrieved, answer) triple with the three cheap metrics.

    Does NOT re-run the pipeline; the caller hands in the answers it just got.
    """
    log.info("RAGAS scoring triple (q len=%d, ctx=%d, ans len=%d)",
             len(query), len(contexts), len(answer))
    ds = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference="",
            reference_contexts=[],
        ),
    ])
    metrics = [faithfulness, answer_relevancy, context_precision]
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    mean = {
        name: (sum(result[name]) / len(result[name])) if result[name] else 0.0
        for name in METRIC_NAMES
    }
    return {"mean": mean, "scores": result.scores[0]}


def evaluate_pipeline(
    qa_set: list[dict],
    conn_str: str,
    judge_llm: LangchainLLMWrapper,
) -> dict:
    """Run the pipeline for each Q in qa_set, score with RAGAS, return
    {"per_row": [...], "mean": {...}, "elapsed_seconds": float}.
    """
    samples: list[SingleTurnSample] = []
    per_row: list[dict] = []
    t0 = time.perf_counter()
    for qa in qa_set:
        log.info("=== running QA id=%s ===", qa["id"])
        log.info("Q: %s", qa["question"][:120])
        contexts, answer = run_rag_pipeline(qa["question"], conn_str)
        samples.append(SingleTurnSample(
            user_input=qa["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=qa.get("ground_truth") or "",
            reference_contexts=qa.get("reference_contexts") or [],
        ))
        per_row.append({
            "id": qa["id"],
            "question": qa["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": qa.get("ground_truth"),
        })

    log.info("running ragas.evaluate (%d metrics)...", len(METRIC_NAMES))
    ds = EvaluationDataset(samples=samples)
    metrics = [faithfulness, answer_relevancy, context_precision]
    embeddings = build_judge_embeddings()
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    elapsed = time.perf_counter() - t0

    # scores is a list of dicts, one per row; order matches samples / per_row.
    scores = result.scores
    per_row_scored: list[dict] = []
    for row, score in zip(per_row, scores):
        per_row_scored.append({
            **{k: row[k] for k in ("id", "question", "ground_truth", "answer")},
            "contexts": row["contexts"],
            "scores": {name: score.get(name) for name in METRIC_NAMES},
        })

    mean = {
        name: (sum(result[name]) / len(result[name])) if result[name] else 0.0
        for name in METRIC_NAMES
    }
    return {
        "per_row": per_row_scored,
        "mean": mean,
        "elapsed_seconds": elapsed,
    }


def write_jsonl(records: Iterable[dict], path: str) -> None:
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
