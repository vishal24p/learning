"""CLI: build a small Q set from the corpus, run the RAG pipeline, score with RAGAS.

Usage:
    python scripts\run_ragas.py [--corpus PATH] [--rows N] [--out PATH]

Default corpus is website-main/content/en (the docs we ingested).
Results are appended to scripts/ragas_results.jsonl so each run accumulates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.evaluation.dataset_gen import build_qa_set  # noqa: E402
from src.evaluation.judge import build_judge_llm  # noqa: E402
from src.evaluation.ragas_runner import (  # noqa: E402
    evaluate_pipeline,
    METRIC_NAMES,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="run RAGAS evaluation against the local pipeline")
    ap.add_argument("--corpus", default=r"C:\Users\visha\rag\website-main\content\en",
                    help="markdown corpus directory")
    ap.add_argument("--rows", type=int, default=5, help="number of QA rows")
    ap.add_argument("--out", default="scripts/ragas_results.jsonl", help="append output JSONL")
    args = ap.parse_args()

    print(f"[run_ragas] corpus: {args.corpus}")
    print(f"[run_ragas] rows:   {args.rows}")
    qa_set = build_qa_set(args.corpus, n=args.rows)
    print(f"[run_ragas] built {len(qa_set)} QA pairs")
    for qa in qa_set:
        print(f"  - {qa['id']}: {qa['question'][:80]}...")

    judge_llm = build_judge_llm()
    result = evaluate_pipeline(qa_set, settings.db_url, judge_llm)

    print("\n== RAGAS mean ==")
    for name, value in result["mean"].items():
        print(f"  {name}: {value:.4f}")
    print(f"\n[run_ragas] elapsed: {result['elapsed_seconds']:.1f}s")

    rows = []
    for r in result["per_row"]:
        flat = {
            **{k: r[k] for k in ("id", "question", "answer", "ground_truth")},
            "scores": r["scores"],
        }
        rows.append(flat)
    write_jsonl(rows, args.out)
    print(f"[run_ragas] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
