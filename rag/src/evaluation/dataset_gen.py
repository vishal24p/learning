"""Build a small Q set from the markdown corpus for RAGAS evaluation.

Each row is one question-answer pair derived from a random markdown page.
We use the first H2 / H3 heading as the question and the first paragraph
after it as the expected answer. This is enough signal for context_precision /
faithfulness / answer_relevancy (all three judges only need the question and
the answer from the system; ground_truth lets ragas also know the reference).

Each row's `reference_contexts` is the same paragraph (or two) so context_recall
can also be computed if needed.

The walker is funneled through ``load_markdown_files`` so the curated
exclusion list (blog, careers, includes, images, etc.) is respected --
otherwise the Q-set ends up built from irrelevant pages.
"""
from __future__ import annotations

import random
import re
from collections.abc import Iterable
from pathlib import Path

from src.loaders.markdown_loader import load_markdown_files

QUESTION_HEADING_RE = re.compile(r"^#{2,3}\s+(.+)$", re.MULTILINE)


def _first_paragraph(text: str, start_at: int) -> str:
    """Return the first paragraph after `start_at`, collapsed to single spaces."""
    block = text[start_at:].lstrip("\n").split("\n\n", 1)[0]
    block = block.strip()
    return re.sub(r"\s+", " ", block)


def build_qa_set(root_dir: str, n: int = 5, seed: int = 7) -> list[dict]:
    """Generate a deterministic list of {id, question, ground_truth, reference_contexts}.

    Returns an empty list if no usable questions could be derived (rather
    than raising), so callers can decide how to proceed.
    """
    docs = load_markdown_files(root_dir, include_subdirs=("docs",))
    if not docs:
        return []

    rng = random.Random(seed)
    rng.shuffle(docs)
    chosen: list[dict] = []
    for d in docs:
        if len(chosen) >= n:
            break
        path = Path(d["path"])
        content = d["content"]
        m = QUESTION_HEADING_RE.search(content)
        if not m:
            continue
        heading = m.group(1).strip()
        paragraph = _first_paragraph(content, m.end())
        if len(paragraph) < 80 or len(heading) < 8:
            continue
        chosen.append(
            {
                "id": path.stem,
                "question": f"How is '{heading}' described in this doc?",
                "ground_truth": paragraph[:300],
                "reference_contexts": [paragraph[:300]],
            }
        )
    return chosen
