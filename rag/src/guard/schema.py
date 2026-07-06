"""Guard prompt contract.

The contract is *plain-text classification*, not tool calls:

  - llama-guard3:1b is called with a system prompt that asks the model
    to emit ONLY ``safe`` or ``unsafe`` on the first line.
  - On ``unsafe``, the model may emit a second line of comma-separated
    Llama Guard category codes (S1..S13).

Everything below this layer is just text parsing. The actual prompt
text lives in :mod:`src.guard.guard` (it is the trained prompt format
and should not be edited casually).

This module exists so other code (e.g. logging, tests) can refer to
the canonical refusal string and the system prompt by import instead
of by string-literal duplication.
"""
from __future__ import annotations


from src.guard.guard import (
    DEFAULT_REFUSAL,
    LLAMA_GUARD3_SYSTEM_PROMPT,
)


__all__ = [
    "DEFAULT_REFUSAL",
    "LLAMA_GUARD3_SYSTEM_PROMPT",
]
