"""Decision dataclass for the guard.

Carries the parsed outcome of a single guard call. A separation of
"raw model output" and "structured decision" keeps the rest of the
pipeline from caring about Ollama's specific tool-call shape.
"""
from __future__ import annotations

from dataclasses import dataclass


# Stable sentinel action names. Using constants instead of free strings
# prevents typos in the pipeline layer.
ACTION_SAFE = "safe_to_continue"
ACTION_TERMINATE = "terminate"


@dataclass(frozen=True)
class GuardDecision:
    """One guard verdict.

    `action` is one of ``ACTION_SAFE`` or ``ACTION_TERMINATE``.

    `reason` is always populated (audit trails want a reason even when
    the prompt was safe; it makes refactor decisions in the future
    easier).

    `refusal` is populated only when `action == ACTION_TERMINATE`.
    """

    action: str
    reason: str
    refusal: str | None = None

    @property
    def is_terminate(self) -> bool:
        return self.action == ACTION_TERMINATE

    @property
    def is_safe(self) -> bool:
        return self.action == ACTION_SAFE


class PipelineTerminated(Exception):
    """Raised by ``guard_or_terminate`` when the guard decides to stop.

    Carrying the decision on the exception means callers (the CLI scripts
    and the Streamlit app) can render the refusal text without any extra
    API surface, and the audit row can be written from a single
    ``except`` block.
    """

    def __init__(self, query: str, decision: GuardDecision) -> None:
        super().__init__(f"guard terminated: {decision.reason}")
        self.query = query
        self.decision = decision
