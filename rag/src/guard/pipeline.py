"""Pipeline entrypoint: ``guard_or_terminate(query) -> query``.

This is the single function the rest of the codebase calls. It either:
  - returns the original ``query`` (the guard said ``safe``), or
  - raises ``PipelineTerminated`` carrying the LLM-generated refusal
    text + an audit reason (the guard said ``terminate``).

It is also responsible for honoring the ``settings.guard_enabled``
flag: when the flag is False, we skip the guard entirely (no Ollama
call, no latency cost, no log noise). This keeps A/B comparison with
an unguarded baseline a one-env-var toggle.
"""
from __future__ import annotations

import logging
import time

from src.config import settings
from src.guard.guard import LlamaGuard3
from src.guard.reason import (
    ACTION_SAFE,
    ACTION_TERMINATE,
    GuardDecision,
    PipelineTerminated,
)

log = logging.getLogger(__name__)


def guard_or_terminate(query: str, *, guard: LlamaGuard3 | None = None) -> str:
    """Run the pre-pipeline guard. Returns the (possibly unchanged) query.

    Raises:
      PipelineTerminated: if the guard decides to terminate.
    """
    if not settings.guard_enabled:
        log.debug("guard disabled via GUARD_ENABLED=false; passing query through")
        return query

    g = guard or LlamaGuard3()
    t = time.perf_counter()
    decision = g.decide(query)
    _audit_log(query, decision, (time.perf_counter() - t) * 1000.0)

    if decision.is_terminate:
        # The audit row is written BEFORE we raise so the log line
        # is on disk regardless of how the caller catches it.
        raise PipelineTerminated(query=query, decision=decision)

    return query


def _audit_log(query: str, decision: GuardDecision, elapsed_ms: float) -> None:
    """Write one structured-ish log line per guard verdict.

    Format chosen so the line is greppable and parseable without
    pulling in JSON:
        guard_decision action=<safe|terminate> reason=<...> elapsed_ms=<...> query_preview=<first 80 chars>
    """
    preview = " ".join(query.split())[:80]
    log.info(
        "guard_decision action=%s reason=%r elapsed_ms=%.1f query_preview=%r",
        decision.action,
        decision.reason,
        elapsed_ms,
        preview,
    )
