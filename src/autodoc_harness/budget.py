"""Run-wide cost/time guardrail tracking, shared across all concurrently-running
agentic-loop stages in a single `generate` invocation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised when a run-wide guardrail (cost or wall-clock time) is exceeded."""


@dataclass
class BudgetTracker:
    """Tracks cost/time/tool-call totals across every stage of a single run.

    Safe to share, unlocked, across concurrently-running asyncio tasks: asyncio's
    single-threaded cooperative scheduling makes a "check, then increment" sequence
    with no `await` in between atomic by construction - there is no interleaving
    point between two tasks both reading `total_cost_usd` and one of them writing
    it back.
    """

    max_total_cost_usd: float
    max_run_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    total_cost_usd: float = 0.0
    total_tool_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    cost_unknown_warned: bool = False

    def check_or_raise(self) -> None:
        if self.elapsed_seconds > self.max_run_seconds:
            raise BudgetExceededError(f"run exceeded max_run_seconds ({self.max_run_seconds}s)")
        if self.total_cost_usd > self.max_total_cost_usd:
            raise BudgetExceededError(
                f"run exceeded max_total_cost_usd (${self.max_total_cost_usd:.2f})"
            )

    def record_llm_call(self, *, cost_usd: float | None, tokens_in: int, tokens_out: int) -> None:
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        if cost_usd is None:
            # Unknown pricing (self-hosted/uncommon provider) - don't crash, just
            # stop enforcing the cost ceiling for this run; tool-call-count and
            # wall-clock guardrails remain the backstop.
            self.cost_unknown_warned = True
            return
        self.total_cost_usd += cost_usd

    def record_tool_call(self) -> None:
        self.total_tool_calls += 1

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at
