import pytest

from autodoc_harness.budget import BudgetExceededError, BudgetTracker


def test_records_cost_and_tokens() -> None:
    tracker = BudgetTracker(max_total_cost_usd=10.0, max_run_seconds=60)
    tracker.record_llm_call(cost_usd=0.5, tokens_in=100, tokens_out=50)
    tracker.record_llm_call(cost_usd=0.25, tokens_in=10, tokens_out=5)
    assert tracker.total_cost_usd == pytest.approx(0.75)
    assert tracker.total_tokens_in == 110
    assert tracker.total_tokens_out == 55


def test_unknown_cost_does_not_crash_or_raise_cost_ceiling() -> None:
    tracker = BudgetTracker(max_total_cost_usd=1.0, max_run_seconds=60)
    tracker.record_llm_call(cost_usd=None, tokens_in=100, tokens_out=50)
    assert tracker.cost_unknown_warned is True
    assert tracker.total_cost_usd == 0.0
    tracker.check_or_raise()


def test_check_or_raise_trips_on_cost_ceiling() -> None:
    tracker = BudgetTracker(max_total_cost_usd=1.0, max_run_seconds=60)
    tracker.record_llm_call(cost_usd=2.0, tokens_in=0, tokens_out=0)
    with pytest.raises(BudgetExceededError, match="max_total_cost_usd"):
        tracker.check_or_raise()


def test_check_or_raise_trips_on_time_ceiling() -> None:
    tracker = BudgetTracker(max_total_cost_usd=100.0, max_run_seconds=0)
    with pytest.raises(BudgetExceededError, match="max_run_seconds"):
        tracker.check_or_raise()


def test_record_tool_call_increments_total() -> None:
    tracker = BudgetTracker(max_total_cost_usd=10.0, max_run_seconds=60)
    tracker.record_tool_call()
    tracker.record_tool_call()
    assert tracker.total_tool_calls == 2
