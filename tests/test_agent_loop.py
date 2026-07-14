from typing import Any

import pytest
from pydantic import BaseModel

from autodoc_harness.agent_loop import (
    LlmCall,
    LlmTurn,
    StructuredCall,
    StructuredTurn,
    ToolCallRequest,
    run_agentic_loop,
)
from autodoc_harness.budget import BudgetExceededError, BudgetTracker
from autodoc_harness.config import ResolvedModelConfig
from autodoc_harness.tools.base import ToolSpec

MODEL_CONFIG = ResolvedModelConfig(
    name="test-model", api_key_env="TEST_KEY", temperature=0.0, max_tokens=100, timeout=30.0
)


class DemoResult(BaseModel):
    answer: str


async def _echo_handler(args: dict[str, Any]) -> str:
    return f"echo: {args.get('text')}"


ECHO_TOOL = ToolSpec(
    name="echo",
    description="Echoes back text.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    handler=_echo_handler,
)


def make_scripted_llm_call(turns: list[LlmTurn]) -> LlmCall:
    turns_iter = iter(turns)

    async def _call(**_kwargs: Any) -> LlmTurn:
        try:
            return next(turns_iter)
        except StopIteration:
            raise AssertionError("scripted llm_call ran out of turns") from None

    return _call


def make_scripted_structured_call(contents: list[str | None]) -> StructuredCall:
    contents_iter = iter(contents)

    async def _call(**_kwargs: Any) -> StructuredTurn:
        try:
            content = next(contents_iter)
        except StopIteration:
            raise AssertionError("scripted structured_call ran out of turns") from None
        return StructuredTurn(content=content, cost_usd=0.01, tokens_in=10, tokens_out=5)

    return _call


def _new_budget() -> BudgetTracker:
    return BudgetTracker(max_total_cost_usd=100.0, max_run_seconds=600)


async def test_extracts_once_model_stops_calling_tools() -> None:
    turns = [
        LlmTurn(content="I'm done.", tool_calls=[], cost_usd=0.01, tokens_in=10, tokens_out=5),
    ]
    result = await run_agentic_loop(
        system_prompt="sys",
        user_prompt="user",
        tools=[ECHO_TOOL],
        result_model=DemoResult,
        model_config=MODEL_CONFIG,
        budget=_new_budget(),
        max_tool_calls=10,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call(['{"answer": "hi"}']),
    )
    assert result.status == "ok"
    assert isinstance(result.data, DemoResult)
    assert result.data.answer == "hi"
    assert result.tool_calls_made == 0


async def test_calls_tool_then_extracts() -> None:
    turns = [
        LlmTurn(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="echo", arguments_json='{"text": "hello"}')],
            cost_usd=0.01,
            tokens_in=10,
            tokens_out=5,
        ),
        LlmTurn(
            content="Done exploring.", tool_calls=[], cost_usd=0.01, tokens_in=10, tokens_out=5
        ),
    ]
    result = await run_agentic_loop(
        system_prompt="sys",
        user_prompt="user",
        tools=[ECHO_TOOL],
        result_model=DemoResult,
        model_config=MODEL_CONFIG,
        budget=_new_budget(),
        max_tool_calls=10,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call(['{"answer": "done"}']),
    )
    assert result.status == "ok"
    assert result.data is not None
    assert result.data.answer == "done"
    assert result.tool_calls_made == 1


async def test_retries_extraction_on_invalid_schema() -> None:
    turns = [
        LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=5, tokens_out=5),
    ]
    result = await run_agentic_loop(
        system_prompt="sys",
        user_prompt="user",
        tools=[],
        result_model=DemoResult,
        model_config=MODEL_CONFIG,
        budget=_new_budget(),
        max_tool_calls=10,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call(
            ['{"wrong_field": "x"}', '{"answer": "fixed"}']
        ),
    )
    assert result.status == "ok"
    assert result.data is not None
    assert result.data.answer == "fixed"


async def test_persistently_invalid_extraction_exhausts_attempts() -> None:
    # Regression case for the old submit-tool design's bug (invalid submissions
    # not counting toward a budget) - here it's explicit: extraction retries are
    # bounded by max_extraction_attempts regardless of how many bad responses the
    # structured_call could theoretically keep returning.
    turns = [LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=1, tokens_out=1)]
    result = await run_agentic_loop(
        system_prompt="sys",
        user_prompt="user",
        tools=[],
        result_model=DemoResult,
        model_config=MODEL_CONFIG,
        budget=_new_budget(),
        max_tool_calls=10,
        max_extraction_attempts=3,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call(['{"wrong_field": "x"}'] * 10),
    )
    assert result.status == "extraction_failed"
    assert result.data is None


async def test_empty_extraction_response_is_treated_as_invalid_and_retried() -> None:
    turns = [LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=1, tokens_out=1)]
    result = await run_agentic_loop(
        system_prompt="sys",
        user_prompt="user",
        tools=[],
        result_model=DemoResult,
        model_config=MODEL_CONFIG,
        budget=_new_budget(),
        max_tool_calls=10,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call([None, '{"answer": "ok"}']),
    )
    assert result.status == "ok"
    assert result.data is not None
    assert result.data.answer == "ok"


async def test_max_tool_calls_exceeded() -> None:
    turns = [
        LlmTurn(
            content=None,
            tool_calls=[ToolCallRequest(id=str(i), name="echo", arguments_json='{"text": "x"}')],
            cost_usd=0.0,
            tokens_in=1,
            tokens_out=1,
        )
        for i in range(10)
    ]
    result = await run_agentic_loop(
        system_prompt="sys",
        user_prompt="user",
        tools=[ECHO_TOOL],
        result_model=DemoResult,
        model_config=MODEL_CONFIG,
        budget=_new_budget(),
        max_tool_calls=3,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call([]),  # should never be reached
    )
    assert result.status == "max_tool_calls_exceeded"
    assert result.data is None


async def test_budget_exceeded_propagates() -> None:
    budget = BudgetTracker(max_total_cost_usd=100.0, max_run_seconds=0)
    with pytest.raises(BudgetExceededError):
        await run_agentic_loop(
            system_prompt="sys",
            user_prompt="user",
            tools=[],
            result_model=DemoResult,
            model_config=MODEL_CONFIG,
            budget=budget,
            max_tool_calls=10,
            llm_call=make_scripted_llm_call([]),
            structured_call=make_scripted_structured_call([]),
        )
