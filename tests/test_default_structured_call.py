"""Tests for `default_structured_call`, the one piece of code that actually talks
to litellm for the schema-constrained extraction phase. Every stage test injects a
scripted fake `structured_call` and never exercises this adapter, so it's covered
separately here with a mocked `litellm.acompletion`.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from autodoc_harness.agent_loop import default_structured_call


class DemoResult(BaseModel):
    answer: str


def _fake_response(
    content: str | None, prompt_tokens: int, completion_tokens: int
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


async def test_passes_response_format_and_normalizes_content() -> None:
    response = _fake_response('{"answer": "hi"}', 10, 5)
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)) as mock_acompletion,
        patch("litellm.completion_cost", return_value=0.001),
    ):
        turn = await default_structured_call(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            result_model=DemoResult,
            temperature=0.2,
            max_tokens=100,
            timeout=30.0,
        )

    assert turn.content == '{"answer": "hi"}'
    assert turn.cost_usd == 0.001
    assert turn.tokens_in == 10
    assert turn.tokens_out == 5
    mock_acompletion.assert_awaited_once()
    _, kwargs = mock_acompletion.call_args
    assert kwargs["response_format"] is DemoResult
    assert "tools" not in kwargs
    assert kwargs["num_retries"] == 3


async def test_passes_api_base_through() -> None:
    response = _fake_response('{"answer": "hi"}', 1, 1)
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)) as mock_acompletion,
        patch("litellm.completion_cost", return_value=0.0),
    ):
        await default_structured_call(
            model="ollama_chat/gemma4:12b",
            messages=[],
            result_model=DemoResult,
            temperature=0.2,
            max_tokens=100,
            timeout=30.0,
            api_base="http://localhost:11434",
        )

    _, kwargs = mock_acompletion.call_args
    assert kwargs["api_base"] == "http://localhost:11434"


async def test_unknown_cost_becomes_none_instead_of_crashing() -> None:
    response = _fake_response('{"answer": "hi"}', 1, 1)
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)),
        patch("litellm.completion_cost", side_effect=Exception("no pricing data for this model")),
    ):
        turn = await default_structured_call(
            model="ollama_chat/gemma4:12b",
            messages=[],
            result_model=DemoResult,
            temperature=0.2,
            max_tokens=100,
            timeout=30.0,
        )

    assert turn.cost_usd is None
