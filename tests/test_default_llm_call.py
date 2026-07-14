"""Tests for `default_llm_call`, the one piece of code that actually talks to
litellm. Every other test injects a scripted fake `llm_call` and never exercises
this adapter, so it's covered separately here with a mocked `litellm.acompletion`.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from autodoc_harness.agent_loop import default_llm_call


def _fake_response(
    content: str | None,
    tool_calls: list[Any] | None,
    prompt_tokens: int,
    completion_tokens: int,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


async def test_normalizes_plain_content_response() -> None:
    response = _fake_response("hello", None, 10, 5)
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)) as mock_acompletion,
        patch("litellm.completion_cost", return_value=0.001),
    ):
        turn = await default_llm_call(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            temperature=0.2,
            max_tokens=100,
            timeout=30.0,
        )

    assert turn.content == "hello"
    assert turn.tool_calls == []
    assert turn.cost_usd == 0.001
    assert turn.tokens_in == 10
    assert turn.tokens_out == 5
    mock_acompletion.assert_awaited_once()
    _, kwargs = mock_acompletion.call_args
    assert kwargs["model"] == "anthropic/claude-sonnet-4-5"
    assert kwargs["num_retries"] == 3


async def test_normalizes_tool_calls() -> None:
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="read_file", arguments='{"path": "a.py"}')
    )
    response = _fake_response(None, [tool_call], 20, 10)
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)),
        patch("litellm.completion_cost", return_value=0.002),
    ):
        turn = await default_llm_call(
            model="anthropic/claude-sonnet-4-5",
            messages=[],
            tool_schemas=[{"type": "function", "function": {"name": "read_file"}}],
            temperature=0.2,
            max_tokens=100,
            timeout=30.0,
        )

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments_json == '{"path": "a.py"}'


async def test_missing_arguments_defaults_to_empty_object() -> None:
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="list_dir", arguments=None)
    )
    response = _fake_response(None, [tool_call], 1, 1)
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)),
        patch("litellm.completion_cost", return_value=0.0),
    ):
        turn = await default_llm_call(
            model="anthropic/claude-sonnet-4-5",
            messages=[],
            tool_schemas=[],
            temperature=0.2,
            max_tokens=100,
            timeout=30.0,
        )

    assert turn.tool_calls[0].arguments_json == "{}"


async def test_unknown_cost_becomes_none_instead_of_crashing() -> None:
    response = _fake_response("hi", None, 1, 1)
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)),
        patch("litellm.completion_cost", side_effect=Exception("no pricing data for this model")),
    ):
        turn = await default_llm_call(
            model="ollama/llama3",
            messages=[],
            tool_schemas=[],
            temperature=0.2,
            max_tokens=100,
            timeout=30.0,
        )

    assert turn.cost_usd is None
