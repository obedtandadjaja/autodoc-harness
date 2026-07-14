"""The model-directed agentic loop shared by Master Explorer, Code Explorer, and
Reviewer. The three stages differ only in prompt/tools/result-model/limits (data,
not behavior), so they all call `run_agentic_loop()` directly rather than going
through a class hierarchy.

Two phases, deliberately kept separate:

1. Explore - a normal tool-calling loop (`read_file`/`list_dir`/`grep_search`).
   Continues as long as the model keeps calling tools.
2. Extract - once a turn comes back with no tool calls (the model is done
   exploring), a single dedicated completion call asks for the structured result
   via `response_format=<pydantic model>` instead of a "submit" tool call.

Earlier this used a "submit tool" whose arguments were the structured payload,
alongside the explore tools in the same request. That put structured-output
reliability entirely on the model correctly filling in a tool's JSON Schema
`parameters` from scratch. Providers' native grammar-constrained/structured-output
support (OpenAI, Anthropic, and - critically for local models - Ollama's
token-masking constrained decoding) is built around `response_format`, not around
an ad-hoc tool schema, so a real model test surfaced smaller/local models
inventing their own field names for a same-batch submit tool despite reasoning
about the underlying content correctly. Splitting extraction into its own
non-tool call lets the provider's schema-constrained decoding actually apply.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import litellm
import pydantic
from pydantic import BaseModel

from autodoc_harness.budget import BudgetTracker
from autodoc_harness.config import ResolvedModelConfig
from autodoc_harness.tools.base import ToolSpec


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class LlmTurn:
    """Normalized shape of a single tool-calling-loop model turn.

    This is the seam that makes the loop testable without litellm-specific mocking:
    tests construct `LlmTurn` objects directly and hand them back from a scripted
    `llm_call`, instead of needing to fabricate litellm response objects.
    """

    content: str | None
    tool_calls: list[ToolCallRequest]
    cost_usd: float | None
    tokens_in: int
    tokens_out: int


LlmCall = Callable[..., Awaitable[LlmTurn]]


async def default_llm_call(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    api_base: str | None = None,
) -> LlmTurn:
    """Default `llm_call`: wraps `litellm.acompletion` and normalizes its response
    into an `LlmTurn`. This is the only place in the codebase that touches litellm's
    response shape directly."""
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        tools=tool_schemas,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        api_base=api_base,
        num_retries=3,
    )
    message = response.choices[0].message
    tool_calls = [
        ToolCallRequest(
            id=tc.id, name=tc.function.name, arguments_json=tc.function.arguments or "{}"
        )
        for tc in (message.tool_calls or [])
    ]
    usage = response.usage
    try:
        cost: float | None = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None
    return LlmTurn(
        content=message.content,
        tool_calls=tool_calls,
        cost_usd=cost,
        tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
        tokens_out=getattr(usage, "completion_tokens", 0) or 0,
    )


@dataclass(frozen=True)
class StructuredTurn:
    """Normalized shape of a single schema-constrained (non-tool) completion call.

    Kept separate from `LlmTurn` (rather than reusing it) since this call never
    has tool_calls - it's a plain-content response that's expected to be JSON text
    matching a schema.
    """

    content: str | None
    cost_usd: float | None
    tokens_in: int
    tokens_out: int


StructuredCall = Callable[..., Awaitable[StructuredTurn]]


async def default_structured_call(
    *,
    model: str,
    messages: list[dict[str, Any]],
    result_model: type[BaseModel],
    temperature: float,
    max_tokens: int,
    timeout: float,
    api_base: str | None = None,
) -> StructuredTurn:
    """Default `structured_call`: wraps `litellm.acompletion` with
    `response_format=result_model` for schema-constrained structured output.

    litellm accepts a pydantic model class directly as `response_format` and
    handles the provider-specific schema translation (OpenAI/Anthropic structured
    outputs, Ollama's grammar-constrained JSON, etc.). The response is still raw
    JSON text in `message.content` - not all providers guarantee strict schema
    compliance, so callers still validate/retry rather than trusting this blindly.
    """
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        response_format=result_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        api_base=api_base,
        num_retries=3,
    )
    message = response.choices[0].message
    usage = response.usage
    try:
        cost: float | None = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None
    return StructuredTurn(
        content=message.content,
        cost_usd=cost,
        tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
        tokens_out=getattr(usage, "completion_tokens", 0) or 0,
    )


@dataclass(frozen=True)
class AgentLoopResult:
    status: Literal["ok", "max_tool_calls_exceeded", "extraction_failed"]
    data: BaseModel | None
    tool_calls_made: int


EXTRACTION_PROMPT = "Now provide your complete findings in the required structured format."


async def run_agentic_loop(
    *,
    system_prompt: str,
    user_prompt: str,
    tools: list[ToolSpec],
    result_model: type[BaseModel],
    model_config: ResolvedModelConfig,
    budget: BudgetTracker,
    max_tool_calls: int,
    llm_call: LlmCall = default_llm_call,
    structured_call: StructuredCall = default_structured_call,
    max_extraction_attempts: int = 3,
) -> AgentLoopResult:
    """Run a model-directed explore loop, then a schema-constrained extraction call.

    Explore: the model calls `tools` freely until a turn comes back with none,
    which signals it's done. Bounded by `max_tool_calls`.

    Extract: a dedicated `response_format=result_model` call over the full explore
    transcript. Retries (bounded by `max_extraction_attempts`) on schema-validation
    failure - some providers only offer best-effort JSON mode, not hard grammar
    constraints, so this stays a safety net even though extraction is expected to
    succeed far more often than the old submit-tool-call approach did.

    Raises `BudgetExceededError` (propagated from `budget.check_or_raise()`) if a
    run-wide guardrail trips - callers (the Coordinator) are expected to catch this
    per-task rather than let it kill the whole run.
    """
    tools_by_name = {t.name: t for t in tools}
    tool_schemas = [t.openai_schema for t in tools]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tool_calls_made = 0

    while True:
        budget.check_or_raise()

        turn = await llm_call(
            model=model_config.name,
            messages=messages,
            tool_schemas=tool_schemas,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout,
            api_base=model_config.api_base,
        )
        budget.record_llm_call(
            cost_usd=turn.cost_usd, tokens_in=turn.tokens_in, tokens_out=turn.tokens_out
        )

        assistant_message: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for tc in turn.tool_calls
            ]
        messages.append(assistant_message)

        if not turn.tool_calls:
            break  # model is done exploring - move to the extraction phase

        for tool_call in turn.tool_calls:
            tool_calls_made += 1
            budget.record_tool_call()
            tool = tools_by_name.get(tool_call.name)
            if tool is None:
                result_text = f"error: unknown tool '{tool_call.name}'"
            else:
                try:
                    tool_args = json.loads(tool_call.arguments_json)
                except json.JSONDecodeError as e:
                    result_text = f"error: could not parse tool arguments as JSON: {e}"
                else:
                    result_text = await tool.handler(tool_args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})

        if tool_calls_made > max_tool_calls:
            return AgentLoopResult(
                status="max_tool_calls_exceeded", data=None, tool_calls_made=tool_calls_made
            )

    messages.append({"role": "user", "content": EXTRACTION_PROMPT})

    for _ in range(max_extraction_attempts):
        budget.check_or_raise()

        structured_turn = await structured_call(
            model=model_config.name,
            messages=messages,
            result_model=result_model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout,
            api_base=model_config.api_base,
        )
        budget.record_llm_call(
            cost_usd=structured_turn.cost_usd,
            tokens_in=structured_turn.tokens_in,
            tokens_out=structured_turn.tokens_out,
        )

        content = structured_turn.content or ""
        try:
            data = result_model.model_validate_json(content)
        except pydantic.ValidationError as e:
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your response didn't match the required schema: {e}. "
                        "Please provide the corrected structured findings."
                    ),
                }
            )
            continue

        return AgentLoopResult(status="ok", data=data, tool_calls_made=tool_calls_made)

    return AgentLoopResult(status="extraction_failed", data=None, tool_calls_made=tool_calls_made)
