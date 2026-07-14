import json
from pathlib import Path
from typing import Any

import pytest

from autodoc_harness.agent_loop import (
    LlmCall,
    LlmTurn,
    StructuredCall,
    StructuredTurn,
    ToolCallRequest,
)
from autodoc_harness.budget import BudgetExceededError, BudgetTracker
from autodoc_harness.config import Config
from autodoc_harness.models import ComponentNotes, TraversalStats
from autodoc_harness.stages.reviewer import run_reviewer
from autodoc_harness.stages.synthesizer import SynthesizedDocs


def make_scripted_structured_call(contents: list[str]) -> StructuredCall:
    contents_iter = iter(contents)

    async def _call(**_kwargs: Any) -> StructuredTurn:
        try:
            content = next(contents_iter)
        except StopIteration:
            raise AssertionError("scripted structured_call ran out of turns") from None
        return StructuredTurn(content=content, cost_usd=0.01, tokens_in=50, tokens_out=25)

    return _call


def make_review_llm_call(corrections: list[str]) -> LlmCall:
    """Explore calls (tool_schemas non-empty) always signal "done" immediately -
    findings extraction is handled separately by structured_call, and actual tool
    dispatch is covered by the explorer-stage tests. Correction calls
    (tool_schemas empty, only made when findings are non-empty) return the next
    scripted corrected-content string, in order."""
    corrections_iter = iter(corrections)

    async def _call(**kwargs: Any) -> LlmTurn:
        if kwargs.get("tool_schemas"):
            return LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=5, tokens_out=5)
        try:
            content = next(corrections_iter)
        except StopIteration:
            raise AssertionError("scripted llm_call ran out of corrections") from None
        return LlmTurn(content=content, tool_calls=[], cost_usd=0.01, tokens_in=20, tokens_out=10)

    return _call


@pytest.fixture
def config(tmp_path: Path) -> Config:
    (tmp_path / "main.py").write_text("def greet():\n    print('hi')\n")
    return Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["main.py"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )


@pytest.fixture
def docs() -> SynthesizedDocs:
    return SynthesizedDocs(
        architecture_md="# Architecture\n\nThe system prints a greeting.",
        api_reference_md="# API Reference\n\n`greet()` - prints a greeting.",
        module_docs={"main": "# Main\n\nCalls greet(), which fabricates a network call."},
    )


@pytest.fixture
def component_notes() -> list[ComponentNotes]:
    stats = TraversalStats(tool_calls=0, tokens_in=0, tokens_out=0, cost_usd=0.0, duration_s=0.0)
    return [
        ComponentNotes(
            component_id="main",
            name="Main",
            status="ok",
            summary="Prints a greeting.",
            stats=stats,
        )
    ]


def _new_budget(config: Config) -> BudgetTracker:
    return BudgetTracker(
        max_total_cost_usd=config.guardrails.max_total_cost_usd,
        max_run_seconds=config.guardrails.max_run_seconds,
    )


async def test_reviewer_reviews_all_three_documents(
    config: Config, docs: SynthesizedDocs, component_notes: list[ComponentNotes]
) -> None:
    # architecture.md and api-reference.md have no findings -> no correction call,
    # original content passed through unchanged. main's doc has one finding ->
    # triggers a plain-text correction call.
    structured_contents = [
        json.dumps({"findings": []}),
        json.dumps({"findings": []}),
        json.dumps(
            {
                "findings": [
                    {
                        "category": "hallucination",
                        "description": "No network call exists in main.py.",
                        "location": "Main",
                    }
                ]
            }
        ),
    ]
    budget = _new_budget(config)
    reviewed = await run_reviewer(
        component_notes,
        docs,
        config,
        budget,
        llm_call=make_review_llm_call(["# Main\n\nCalls greet(), which prints a greeting."]),
        structured_call=make_scripted_structured_call(structured_contents),
    )

    assert reviewed.architecture_md == docs.architecture_md
    assert reviewed.api_reference_md == docs.api_reference_md
    assert reviewed.module_docs["main"] == "# Main\n\nCalls greet(), which prints a greeting."

    assert reviewed.report.findings_by_document["architecture.md"] == []
    assert reviewed.report.findings_by_document["api-reference.md"] == []
    main_findings = reviewed.report.findings_by_document["modules/main.md"]
    assert len(main_findings) == 1
    assert main_findings[0].category == "hallucination"


async def test_reviewer_skips_correction_call_when_no_findings(
    config: Config, docs: SynthesizedDocs, component_notes: list[ComponentNotes]
) -> None:
    async def llm_call_no_corrections_allowed(**kwargs: Any) -> LlmTurn:
        if not kwargs.get("tool_schemas"):
            raise AssertionError("correction call should have been skipped - no findings")
        return LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=5, tokens_out=5)

    structured_contents = [json.dumps({"findings": []})] * 3
    budget = _new_budget(config)
    reviewed = await run_reviewer(
        component_notes,
        docs,
        config,
        budget,
        llm_call=llm_call_no_corrections_allowed,
        structured_call=make_scripted_structured_call(structured_contents),
    )
    assert reviewed.architecture_md == docs.architecture_md
    assert reviewed.api_reference_md == docs.api_reference_md
    assert reviewed.module_docs["main"] == docs.module_docs["main"]


async def test_reviewer_keeps_original_content_on_incomplete_review(
    config: Config, docs: SynthesizedDocs, component_notes: list[ComponentNotes]
) -> None:
    # architecture.md and api-reference.md complete normally; main's explore phase
    # never stops calling tools, exhausting max_tool_calls_reviewer_per_doc before
    # it ever reaches extraction.
    tool_call_count = {"n": 0}

    async def flaky_llm_call(**kwargs: Any) -> LlmTurn:
        if not kwargs.get("tool_schemas"):
            raise AssertionError("no correction call should happen for this test")
        prompt = str(kwargs["messages"][1]["content"])
        if "modules/main.md" not in prompt:
            return LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=5, tokens_out=5)
        tool_call_count["n"] += 1
        return LlmTurn(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id=str(tool_call_count["n"]),
                    name="read_file",
                    arguments_json='{"path": "main.py"}',
                )
            ],
            cost_usd=0.0,
            tokens_in=1,
            tokens_out=1,
        )

    structured_contents = [
        json.dumps({"findings": []}),
        json.dumps({"findings": []}),
    ]
    budget = _new_budget(config)
    reviewed = await run_reviewer(
        component_notes,
        docs,
        config,
        budget,
        llm_call=flaky_llm_call,
        structured_call=make_scripted_structured_call(structured_contents),
    )

    assert reviewed.module_docs["main"] == docs.module_docs["main"]
    findings = reviewed.report.findings_by_document["modules/main.md"]
    assert any("did not complete" in f.description for f in findings)


async def test_reviewer_keeps_original_content_on_extraction_failure(
    config: Config, docs: SynthesizedDocs, component_notes: list[ComponentNotes]
) -> None:
    async def done_llm_call(**kwargs: Any) -> LlmTurn:
        if not kwargs.get("tool_schemas"):
            raise AssertionError("no correction call should happen for this test")
        return LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=5, tokens_out=5)

    structured_contents = [
        json.dumps({"findings": []}),
        json.dumps({"findings": []}),
        # "findings" present but wrong type - `findings` defaults to [] only when
        # *absent*, so a wrong-type value here genuinely fails schema validation,
        # exhausting max_extraction_attempts (default 3) for main.md.
        '{"findings": "not a list"}',
        '{"findings": "not a list"}',
        '{"findings": "not a list"}',
    ]
    budget = _new_budget(config)
    reviewed = await run_reviewer(
        component_notes,
        docs,
        config,
        budget,
        llm_call=done_llm_call,
        structured_call=make_scripted_structured_call(structured_contents),
    )

    assert reviewed.module_docs["main"] == docs.module_docs["main"]
    findings = reviewed.report.findings_by_document["modules/main.md"]
    assert any("did not complete" in f.description for f in findings)


async def test_reviewer_propagates_budget_exceeded(
    config: Config, docs: SynthesizedDocs, component_notes: list[ComponentNotes]
) -> None:
    budget = BudgetTracker(max_total_cost_usd=100.0, max_run_seconds=0)

    async def unreachable_llm_call(**_kwargs: Any) -> LlmTurn:
        raise AssertionError("should not be called - budget check happens first")

    with pytest.raises(BudgetExceededError):
        await run_reviewer(
            component_notes,
            docs,
            config,
            budget,
            llm_call=unreachable_llm_call,
            structured_call=make_scripted_structured_call([]),
        )
