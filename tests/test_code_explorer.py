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
from autodoc_harness.models import ComponentRef
from autodoc_harness.stages.code_explorer import run_code_explorer


def make_scripted_llm_call(turns: list[LlmTurn]) -> LlmCall:
    turns_iter = iter(turns)

    async def _call(**_kwargs: Any) -> LlmTurn:
        try:
            return next(turns_iter)
        except StopIteration:
            raise AssertionError("scripted llm_call ran out of turns") from None

    return _call


def make_scripted_structured_call(contents: list[str]) -> StructuredCall:
    contents_iter = iter(contents)

    async def _call(**_kwargs: Any) -> StructuredTurn:
        try:
            content = next(contents_iter)
        except StopIteration:
            raise AssertionError("scripted structured_call ran out of turns") from None
        return StructuredTurn(content=content, cost_usd=0.01, tokens_in=100, tokens_out=50)

    return _call


def _done_turn() -> LlmTurn:
    return LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=5, tokens_out=5)


def _extraction_content(**submission_fields: Any) -> str:
    return json.dumps({"summary": "does things", **submission_fields})


@pytest.fixture
def config(tmp_path: Path) -> Config:
    (tmp_path / "main.py").write_text("print('hi')\n")
    return Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["main.py"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )


@pytest.fixture
def component() -> ComponentRef:
    return ComponentRef(
        component_id="main",
        name="Main entrypoint",
        summary="Prints a greeting.",
        seed_paths=["main.py"],
        related_component_ids=[],
        role="core",
    )


def _new_budget(config: Config) -> BudgetTracker:
    return BudgetTracker(
        max_total_cost_usd=config.guardrails.max_total_cost_usd,
        max_run_seconds=config.guardrails.max_run_seconds,
    )


async def test_code_explorer_returns_ok_notes(config: Config, component: ComponentRef) -> None:
    budget = _new_budget(config)
    notes = await run_code_explorer(
        component,
        config,
        budget,
        llm_call=make_scripted_llm_call([_done_turn()]),
        structured_call=make_scripted_structured_call(
            [
                _extraction_content(
                    paths=[
                        {
                            "kind": "green",
                            "title": "Happy path",
                            "narrative": "Prints hi and exits 0.",
                            "citations": [{"file": "main.py", "lines": "1"}],
                        }
                    ]
                )
            ]
        ),
    )
    assert notes.status == "ok"
    assert notes.component_id == "main"
    assert notes.summary == "does things"
    assert len(notes.paths) == 1
    assert notes.paths[0].kind == "green"


async def test_code_explorer_marks_partial_on_incomplete_run(
    config: Config, component: ComponentRef
) -> None:
    turns = [
        LlmTurn(
            content=None,
            tool_calls=[
                ToolCallRequest(id=str(i), name="read_file", arguments_json='{"path": "main.py"}')
            ],
            cost_usd=0.0,
            tokens_in=1,
            tokens_out=1,
        )
        for i in range(config.guardrails.max_tool_calls_per_code_explorer + 5)
    ]
    budget = _new_budget(config)
    notes = await run_code_explorer(
        component,
        config,
        budget,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call([]),  # never reached
    )
    assert notes.status == "partial"
    assert notes.error is not None
    assert "max_tool_calls_exceeded" in notes.error


async def test_code_explorer_marks_partial_on_extraction_failure(
    config: Config, component: ComponentRef
) -> None:
    budget = _new_budget(config)
    notes = await run_code_explorer(
        component,
        config,
        budget,
        llm_call=make_scripted_llm_call([_done_turn()]),
        structured_call=make_scripted_structured_call(['{"wrong": "shape"}'] * 3),
    )
    assert notes.status == "partial"
    assert notes.error is not None
    assert "extraction_failed" in notes.error


async def test_code_explorer_propagates_budget_exceeded(
    config: Config, component: ComponentRef
) -> None:
    budget = BudgetTracker(max_total_cost_usd=100.0, max_run_seconds=0)
    with pytest.raises(BudgetExceededError):
        await run_code_explorer(
            component,
            config,
            budget,
            llm_call=make_scripted_llm_call([]),
            structured_call=make_scripted_structured_call([]),
        )
