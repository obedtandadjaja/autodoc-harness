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
from autodoc_harness.stages.master_explorer import run_master_explorer


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
    """A turn with no tool calls - signals the explore phase is over."""
    return LlmTurn(
        content="I'm done exploring.", tool_calls=[], cost_usd=0.0, tokens_in=5, tokens_out=5
    )


def _extraction_content(
    components: list[dict[str, Any]], unresolved_notes: list[str] | None = None
) -> str:
    return json.dumps({"components": components, "unresolved_notes": unresolved_notes or []})


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


def _new_budget(config: Config) -> BudgetTracker:
    return BudgetTracker(
        max_total_cost_usd=config.guardrails.max_total_cost_usd,
        max_run_seconds=config.guardrails.max_run_seconds,
    )


async def test_master_explorer_returns_component_map(config: Config) -> None:
    budget = _new_budget(config)
    component_map = await run_master_explorer(
        config,
        budget,
        llm_call=make_scripted_llm_call([_done_turn()]),
        structured_call=make_scripted_structured_call(
            [
                _extraction_content(
                    [
                        {
                            "component_id": "main",
                            "name": "Main entrypoint",
                            "summary": "Prints a greeting.",
                            "seed_paths": ["main.py"],
                            "related_component_ids": [],
                            "role": "core",
                        }
                    ]
                )
            ]
        ),
    )

    assert component_map.target_repo == str(config.target_repo)
    assert component_map.entry_points == ["main.py"]
    assert len(component_map.components) == 1
    assert component_map.components[0].component_id == "main"
    assert component_map.model == "anthropic/claude-sonnet-4-5"
    # 5 (explore "done" turn) + 100 (extraction call)
    assert component_map.stats.tokens_in == 105
    assert component_map.stats.cost_usd == pytest.approx(0.01)


async def test_master_explorer_calls_tools_then_extracts(config: Config) -> None:
    turns = [
        LlmTurn(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="list_dir", arguments_json='{"path": "."}')],
            cost_usd=0.0,
            tokens_in=10,
            tokens_out=5,
        ),
        _done_turn(),
    ]
    budget = _new_budget(config)
    component_map = await run_master_explorer(
        config,
        budget,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call([_extraction_content([])]),
    )
    assert component_map.components == []
    assert component_map.stats.tool_calls == 1


async def test_master_explorer_records_failure_as_unresolved_note(config: Config) -> None:
    turns = [
        LlmTurn(
            content=None,
            tool_calls=[
                ToolCallRequest(id=str(i), name="list_dir", arguments_json='{"path": "."}')
            ],
            cost_usd=0.0,
            tokens_in=1,
            tokens_out=1,
        )
        for i in range(config.guardrails.max_tool_calls_master_explorer + 5)
    ]
    budget = _new_budget(config)
    component_map = await run_master_explorer(
        config,
        budget,
        llm_call=make_scripted_llm_call(turns),
        structured_call=make_scripted_structured_call([]),  # never reached
    )
    assert component_map.components == []
    assert any("did not complete" in note for note in component_map.unresolved_notes)


async def test_master_explorer_records_extraction_failure_as_unresolved_note(
    config: Config,
) -> None:
    budget = _new_budget(config)
    component_map = await run_master_explorer(
        config,
        budget,
        llm_call=make_scripted_llm_call([_done_turn()]),
        structured_call=make_scripted_structured_call(
            ['{"totally": "wrong shape"}'] * 3  # exhausts max_extraction_attempts default of 3
        ),
    )
    assert component_map.components == []
    assert any("did not complete" in note for note in component_map.unresolved_notes)


async def test_master_explorer_prompt_includes_description_and_hints(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n")
    (tmp_path / "config.py").write_text("SETTING = 1\n")
    config = Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "description": "A CLI tool that syncs files to S3.",
            "entry_points": [{"path": "main.py", "note": "CLI entry point"}],
            "hints": [{"path": "config.py", "note": "Runtime configuration"}],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    budget = _new_budget(config)
    captured: dict[str, Any] = {}

    async def capturing_llm_call(**kwargs: Any) -> LlmTurn:
        captured["messages"] = kwargs["messages"]
        return _done_turn()

    await run_master_explorer(
        config,
        budget,
        llm_call=capturing_llm_call,
        structured_call=make_scripted_structured_call([_extraction_content([])]),
    )

    user_prompt = captured["messages"][1]["content"]
    assert "A CLI tool that syncs files to S3." in user_prompt
    assert "main.py" in user_prompt and "CLI entry point" in user_prompt
    assert "config.py" in user_prompt and "Runtime configuration" in user_prompt


async def test_master_explorer_propagates_budget_exceeded(config: Config) -> None:
    budget = BudgetTracker(max_total_cost_usd=100.0, max_run_seconds=0)
    with pytest.raises(BudgetExceededError):
        await run_master_explorer(
            config,
            budget,
            llm_call=make_scripted_llm_call([]),
            structured_call=make_scripted_structured_call([]),
        )
