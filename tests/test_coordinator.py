import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from autodoc_harness.agent_loop import LlmCall, LlmTurn, StructuredCall, StructuredTurn
from autodoc_harness.config import Config
from autodoc_harness.coordinator import run_pipeline_through_code_explorer


def _component_ref(component_id: str) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "name": component_id,
        "summary": "a component",
        "seed_paths": ["main.py"],
        "related_component_ids": [],
        "role": "core",
    }


def _user_prompt(kwargs: dict[str, Any]) -> str:
    return str(kwargs["messages"][1]["content"])


def make_pipeline_llm_call(outcomes: dict[str, Exception]) -> LlmCall:
    """Scripted explore-phase `llm_call` shared across Master Explorer and every
    Code Explorer (the Coordinator shares one `llm_call` across every stage/task in
    a run). Master Explorer's prompt never mentions a component_id, so it always
    just signals "done" immediately. A Code Explorer's prompt does mention its
    component_id - if that component has an Exception outcome, raise it here to
    simulate a crash mid-exploration; otherwise also signal "done"."""

    async def _call(**kwargs: Any) -> LlmTurn:
        prompt = _user_prompt(kwargs)
        for component_id, outcome in outcomes.items():
            if f"component_id: {component_id}" in prompt:
                raise outcome
        return LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=10, tokens_out=5)

    return _call


def make_pipeline_structured_call(
    component_ids: list[str], extraction_outcomes: dict[str, dict[str, Any]]
) -> StructuredCall:
    """Scripted extraction-phase `structured_call`. Branches the same way: no
    component_id in the prompt means this is Master Explorer's extraction."""

    async def _call(**kwargs: Any) -> StructuredTurn:
        prompt = _user_prompt(kwargs)
        for component_id, outcome in extraction_outcomes.items():
            if f"component_id: {component_id}" in prompt:
                payload = {"summary": "does things", **outcome}
                return StructuredTurn(
                    content=json.dumps(payload), cost_usd=0.0, tokens_in=10, tokens_out=5
                )
        payload = {
            "components": [_component_ref(cid) for cid in component_ids],
            "unresolved_notes": [],
        }
        return StructuredTurn(content=json.dumps(payload), cost_usd=0.0, tokens_in=10, tokens_out=5)

    return _call


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


async def test_broken_component_isolated_siblings_succeed(config: Config) -> None:
    component_ids = ["good-a", "broken", "good-b"]
    explore_outcomes = {"broken": RuntimeError("simulated crash exploring this component")}
    extraction_outcomes: dict[str, dict[str, Any]] = {"good-a": {}, "good-b": {}}

    run = await run_pipeline_through_code_explorer(
        config,
        llm_call=make_pipeline_llm_call(explore_outcomes),
        structured_call=make_pipeline_structured_call(component_ids, extraction_outcomes),
    )

    by_id = {n.component_id: n for n in run.component_notes}
    assert by_id["good-a"].status == "ok"
    assert by_id["good-b"].status == "ok"
    assert by_id["broken"].status == "failed"
    assert "simulated crash" in (by_id["broken"].error or "")


async def test_respects_max_parallel_code_explorers(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n")
    config = Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["main.py"],
            "model": {
                "name": "anthropic/claude-sonnet-4-5",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
            "guardrails": {"max_parallel_code_explorers": 2},
        }
    )

    component_ids = [f"c{i}" for i in range(6)]
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fast_llm_call(**_kwargs: Any) -> LlmTurn:
        return LlmTurn(content="done", tool_calls=[], cost_usd=0.0, tokens_in=1, tokens_out=1)

    async def slow_structured_call(**kwargs: Any) -> StructuredTurn:
        nonlocal concurrent, max_concurrent
        prompt = _user_prompt(kwargs)
        if "component_id:" not in prompt:
            # Master Explorer's extraction - not part of the concurrency being measured.
            payload = {
                "components": [_component_ref(cid) for cid in component_ids],
                "unresolved_notes": [],
            }
            return StructuredTurn(
                content=json.dumps(payload), cost_usd=0.0, tokens_in=1, tokens_out=1
            )

        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.05)
        async with lock:
            concurrent -= 1

        payload = {"summary": "does things"}
        return StructuredTurn(content=json.dumps(payload), cost_usd=0.0, tokens_in=1, tokens_out=1)

    run = await run_pipeline_through_code_explorer(
        config, llm_call=fast_llm_call, structured_call=slow_structured_call
    )

    assert max_concurrent <= 2
    assert all(n.status == "ok" for n in run.component_notes)
    assert {n.component_id for n in run.component_notes} == set(component_ids)
