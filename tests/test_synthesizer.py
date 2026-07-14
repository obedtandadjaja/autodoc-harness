from datetime import UTC, datetime
from typing import Any

import pytest

from autodoc_harness.agent_loop import LlmCall, LlmTurn
from autodoc_harness.budget import BudgetExceededError, BudgetTracker
from autodoc_harness.config import Config
from autodoc_harness.models import ComponentMap, ComponentNotes, ComponentRef, TraversalStats
from autodoc_harness.stages.synthesizer import run_synthesizer


def make_scripted_llm_call(contents: list[str]) -> LlmCall:
    contents_iter = iter(contents)

    async def _call(**_kwargs: Any) -> LlmTurn:
        try:
            content = next(contents_iter)
        except StopIteration:
            raise AssertionError("scripted llm_call ran out of turns") from None
        return LlmTurn(content=content, tool_calls=[], cost_usd=0.02, tokens_in=50, tokens_out=25)

    return _call


@pytest.fixture
def config(tmp_path: Any) -> Config:
    return Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["main.py"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )


@pytest.fixture
def component_map() -> ComponentMap:
    return ComponentMap(
        target_repo="/repo",
        entry_points=["main.py"],
        components=[
            ComponentRef(
                component_id="alpha",
                name="Alpha",
                summary="Does alpha things.",
                seed_paths=["alpha.py"],
                related_component_ids=[],
                role="core",
            ),
            ComponentRef(
                component_id="beta",
                name="Beta",
                summary="Does beta things.",
                seed_paths=["beta.py"],
                related_component_ids=["alpha"],
                role="core",
            ),
        ],
        unresolved_notes=[],
        generated_at=datetime.now(UTC),
        model="anthropic/claude-sonnet-4-5",
        stats=TraversalStats(tool_calls=1, tokens_in=1, tokens_out=1, cost_usd=0.0, duration_s=0.1),
    )


@pytest.fixture
def component_notes() -> list[ComponentNotes]:
    empty_stats = TraversalStats(
        tool_calls=0, tokens_in=0, tokens_out=0, cost_usd=0.0, duration_s=0.0
    )
    return [
        ComponentNotes(
            component_id="alpha",
            name="Alpha",
            status="ok",
            summary="alpha summary",
            stats=empty_stats,
        ),
        ComponentNotes(
            component_id="beta",
            name="Beta",
            status="ok",
            summary="beta summary",
            stats=empty_stats,
        ),
    ]


def _new_budget(config: Config) -> BudgetTracker:
    return BudgetTracker(
        max_total_cost_usd=config.guardrails.max_total_cost_usd,
        max_run_seconds=config.guardrails.max_run_seconds,
    )


async def test_synthesizer_produces_all_documents(
    config: Config, component_map: ComponentMap, component_notes: list[ComponentNotes]
) -> None:
    contents = ["# Architecture", "# API Reference", "# Alpha module doc", "# Beta module doc"]
    budget = _new_budget(config)
    docs = await run_synthesizer(
        component_map, component_notes, config, budget, llm_call=make_scripted_llm_call(contents)
    )
    assert docs.architecture_md == "# Architecture"
    assert docs.api_reference_md == "# API Reference"
    assert docs.module_docs["alpha"] == "# Alpha module doc"
    assert docs.module_docs["beta"] == "# Beta module doc"


async def test_synthesizer_tracks_budget(
    config: Config, component_map: ComponentMap, component_notes: list[ComponentNotes]
) -> None:
    contents = ["# A", "# B", "# C", "# D"]
    budget = _new_budget(config)
    await run_synthesizer(
        component_map, component_notes, config, budget, llm_call=make_scripted_llm_call(contents)
    )
    assert budget.total_cost_usd == pytest.approx(0.08)
    assert budget.total_tokens_in == 200


async def test_synthesizer_propagates_budget_exceeded(
    config: Config, component_map: ComponentMap, component_notes: list[ComponentNotes]
) -> None:
    budget = BudgetTracker(max_total_cost_usd=100.0, max_run_seconds=0)
    with pytest.raises(BudgetExceededError):
        await run_synthesizer(
            component_map, component_notes, config, budget, llm_call=make_scripted_llm_call([])
        )
