"""Coordinator: orchestrates a full pipeline run. Owns concurrency (bounded via a
`Semaphore`) and partial-failure isolation for the Code Explorer fan-out - one
failed component must not kill the run.

Deliberately does NOT use `asyncio.TaskGroup` for the fan-out: `TaskGroup` cancels
every sibling task the moment one raises, which is the opposite of the tolerance
this pipeline needs. Each Code Explorer call is wrapped in its own try/except
*inside* the gathered coroutine, converting any failure into a
`ComponentNotes(status="failed")` value rather than propagating it, so
`asyncio.gather` runs every component to completion regardless of individual
outcomes.

A hard wall-clock/cost ceiling on the whole run is enforced by the shared
`BudgetTracker` (checked at the top of every agent-loop iteration, across every
concurrently-running stage) rather than an outer `asyncio.wait_for` - the latter
would cancel `asyncio.gather` on timeout and discard whatever components had
already finished, which is worse than letting the shared budget check trip
per-task once it's exhausted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from autodoc_harness.agent_loop import (
    LlmCall,
    StructuredCall,
    default_llm_call,
    default_structured_call,
)
from autodoc_harness.budget import BudgetTracker
from autodoc_harness.config import Config
from autodoc_harness.models import ComponentMap, ComponentNotes, ComponentRef, TraversalStats
from autodoc_harness.stages.code_explorer import run_code_explorer
from autodoc_harness.stages.master_explorer import run_master_explorer


@dataclass
class PipelineRun:
    component_map: ComponentMap
    component_notes: list[ComponentNotes]
    budget: BudgetTracker


async def _run_code_explorer_safe(
    component: ComponentRef,
    config: Config,
    budget: BudgetTracker,
    semaphore: asyncio.Semaphore,
    llm_call: LlmCall,
    structured_call: StructuredCall,
    on_component_done: Callable[[ComponentNotes], None] | None,
) -> ComponentNotes:
    async with semaphore:
        try:
            notes = await run_code_explorer(
                component, config, budget, llm_call=llm_call, structured_call=structured_call
            )
        except Exception as e:
            notes = ComponentNotes(
                component_id=component.component_id,
                name=component.name,
                status="failed",
                error=str(e),
                stats=TraversalStats(
                    tool_calls=0, tokens_in=0, tokens_out=0, cost_usd=0.0, duration_s=0.0
                ),
            )
    if on_component_done is not None:
        on_component_done(notes)
    return notes


async def run_pipeline_through_code_explorer(
    config: Config,
    *,
    llm_call: LlmCall = default_llm_call,
    structured_call: StructuredCall = default_structured_call,
    on_master_explorer_done: Callable[[ComponentMap], None] | None = None,
    on_component_done: Callable[[ComponentNotes], None] | None = None,
) -> PipelineRun:
    """Run Master Explorer, then dispatch a Code Explorer per component in parallel.

    If Master Explorer itself fails (including `BudgetExceededError`), that
    propagates - without a component map there is nothing to dispatch, so this is a
    genuinely fatal condition for the run rather than a per-component partial
    failure.

    `on_master_explorer_done`/`on_component_done` are optional progress-reporting
    hooks (e.g. for a CLI progress bar) - purely observational, they don't affect
    control flow.
    """
    budget = BudgetTracker(
        max_total_cost_usd=config.guardrails.max_total_cost_usd,
        max_run_seconds=config.guardrails.max_run_seconds,
    )

    component_map = await run_master_explorer(
        config, budget, llm_call=llm_call, structured_call=structured_call
    )
    if on_master_explorer_done is not None:
        on_master_explorer_done(component_map)

    semaphore = asyncio.Semaphore(config.guardrails.max_parallel_code_explorers)
    component_notes = await asyncio.gather(
        *(
            _run_code_explorer_safe(
                component, config, budget, semaphore, llm_call, structured_call, on_component_done
            )
            for component in component_map.components
        )
    )

    return PipelineRun(
        component_map=component_map, component_notes=list(component_notes), budget=budget
    )
