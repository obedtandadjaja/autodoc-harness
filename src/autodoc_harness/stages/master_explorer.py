"""Master Explorer: high-level, model-directed traversal from entry points to a
component map. Deliberately shallow - a separate Code Explorer stage (per component)
does the deep dive."""

from __future__ import annotations

from datetime import UTC, datetime

from autodoc_harness.agent_loop import (
    LlmCall,
    StructuredCall,
    default_llm_call,
    default_structured_call,
    run_agentic_loop,
)
from autodoc_harness.budget import BudgetTracker
from autodoc_harness.config import Config
from autodoc_harness.models import ComponentMap, ComponentMapSubmission, TraversalStats
from autodoc_harness.tools.base import RepoBoundary, ToolSpec
from autodoc_harness.tools.grep_search import make_grep_search_tool
from autodoc_harness.tools.list_dir import make_list_dir_tool
from autodoc_harness.tools.read_file import make_read_file_tool

SYSTEM_PROMPT = """\
You are the Master Explorer stage of a documentation-generation pipeline. Your job \
is a HIGH-LEVEL traversal only: identify the major components/modules reachable \
from the given entry point files and how they relate to each other. Do not dive \
into implementation detail - a separate Code Explorer stage will do that for each \
component you identify.

Rules:
- Use `list_dir`, `read_file`, and `grep_search` to explore the repository, \
  starting from the given entry points. All paths are relative to the target \
  repository root.
- Stop at the repository boundary: if a reference points to a third-party/installed \
  dependency, note it by name only - never assume details about code you have not \
  actually read via these tools.
- For each major component you find, you will need to record: a short kebab-case \
  `component_id`, a `name`, a `summary`, the `seed_paths` where you found it (this \
  becomes the Code Explorer's starting point - be precise), `related_component_ids`, \
  and a `role`.
- If something is ambiguous, or you run out of budget before fully mapping the \
  repository, you will need to note it in `unresolved_notes` rather than guessing.
- When you have explored enough to answer, stop calling tools - simply state that \
  you are done. You will then be asked to provide your complete findings in a \
  structured format.\
"""


def _build_tools(boundary: RepoBoundary, max_file_bytes: int) -> list[ToolSpec]:
    return [
        make_read_file_tool(boundary, max_file_bytes),
        make_list_dir_tool(boundary),
        make_grep_search_tool(boundary),
    ]


def _user_prompt(config: Config) -> str:
    entry_points = "\n".join(f"- {ep}" for ep in config.entry_points)
    return (
        f"Target repository root (use paths relative to this root in every tool "
        f"call): {config.target_repo}\n\n"
        f"Entry points to start traversal from:\n{entry_points}\n\n"
        "Explore from these entry points and build the component map."
    )


async def run_master_explorer(
    config: Config,
    budget: BudgetTracker,
    *,
    llm_call: LlmCall = default_llm_call,
    structured_call: StructuredCall = default_structured_call,
) -> ComponentMap:
    """Run the Master Explorer stage, returning a `ComponentMap`.

    Raises `BudgetExceededError` (propagated from the shared agent loop) if a
    run-wide guardrail trips - the Coordinator is expected to catch this per-task
    rather than let it kill the whole run.
    """
    boundary = RepoBoundary(
        config.target_repo, config.all_ignore_globs, honor_gitignore=config.honor_gitignore
    )
    tools = _build_tools(boundary, config.guardrails.max_file_bytes)
    model_config = config.model.resolved_for_stage("master_explorer")

    started_at = datetime.now(UTC)
    tool_calls_before = budget.total_tool_calls
    tokens_in_before = budget.total_tokens_in
    tokens_out_before = budget.total_tokens_out
    cost_before = budget.total_cost_usd

    result = await run_agentic_loop(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_user_prompt(config),
        tools=tools,
        result_model=ComponentMapSubmission,
        model_config=model_config,
        budget=budget,
        max_tool_calls=config.guardrails.max_tool_calls_master_explorer,
        llm_call=llm_call,
        structured_call=structured_call,
        max_extraction_attempts=config.guardrails.max_extraction_attempts,
    )

    stats = TraversalStats(
        tool_calls=budget.total_tool_calls - tool_calls_before,
        tokens_in=budget.total_tokens_in - tokens_in_before,
        tokens_out=budget.total_tokens_out - tokens_out_before,
        cost_usd=budget.total_cost_usd - cost_before,
        duration_s=(datetime.now(UTC) - started_at).total_seconds(),
    )

    if result.status != "ok" or result.data is None:
        return ComponentMap(
            target_repo=str(config.target_repo),
            entry_points=config.entry_points,
            components=[],
            unresolved_notes=[
                f"Master Explorer did not complete: {result.status} "
                f"(tool_calls_made={result.tool_calls_made})"
            ],
            generated_at=started_at,
            model=model_config.name,
            stats=stats,
        )

    submission = result.data
    assert isinstance(submission, ComponentMapSubmission)
    return ComponentMap(
        target_repo=str(config.target_repo),
        entry_points=config.entry_points,
        components=submission.components,
        unresolved_notes=submission.unresolved_notes,
        generated_at=started_at,
        model=model_config.name,
        stats=stats,
    )
