"""Code Explorer: deep dive into a single component, producing structured notes
that cover green/yellow/red paths with citations. One instance runs per component
identified by the Master Explorer, dispatched in parallel by the Coordinator."""

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
from autodoc_harness.models import (
    ComponentNotes,
    ComponentNotesSubmission,
    ComponentRef,
    TraversalStats,
)
from autodoc_harness.tools.base import RepoBoundary, ToolSpec
from autodoc_harness.tools.grep_search import make_grep_search_tool
from autodoc_harness.tools.list_dir import make_list_dir_tool
from autodoc_harness.tools.read_file import make_read_file_tool

SYSTEM_PROMPT = """\
You are the Code Explorer stage of a documentation-generation pipeline. Your job is \
to deep-dive into a SINGLE component and document how it actually works, grounded in \
the real code you read.

Rules:
- Use `list_dir`, `read_file`, and `grep_search` to explore the repository, starting \
  from the component's seed paths. All paths are relative to the target repository \
  root.
- Stop at the repository boundary: if a reference points to a third-party/installed \
  dependency, note it in `external_dependencies` by name only - never assume details \
  about code you have not actually read via these tools.
- You MUST document at least one "green" (happy) path. Also document "yellow" \
  (edge-case/warning) and "red" (error/failure) paths wherever the code actually \
  handles them - do not fabricate a yellow/red path that isn't really there, but do \
  not skip ones that are.
- Every entry in `paths` and `public_interfaces` must include a `citations` entry \
  (file, and a best-effort line range if you can tell) pointing at the code that \
  supports it.
- If something is ambiguous, or you run out of budget before fully understanding the \
  component, you will need to note it in `open_questions` rather than guessing.
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


def _user_prompt(component: ComponentRef) -> str:
    seed_paths = "\n".join(f"- {p}" for p in component.seed_paths)
    return (
        "Component to document:\n"
        f"- component_id: {component.component_id}\n"
        f"- name: {component.name}\n"
        f"- summary (from a prior high-level pass): {component.summary}\n\n"
        f"Seed paths to start from:\n{seed_paths}\n\n"
        "Explore this component in depth and document it."
    )


async def run_code_explorer(
    component: ComponentRef,
    config: Config,
    budget: BudgetTracker,
    *,
    llm_call: LlmCall = default_llm_call,
    structured_call: StructuredCall = default_structured_call,
) -> ComponentNotes:
    """Run the Code Explorer stage for a single component.

    Returns `ComponentNotes` with `status="ok"` or `status="partial"` for an
    incomplete/invalid submission - it never raises for that. Only
    `BudgetExceededError` propagates (from the shared agent loop), for the
    Coordinator to catch per-task.
    """
    boundary = RepoBoundary(
        config.target_repo, config.all_ignore_globs, honor_gitignore=config.honor_gitignore
    )
    tools = _build_tools(boundary, config.guardrails.max_file_bytes)
    model_config = config.model.resolved_for_stage("code_explorer")

    started_at = datetime.now(UTC)
    tool_calls_before = budget.total_tool_calls
    tokens_in_before = budget.total_tokens_in
    tokens_out_before = budget.total_tokens_out
    cost_before = budget.total_cost_usd

    result = await run_agentic_loop(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_user_prompt(component),
        tools=tools,
        result_model=ComponentNotesSubmission,
        model_config=model_config,
        budget=budget,
        max_tool_calls=config.guardrails.max_tool_calls_per_code_explorer,
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

    files_visited = sorted(boundary.visited_files)

    if result.status != "ok" or result.data is None:
        return ComponentNotes(
            component_id=component.component_id,
            name=component.name,
            status="partial",
            error=f"{result.status} (tool_calls_made={result.tool_calls_made})",
            files_visited=files_visited,
            stats=stats,
        )

    submission = result.data
    assert isinstance(submission, ComponentNotesSubmission)
    return ComponentNotes(
        component_id=component.component_id,
        name=component.name,
        status="ok",
        summary=submission.summary,
        responsibilities=submission.responsibilities,
        public_interfaces=submission.public_interfaces,
        external_dependencies=submission.external_dependencies,
        internal_dependencies=submission.internal_dependencies,
        paths=submission.paths,
        open_questions=submission.open_questions,
        files_visited=files_visited,
        stats=stats,
    )
