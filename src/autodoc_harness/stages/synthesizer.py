"""Synthesizer: turns the Master Explorer's component map and every Code Explorer's
notes into the final narrative markdown doc set.

Unlike the explorer stages, the Synthesizer needs no repo access - it only consumes
already-gathered structured notes - so it makes plain, tool-free LLM calls (reusing
the same `LlmCall`/`LlmTurn` seam as the agentic loop for consistency and
testability) rather than running `run_agentic_loop`. One call per output document
keeps each response trivial to use directly as that document's contents, and keeps
the calls naturally parallelizable later if needed (kept sequential for MVP).
"""

from __future__ import annotations

from dataclasses import dataclass

from autodoc_harness.agent_loop import LlmCall, default_llm_call
from autodoc_harness.budget import BudgetTracker
from autodoc_harness.config import Config, ResolvedModelConfig
from autodoc_harness.models import ComponentMap, ComponentNotes

SYSTEM_PROMPT = """\
You are the Synthesizer stage of a documentation-generation pipeline. You are given \
structured notes gathered by prior exploration stages: a high-level component map \
and, for each component, detailed notes covering how it works, including green \
(happy), yellow (edge-case), and red (error/failure) paths, each grounded with file \
citations.

Write clear, detailed, narrative technical documentation in Markdown from this \
structured input. Rules:
- Do not invent facts beyond what's in the structured input - you have no access to \
  the actual source code at this stage, only these notes.
- Preserve and surface the green/yellow/red path distinctions in your narrative \
  rather than collapsing them into a single description.
- Where the input includes citations, mention the referenced files so a reader can \
  go verify a claim themselves.
- Output raw Markdown only - no diagrams, no code fence wrapping the whole \
  document, just the document body.\
"""

ARCHITECTURE_INSTRUCTIONS = (
    "This is the top-level architecture overview: how the system is organized, "
    "what the major components are, how they relate and interact, and how a "
    "request/call flows from an entry point through the components. Do not restate "
    "implementation details of any single component in depth - that belongs in its "
    "own module doc."
)

API_REFERENCE_INSTRUCTIONS = (
    "This is the API/interface reference: list every public interface (function, "
    "class, method, CLI command, endpoint) documented across all components, "
    "grouped by component, with its signature and description."
)


def _module_instructions(component_id: str) -> str:
    return (
        f"This is the per-module documentation for the '{component_id}' component: "
        "its responsibilities, public interfaces, dependencies, and - most "
        "importantly - a narrative walkthrough of its green (happy), yellow "
        "(edge-case), and red (error/failure) paths, grounded in the citations "
        "provided."
    )


@dataclass(frozen=True)
class SynthesizedDocs:
    architecture_md: str
    api_reference_md: str
    module_docs: dict[str, str]  # component_id -> markdown


def _format_component_map(component_map: ComponentMap) -> str:
    lines = [
        f"Target repo: {component_map.target_repo}",
        f"Entry points: {component_map.entry_points}",
        "",
        "Components:",
    ]
    for c in component_map.components:
        lines.append(f"- {c.component_id} ({c.role}): {c.summary}")
        lines.append(f"  seed_paths: {c.seed_paths}")
        lines.append(f"  related: {c.related_component_ids}")
    if component_map.unresolved_notes:
        lines.append("")
        lines.append("Unresolved notes from traversal:")
        lines.extend(f"- {note}" for note in component_map.unresolved_notes)
    return "\n".join(lines)


def _format_citations(citations: list[object]) -> str:
    parts = []
    for c in citations:
        file_ = getattr(c, "file", None)
        lines_ = getattr(c, "lines", None)
        if file_ is None:
            continue
        parts.append(f"{file_}:{lines_}" if lines_ else str(file_))
    return "; ".join(parts)


def format_component_notes(notes: ComponentNotes) -> str:
    lines = [f"## {notes.name} ({notes.component_id})", f"Status: {notes.status}"]
    if notes.error:
        lines.append(f"Error: {notes.error}")
    lines.append(f"Summary: {notes.summary}")
    if notes.responsibilities:
        lines.append("Responsibilities: " + "; ".join(notes.responsibilities))
    if notes.public_interfaces:
        lines.append("Public interfaces:")
        for iface in notes.public_interfaces:
            cites = _format_citations(list(iface.citations))
            lines.append(
                f"- [{iface.kind}] {iface.name}: {iface.signature_text} - "
                f"{iface.description} (cites: {cites})"
            )
    if notes.external_dependencies:
        lines.append("External dependencies: " + ", ".join(notes.external_dependencies))
    if notes.internal_dependencies:
        lines.append("Internal dependencies: " + ", ".join(notes.internal_dependencies))
    if notes.paths:
        lines.append("Paths:")
        for p in notes.paths:
            cites = _format_citations(list(p.citations))
            lines.append(f"- [{p.kind}] {p.title}: {p.narrative} (cites: {cites})")
    if notes.open_questions:
        lines.append("Open questions: " + "; ".join(notes.open_questions))
    return "\n".join(lines)


async def _synthesize_document(
    *,
    user_prompt: str,
    model_config: ResolvedModelConfig,
    budget: BudgetTracker,
    llm_call: LlmCall,
) -> str:
    budget.check_or_raise()
    turn = await llm_call(
        model=model_config.name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tool_schemas=[],
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout=model_config.timeout,
        api_base=model_config.api_base,
    )
    budget.record_llm_call(
        cost_usd=turn.cost_usd, tokens_in=turn.tokens_in, tokens_out=turn.tokens_out
    )
    return (turn.content or "").strip()


async def run_synthesizer(
    component_map: ComponentMap,
    component_notes: list[ComponentNotes],
    config: Config,
    budget: BudgetTracker,
    *,
    llm_call: LlmCall = default_llm_call,
) -> SynthesizedDocs:
    """Stitch a component map and every component's notes into the final markdown
    doc set. Raises `BudgetExceededError` if the shared run-wide budget is already
    exhausted - callers are expected to handle that the same way as the explorer
    stages.
    """
    model_config = config.model.resolved_for_stage("synthesizer")
    map_context = _format_component_map(component_map)
    all_notes_context = "\n\n".join(format_component_notes(n) for n in component_notes)
    description_prefix = (
        f"System description: {config.description}\n\n" if config.description else ""
    )

    architecture_md = await _synthesize_document(
        user_prompt=(
            f"{description_prefix}"
            f"Write the architecture overview document.\n\n{ARCHITECTURE_INSTRUCTIONS}\n\n"
            f"Component map:\n{map_context}\n\nComponent notes:\n{all_notes_context}"
        ),
        model_config=model_config,
        budget=budget,
        llm_call=llm_call,
    )

    api_reference_md = await _synthesize_document(
        user_prompt=(
            f"{description_prefix}"
            f"Write the API reference document.\n\n{API_REFERENCE_INSTRUCTIONS}\n\n"
            f"Component notes:\n{all_notes_context}"
        ),
        model_config=model_config,
        budget=budget,
        llm_call=llm_call,
    )

    module_docs: dict[str, str] = {}
    for notes in component_notes:
        module_docs[notes.component_id] = await _synthesize_document(
            user_prompt=(
                f"{description_prefix}"
                f"Write the module documentation for '{notes.component_id}'.\n\n"
                f"{_module_instructions(notes.component_id)}\n\n"
                f"Component notes:\n{format_component_notes(notes)}"
            ),
            model_config=model_config,
            budget=budget,
            llm_call=llm_call,
        )

    return SynthesizedDocs(
        architecture_md=architecture_md,
        api_reference_md=api_reference_md,
        module_docs=module_docs,
    )
