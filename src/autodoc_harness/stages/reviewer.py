"""Reviewer: the final quality gate before documentation is written out.

Reviews each synthesized document against the component notes it was based on,
using the same read-only repo tools as the explorer stages to fact-check citations
against real source. Runs one review pass per document - mirroring the
Synthesizer's one-call-per-document structure - rather than one giant pass over the
whole doc set, so a single response never has to carry every document's corrected
content in one payload.

Each review pass is itself two steps:

1. Explore + extract `findings` (via the shared agentic loop, schema-constrained).
2. If there were findings, a plain-text "apply the findings" call - no JSON, no
   tools - to produce the corrected document, the same way the Synthesizer
   produces markdown.

Real-model testing showed that asking for the corrected document as a JSON string
field (escaping a large markdown block: quotes, backslashes, newlines) was a much
harder generation task than the small structured `findings` list, and was where
most review passes exhausted their extraction retries - even though the model
reasoned about the content correctly. Splitting it into a separate plain-text call
removes the escaping burden entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from autodoc_harness.agent_loop import (
    LlmCall,
    StructuredCall,
    default_llm_call,
    default_structured_call,
    run_agentic_loop,
)
from autodoc_harness.budget import BudgetTracker
from autodoc_harness.config import Config, ResolvedModelConfig
from autodoc_harness.models import (
    ComponentNotes,
    ReviewFinding,
    ReviewFindingsSubmission,
    ReviewReport,
    TraversalStats,
)
from autodoc_harness.stages.synthesizer import SynthesizedDocs, format_component_notes
from autodoc_harness.tools.base import RepoBoundary, ToolSpec
from autodoc_harness.tools.grep_search import make_grep_search_tool
from autodoc_harness.tools.list_dir import make_list_dir_tool
from autodoc_harness.tools.read_file import make_read_file_tool

SYSTEM_PROMPT = """\
You are the Reviewer stage of a documentation-generation pipeline - the final \
quality gate before documentation is published. You are given ONE generated \
document plus the structured component notes it was based on. Your job:

1. Fact-check: for every claim in the document that cites a file, use `read_file` \
   to actually read that file and verify the claim is supported. If a claim is \
   unsupported, fabricated, or the citation doesn't check out, record a \
   "hallucination" finding describing what's wrong.
2. Path coverage: check whether the document covers green (happy), yellow \
   (edge-case), and red (error/failure) paths, per the underlying component \
   notes. If a path kind the notes actually documented is missing from the prose, \
   record a "missing_path_coverage" finding. Do not fabricate a path that isn't in \
   the notes.
3. Formatting/style: note heading hierarchy issues, inconsistent terminology, and \
   other style problems as "formatting" findings.

Use `list_dir`, `read_file`, and `grep_search` to verify claims - all paths are \
relative to the target repository root. When you have finished reviewing, stop \
calling tools - simply state that you are done. You will then be asked to provide \
your findings in a structured format. Just describe what's wrong here - a \
separate step will apply your findings to produce the corrected document, so do \
not try to rewrite the document yourself in this response.\
"""

CORRECTION_SYSTEM_PROMPT = """\
You are finalizing a reviewed document for a documentation-generation pipeline. \
You are given a document and a list of findings from a prior review pass. Apply \
the findings to produce the corrected document.

Rules:
- Output ONLY the corrected Markdown document content - no commentary, no code \
  fence wrapping the whole document, no JSON, no preamble like "Here is the \
  corrected document:".
- Do not introduce new claims or restructure sections the findings didn't flag - \
  apply only the fixes the findings describe.\
"""


@dataclass(frozen=True)
class ReviewedDocs:
    architecture_md: str
    api_reference_md: str
    module_docs: dict[str, str]
    report: ReviewReport


@dataclass(frozen=True)
class DocReviewResult:
    findings: list[ReviewFinding]
    corrected_content: str


def _build_tools(boundary: RepoBoundary, max_file_bytes: int) -> list[ToolSpec]:
    return [
        make_read_file_tool(boundary, max_file_bytes),
        make_list_dir_tool(boundary),
        make_grep_search_tool(boundary),
    ]


def _user_prompt(*, doc_name: str, content: str, context: str, description: str | None) -> str:
    sections = []
    if description:
        sections.append(f"System description: {description}")
    sections.append(
        f"Document under review: {doc_name}\n\n"
        f"--- BEGIN DOCUMENT ---\n{content}\n--- END DOCUMENT ---"
    )
    sections.append(f"Structured notes this document was based on:\n{context}")
    return "\n\n".join(sections)


def _format_findings(findings: list[ReviewFinding]) -> str:
    return "\n".join(f"- [{f.category}] {f.description}" for f in findings)


async def _correct_document(
    *,
    doc_name: str,
    content: str,
    findings: list[ReviewFinding],
    model_config: ResolvedModelConfig,
    budget: BudgetTracker,
    llm_call: LlmCall,
) -> str:
    """Apply findings to produce the corrected document, as a plain-text
    (non-JSON, tool-free) call - reuses the same `LlmCall` seam the Synthesizer
    uses for markdown generation."""
    budget.check_or_raise()
    turn = await llm_call(
        model=model_config.name,
        messages=[
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Document: {doc_name}\n\n"
                    f"Findings:\n{_format_findings(findings)}\n\n"
                    f"--- BEGIN DOCUMENT ---\n{content}\n--- END DOCUMENT ---"
                ),
            },
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
    corrected = (turn.content or "").strip()
    return corrected or content


async def _review_document(
    *,
    doc_name: str,
    content: str,
    context: str,
    description: str | None,
    tools: list[ToolSpec],
    model_config: ResolvedModelConfig,
    budget: BudgetTracker,
    max_tool_calls: int,
    max_extraction_attempts: int,
    llm_call: LlmCall,
    structured_call: StructuredCall,
) -> DocReviewResult:
    result = await run_agentic_loop(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_user_prompt(
            doc_name=doc_name, content=content, context=context, description=description
        ),
        tools=tools,
        result_model=ReviewFindingsSubmission,
        model_config=model_config,
        budget=budget,
        max_tool_calls=max_tool_calls,
        llm_call=llm_call,
        structured_call=structured_call,
        max_extraction_attempts=max_extraction_attempts,
    )
    if result.status != "ok" or result.data is None:
        # Review didn't complete for this doc - keep the original content rather
        # than losing it, and record that as a finding so it's visible in the
        # report.
        return DocReviewResult(
            findings=[
                ReviewFinding(
                    category="formatting",
                    description=f"Review did not complete: {result.status}",
                )
            ],
            corrected_content=content,
        )
    findings_submission = result.data
    assert isinstance(findings_submission, ReviewFindingsSubmission)

    if not findings_submission.findings:
        # Nothing to fix - skip the correction call entirely rather than risk the
        # model degrading an already-fine document.
        return DocReviewResult(findings=[], corrected_content=content)

    corrected_content = await _correct_document(
        doc_name=doc_name,
        content=content,
        findings=findings_submission.findings,
        model_config=model_config,
        budget=budget,
        llm_call=llm_call,
    )
    return DocReviewResult(
        findings=findings_submission.findings, corrected_content=corrected_content
    )


async def run_reviewer(
    component_notes: list[ComponentNotes],
    docs: SynthesizedDocs,
    config: Config,
    budget: BudgetTracker,
    *,
    llm_call: LlmCall = default_llm_call,
    structured_call: StructuredCall = default_structured_call,
) -> ReviewedDocs:
    """Review every synthesized document, fact-checking against the target repo and
    returning corrected content plus a structured report.

    Raises `BudgetExceededError` (propagated from the shared agent loop) if the
    run-wide budget is already exhausted - callers handle this the same way as
    every other stage.
    """
    boundary = RepoBoundary(
        config.target_repo, config.all_ignore_globs, honor_gitignore=config.honor_gitignore
    )
    tools = _build_tools(boundary, config.guardrails.max_file_bytes)
    model_config = config.model.resolved_for_stage("reviewer")
    max_tool_calls = config.guardrails.max_tool_calls_reviewer_per_doc
    max_extraction_attempts = config.guardrails.max_extraction_attempts

    started_at = datetime.now(UTC)
    tool_calls_before = budget.total_tool_calls
    tokens_in_before = budget.total_tokens_in
    tokens_out_before = budget.total_tokens_out
    cost_before = budget.total_cost_usd

    all_notes_context = "\n\n".join(format_component_notes(n) for n in component_notes)
    findings_by_document: dict[str, list[ReviewFinding]] = {}

    architecture_review = await _review_document(
        doc_name="architecture.md",
        content=docs.architecture_md,
        context=all_notes_context,
        description=config.description,
        tools=tools,
        model_config=model_config,
        budget=budget,
        max_tool_calls=max_tool_calls,
        max_extraction_attempts=max_extraction_attempts,
        llm_call=llm_call,
        structured_call=structured_call,
    )
    findings_by_document["architecture.md"] = architecture_review.findings

    api_reference_review = await _review_document(
        doc_name="api-reference.md",
        content=docs.api_reference_md,
        context=all_notes_context,
        description=config.description,
        tools=tools,
        model_config=model_config,
        budget=budget,
        max_tool_calls=max_tool_calls,
        max_extraction_attempts=max_extraction_attempts,
        llm_call=llm_call,
        structured_call=structured_call,
    )
    findings_by_document["api-reference.md"] = api_reference_review.findings

    notes_by_id = {n.component_id: n for n in component_notes}
    module_docs: dict[str, str] = {}
    for component_id, content in docs.module_docs.items():
        doc_name = f"modules/{component_id}.md"
        notes = notes_by_id.get(component_id)
        context = format_component_notes(notes) if notes is not None else ""
        review = await _review_document(
            doc_name=doc_name,
            content=content,
            context=context,
            description=config.description,
            tools=tools,
            model_config=model_config,
            budget=budget,
            max_tool_calls=max_tool_calls,
            max_extraction_attempts=max_extraction_attempts,
            llm_call=llm_call,
            structured_call=structured_call,
        )
        findings_by_document[doc_name] = review.findings
        module_docs[component_id] = review.corrected_content

    stats = TraversalStats(
        tool_calls=budget.total_tool_calls - tool_calls_before,
        tokens_in=budget.total_tokens_in - tokens_in_before,
        tokens_out=budget.total_tokens_out - tokens_out_before,
        cost_usd=budget.total_cost_usd - cost_before,
        duration_s=(datetime.now(UTC) - started_at).total_seconds(),
    )

    return ReviewedDocs(
        architecture_md=architecture_review.corrected_content,
        api_reference_md=api_reference_review.corrected_content,
        module_docs=module_docs,
        report=ReviewReport(
            findings_by_document=findings_by_document,
            generated_at=started_at,
            model=model_config.name,
            stats=stats,
        ),
    )
