"""Shared pydantic vocabulary passed between pipeline stages.

Each stage that submits structured output via the agentic loop has a narrow
"*Submission" model - exactly what the LLM is asked to produce - kept separate from
the full, harness-attached record (`ComponentMap`, `ComponentNotes`). Metadata like
timestamps, model name, and traversal stats always comes from our own tracking
(`BudgetTracker`, wall-clock time), never from asking the model to self-report data
it has no reliable way to know.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    file: str = Field(description="Path relative to the target repo root.")
    lines: str | None = Field(
        default=None,
        description="Best-effort line range, e.g. '42-58'. Not guaranteed exact - no AST is used.",
    )
    note: str | None = None


class PathNarrative(BaseModel):
    kind: Literal["green", "yellow", "red"] = Field(
        description="green = happy path, yellow = edge/warning case, red = error/failure path."
    )
    title: str
    narrative: str
    citations: list[Citation] = Field(default_factory=list)


class InterfaceSignature(BaseModel):
    name: str
    kind: Literal["function", "class", "method", "cli_command", "endpoint", "other"]
    signature_text: str
    description: str
    file: str
    citations: list[Citation] = Field(default_factory=list)


class ComponentRef(BaseModel):
    component_id: str = Field(description="Short kebab-case slug, e.g. 'auth-service'.")
    name: str
    summary: str
    seed_paths: list[str] = Field(
        description=(
            "File(s) where this component was found - this becomes the Code "
            "Explorer's starting point for a deep dive into this component."
        )
    )
    related_component_ids: list[str] = Field(default_factory=list)
    role: Literal["core", "library", "integration", "config", "test", "other"] = "other"


class ComponentMapSubmission(BaseModel):
    """What the Master Explorer's submit tool actually asks the model to produce."""

    components: list[ComponentRef]
    unresolved_notes: list[str] = Field(default_factory=list)


class TraversalStats(BaseModel):
    tool_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    duration_s: float


class ComponentMap(BaseModel):
    """Master Explorer's submission plus harness-attached run metadata."""

    target_repo: str
    entry_points: list[str]
    components: list[ComponentRef]
    unresolved_notes: list[str]
    generated_at: datetime
    model: str
    stats: TraversalStats


class ComponentNotesSubmission(BaseModel):
    """What each Code Explorer's submit tool actually asks the model to produce."""

    summary: str
    responsibilities: list[str] = Field(default_factory=list)
    public_interfaces: list[InterfaceSignature] = Field(default_factory=list)
    external_dependencies: list[str] = Field(default_factory=list)
    internal_dependencies: list[str] = Field(default_factory=list)
    paths: list[PathNarrative] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ComponentNotes(BaseModel):
    """A Code Explorer's submission plus harness-attached identity/status/stats.

    `status`/`error` are set by the Coordinator's own success/failure handling, not
    reported by the model - a component's own explorer can't reliably self-diagnose
    a crash or budget cutoff that happened to it.
    """

    component_id: str
    name: str
    status: Literal["ok", "partial", "failed"]
    error: str | None = None
    summary: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    public_interfaces: list[InterfaceSignature] = Field(default_factory=list)
    external_dependencies: list[str] = Field(default_factory=list)
    internal_dependencies: list[str] = Field(default_factory=list)
    paths: list[PathNarrative] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    files_visited: list[str] = Field(
        default_factory=list,
        description=(
            "Files actually read via the read_file tool during this component's "
            "exploration - harness-tracked, not model-reported."
        ),
    )
    stats: TraversalStats


class ReviewFinding(BaseModel):
    category: Literal["hallucination", "missing_path_coverage", "formatting"]
    description: str
    location: str | None = Field(
        default=None, description="A heading/section within the document, if applicable."
    )


class ReviewFindingsSubmission(BaseModel):
    """What the Reviewer's structured extraction call asks the model to produce.

    Deliberately does NOT include the corrected document content. Asking a model
    to echo back a large markdown block inside a JSON string field is a much
    harder generation task - correctly escaping quotes/backslashes/newlines across
    a long string - than a small structured list, even with schema-constrained
    decoding: real-model testing showed most review passes exhausting their
    extraction retries specifically on this field. The corrected content is
    obtained separately as plain text (see `stages/reviewer.py`), the same way the
    Synthesizer produces markdown - no JSON, no escaping burden.
    """

    findings: list[ReviewFinding] = Field(default_factory=list)


class ReviewReport(BaseModel):
    """Harness-assembled aggregate of every per-document review pass."""

    findings_by_document: dict[str, list[ReviewFinding]]
    generated_at: datetime
    model: str
    stats: TraversalStats
