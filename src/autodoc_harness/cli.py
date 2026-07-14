import asyncio
import json
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TaskProgressColumn, TextColumn

from autodoc_harness import __version__
from autodoc_harness.budget import BudgetExceededError, BudgetTracker
from autodoc_harness.config import (
    CONFIG_FILENAME,
    Config,
    ConfigError,
    load_config,
    validate_config_semantics,
)
from autodoc_harness.coordinator import run_pipeline_through_code_explorer
from autodoc_harness.models import ComponentMap, ComponentNotes
from autodoc_harness.output_writer import write_docs
from autodoc_harness.stages.master_explorer import run_master_explorer
from autodoc_harness.stages.reviewer import run_reviewer
from autodoc_harness.stages.synthesizer import SynthesizedDocs, run_synthesizer
from autodoc_harness.templates import STARTER_CONFIG_TEMPLATE

app = typer.Typer(
    name="autodoc-harness",
    help="Model-agnostic agentic harness for generating detailed technical "
    "documentation from source code.",
    no_args_is_help=True,
)

# Progress/status narration goes to stderr, kept separate from `console` so
# `--stop-after ... > out.json` and similar redirections of stdout stay clean.
status_console = Console(stderr=True)


def _print_issues(header: str, issues: list[str]) -> None:
    typer.echo(header, err=True)
    for issue in issues:
        typer.echo(f"  - {issue}", err=True)


def _load_and_validate_config(config_path: Path) -> Config:
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        _print_issues("Config validation failed:", e.issues)
        raise typer.Exit(code=1) from e

    issues = validate_config_semantics(cfg)
    if issues:
        _print_issues("Config validation failed:", issues)
        raise typer.Exit(code=1)
    return cfg


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"autodoc-harness {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    pass


@app.command()
def init(
    target: Path = typer.Option(
        Path("."), "--target", help="Directory to write the starter config into."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config file."),
) -> None:
    """Scaffold a starter config file in TARGET (default: current directory)."""
    config_path = target / CONFIG_FILENAME
    if config_path.exists() and not force:
        typer.echo(f"{config_path} already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(STARTER_CONFIG_TEMPLATE)
    typer.echo(f"Wrote starter config to {config_path}")


@app.command()
def validate(
    config: Path = typer.Option(..., "--config", help="Path to the config file to validate."),
) -> None:
    """Validate a config file's schema, entry points, and API key env vars.

    Makes no LLM calls and does not touch the target repo beyond checking that
    entry-point files exist.
    """
    cfg = _load_and_validate_config(config)
    typer.echo("Config is valid.")
    typer.echo(cfg.model_dump_json(indent=2))


class StopAfterStage(StrEnum):
    MASTER_EXPLORER = "master-explorer"
    CODE_EXPLORER = "code-explorer"
    SYNTHESIZER = "synthesizer"


@app.command()
def generate(
    config: Path = typer.Option(..., "--config", help="Path to the config file."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and print the config; make no LLM calls."
    ),
    stop_after: StopAfterStage | None = typer.Option(
        None,
        "--stop-after",
        help=(
            "Stop after a given stage and print its intermediate output instead of "
            "writing docs - useful for milestone debugging/demoing."
        ),
    ),
) -> None:
    """Run the documentation-generation pipeline against the target repo."""
    cfg = _load_and_validate_config(config)

    if dry_run:
        typer.echo("Dry run - no LLM calls will be made.")
        typer.echo(f"target_repo: {cfg.target_repo}")
        typer.echo(f"entry_points: {cfg.entry_points}")
        typer.echo(f"model: {cfg.model.name}")
        typer.echo(f"guardrails: {cfg.guardrails.model_dump()}")
        return

    asyncio.run(_run_generate(cfg, stop_after))


async def _run_generate(cfg: Config, stop_after: StopAfterStage | None) -> None:
    if stop_after is StopAfterStage.MASTER_EXPLORER:
        # Avoid unnecessary cost: don't dispatch Code Explorers when the caller
        # only wants the Master Explorer's output.
        budget = BudgetTracker(
            max_total_cost_usd=cfg.guardrails.max_total_cost_usd,
            max_run_seconds=cfg.guardrails.max_run_seconds,
        )
        try:
            with status_console.status("Running Master Explorer..."):
                component_map = await run_master_explorer(cfg, budget)
        except BudgetExceededError as e:
            typer.echo(f"Run aborted: {e}", err=True)
            raise typer.Exit(code=1) from e
        typer.echo(component_map.model_dump_json(indent=2))
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=status_console,
    )
    master_explorer_task_id = progress.add_task("Running Master Explorer...", total=None)
    code_explorer_task_id: TaskID | None = None

    def _on_master_explorer_done(component_map: ComponentMap) -> None:
        nonlocal code_explorer_task_id
        progress.update(
            master_explorer_task_id,
            completed=1,
            total=1,
            description=f"Master Explorer found {len(component_map.components)} component(s)",
        )
        code_explorer_task_id = progress.add_task(
            "Exploring components...", total=len(component_map.components)
        )

    def _on_component_done(notes: ComponentNotes) -> None:
        if code_explorer_task_id is not None:
            progress.update(code_explorer_task_id, advance=1)

    try:
        with progress:
            run = await run_pipeline_through_code_explorer(
                cfg,
                on_master_explorer_done=_on_master_explorer_done,
                on_component_done=_on_component_done,
            )
    except BudgetExceededError as e:
        typer.echo(f"Run aborted: {e}", err=True)
        raise typer.Exit(code=1) from e

    failed = [n for n in run.component_notes if n.status == "failed"]
    if failed:
        status_console.print(
            f"[yellow]Warning:[/] {len(failed)} component(s) failed exploration and "
            "will be documented with a status note instead: "
            + ", ".join(n.component_id for n in failed)
        )

    if stop_after is StopAfterStage.CODE_EXPLORER:
        typer.echo(
            json.dumps(
                {
                    "component_map": json.loads(run.component_map.model_dump_json()),
                    "component_notes": [
                        json.loads(notes.model_dump_json()) for notes in run.component_notes
                    ],
                },
                indent=2,
            )
        )
        return

    try:
        with status_console.status("Synthesizing documentation..."):
            docs = await run_synthesizer(run.component_map, run.component_notes, cfg, run.budget)
    except BudgetExceededError as e:
        typer.echo(f"Run aborted: {e}", err=True)
        raise typer.Exit(code=1) from e

    if stop_after is StopAfterStage.SYNTHESIZER:
        typer.echo("# architecture.md\n\n" + docs.architecture_md)
        typer.echo("\n\n# api-reference.md\n\n" + docs.api_reference_md)
        for component_id, module_md in docs.module_docs.items():
            typer.echo(f"\n\n# modules/{component_id}.md\n\n" + module_md)
        return

    try:
        with status_console.status("Reviewing generated documentation..."):
            reviewed = await run_reviewer(run.component_notes, docs, cfg, run.budget)
    except BudgetExceededError as e:
        typer.echo(f"Run aborted: {e}", err=True)
        raise typer.Exit(code=1) from e

    total_findings = sum(len(f) for f in reviewed.report.findings_by_document.values())
    if total_findings:
        status_console.print(
            f"Reviewer recorded {total_findings} finding(s) - see review-report.json."
        )

    final_docs = SynthesizedDocs(
        architecture_md=reviewed.architecture_md,
        api_reference_md=reviewed.api_reference_md,
        module_docs=reviewed.module_docs,
    )
    docs_dir = write_docs(
        run.component_map, run.component_notes, final_docs, cfg, review_report=reviewed.report
    )
    status_console.print(f"[bold green]Wrote documentation to {docs_dir}[/]")


if __name__ == "__main__":
    app()
