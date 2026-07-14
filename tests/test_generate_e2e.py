"""End-to-end smoke test of the CLI `generate` command wired through
asyncio -> Coordinator -> stages -> output writer, using a mocked
`litellm.acompletion` instead of a real API key so it runs in CI. Complements
(does not replace) the per-stage scripted-fake unit tests - this is the first test
that exercises everything wired together the way a real `autodoc-harness generate`
invocation would.

Since `default_llm_call` (tool-calling explore phase) and `default_structured_call`
(schema-constrained extraction phase) both call `litellm.acompletion`, the fake
below distinguishes them by kwargs: `response_format` present means extraction,
`tools` present (and non-empty) means explore.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from autodoc_harness.cli import app
from autodoc_harness.config import CONFIG_FILENAME

runner = CliRunner()


def _response(content: str | None, tool_calls: list[Any] | None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _original_user_prompt(kwargs: dict[str, Any]) -> str:
    return str(kwargs["messages"][1]["content"])


async def _fake_acompletion(**kwargs: Any) -> SimpleNamespace:
    if "response_format" in kwargs:
        prompt = _original_user_prompt(kwargs)
        if "Document under review" in prompt:
            # One finding, to also exercise the Reviewer's separate plain-text
            # correction call below (rather than the no-findings/skip-correction
            # path already covered by the per-stage reviewer tests).
            payload = (
                '{"findings": [{"category": "formatting", '
                '"description": "test finding", "location": null}]}'
            )
        elif "component_id:" in prompt:
            payload = (
                '{"summary": "prints a greeting", "paths": [{"kind": "green", '
                '"title": "Happy path", "narrative": "Prints hi.", "citations": []}]}'
            )
        else:
            payload = (
                '{"components": [{"component_id": "main", "name": "Main", '
                '"summary": "entrypoint", "seed_paths": ["src/main.py"], '
                '"related_component_ids": [], "role": "core"}], "unresolved_notes": []}'
            )
        return _response(payload, None)

    if kwargs.get("tools"):
        # Explore phase: immediately signal "done" - actual tool dispatch is
        # covered by the per-stage scripted-fake unit tests, not this smoke test.
        return _response("done exploring", None)

    # No tools, no response_format: either the Synthesizer's markdown generation,
    # or the Reviewer's plain-text correction call - both use the same LlmCall
    # interface, distinguished here by the correction call's distinctive prompt.
    prompt = _original_user_prompt(kwargs)
    if "Findings:" in prompt and "BEGIN DOCUMENT" in prompt:
        return _response("# Reviewed section\n\nLooks good.", None)
    return _response("# Generated section\n\nSome content.", None)


def test_generate_end_to_end_writes_docs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    runner.invoke(app, ["init", "--target", str(tmp_path)])
    config_path = tmp_path / CONFIG_FILENAME

    with (
        patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        result = runner.invoke(
            app,
            ["generate", "--config", str(config_path)],
            env={"ANTHROPIC_API_KEY": "sk-test"},
        )

    assert result.exit_code == 0, result.output
    docs_dir = tmp_path / "docs"
    assert (docs_dir / "architecture.md").is_file()
    assert (docs_dir / "api-reference.md").is_file()
    assert (docs_dir / "modules" / "main.md").is_file()
    assert (docs_dir / ".autodoc-harness" / "component-map.json").is_file()
    assert (docs_dir / ".autodoc-harness" / "component-notes" / "main.json").is_file()
    assert (docs_dir / ".autodoc-harness" / "run-manifest.json").is_file()
    assert (docs_dir / ".autodoc-harness" / "review-report.json").is_file()
    assert "Reviewed section" in (docs_dir / "architecture.md").read_text()
