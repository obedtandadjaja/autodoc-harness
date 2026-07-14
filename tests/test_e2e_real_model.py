"""Opt-in end-to-end test against a REAL model - skipped by default.

Every other test in this suite injects a scripted or mocked `llm_call`/
`litellm.acompletion`; none of them prove the prompts actually elicit good
behavior from a real model. This test does, but it costs real money and depends
on live model behavior, so it never runs automatically (not in CI, not in a
plain `pytest` invocation).

Run it explicitly once you have a provider API key configured:

    AUTODOC_E2E=1 ANTHROPIC_API_KEY=... uv run pytest tests/test_e2e_real_model.py -v

Override the model/provider via AUTODOC_E2E_MODEL / AUTODOC_E2E_API_KEY_ENV if
you want to point it at something other than Anthropic.
"""

import os
import shutil
from pathlib import Path

import pytest

from autodoc_harness.config import Config
from autodoc_harness.coordinator import run_pipeline_through_code_explorer
from autodoc_harness.output_writer import write_docs
from autodoc_harness.stages.reviewer import run_reviewer
from autodoc_harness.stages.synthesizer import SynthesizedDocs, run_synthesizer

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTODOC_E2E") != "1",
    reason=(
        "opt-in real-model end-to-end test; set AUTODOC_E2E=1 plus a provider API "
        "key (e.g. ANTHROPIC_API_KEY) to run it"
    ),
)

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


async def test_generate_against_sample_repo_with_real_model(tmp_path: Path) -> None:
    # Copy into a scratch dir rather than running in-place, so the checked-in
    # fixture repo never ends up with a generated docs/ directory in it.
    target_repo = tmp_path / "sample_repo"
    shutil.copytree(FIXTURE_REPO, target_repo)

    config = Config.model_validate(
        {
            "target_repo": str(target_repo),
            "entry_points": ["src/main.py"],
            "model": {
                "name": os.environ.get("AUTODOC_E2E_MODEL", "anthropic/claude-sonnet-4-5"),
                "api_key_env": os.environ.get("AUTODOC_E2E_API_KEY_ENV", "ANTHROPIC_API_KEY"),
            },
        }
    )

    run = await run_pipeline_through_code_explorer(config)
    assert run.component_map.components, "Master Explorer found no components"
    assert any(n.status == "ok" for n in run.component_notes), (
        "no Code Explorer completed successfully"
    )

    all_path_kinds = {p.kind for n in run.component_notes for p in n.paths}
    assert "green" in all_path_kinds, "expected at least a green (happy) path to be documented"

    docs = await run_synthesizer(run.component_map, run.component_notes, config, run.budget)
    assert docs.architecture_md.strip()

    reviewed = await run_reviewer(run.component_notes, docs, config, run.budget)
    final_docs = SynthesizedDocs(
        architecture_md=reviewed.architecture_md,
        api_reference_md=reviewed.api_reference_md,
        module_docs=reviewed.module_docs,
    )
    docs_dir = write_docs(
        run.component_map, run.component_notes, final_docs, config, review_report=reviewed.report
    )
    assert (docs_dir / "architecture.md").is_file()
