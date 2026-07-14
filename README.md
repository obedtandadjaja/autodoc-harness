# autodoc-harness

A model-agnostic, agentic CLI that generates detailed technical documentation from a
codebase's actual source code, grounded in real control flow rather than docstrings or
comments.

Point it at a repo's entry points and it produces narrative markdown covering not just
the happy path through the code, but edge cases and failure handling too (green /
yellow / red paths).

## Status

MVP: all five pipeline stages are implemented, covered by scripted-fake/mocked tests,
and validated end-to-end against a real local model (see
[`examples/sample-repo`](examples/sample-repo) for unedited real output). Still worth
running against a repo you know well before trusting it on something important - see
[Validating against a real model](#validating-against-a-real-model).

## Usage

```sh
uv sync --all-groups

# Scaffold a starter config in the repo you want to document
uv run autodoc-harness init --target /path/to/some-project

# Edit /path/to/some-project/.autodoc.yaml: set entry_points to the files you
# want traversal to start from, confirm the model/API key env var, and fill in
# `description` (a sentence about what the system does) - it noticeably improves
# how components get named/framed, since otherwise the model has nothing but file
# contents to infer intent from. `hints` is for locations worth checking that
# aren't traversal starting points themselves (e.g. config files not imported by
# any entry point).

# Check the config without spending anything
uv run autodoc-harness validate --config /path/to/some-project/.autodoc.yaml

# Run the full pipeline - writes docs/ into the target repo
export ANTHROPIC_API_KEY=...  # or whatever provider/env var your config names
uv run autodoc-harness generate --config /path/to/some-project/.autodoc.yaml
```

`generate` accepts `--dry-run` (resolve and print the config, no LLM calls) and
`--stop-after {master-explorer,code-explorer,synthesizer}` to inspect intermediate
output instead of writing files - useful when debugging a single stage.

## Architecture

A **Coordinator** orchestrates four pipeline stages, each an LLM call (via
[litellm](https://github.com/BerriAI/litellm), so any supported provider works):

1. **Master Explorer** - traverses from configured entry-point files to build a
   high-level component map.
2. **Code Explorer** - one instance per component, dispatched in parallel, deep-dives
   into how each component works and documents green/yellow/red paths with citations.
3. **Synthesizer** - stitches the component map and all component notes into the final
   markdown doc set (architecture overview, per-module docs, API reference).
4. **Reviewer** - fact-checks citations against real source, flags missing path
   coverage, and fixes formatting/style consistency, one document at a time.

The Coordinator's Code Explorer fan-out is bounded by a `Semaphore`
(`guardrails.max_parallel_code_explorers`) and isolates per-component failures - one
broken component is recorded as `status: "failed"` rather than aborting the run.

Output lands in `<target_repo>/<output.dir>` (default `docs/`):

```
docs/
├── architecture.md
├── api-reference.md
├── modules/
│   └── <component>.md
└── .autodoc-harness/       # machine-readable audit trail
    ├── component-map.json
    ├── component-notes/<component>.json
    ├── review-report.json
    └── run-manifest.json   # file -> content hash for every file actually read
```

MVP scope: one-shot generation (no incremental updates yet - `run-manifest.json` is a
forward-compat hook for that), raw source text fed to the model (no AST parsing), pure
narrative markdown (no diagrams).

## Validating against a real model

Every test in this repo uses a scripted-fake or mocked `llm_call`/`litellm.acompletion`
- none of them prove the prompts actually produce good documentation from a real
model. Before trusting this against a real project:

1. Run `autodoc-harness generate` against a repo you know well (or one with existing
   good documentation to compare against) and read the output critically.
2. Optionally run the opt-in end-to-end test, which exercises the full pipeline
   against a tiny fixture repo (`tests/fixtures/sample_repo`) with deliberate
   green/yellow/red branches:

   ```sh
   AUTODOC_E2E=1 ANTHROPIC_API_KEY=... uv run pytest tests/test_e2e_real_model.py -v
   ```

   This costs real money and is never run automatically (not in CI, not in a plain
   `pytest` invocation).

### Using a local model (Ollama)

Confirmed working against both `ollama_chat/gemma4:e4b` and `gemma4:12b` - see
[`examples/sample-repo`](examples/sample-repo) for real, unedited output from `e4b`.
Notes from that testing:

- Use the `ollama_chat/` prefix, not `ollama/` - the latter doesn't reliably support
  tool calling and litellm's own docs warn it can cause infinite tool-call loops.
- Structured output is obtained via a dedicated schema-constrained extraction call
  (`response_format=`), separate from the tool-calling explore loop, rather than a
  "submit tool" alongside the explore tools. An earlier submit-tool design worked
  fine with Claude/GPT but real-model testing showed `gemma4:e4b` inventing its own
  field names for it despite reasoning about the content correctly; splitting
  extraction into its own call lets Ollama's grammar-constrained decoding actually
  apply, and fixed it. See `agent_loop.py`'s module docstring for the full story.
- Local inference is slow enough on a laptop that the default 120s per-call `timeout`
  can be too tight, especially for 12b+ models - set `model.timeout` to `250`-`300`
  or higher in `.autodoc.yaml`.
- litellm has no pricing data for Ollama models, so `guardrails.max_total_cost_usd`
  won't meaningfully bound a local run - `max_run_seconds` and the `max_tool_calls_*`
  settings are the guardrails that actually apply.

```yaml
model:
  name: ollama_chat/gemma4:e4b
  api_base: http://localhost:11434
  timeout: 250
  # api_key_env omitted entirely - local providers don't need one
```

## Development

```sh
uv sync --all-groups
uv run autodoc-harness --version
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
```
