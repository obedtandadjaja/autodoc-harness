# Example: real output from a real model

This is unedited output from a real `autodoc-harness generate` run - not a mocked
or hand-written sample. Nothing here was cleaned up afterward, including the one
finding where the Reviewer's correction call failed and the original content was
kept (see `docs/.autodoc-harness/review-report.json`).

- **Target repo**: [`tests/fixtures/sample_repo`](../../tests/fixtures/sample_repo) -
  a tiny two-file CLI tool (`divide(a, b)`) with a deliberately planted green
  (happy), yellow (edge-case warning), and red (error) path.
- **Model**: `ollama_chat/gemma4:e4b`, running locally via Ollama - free, no API
  key, and the smaller of the two Gemma 4 sizes tested during development.
- **Config used**: [`.autodoc.yaml`](./.autodoc.yaml) in this directory, including
  the `description` field that gives the model context beyond just file contents.

## Layout

```
docs/
├── architecture.md          # system-level overview
├── api-reference.md         # every public interface across components
├── modules/
│   ├── cli-interface.md     # per-component deep dive
│   └── calculator-logic.md
└── .autodoc-harness/        # machine-readable audit trail
    ├── component-map.json   # Master Explorer's output
    ├── component-notes/     # each Code Explorer's output
    ├── review-report.json   # Reviewer's findings per document
    └── run-manifest.json    # files actually read, with content hashes
```

## Regenerating this example

`output.dir` is always relative to `target_repo`, so running `.autodoc.yaml` as-is
would write into `tests/fixtures/sample_repo/docs/` rather than here. To refresh
this example: copy `tests/fixtures/sample_repo` to a scratch directory, point a
copy of `.autodoc.yaml` at that scratch directory instead, run `generate` there,
then copy the resulting `docs/` back into this directory.

```sh
uv sync --all-groups
# have Ollama running locally with: ollama pull gemma4:e4b
```
