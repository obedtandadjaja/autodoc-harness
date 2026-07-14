"""Static templates (starter config, etc.) kept out of cli.py for readability."""

STARTER_CONFIG_TEMPLATE = """\
# autodoc-harness config

# Path to the repository this config documents. Absolute, or relative to this file.
target_repo: .

# Free-text context about what this system does. Without it, the model has
# nothing but file contents to infer intent from - a one-line description
# noticeably improves how it names/frames components. Strongly recommended:
# fill this in before running `generate`.
description: null
# description: >
#   A CLI tool that syncs local files to S3, used by the data team's
#   nightly backup job.

# Files to start traversal from (relative to target_repo). The harness explores
# everything reachable from these files - it does not scan the whole repo.
# Each entry can be a bare path, or {path, note} to explain that entry point's
# role (useful when there's more than one, e.g. a CLI vs. a web server):
entry_points:
  - src/main.py
  # - path: src/api/server.py
  #   note: HTTP server entry point, separate from the CLI above

# Optional: locations worth checking that aren't traversal starting points
# themselves - e.g. config files or anything wired up dynamically rather than
# imported statically, so they wouldn't otherwise be discovered by following
# imports from the entry points. May be files or directories.
hints: []
# hints:
#   - path: src/config.py
#     note: Runtime configuration - not imported directly by main.py

output:
  dir: docs        # written relative to target_repo
  overwrite: true  # this MVP is one-shot only; must stay true

model:
  name: anthropic/claude-sonnet-4-5   # litellm-format model id
  api_key_env: ANTHROPIC_API_KEY      # name of the env var holding the API key
  temperature: 0.2
  max_tokens: 8192
  # For a local/self-hosted model (e.g. Ollama) instead, use something like:
  #   name: ollama_chat/gemma4:e4b   # "ollama_chat/", not "ollama/" - the latter
  #                                  # doesn't reliably support tool calling
  #   api_base: http://localhost:11434
  #   api_key_env omitted entirely - local providers don't need one
  # api_base: http://localhost:11434
  # stage_overrides:
  #   reviewer:
  #     name: anthropic/claude-opus-4-1
  #     temperature: 0.0

guardrails:
  max_files_per_component: 40
  max_tool_calls_master_explorer: 60
  max_tool_calls_per_code_explorer: 30
  max_tool_calls_reviewer_per_doc: 20
  max_extraction_attempts: 3   # retries for the final schema-constrained extraction call
  max_file_bytes: 51200
  max_total_cost_usd: 5.00
  max_run_seconds: 1800
  max_parallel_code_explorers: 5
  max_traversal_depth: 6

# Extra glob patterns to exclude from traversal, merged with a built-in default
# list (node_modules, .venv, dist, build, vendor, .git, __pycache__, etc.)
ignore_globs: []
honor_gitignore: true

logging:
  level: info
  log_file: null
"""
