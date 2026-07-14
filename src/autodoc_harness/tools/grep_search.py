from __future__ import annotations

import re
from typing import Any

from autodoc_harness.tools.base import RepoBoundary, RepoBoundaryError, ToolSpec

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Python-syntax regular expression to search for.",
        },
        "path": {
            "type": "string",
            "description": "Directory (or file) to search under, relative to repo root.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of matches to return. Defaults to 50.",
        },
        "case_sensitive": {
            "type": "boolean",
            "description": "Defaults to true.",
        },
    },
    "required": ["pattern"],
}

DESCRIPTION = (
    "Search file contents in the target repository for a Python-syntax regular "
    "expression. Returns matching file paths, line numbers, and short snippets. "
    "Files/directories excluded by ignore rules are never searched."
)


def make_grep_search_tool(boundary: RepoBoundary) -> ToolSpec:
    async def handler(args: dict[str, Any]) -> str:
        pattern_str = str(args["pattern"])
        path_arg = str(args.get("path", "."))
        max_results = int(args.get("max_results", 50))
        case_sensitive = bool(args.get("case_sensitive", True))

        try:
            resolved = boundary.resolve(path_arg)
        except RepoBoundaryError as e:
            return f"error: {e}"
        if not resolved.exists():
            return f"error: '{path_arg}' does not exist"

        try:
            regex = re.compile(pattern_str, 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            return f"error: invalid regular expression: {e}"

        candidates = (
            [resolved]
            if resolved.is_file()
            else (p for p in boundary.iter_tree(resolved, recursive=True) if p.is_file())
        )
        matches: list[str] = []
        for file_path in candidates:
            if len(matches) >= max_results:
                break
            rel = file_path.relative_to(boundary.target_repo)
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if len(matches) >= max_results:
                    break
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")

        if not matches:
            return "(no matches found)"
        return "\n".join(matches)

    return ToolSpec(
        name="grep_search", description=DESCRIPTION, parameters=PARAMETERS, handler=handler
    )
