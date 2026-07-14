from __future__ import annotations

from typing import Any

from autodoc_harness.tools.base import RepoBoundary, RepoBoundaryError, ToolSpec

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path relative to the target repository root.",
        },
        "start_line": {
            "type": "integer",
            "description": "Optional 1-indexed start line (inclusive).",
        },
        "end_line": {
            "type": "integer",
            "description": "Optional 1-indexed end line (inclusive).",
        },
    },
    "required": ["path"],
}

DESCRIPTION = (
    "Read a file within the target repository. Returns the text (optionally a "
    "line range), a truncation notice if it exceeds the configured size limit, "
    "or an error if the path doesn't exist, isn't a text file, or falls outside "
    "the repository/ignore boundary."
)


def make_read_file_tool(boundary: RepoBoundary, max_file_bytes: int) -> ToolSpec:
    async def handler(args: dict[str, Any]) -> str:
        path_arg = str(args["path"])
        try:
            resolved = boundary.resolve(path_arg)
        except RepoBoundaryError as e:
            return f"error: {e}"
        if not resolved.is_file():
            return f"error: '{path_arg}' is not a file (or does not exist)"

        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"error: '{path_arg}' does not appear to be a UTF-8 text file (binary?)"

        boundary.mark_visited(resolved.relative_to(boundary.target_repo))

        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if start_line is not None or end_line is not None:
            lines = text.splitlines()
            start = max(int(start_line) - 1, 0) if start_line is not None else 0
            end = int(end_line) if end_line is not None else len(lines)
            text = "\n".join(lines[start:end])

        encoded = text.encode("utf-8")
        if len(encoded) > max_file_bytes:
            text = encoded[:max_file_bytes].decode("utf-8", errors="ignore")
            text += f"\n\n[... truncated: file exceeds the {max_file_bytes} byte limit ...]"
        return text

    return ToolSpec(
        name="read_file", description=DESCRIPTION, parameters=PARAMETERS, handler=handler
    )
