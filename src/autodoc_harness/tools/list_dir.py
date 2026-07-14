from __future__ import annotations

from typing import Any

from autodoc_harness.tools.base import RepoBoundary, RepoBoundaryError, ToolSpec

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory path relative to repo root; '.' for root.",
        },
        "recursive": {
            "type": "boolean",
            "description": "List subdirectories recursively. Defaults to false.",
        },
    },
    "required": ["path"],
}

DESCRIPTION = (
    "List files and subdirectories at a path within the target repository. "
    "Entries excluded by ignore rules (vendor/build/dependency directories) are "
    "omitted."
)


def make_list_dir_tool(boundary: RepoBoundary) -> ToolSpec:
    async def handler(args: dict[str, Any]) -> str:
        path_arg = str(args.get("path", "."))
        try:
            resolved = boundary.resolve(path_arg)
        except RepoBoundaryError as e:
            return f"error: {e}"
        if not resolved.is_dir():
            return f"error: '{path_arg}' is not a directory (or does not exist)"

        recursive = bool(args.get("recursive", False))
        entries: list[str] = []
        for entry in boundary.iter_tree(resolved, recursive=recursive):
            rel = entry.relative_to(boundary.target_repo)
            entries.append(f"{rel}/" if entry.is_dir() else str(rel))

        if not entries:
            return "(empty directory, or all entries are excluded by ignore rules)"
        return "\n".join(entries)

    return ToolSpec(
        name="list_dir", description=DESCRIPTION, parameters=PARAMETERS, handler=handler
    )
