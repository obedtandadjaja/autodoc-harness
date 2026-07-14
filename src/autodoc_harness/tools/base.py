"""Shared tool plumbing: the OpenAI-style tool schema wrapper and the repo-boundary
enforcement every read-only filesystem tool is built on."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathspec import GitIgnoreSpec


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]

    @property
    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class RepoBoundaryError(Exception):
    """Raised when a tool call attempts to escape the target repo or touch an
    ignored (third-party/vendor/build) path."""


class RepoBoundary:
    """Resolves and validates paths against a target repo root plus ignore rules.

    Centralizes the two rules every filesystem tool must enforce: never let the
    model read/list outside `target_repo` (path traversal or symlink escape), and
    never let it open a path that's supposed to be a black-box dependency
    (node_modules, .venv, vendor, etc.) even when those live inside the repo root.
    """

    def __init__(
        self, target_repo: Path, ignore_globs: list[str], honor_gitignore: bool = True
    ) -> None:
        self.target_repo = target_repo.resolve()
        patterns = list(ignore_globs)
        if honor_gitignore:
            gitignore_path = self.target_repo / ".gitignore"
            if gitignore_path.is_file():
                patterns.extend(gitignore_path.read_text(encoding="utf-8").splitlines())
        self._spec = GitIgnoreSpec.from_lines(patterns)
        # Harness-tracked (not model-reported) record of files actually read via
        # the read_file tool - grounds the eventual run manifest in real tool
        # calls rather than trusting the model's self-reported citations.
        self.visited_files: set[str] = set()

    def mark_visited(self, rel_path: Path) -> None:
        self.visited_files.add(str(rel_path))

    def resolve(self, relative_path: str) -> Path:
        """Resolve a model-supplied path to an absolute path inside the target repo.

        Raises `RepoBoundaryError` if it escapes the repo (including via symlinks,
        since `.resolve()` follows them before the containment check) or matches an
        ignore pattern.
        """
        candidate = (self.target_repo / relative_path).resolve()
        try:
            rel = candidate.relative_to(self.target_repo)
        except ValueError as e:
            raise RepoBoundaryError(
                f"path '{relative_path}' resolves outside the target repository"
            ) from e
        if self.is_ignored(rel, is_dir=candidate.is_dir()):
            raise RepoBoundaryError(
                f"path '{relative_path}' is excluded by ignore rules - treat it as "
                "an external/black-box dependency, do not attempt to read it"
            )
        return candidate

    def is_ignored(self, rel_path: Path, *, is_dir: bool = False) -> bool:
        """Check whether `rel_path` matches an ignore pattern.

        `is_dir` matters: gitignore patterns written with a trailing slash (as most
        real .gitignore files write directory entries, e.g. "node_modules/") only
        match when the candidate is known to be a directory - pathspec has no way
        to infer that from a bare string, so directory candidates must be checked
        with a trailing slash appended.
        """
        rel_str = str(rel_path)
        if rel_str == ".":
            return False
        return self._spec.match_file(f"{rel_str}/" if is_dir else rel_str)

    def iter_tree(self, start: Path, *, recursive: bool) -> Iterator[Path]:
        """Yield non-ignored paths under `start`.

        For recursive walks, ignored directories are pruned *before* descending
        into them (via `os.walk`'s in-place `dirnames` mutation) rather than
        enumerated and filtered after the fact - important for real repos, where an
        ignored `node_modules`/`.venv` can contain far more files than the rest of
        the repo combined.
        """
        if not recursive:
            for entry in sorted(start.iterdir()):
                rel = entry.relative_to(self.target_repo)
                if self.is_ignored(rel, is_dir=entry.is_dir()):
                    continue
                yield entry
            return

        for dirpath, dirnames, filenames in os.walk(start):
            current = Path(dirpath)
            dirnames.sort()
            kept_dirnames = []
            for name in dirnames:
                rel = (current / name).relative_to(self.target_repo)
                if self.is_ignored(rel, is_dir=True):
                    continue
                kept_dirnames.append(name)
            dirnames[:] = kept_dirnames

            for name in dirnames:
                yield current / name
            for name in sorted(filenames):
                file_path = current / name
                rel = file_path.relative_to(self.target_repo)
                if self.is_ignored(rel, is_dir=False):
                    continue
                yield file_path
