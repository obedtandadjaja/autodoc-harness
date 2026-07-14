from pathlib import Path

import pytest

from autodoc_harness.tools.base import RepoBoundary
from autodoc_harness.tools.grep_search import make_grep_search_tool
from autodoc_harness.tools.list_dir import make_list_dir_tool
from autodoc_harness.tools.read_file import make_read_file_tool


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("line1\nline2\nline3\nsecret_token\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("module.exports = {}")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    return tmp_path


async def test_read_file_returns_contents(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=[])
    tool = make_read_file_tool(boundary, max_file_bytes=1_000_000)
    result = await tool.handler({"path": "src/main.py"})
    assert "line1" in result
    assert "secret_token" in result


async def test_read_file_line_range(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=[])
    tool = make_read_file_tool(boundary, max_file_bytes=1_000_000)
    result = await tool.handler({"path": "src/main.py", "start_line": 2, "end_line": 3})
    assert result == "line2\nline3"


async def test_read_file_truncates_oversized_files(repo: Path) -> None:
    (repo / "big.txt").write_text("x" * 1000)
    boundary = RepoBoundary(repo, ignore_globs=[])
    tool = make_read_file_tool(boundary, max_file_bytes=100)
    result = await tool.handler({"path": "big.txt"})
    assert "truncated" in result
    assert len(result) < 1000


async def test_read_file_rejects_path_traversal(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=[])
    tool = make_read_file_tool(boundary, max_file_bytes=1_000_000)
    result = await tool.handler({"path": "../outside.txt"})
    assert "error" in result
    assert "outside the target repository" in result


async def test_read_file_rejects_ignored_paths(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=["node_modules"])
    tool = make_read_file_tool(boundary, max_file_bytes=1_000_000)
    result = await tool.handler({"path": "node_modules/dep.js"})
    assert "error" in result
    assert "excluded" in result


async def test_read_file_missing_file(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=[])
    tool = make_read_file_tool(boundary, max_file_bytes=1_000_000)
    result = await tool.handler({"path": "nope.py"})
    assert "error" in result


async def test_list_dir_excludes_ignored_entries(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=["node_modules"], honor_gitignore=False)
    tool = make_list_dir_tool(boundary)
    result = await tool.handler({"path": "."})
    assert "src" in result
    assert "node_modules" not in result


async def test_list_dir_honors_gitignore(repo: Path) -> None:
    (repo / ".gitignore").write_text("node_modules/\n")
    boundary = RepoBoundary(repo, ignore_globs=[], honor_gitignore=True)
    tool = make_list_dir_tool(boundary)
    result = await tool.handler({"path": "."})
    assert "node_modules" not in result


async def test_list_dir_recursive(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=["node_modules"], honor_gitignore=False)
    tool = make_list_dir_tool(boundary)
    result = await tool.handler({"path": ".", "recursive": True})
    assert "src/main.py" in result


async def test_grep_search_finds_matches(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=["node_modules"], honor_gitignore=False)
    tool = make_grep_search_tool(boundary)
    result = await tool.handler({"pattern": "secret_token"})
    assert "src/main.py:4" in result


async def test_grep_search_skips_ignored_paths(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=["node_modules"], honor_gitignore=False)
    tool = make_grep_search_tool(boundary)
    result = await tool.handler({"pattern": "exports"})
    assert "no matches found" in result


async def test_grep_search_invalid_regex(repo: Path) -> None:
    boundary = RepoBoundary(repo, ignore_globs=[])
    tool = make_grep_search_tool(boundary)
    result = await tool.handler({"pattern": "["})
    assert "error" in result
