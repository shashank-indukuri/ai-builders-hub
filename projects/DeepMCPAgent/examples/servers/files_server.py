from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

from fastmcp import FastMCP

# A safe, sandboxed file tools server for the current repository
mcp = FastMCP("Files")

# Root is the directory where this server is launched
ROOT = Path(os.environ.get("FILES_SERVER_ROOT", Path.cwd())).resolve()


def _safe_resolve(rel_path: str) -> Path:
    """Resolve a user-supplied path safely under ROOT to prevent traversal."""
    p = (ROOT / rel_path).resolve()
    if not str(p).startswith(str(ROOT)):
        raise ValueError("Path escapes server root")
    return p


@mcp.tool()
def list_dir(path: str = ".", glob: Optional[str] = None, max_items: int | float = 200) -> str:
    """List files and directories under a path relative to repo root.

    Args:
        path: Relative path from repository root.
        glob: Optional glob to filter entries (e.g., "**/*.py").
        max_items: Maximum number of entries to return.
    Returns:
        JSON string: list of {path, name, type, size_bytes}.
    """
    base = _safe_resolve(path)
    try:
        max_items_int = int(max_items)
    except Exception:
        max_items_int = 200
    results: List[Dict] = []

    if not base.exists():
        return json.dumps([])

    if glob:
        iter_paths = base.rglob(glob)
    else:
        iter_paths = base.iterdir()

    for p in iter_paths:
        try:
            item = {
                "path": str(p.relative_to(ROOT).as_posix()),
                "name": p.name,
                "type": "dir" if p.is_dir() else "file",
                "size_bytes": p.stat().st_size if p.is_file() else None,
            }
            results.append(item)
            if len(results) >= max_items_int:
                break
        except Exception:
            # Skip unreadable entries
            continue

    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
def read_file(path: str, start_line: int | float = 1, end_line: int | float = 2000) -> str:
    """Read a text file under the repo root and return a line-range snippet.

    Args:
        path: Relative file path from repo root.
        start_line: 1-indexed start line.
        end_line: 1-indexed end line (inclusive). Max 2000 lines per call.
    Returns:
        JSON string: {path, start_line, end_line, total_lines, content}.
    """
    file_path = _safe_resolve(path)
    if not file_path.is_file():
        raise FileNotFoundError("Not a file")

    # Safety limits
    try:
        start_i = int(start_line)
    except Exception:
        start_i = 1
    try:
        end_i = int(end_line)
    except Exception:
        end_i = start_i + 2000 - 1
    start = max(1, start_i)
    end = max(start, min(end_i, start + 2000 - 1))

    content_lines: List[str] = []
    total = 0
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            total = i
            if start <= i <= end:
                content_lines.append(line)
            if i > end:
                break

    return json.dumps({
        "path": str(file_path.relative_to(ROOT).as_posix()),
        "start_line": start,
        "end_line": end,
        "total_lines": total,
        "content": "".join(content_lines),
    }, ensure_ascii=False)


@mcp.tool()
def grep(
    query: str,
    path: str = ".",
    regex: bool = False,
    case_insensitive: bool = True,
    include: str = "**/*",
    max_matches: int | float = 200,
    max_file_size_bytes: int | float = 1_000_000,
) -> str:
    """Search files for a pattern under a path.

    Args:
        query: String or regex pattern to search for.
        path: Relative path from repo root.
        regex: Treat the query as a Python regex.
        case_insensitive: Case-insensitive search if True.
        include: Glob pattern to include (e.g., "**/*.py").
        max_matches: Max total matches to return.
        max_file_size_bytes: Skip files bigger than this.
    Returns:
        JSON string: list of {path, line, line_number}.
    """
    base = _safe_resolve(path)
    if not base.exists():
        return json.dumps([])

    flags = re.IGNORECASE if case_insensitive else 0
    pattern = re.compile(query, flags) if regex else None

    matches: List[Dict] = []
    try:
        max_matches_int = int(max_matches)
    except Exception:
        max_matches_int = 200
    try:
        max_file_size_int = int(max_file_size_bytes)
    except Exception:
        max_file_size_int = 1_000_000

    for file in base.rglob(include):
        try:
            if not file.is_file():
                continue
            if file.stat().st_size > max_file_size_int:
                continue

            rel = str(file.relative_to(ROOT).as_posix())
            with file.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    hay = line
                    found = False
                    if regex:
                        if pattern and pattern.search(hay):
                            found = True
                    else:
                        if (hay.lower().find(query.lower()) != -1) if case_insensitive else (query in hay):
                            found = True

                    if found:
                        matches.append({
                            "path": rel,
                            "line_number": i,
                            "line": hay.rstrip("\n"),
                        })
                        if len(matches) >= max_matches_int:
                            return json.dumps(matches, ensure_ascii=False)
        except Exception:
            continue

    return json.dumps(matches, ensure_ascii=False)


if __name__ == "__main__":
    # Serve over HTTP at /mcp on port 8001 to avoid clashing with math_server
    mcp.run(transport="http", host="127.0.0.1", port=8001, path="/mcp")
