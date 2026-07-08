"""P4.0 Phase C — Read-only Tool Layer.

Typed Observation models and four read-only tool functions that operate
on a ``MicroTaskWorkspace``. Mutating tools (apply_patch, rollback_patch,
run_tests) are added in later phases.

See docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §5.
"""
from __future__ import annotations

from pydantic import BaseModel

from src.agent_workspace import MicroTaskWorkspace


_READ_SIZE_LIMIT = 64 * 1024  # 64 KB
_BINARY_SNIFF_BYTES = 8 * 1024  # 8 KB NUL-byte sniff window
_SEARCH_MAX_RESULTS_CAP = 100


class FileListObservation(BaseModel):
    """Result of ``tool_list_files``."""
    paths: list[str]


class FileContentObservation(BaseModel):
    """Result of ``tool_read_file``."""
    path: str
    content: str
    line_count: int
    truncated: bool


class SearchHit(BaseModel):
    """A single literal-substring search match."""
    file_path: str
    line_number: int
    line_text: str


class SearchObservation(BaseModel):
    """Result of ``tool_search_text``."""
    hits: list[SearchHit]


class TaskObservation(BaseModel):
    """Result of ``tool_inspect_task`` — parsed README.md fields."""
    goal: str
    constraints: list[str]
    hints: list[str]


class TaskError(ValueError):
    """Raised when ``tool_inspect_task`` cannot parse README.md."""
    pass


def tool_list_files(
    workspace: MicroTaskWorkspace, pattern: str | None = None
) -> FileListObservation:
    """List files under ``workspace_root``, respecting the ignore list."""
    return FileListObservation(paths=workspace.list_files(pattern))


def tool_read_file(
    workspace: MicroTaskWorkspace,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> FileContentObservation:
    """Read a workspace-relative text file with optional line range.

    Files larger than 64 KB are truncated to the first 64 KB and the
    ``truncated`` flag is set. Files containing a NUL byte or failing
    UTF-8 decode raise ``ValueError``.
    """
    abs_path = workspace.resolve_path(path)
    raw = abs_path.read_bytes()
    truncated = False
    if len(raw) > _READ_SIZE_LIMIT:
        raw = raw[:_READ_SIZE_LIMIT]
        truncated = True
    if b"\x00" in raw:
        raise ValueError(f"binary file: {path}")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"binary file: {path}") from exc

    if start_line is not None or end_line is not None:
        lines = content.split("\n")
        start = (start_line - 1) if start_line is not None else 0
        end = end_line if end_line is not None else len(lines)
        lines = lines[start:end]
        content = "\n".join(lines)

    line_count = len(content.splitlines())
    return FileContentObservation(
        path=path,
        content=content,
        line_count=line_count,
        truncated=truncated,
    )


def tool_search_text(
    workspace: MicroTaskWorkspace,
    query: str,
    file_glob: str | None = None,
    max_results: int = 20,
) -> SearchObservation:
    """Literal-substring search across workspace text files.

    Skips binary files (NUL byte in first 8 KB). Stops after
    ``max_results`` hits (capped at 100).
    """
    if max_results > _SEARCH_MAX_RESULTS_CAP:
        max_results = _SEARCH_MAX_RESULTS_CAP
    if max_results < 0:
        max_results = 0

    hits: list[SearchHit] = []
    for rel_path in workspace.list_files(file_glob):
        if len(hits) >= max_results:
            break
        abs_path = workspace.resolve_path(rel_path)
        try:
            raw = abs_path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for idx, line in enumerate(content.split("\n"), start=1):
            if query in line:
                hits.append(
                    SearchHit(
                        file_path=rel_path,
                        line_number=idx,
                        line_text=line,
                    )
                )
                if len(hits) >= max_results:
                    break
    return SearchObservation(hits=hits)


def tool_inspect_task(workspace: MicroTaskWorkspace) -> TaskObservation:
    """Parse ``README.md`` at workspace root into a ``TaskObservation``.

    Raises ``TaskError`` if the ``## Goal`` section is missing.
    """
    readme_path = workspace.resolve_path("README.md")
    text = readme_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    sections: dict[str, list[str]] = {
        "Goal": [],
        "Constraints": [],
        "Hints": [],
    }
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            header = stripped[3:].strip()
            current = header if header in sections else None
            continue
        if current is None:
            continue
        sections[current].append(line)

    if not sections["Goal"]:
        raise TaskError("README.md missing Goal section")

    goal_text = ""
    for line in sections["Goal"]:
        if line.strip():
            goal_text = line.strip()
            break

    constraints = [
        line.strip()[2:].strip()
        for line in sections["Constraints"]
        if line.strip().startswith("- ")
    ]
    hints = [
        line.strip()[2:].strip()
        for line in sections["Hints"]
        if line.strip().startswith("- ")
    ]

    return TaskObservation(
        goal=goal_text,
        constraints=constraints,
        hints=hints,
    )
