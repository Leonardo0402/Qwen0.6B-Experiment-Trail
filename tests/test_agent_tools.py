"""Phase C: Workspace + tool tests.

Task 4 adds the MicroTaskWorkspace tests below. Tasks 5-7 will append
tool-layer tests (list_files / read_file / apply_patch / run_tests) to
this same file.
"""
import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from src.agent_actions import PathValidationError
from src.agent_workspace import MicroTaskWorkspace
from src.agent_tools import (
    FileListObservation,
    FileContentObservation,
    SearchObservation,
    SearchHit,
    TaskObservation,
    TaskError,
    tool_list_files,
    tool_read_file,
    tool_search_text,
    tool_inspect_task,
)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_workspace_copy_to_temp():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "solution.py", b"def solve():\n    return 0\n")
        _write(source_dir / "test_solution.py", b"def test_solve():\n    assert solve() == 0\n")

        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            assert workspace.workspace_root.exists()
            assert workspace.workspace_root.is_dir()
            assert workspace.workspace_root != source_dir
            assert (workspace.workspace_root / "solution.py").exists()
            assert (workspace.workspace_root / "test_solution.py").exists()
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_workspace_ignore_list():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "solution.py", b"pass\n")
        _write(source_dir / ".git" / "config", b"[user]\n")
        _write(source_dir / "__pycache__" / "x.pyc", b"\x00\x01")
        _write(source_dir / "sub" / "utils.py", b"pass\n")

        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            files = workspace.list_files()
            assert "solution.py" in files
            assert "sub/utils.py" in files
            assert not any(".git" in p for p in files)
            assert not any("__pycache__" in p for p in files)
            assert not any(p.endswith(".pyc") for p in files)
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_workspace_resolve_path_validation():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "solution.py", b"pass\n")

        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            p = workspace.resolve_path("solution.py")
            assert p.name == "solution.py"

            with pytest.raises(PathValidationError):
                workspace.resolve_path("../escape")
            with pytest.raises(PathValidationError):
                workspace.resolve_path("/etc/passwd")
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_workspace_resolve_path_absolute_return():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "src" / "foo.py", b"pass\n")

        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            p = workspace.resolve_path("src/foo.py")
            assert p.is_absolute()
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_workspace_file_sha256_stability():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "data.txt", b"hello world")

        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            h1 = workspace.file_sha256("data.txt")
            h2 = workspace.file_sha256("data.txt")
            assert h1 == h2
            assert h1 == hashlib.sha256(b"hello world").hexdigest()
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


# --- Task 5: read-only tool tests ---


def test_tool_list_files_basic():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"def solve():\n    return 0\n")
    _write(source_dir / "README.md", b"# Task: t\n\n## Goal\nDo something\n")
    _write(source_dir / "test_solution.py", b"def test_solve():\n    assert solve() == 0\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_list_files(ws)
    assert isinstance(obs, FileListObservation)
    assert "solution.py" in obs.paths
    assert "README.md" in obs.paths
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_tool_read_file_basic():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "foo.py", b"def add(a, b):\n    return a + b\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_read_file(ws, "foo.py")
    assert obs.path == "foo.py"
    assert "def add" in obs.content
    assert obs.line_count == 2
    assert obs.truncated is False
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_tool_read_file_line_range():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "multi.py", b"line1\nline2\nline3\nline4\nline5\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_read_file(ws, "multi.py", start_line=2, end_line=4)
    assert "line2" in obs.content
    assert "line4" in obs.content
    assert "line1" not in obs.content
    assert "line5" not in obs.content
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_tool_read_file_binary_rejected():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "bin.dat", b"\x00\x01\x02binary")

    ws = MicroTaskWorkspace.from_task(source_dir)
    with pytest.raises(ValueError):
        tool_read_file(ws, "bin.dat")
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_tool_search_text_basic():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(
        source_dir / "code.py",
        b"def foo():\n    return 'hello'\n\ndef bar():\n    return 'world'\n",
    )

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_search_text(ws, "def")
    assert len(obs.hits) == 2
    assert obs.hits[0].file_path == "code.py"
    assert obs.hits[0].line_number == 1
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_tool_search_text_max_results():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    content = "".join(f"match line {i}\n" for i in range(30))
    _write(source_dir / "many.py", content.encode())

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_search_text(ws, "match", max_results=5)
    assert len(obs.hits) == 5
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_tool_inspect_task_readme():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    readme = (
        "# Task: test_task\n\n"
        "## Goal\n"
        "Fix the add function\n\n"
        "## Constraints\n"
        "- Do not modify tests\n"
        "- Keep it simple\n\n"
        "## Hints\n"
        "- The bug is in the return statement\n"
    )
    _write(source_dir / "README.md", readme.encode())

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_inspect_task(ws)
    assert obs.goal == "Fix the add function"
    assert len(obs.constraints) == 2
    assert len(obs.hints) == 1
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)
