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
