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

from src.agent_actions import PathValidationError, TaskSuccessCriterion
from src.agent_state import AgentMemory
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
    PatchObservation,
    PatchProposalObservation,
    RollbackError,
    tool_apply_patch,
    tool_propose_patch,
    tool_rollback_patch,
    TestObservation,
    ErrorObservation,
    MemoryObservation,
    FinishObservation,
    ToolUnavailableError,
    tool_run_tests,
    tool_inspect_error,
    tool_write_memory,
    tool_finish,
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


# --- Task 6: patch transaction tests ---


def test_apply_patch_success():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"def add(a, b):\n    return a - b\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_apply_patch(ws, "solution.py", "return a - b", "return a + b", action_id="test-patch-1")
    assert isinstance(obs, PatchObservation)
    assert obs.success is True
    assert obs.before_sha256 != obs.after_sha256
    assert obs.action_id == "test-patch-1"
    assert "return a + b" in (ws.workspace_root / "solution.py").read_text()
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_apply_patch_not_found():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"def add(a, b):\n    return a + b\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_apply_patch(ws, "solution.py", "return a - b", "return a * b")
    assert obs.success is False
    assert "not found" in (obs.error or "")
    assert (ws.workspace_root / "solution.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_apply_patch_ambiguous():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "dup.py", b"x = 1\nx = 1\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_apply_patch(ws, "dup.py", "x = 1", "x = 2")
    assert obs.success is False
    assert "unique" in (obs.error or "")
    assert (ws.workspace_root / "dup.py").read_bytes() == b"x = 1\nx = 1\n"
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_apply_patch_sha_mismatch():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"hello\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    obs = tool_apply_patch(
        ws,
        "solution.py",
        "hello",
        "world",
        expected_before_sha256="0000000000000000000000000000000000000000000000000000000000000000",
    )
    assert obs.success is False
    assert "SHA" in (obs.error or "") or "mismatch" in (obs.error or "")
    assert (ws.workspace_root / "solution.py").read_bytes() == b"hello\n"
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_apply_patch_audit_record():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"a\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    tool_apply_patch(ws, "solution.py", "a", "b", action_id="audit-test")

    audit_file = ws.workspace_root / ".audit" / "patches.jsonl"
    assert audit_file.exists()
    import json as _json
    found = False
    for line in audit_file.read_text().splitlines():
        if not line.strip():
            continue
        record = _json.loads(line)
        if record.get("action_id") == "audit-test":
            assert record.get("success") is True
            assert record.get("rolled_back") is False
            found = True
    assert found, "audit-test record not found in patches.jsonl"
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_propose_patch_does_not_modify():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"a\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    original_sha = ws.file_sha256("solution.py")
    obs = tool_propose_patch(ws, "solution.py", "a", "b")
    assert isinstance(obs, PatchProposalObservation)
    assert obs.would_succeed is True
    assert obs.before_sha256 == original_sha
    assert obs.after_sha256 != original_sha
    assert ws.file_sha256("solution.py") == original_sha
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_rollback_patch_success():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"a\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    original_sha = ws.file_sha256("solution.py")
    patch_obs = tool_apply_patch(ws, "solution.py", "a", "b", action_id="rollback-test")
    assert patch_obs.success is True
    assert ws.file_sha256("solution.py") != original_sha

    rollback_obs = tool_rollback_patch(ws, "rollback-test")
    assert isinstance(rollback_obs, PatchObservation)
    assert rollback_obs.success is True
    assert ws.file_sha256("solution.py") == original_sha
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_rollback_patch_unknown_action_id():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"a\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    with pytest.raises(RollbackError):
        tool_rollback_patch(ws, "nonexistent-id")
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


def test_rollback_patch_double_rollback():
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    _write(source_dir / "solution.py", b"a\n")

    ws = MicroTaskWorkspace.from_task(source_dir)
    tool_apply_patch(ws, "solution.py", "a", "b", action_id="double-test")
    tool_rollback_patch(ws, "double-test")  # first rollback succeeds
    with pytest.raises(RollbackError):
        tool_rollback_patch(ws, "double-test")  # already rolled back
    ws.cleanup()
    shutil.rmtree(source_dir, ignore_errors=True)


# --- Task 7: run_tests / inspect_error / write_memory / finish tests ---


def test_run_tests_passing(monkeypatch):
    """A correct solution + test → TestObservation(passed=True)."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "solution.py", b"def add(a, b):\n    return a + b\n")
        _write(source_dir / "test_solution.py",
               b"from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            obs = tool_run_tests(workspace)
            assert obs.passed is True
            assert obs.num_collected >= 1
            assert obs.num_passed >= 1
            assert obs.num_failed == 0
            assert obs.timed_out is False
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_run_tests_failing(monkeypatch):
    """A broken solution + test → TestObservation(passed=False, num_failed>=1)."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "solution.py", b"def add(a, b):\n    return a - b\n")
        _write(source_dir / "test_solution.py",
               b"from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            obs = tool_run_tests(workspace)
            assert obs.passed is False
            assert obs.num_failed >= 1
            assert len(obs.stderr) > 0 or len(obs.stdout) > 0
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_run_tests_timeout(monkeypatch):
    """A solution that hangs → TestObservation(timed_out=True)."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "solution.py",
               b"import time\n\ndef slow():\n    time.sleep(10)\n    return 42\n")
        _write(source_dir / "test_solution.py",
               b"from solution import slow\n\ndef test_slow():\n    assert slow() == 42\n")
        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            obs = tool_run_tests(workspace, timeout_s=2.0)
            assert obs.timed_out is True
            assert obs.passed is False
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_run_tests_network_hard_fail(monkeypatch):
    """Without P4_ALLOW_NETWORK=0, run_tests raises ToolUnavailableError."""
    monkeypatch.delenv("P4_ALLOW_NETWORK", raising=False)
    source_dir = Path(tempfile.mkdtemp(prefix="p4_src_"))
    try:
        _write(source_dir / "solution.py", b"def f():\n    return 0\n")
        _write(source_dir / "test_solution.py",
               b"from solution import f\n\ndef test_f():\n    assert f() == 0\n")
        workspace = MicroTaskWorkspace.from_task(source_dir)
        try:
            with pytest.raises(ToolUnavailableError, match="network isolation unavailable"):
                tool_run_tests(workspace)
        finally:
            workspace.cleanup()
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_inspect_error_last_test():
    """inspect_error with error_source='last_test' returns the test stderr."""
    test_obs = TestObservation(
        passed=False, num_collected=1, num_passed=0, num_failed=1,
        timed_out=False, stdout="", stderr="AssertionError: expected 3",
        duration_s=0.1,
    )
    err = tool_inspect_error(
        error_source="last_test",
        last_test_observation=test_obs,
        last_patch_observation=None,
    )
    assert err.source == "last_test"
    assert "AssertionError" in err.content


def test_write_memory():
    """write_memory returns MemoryObservation with before + after."""
    before = AgentMemory(notes="", hypothesis="", failed_attempts=[], last_test_summary="")
    after = AgentMemory(
        notes="bug is in add()", hypothesis="sign flipped",
        failed_attempts=["tried subtraction"], last_test_summary="1 failed",
    )
    obs = tool_write_memory(memory_before=before, memory=after)
    assert obs.memory_before.notes == ""
    assert obs.memory_after.hypothesis == "sign flipped"
    assert obs.memory_after.failed_attempts == ["tried subtraction"]


def test_finish():
    """finish returns FinishObservation with declared values."""
    obs = tool_finish(
        success_criterion=TaskSuccessCriterion.TEST_PASS,
        tests_passed=True,
        identification_verified=False,
        summary="Fixed the add() function by changing subtraction to addition.",
    )
    assert obs.success_criterion == TaskSuccessCriterion.TEST_PASS
    assert obs.tests_passed is True
    assert obs.identification_verified is False
    assert "add()" in obs.summary


# --- Task 4: inspect_error returns stdout+stderr capped 8KB ---

def test_inspect_error_returns_stdout_on_test_failure():
    """Failed run_tests writes traceback to stdout; inspect_error must
    surface it, not return empty content."""
    test_obs = TestObservation(
        passed=False, num_collected=1, num_passed=0, num_failed=1,
        timed_out=False,
        stdout="AssertionError: expected 5 but got 4",
        stderr="",
        duration_s=0.1,
    )
    result = tool_inspect_error(
        error_source="last_test",
        last_test_observation=test_obs,
        last_patch_observation=None,
    )
    assert result.content != "", "inspect_error returned empty content for stdout-only failure"
    assert "AssertionError" in result.content


def test_inspect_error_caps_at_8kb():
    """stdout+stderr > 8KB -> content truncated to exactly 8192 chars."""
    big_stdout = "x" * 10000
    test_obs = TestObservation(
        passed=False, num_collected=1, num_passed=0, num_failed=1,
        timed_out=False,
        stdout=big_stdout, stderr="",
        duration_s=0.1,
    )
    result = tool_inspect_error(
        error_source="last_test",
        last_test_observation=test_obs,
        last_patch_observation=None,
    )
    assert len(result.content) == 8192, \
        f"expected 8192 chars (8KB cap), got {len(result.content)}"
