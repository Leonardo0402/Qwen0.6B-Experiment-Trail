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


# --- Phase C Part 3: patch transactions (apply_patch / propose_patch / rollback_patch) ---
#
# Imports below are placed here so the appended block does not modify the
# existing top-of-file imports. Standard library + Pydantic only.
import hashlib
import json
import shutil
from datetime import datetime
from uuid import uuid4


class PatchObservation(BaseModel):
    """Result of ``tool_apply_patch`` and ``tool_rollback_patch``."""
    file_path: str
    before_sha256: str
    after_sha256: str
    backup_path: str | None
    action_id: str
    success: bool
    error: str | None = None


class PatchProposalObservation(BaseModel):
    """Result of ``tool_propose_patch`` — a dry-run of ``tool_apply_patch``."""
    file_path: str
    before_sha256: str
    after_sha256: str
    would_succeed: bool
    error: str | None = None


class RollbackError(ValueError):
    """Raised when ``tool_rollback_patch`` cannot reverse an action_id."""
    pass


def _validate_patch_preconditions(
    workspace: MicroTaskWorkspace,
    file_path: str,
    old_text: str,
    expected_before_sha256: str | None,
) -> tuple[bytes, str, str | None, str | None, str]:
    """Shared preconditions for ``tool_apply_patch`` and ``tool_propose_patch``.

    Returns ``(content_bytes, before_sha256, error, content_str,
    normalized_old_text)``. When ``error`` is not None the caller must fail
    without modifying the file and ``content_str`` will be None.
    ``normalized_old_text`` has its line endings adjusted to match the file's
    style (CRLF vs LF) so that exact-string matching succeeds.
    """
    abs_path = workspace.resolve_path(file_path)
    content_bytes = abs_path.read_bytes()
    before_sha256 = hashlib.sha256(content_bytes).hexdigest()

    if b"\x00" in content_bytes:
        return content_bytes, before_sha256, "binary file rejected", None, old_text

    if expected_before_sha256 is not None and expected_before_sha256 != before_sha256:
        return content_bytes, before_sha256, "SHA mismatch", None, old_text

    try:
        content_str = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return content_bytes, before_sha256, "binary file rejected (non-UTF-8)", None, old_text

    # Normalize old_text line endings to match the file's style
    if "\r\n" in content_str and "\r\n" not in old_text:
        old_text = old_text.replace("\n", "\r\n")
    elif "\r\n" not in content_str and "\r\n" in old_text:
        old_text = old_text.replace("\r\n", "\n")

    occurrences = content_str.count(old_text)
    if occurrences == 0:
        return content_bytes, before_sha256, "old_text not found", None, old_text
    if occurrences > 1:
        return content_bytes, before_sha256, "old_text not unique", None, old_text

    return content_bytes, before_sha256, None, content_str, old_text


def tool_apply_patch(
    workspace: MicroTaskWorkspace,
    file_path: str,
    old_text: str,
    new_text: str,
    expected_before_sha256: str | None = None,
    action_id: str | None = None,
) -> PatchObservation:
    """Apply a unique-occurrence text patch with backup + audit trail.

    Fails (returns ``success=False``) without modifying the file when the
    file is binary (NUL byte), ``expected_before_sha256`` does not match,
    or ``old_text`` is not found / not unique. On success: backs up the
    file to ``<file_path>.bak.<before_sha256[:8]>``, writes the new
    content, and appends an audit record to
    ``workspace_root/.audit/patches.jsonl``.
    """
    if action_id is None:
        action_id = f"patch_{uuid4().hex[:8]}"

    content_bytes, before_sha256, error, content_str, old_text = _validate_patch_preconditions(
        workspace, file_path, old_text, expected_before_sha256
    )

    if error is not None:
        return PatchObservation(
            file_path=file_path,
            before_sha256=before_sha256,
            after_sha256=before_sha256,
            backup_path=None,
            action_id=action_id,
            success=False,
            error=error,
        )

    new_content = content_str.replace(old_text, new_text, 1)
    new_bytes = new_content.encode("utf-8")
    after_sha256 = hashlib.sha256(new_bytes).hexdigest()

    abs_path = workspace.resolve_path(file_path)
    backup_rel = f"{file_path}.bak.{before_sha256[:8]}"
    backup_abs = workspace.resolve_path(backup_rel)
    shutil.copy2(abs_path, backup_abs)

    abs_path.write_bytes(new_bytes)

    audit_dir = workspace.workspace_root / ".audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_file = audit_dir / "patches.jsonl"
    record = {
        "timestamp": datetime.now().isoformat(),
        "action_id": action_id,
        "file_path": file_path,
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "success": True,
        "rolled_back": False,
    }
    with audit_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return PatchObservation(
        file_path=file_path,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        backup_path=str(backup_abs),
        action_id=action_id,
        success=True,
        error=None,
    )


def tool_propose_patch(
    workspace: MicroTaskWorkspace,
    file_path: str,
    old_text: str,
    new_text: str,
    expected_before_sha256: str | None = None,
) -> PatchProposalObservation:
    """Validate a patch and report what its SHAs would be — without writing.

    Same validation rules as ``tool_apply_patch``. Does not modify the
    file, does not create a backup, does not write an audit record.
    """
    content_bytes, before_sha256, error, content_str, old_text = _validate_patch_preconditions(
        workspace, file_path, old_text, expected_before_sha256
    )

    if error is not None:
        return PatchProposalObservation(
            file_path=file_path,
            before_sha256=before_sha256,
            after_sha256=before_sha256,
            would_succeed=False,
            error=error,
        )

    new_content = content_str.replace(old_text, new_text, 1)
    new_bytes = new_content.encode("utf-8")
    after_sha256 = hashlib.sha256(new_bytes).hexdigest()

    return PatchProposalObservation(
        file_path=file_path,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        would_succeed=True,
        error=None,
    )


def tool_rollback_patch(
    workspace: MicroTaskWorkspace,
    action_id: str,
) -> PatchObservation:
    """Reverse a prior successful ``tool_apply_patch`` by ``action_id``.

    Restores the file from the backup, marks the original audit record
    ``rolled_back=true``, and appends a new audit line for the rollback.
    Raises ``RollbackError`` when the action_id is missing, the original
    patch was not successful, or it was already rolled back.
    """
    audit_file = workspace.workspace_root / ".audit" / "patches.jsonl"

    records: list[dict] = []
    target_idx: int | None = None
    target_record: dict | None = None

    if audit_file.exists():
        with audit_file.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                records.append(record)
                if record.get("action_id") == action_id:
                    target_idx = len(records) - 1
                    target_record = record

    if target_record is None:
        raise RollbackError(f"action_id not reversible: not found: {action_id}")
    if not target_record.get("success"):
        raise RollbackError(
            f"action_id not reversible: patch was not successful: {action_id}"
        )
    if target_record.get("rolled_back"):
        raise RollbackError(
            f"action_id not reversible: already rolled back: {action_id}"
        )

    file_path = target_record["file_path"]
    before_sha256 = target_record["before_sha256"]
    after_sha256 = target_record["after_sha256"]

    backup_rel = f"{file_path}.bak.{before_sha256[:8]}"
    backup_abs = workspace.resolve_path(backup_rel)
    abs_path = workspace.resolve_path(file_path)

    if not backup_abs.exists():
        raise RollbackError(
            f"action_id not reversible: backup file missing: {backup_rel}"
        )

    shutil.copy2(backup_abs, abs_path)

    restored_bytes = abs_path.read_bytes()
    restored_sha = hashlib.sha256(restored_bytes).hexdigest()
    if restored_sha != before_sha256:
        raise RollbackError(
            f"action_id not reversible: restored SHA mismatch "
            f"(expected {before_sha256}, got {restored_sha})"
        )

    records[target_idx]["rolled_back"] = True
    with audit_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    new_action_id = f"rollback_{uuid4().hex[:8]}"
    rollback_record = {
        "timestamp": datetime.now().isoformat(),
        "action_id": new_action_id,
        "file_path": file_path,
        "before_sha256": after_sha256,
        "after_sha256": before_sha256,
        "success": True,
        "rolled_back": False,
    }
    with audit_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rollback_record) + "\n")

    return PatchObservation(
        file_path=file_path,
        before_sha256=after_sha256,
        after_sha256=before_sha256,
        backup_path=None,
        action_id=new_action_id,
        success=True,
        error=None,
    )


# --- Phase C Part 4: run_tests / inspect_error / write_memory / finish ---
import os
import re

from src.agent_state import AgentMemory
from src.agent_actions import TaskSuccessCriterion, validate_path
from src.sandbox import run_pytest


class ToolUnavailableError(Exception):
    """Raised when a tool's runtime preconditions are not met (e.g. no
    network isolation for run_tests).

    NOTE: Inherits from Exception (not ValueError) so it is not caught
    and re-wrapped by Pydantic validators.
    """


class TestObservation(BaseModel):
    """Result of ``tool_run_tests``."""
    passed: bool
    num_collected: int
    num_passed: int
    num_failed: int
    timed_out: bool
    stdout: str
    stderr: str
    duration_s: float


class ErrorObservation(BaseModel):
    """Result of ``tool_inspect_error``."""
    source: str
    content: str


class MemoryObservation(BaseModel):
    """Result of ``tool_write_memory``."""
    memory_before: AgentMemory
    memory_after: AgentMemory


class FinishObservation(BaseModel):
    """Result of ``tool_finish``."""
    success_criterion: str
    tests_passed: bool
    identification_verified: bool
    summary: str


_TEST_PATH_RE = re.compile(r"^(test_.*\.py|.*_test\.py)$")


def tool_run_tests(
    workspace: MicroTaskWorkspace,
    test_path: str | None = None,
    timeout_s: float = 10.0,
) -> TestObservation:
    """Run pytest on the workspace's ``solution.py`` + test file.

    Network hard-fail: if ``P4_ALLOW_NETWORK`` env var is not ``"0"``,
    raises ``ToolUnavailableError`` — no silent fallback.

    Parameters
    ----------
    workspace:
        The micro-task workspace containing solution.py and test files.
    test_path:
        Workspace-relative test file path. Must match ``test_*.py`` or
        ``*_test.py``. Defaults to ``test_solution.py``.
    timeout_s:
        Pytest timeout in seconds. Must be in (0, 30]. Default 10.
    """
    # Network hard-fail (user fix #3)
    if os.environ.get("P4_ALLOW_NETWORK", "1") != "0":
        raise ToolUnavailableError("network isolation unavailable")

    # Default + validate test_path
    if test_path is None:
        test_path = "test_solution.py"
    validate_path(test_path)
    if not _TEST_PATH_RE.match(test_path):
        raise ValueError(
            f"test_path must match test_*.py or *_test.py: {test_path}"
        )

    # Validate timeout
    if timeout_s <= 0 or timeout_s > 30.0:
        raise ValueError(f"timeout_s must be in (0, 30]: {timeout_s}")

    # Read solution + test code from workspace
    solution_path = workspace.resolve_path("solution.py")
    test_abs = workspace.resolve_path(test_path)
    if not solution_path.exists():
        raise FileNotFoundError(f"solution.py not found in workspace")
    if not test_abs.exists():
        raise FileNotFoundError(f"test file not found: {test_path}")

    solution_code = solution_path.read_text(encoding="utf-8")
    test_code = test_abs.read_text(encoding="utf-8")

    # Run via sandbox (8 KB output cap per spec §5 line 501)
    result = run_pytest(
        target_code=solution_code,
        test_code=test_code,
        timeout_s=timeout_s,
        max_output_chars=8 * 1024,
    )

    return TestObservation(
        passed=result.passed,
        num_collected=result.num_collected,
        num_passed=result.num_passed,
        num_failed=result.num_failed,
        timed_out=result.timed_out,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=result.duration_s,
    )


def tool_inspect_error(
    error_source: str,
    last_test_observation: "TestObservation | None",
    last_patch_observation: "PatchObservation | None",
) -> ErrorObservation:
    """Return the error content from the last test or patch observation.

    Stateless: the caller passes the last observations; the tool layer
    does not track history.
    """
    if error_source == "last_test":
        if last_test_observation is None:
            raise ValueError("no prior run_tests observation")
        raw = last_test_observation.stdout + "\n" + last_test_observation.stderr
        capped = raw[:8192]
        return ErrorObservation(source="last_test", content=capped)
    elif error_source == "last_patch":
        if last_patch_observation is None:
            raise ValueError("no prior patch observation")
        content = last_patch_observation.error or ""
        return ErrorObservation(source="last_patch", content=content)
    else:
        raise ValueError(
            f"error_source must be 'last_test' or 'last_patch': {error_source}"
        )


def tool_write_memory(
    memory_before: AgentMemory,
    memory: AgentMemory,
) -> MemoryObservation:
    """Validate and record a new AgentMemory state.

    Stateless: the caller tracks the prior memory; the tool validates
    the new schema (Pydantic enforces on construction) and returns both.
    """
    return MemoryObservation(memory_before=memory_before, memory_after=memory)


def tool_finish(
    success_criterion: TaskSuccessCriterion,
    tests_passed: bool,
    identification_verified: bool,
    summary: str,
) -> FinishObservation:
    """Record the agent's finish declaration.

    Does NOT enforce invariants (tests_passed=True requires last run_tests
    passed, etc.) — that is the evaluator's job (spec §5 line 551-552).
    """
    return FinishObservation(
        success_criterion=success_criterion,
        tests_passed=tests_passed,
        identification_verified=identification_verified,
        summary=summary,
    )
