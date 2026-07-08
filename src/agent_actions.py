"""P4.0 Agent Action Schema — enums and SafetyFlags.

See docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §4.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class ActionType(str, Enum):
    """11 action types (Issue #17 + rollback_patch)."""
    list_files = "list_files"
    read_file = "read_file"
    search_text = "search_text"
    inspect_task = "inspect_task"
    propose_patch = "propose_patch"
    apply_patch = "apply_patch"
    rollback_patch = "rollback_patch"
    run_tests = "run_tests"
    inspect_error = "inspect_error"
    write_memory = "write_memory"
    finish = "finish"


class TaskSuccessCriterion(str, Enum):
    """How a task is judged successful (user fix #1)."""
    TEST_PASS = "test_pass"
    IDENTIFY_BUG = "identify_bug"
    PATCH_APPLIED = "patch_applied"


class EvaluationMode(str, Enum):
    """Evaluator mode (user fix #2)."""
    REPLAY = "replay"
    AGENT_RUN = "agent_run"


class SafetyFlags(BaseModel):
    """Fixed 5-field safety descriptor.

    P4.0 hard-reject: any action with network_required=True or
    reads_sensitive_path=True is rejected at action-validation time.
    """
    model_config = ConfigDict(frozen=True)

    modifies_workspace: bool
    executes_code: bool
    network_required: bool
    reads_sensitive_path: bool
    is_terminal: bool
