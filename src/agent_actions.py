"""P4.0 Agent Action Schema — enums and SafetyFlags.

See docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §4.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agent_state import AgentMemory


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


_SECRET_BASENAMES = {
    ".env", ".env.local", ".git", ".ssh",
    "credentials", "credentials.json",
    "secrets", "secret.key", "token",
}
_SECRET_PATTERNS = [re.compile(r"\.pem$"), re.compile(r"\.key$")]
_URL_SCHEMES = ("http://", "https://", "ftp://", "file://")


class PathValidationError(ValueError):
    """Raised when a path violates P4.0 workspace path rules."""


class P4ForbiddenActionError(Exception):
    """Raised when an action violates P4.0 hard safety rules.

    NOTE: Inherits from Exception (not ValueError) so Pydantic v2
    model_validator(mode="after") does not catch-and-wrap it as
    ValidationError. This lets the raw exception propagate to callers.
    """


def validate_path(rel_path: str) -> str:
    """Validate a workspace-relative path. Returns normalized path on success."""
    if not rel_path or not isinstance(rel_path, str):
        raise PathValidationError("empty path")
    if rel_path.startswith("/") or (len(rel_path) >= 2 and rel_path[1] == ":"):
        raise PathValidationError(f"absolute path not allowed: {rel_path}")
    if rel_path.startswith("\\\\"):
        raise PathValidationError(f"UNC path not allowed: {rel_path}")
    for scheme in _URL_SCHEMES:
        if rel_path.lower().startswith(scheme):
            raise PathValidationError(f"URL not allowed: {rel_path}")
    parts = rel_path.replace("\\", "/").split("/")
    if ".." in parts:
        raise PathValidationError(f"parent traversal not allowed: {rel_path}")
    basename = parts[-1].lower()
    if basename in _SECRET_BASENAMES:
        raise PathValidationError(f"sensitive path not allowed: {rel_path}")
    for pat in _SECRET_PATTERNS:
        if pat.search(basename):
            raise PathValidationError(f"sensitive path not allowed: {rel_path}")
    return rel_path


class ActionBase(BaseModel):
    """Base for all actions."""
    model_config = ConfigDict(use_enum_values=True)

    action_id: str
    action_type: ActionType
    reason_short: str = Field(max_length=120)
    expected_observation: str
    safety_flags: SafetyFlags

    @model_validator(mode="after")
    def _reject_forbidden_safety(self):
        if self.safety_flags.network_required:
            raise P4ForbiddenActionError(
                f"network_required=True is forbidden in P4.0 (action {self.action_id})"
            )
        if self.safety_flags.reads_sensitive_path:
            raise P4ForbiddenActionError(
                f"reads_sensitive_path=True is forbidden in P4.0 (action {self.action_id})"
            )
        return self


class ListFilesArgs(BaseModel):
    pattern: str | None = None


class ReadFileArgs(BaseModel):
    path: str
    start_line: int | None = None
    end_line: int | None = None

    @model_validator(mode="after")
    def _validate(self):
        validate_path(self.path)
        return self


class SearchTextArgs(BaseModel):
    query: str
    file_glob: str | None = None
    max_results: int = 20


class InspectTaskArgs(BaseModel):
    pass


class ProposePatchArgs(BaseModel):
    file_path: str
    old_text: str
    new_text: str

    @model_validator(mode="after")
    def _validate(self):
        validate_path(self.file_path)
        if not self.old_text:
            raise ValueError("old_text must be non-empty")
        return self


class ApplyPatchArgs(BaseModel):
    file_path: str
    old_text: str
    new_text: str
    expected_before_sha256: str | None = None

    @model_validator(mode="after")
    def _validate(self):
        validate_path(self.file_path)
        if not self.old_text:
            raise ValueError("old_text must be non-empty")
        return self


class RollbackPatchArgs(BaseModel):
    action_id: str


class RunTestsArgs(BaseModel):
    test_path: str | None = None
    timeout_s: float = 10.0


class InspectErrorArgs(BaseModel):
    error_source: Literal["last_test", "last_patch"]


class WriteMemoryArgs(BaseModel):
    memory: "AgentMemory"


class FinishArgs(BaseModel):
    success_criterion: TaskSuccessCriterion
    tests_passed: bool
    identification_verified: bool
    summary: str = Field(max_length=500)


class ListFilesAction(ActionBase):
    action_type: Literal[ActionType.list_files] = ActionType.list_files
    arguments: ListFilesArgs = Field(default_factory=ListFilesArgs)


class ReadFileAction(ActionBase):
    action_type: Literal[ActionType.read_file] = ActionType.read_file
    arguments: ReadFileArgs


class SearchTextAction(ActionBase):
    action_type: Literal[ActionType.search_text] = ActionType.search_text
    arguments: SearchTextArgs


class InspectTaskAction(ActionBase):
    action_type: Literal[ActionType.inspect_task] = ActionType.inspect_task
    arguments: InspectTaskArgs = Field(default_factory=InspectTaskArgs)


class ProposePatchAction(ActionBase):
    action_type: Literal[ActionType.propose_patch] = ActionType.propose_patch
    arguments: ProposePatchArgs


class ApplyPatchAction(ActionBase):
    action_type: Literal[ActionType.apply_patch] = ActionType.apply_patch
    arguments: ApplyPatchArgs


class RollbackPatchAction(ActionBase):
    action_type: Literal[ActionType.rollback_patch] = ActionType.rollback_patch
    arguments: RollbackPatchArgs


class RunTestsAction(ActionBase):
    action_type: Literal[ActionType.run_tests] = ActionType.run_tests
    arguments: RunTestsArgs = Field(default_factory=RunTestsArgs)


class InspectErrorAction(ActionBase):
    action_type: Literal[ActionType.inspect_error] = ActionType.inspect_error
    arguments: InspectErrorArgs


class WriteMemoryAction(ActionBase):
    action_type: Literal[ActionType.write_memory] = ActionType.write_memory
    arguments: WriteMemoryArgs


class FinishAction(ActionBase):
    action_type: Literal[ActionType.finish] = ActionType.finish
    arguments: FinishArgs


Action = Annotated[
    Union[
        ListFilesAction, ReadFileAction, SearchTextAction, InspectTaskAction,
        ProposePatchAction, ApplyPatchAction, RollbackPatchAction,
        RunTestsAction, InspectErrorAction, WriteMemoryAction, FinishAction,
    ],
    Field(discriminator="action_type"),
]


WriteMemoryArgs.model_rebuild()
