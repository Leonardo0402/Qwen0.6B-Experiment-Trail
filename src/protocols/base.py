"""P4.1b Protocol abstraction layer.

Defines ProtocolBase ABC and ProtocolDiagnostics used by all protocol
implementations (JSON, Tag, DSL).

ProtocolDiagnostics computes each validity dimension independently,
fixing the P4.1 issue where schema_valid/safety_valid/action_type_valid/
arguments_valid were all set to True together.

Issue #32 Trust Repair: each check_* method validates ONE dimension only,
without silently dropping unknown fields. All Args models now use
ConfigDict(extra="forbid") so unknown fields hard-fail at every level.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter

from src.agent_actions import (
    Action,
    SafetyFlags,
    ListFilesArgs, ReadFileArgs, SearchTextArgs, InspectTaskArgs,
    ProposePatchArgs, ApplyPatchArgs, RollbackPatchArgs,
    RunTestsArgs, InspectErrorArgs, WriteMemoryArgs, FinishArgs,
)

if TYPE_CHECKING:
    from src.agent_model_provider import SentinelAction

_ACTION_ADAPTER = TypeAdapter(Action)

_ALLOWED_ACTION_TYPES = frozenset({
    "list_files", "read_file", "search_text", "inspect_task",
    "propose_patch", "apply_patch", "rollback_patch", "run_tests",
    "inspect_error", "write_memory", "finish",
})

# action_type -> Args model class, for independent arguments validation.
_ACTION_ARGS_MAP = {
    "list_files": ListFilesArgs,
    "read_file": ReadFileArgs,
    "search_text": SearchTextArgs,
    "inspect_task": InspectTaskArgs,
    "propose_patch": ProposePatchArgs,
    "apply_patch": ApplyPatchArgs,
    "rollback_patch": RollbackPatchArgs,
    "run_tests": RunTestsArgs,
    "inspect_error": InspectErrorArgs,
    "write_memory": WriteMemoryArgs,
    "finish": FinishArgs,
}

# Actions whose arguments have default_factory (may be omitted or None).
_ARGS_OPTIONAL = frozenset({"list_files", "inspect_task", "run_tests"})


class ProtocolDiagnostics(BaseModel):
    """Per-step diagnostics computed independently by the protocol layer.

    Issue #32 Trust Repair: each field is computed by an independent
    check_* method. Previously, when validate_action() succeeded, all four
    validity fields were set to True together, and when it failed, all
    were set to False together — making it impossible to tell which
    dimension actually failed. Unknown fields were also silently dropped
    by Pydantic's default extra="ignore", inflating schema_valid_rate.

    Now:
    - format_parse_ok: protocol text parsed to a structured dict
    - action_type_valid: raw action_type is one of the 11 allowed types
    - arguments_valid: raw arguments have no unknown fields, required
      fields present, types correct (validated against the per-action
      Args model with extra="forbid")
    - safety_valid: raw safety_flags has all 5 fields, no unknown fields,
      correct types, AND network_required=False, reads_sensitive_path=False
    - schema_valid: the complete raw payload passes the full Action union
      validation (including extra="forbid" on every submodel) WITHOUT
      dropping any unknown field
    """
    raw_output: str
    format_parse_ok: bool       # Protocol format syntax correct
    schema_valid: bool          # Full Action union validation passed
    safety_valid: bool          # safety_flags independent check passed
    action_type_valid: bool     # action_type in allowed 11 types
    arguments_valid: bool       # Arguments independent check passed
    repair_attempted: bool      # Format repair was attempted
    repair_success: bool        # Format repair succeeded
    latency_ms: int             # Parse latency in milliseconds
    failure_class: str | None = None  # Failure classification


class ProtocolBase(ABC):
    """Abstract base for action protocols.

    A protocol defines:
    - How to instruct the model to format its output (build_system_prompt)
    - How to parse model output into an Action (parse_output)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol identifier (e.g. 'json', 'tag', 'dsl')."""

    @abstractmethod
    def build_system_prompt(self, task_context: str) -> str:
        """Build system prompt with protocol format instructions and tool semantics."""

    @abstractmethod
    def parse_output(self, raw: str) -> tuple[Action | "SentinelAction", ProtocolDiagnostics]:
        """Parse model output into Action or SentinelAction with diagnostics."""

    @staticmethod
    def validate_action(data: dict) -> Action | None:
        """Validate a dict against the Action union. Returns Action or None.

        This performs FULL schema validation (including extra="forbid" on
        all submodels). Use the check_* methods for per-dimension checks.
        """
        try:
            return _ACTION_ADAPTER.validate_python(data)
        except Exception:
            return None

    @staticmethod
    def is_valid_action_type(action_type: str) -> bool:
        """Check if action_type string is one of the 11 allowed types."""
        return action_type in _ALLOWED_ACTION_TYPES

    @staticmethod
    def check_action_type_valid(data: dict) -> bool:
        """Check ONLY if action_type is one of the 11 allowed types.

        Does not validate arguments, safety_flags, or any other field.
        """
        return data.get("action_type") in _ALLOWED_ACTION_TYPES

    @staticmethod
    def check_safety_valid(data: dict) -> bool:
        """Check ONLY safety_flags: 5 fields complete, no unknown fields,
        types correct, AND network_required=False, reads_sensitive_path=False.

        Does not validate action_type or arguments.
        """
        sf = data.get("safety_flags")
        if not isinstance(sf, dict):
            return False
        try:
            flags = SafetyFlags.model_validate(sf)  # extra="forbid" rejects unknown
        except Exception:
            return False
        # P4.0 hard-reject conditions
        if flags.network_required or flags.reads_sensitive_path:
            return False
        return True

    @staticmethod
    def check_arguments_valid(data: dict) -> bool:
        """Check ONLY arguments for the given action_type: no unknown fields,
        required fields present, types correct, value constraints correct.

        Does not validate action_type itself or safety_flags.
        Returns False if action_type is unknown (cannot determine Args class).
        """
        at = data.get("action_type")
        if at not in _ALLOWED_ACTION_TYPES:
            return False
        args = data.get("arguments")
        args_class = _ACTION_ARGS_MAP.get(at)
        if args_class is None:
            return False
        # Actions with default_factory may omit arguments
        if args is None:
            return at in _ARGS_OPTIONAL
        if not isinstance(args, dict):
            return False
        try:
            args_class.model_validate(args)  # extra="forbid" rejects unknown
            return True
        except Exception:
            return False

    @staticmethod
    def check_schema_valid(data: dict) -> bool:
        """Full Action union validation including extra='forbid' on every
        submodel. Returns True only if the complete payload is valid WITHOUT
        dropping any unknown field.

        If this returns True, all other check_* methods also return True.
        If this returns False, use the other check_* methods to identify
        which dimension failed.
        """
        try:
            _ACTION_ADAPTER.validate_python(data)
            return True
        except Exception:
            return False

    @staticmethod
    def classify_failure(data: dict, format_parse_ok: bool) -> str | None:
        """Classify failure based on independent dimension checks.

        Returns None if no failure detected (all dimensions pass).
        Priority order ensures FORMAT_PARSE_FAIL is not double-counted
        as UNKNOWN_ACTION_TYPE.
        """
        if not format_parse_ok:
            return "FORMAT_PARSE_FAIL"
        if not ProtocolBase.check_action_type_valid(data):
            return "UNKNOWN_ACTION_TYPE"
        if not ProtocolBase.check_safety_valid(data):
            # Distinguish FORBIDDEN_ACTION (network/sensitive=True) from
            # SCHEMA_VALIDATION_FAIL (malformed safety_flags)
            sf = data.get("safety_flags")
            if isinstance(sf, dict):
                try:
                    flags = SafetyFlags.model_validate(sf)
                    if flags.network_required or flags.reads_sensitive_path:
                        return "FORBIDDEN_ACTION"
                except Exception:
                    pass
            return "SCHEMA_VALIDATION_FAIL"
        if not ProtocolBase.check_arguments_valid(data):
            return "SCHEMA_VALIDATION_FAIL"
        if not ProtocolBase.check_schema_valid(data):
            return "SCHEMA_VALIDATION_FAIL"
        return None
