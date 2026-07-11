"""P4.1b Protocol abstraction layer.

Defines ProtocolBase ABC and ProtocolDiagnostics used by all protocol
implementations (JSON, Tag, DSL).

ProtocolDiagnostics computes each validity dimension independently,
fixing the P4.1 issue where schema_valid/safety_valid/action_type_valid/
arguments_valid were all set to True together.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter

from src.agent_actions import Action

if TYPE_CHECKING:
    from src.agent_model_provider import SentinelAction

_ACTION_ADAPTER = TypeAdapter(Action)

_ALLOWED_ACTION_TYPES = frozenset({
    "list_files", "read_file", "search_text", "inspect_task",
    "propose_patch", "apply_patch", "rollback_patch", "run_tests",
    "inspect_error", "write_memory", "finish",
})


class ProtocolDiagnostics(BaseModel):
    """Per-step diagnostics computed independently by the protocol layer.

    Unlike ModelStepDiagnostics which sets schema_valid/safety_valid/
    action_type_valid/arguments_valid all-together, each field here is
    computed independently for finer failure classification.
    """
    raw_output: str
    format_parse_ok: bool       # Protocol format syntax correct
    schema_valid: bool          # Pydantic Action validation passed
    safety_valid: bool          # safety_flags check passed
    action_type_valid: bool     # action_type in allowed 11 types
    arguments_valid: bool       # Argument types and values valid
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
        """Validate a dict against the Action union. Returns Action or None."""
        try:
            return _ACTION_ADAPTER.validate_python(data)
        except Exception:
            return None

    @staticmethod
    def is_valid_action_type(action_type: str) -> bool:
        """Check if action_type string is one of the 11 allowed types."""
        return action_type in _ALLOWED_ACTION_TYPES
