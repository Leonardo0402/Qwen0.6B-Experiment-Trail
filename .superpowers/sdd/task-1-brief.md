### Task 1: ProtocolBase ABC + ProtocolDiagnostics

**Files:**
- Create: `src/protocols/__init__.py`
- Create: `src/protocols/base.py`
- Test: `tests/test_protocol_base.py`

**Interfaces:**
- Produces: `ProtocolBase` (ABC with `name`, `build_system_prompt`, `parse_output`, `validate_action`, `is_valid_action_type`), `ProtocolDiagnostics` (Pydantic BaseModel with 10 fields)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protocol_base.py
"""Tests for ProtocolBase and ProtocolDiagnostics."""
import pytest
from src.protocols.base import ProtocolBase, ProtocolDiagnostics


def test_protocol_diagnostics_has_all_fields():
    diag = ProtocolDiagnostics(
        raw_output="test",
        format_parse_ok=True,
        schema_valid=False,
        safety_valid=False,
        action_type_valid=True,
        arguments_valid=False,
        repair_attempted=False,
        repair_success=False,
        latency_ms=42,
    )
    assert diag.failure_class is None
    assert diag.latency_ms == 42
    assert diag.raw_output == "test"


def test_protocol_diagnostics_failure_class_set():
    diag = ProtocolDiagnostics(
        raw_output="",
        format_parse_ok=False,
        schema_valid=False,
        safety_valid=False,
        action_type_valid=False,
        arguments_valid=False,
        repair_attempted=False,
        repair_success=False,
        latency_ms=0,
        failure_class="FORMAT_PARSE_FAIL",
    )
    assert diag.failure_class == "FORMAT_PARSE_FAIL"


def test_protocol_diagnostics_model_dump_works():
    """ProtocolDiagnostics must support model_dump() for trajectory recording."""
    diag = ProtocolDiagnostics(
        raw_output="x", format_parse_ok=True, schema_valid=True,
        safety_valid=True, action_type_valid=True, arguments_valid=True,
        repair_attempted=False, repair_success=False, latency_ms=5,
    )
    d = diag.model_dump()
    assert d["format_parse_ok"] is True
    assert d["schema_valid"] is True
    assert "failure_class" in d


def test_validate_action_returns_none_for_invalid():
    result = ProtocolBase.validate_action({"action_type": "nonexistent"})
    assert result is None


def test_validate_action_returns_none_for_empty():
    result = ProtocolBase.validate_action({})
    assert result is None


def test_is_valid_action_type_recognizes_11_types():
    for at in ["list_files", "read_file", "search_text", "inspect_task",
               "propose_patch", "apply_patch", "rollback_patch", "run_tests",
               "inspect_error", "write_memory", "finish"]:
        assert ProtocolBase.is_valid_action_type(at), f"{at} should be valid"


def test_is_valid_action_type_rejects_unknown():
    assert not ProtocolBase.is_valid_action_type("run_terminal")
    assert not ProtocolBase.is_valid_action_type("")
    assert not ProtocolBase.is_valid_action_type("READ_FILE")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_protocol_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.protocols'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protocols/__init__.py
"""P4.1b Protocol abstraction layer.

Exports ProtocolBase, ProtocolDiagnostics, and protocol implementations.
"""
from src.protocols.base import ProtocolBase, ProtocolDiagnostics

__all__ = ["ProtocolBase", "ProtocolDiagnostics"]
```

```python
# src/protocols/base.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_protocol_base.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/protocols/__init__.py src/protocols/base.py tests/test_protocol_base.py
git commit -m "feat(protocols): add ProtocolBase ABC and ProtocolDiagnostics (P4.1b T1)"
```
