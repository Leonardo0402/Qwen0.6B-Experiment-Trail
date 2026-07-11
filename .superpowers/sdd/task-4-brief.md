### Task 4: DslProtocol

**Files:**
- Create: `src/protocols/dsl_protocol.py`
- Modify: `src/protocols/__init__.py`
- Test: `tests/test_protocol_dsl.py`

**Interfaces:**
- Consumes: `ProtocolBase`, `ProtocolDiagnostics` from Task 1; `SentinelAction` from `src.agent_model_provider`
- Produces: `DslProtocol` class with `name="dsl"`, `build_system_prompt(task_context)`, `parse_output(raw)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_protocol_dsl.py
"""Tests for DslProtocol — one-line DSL action protocol."""
import pytest
from src.protocols.dsl_protocol import DslProtocol
from src.agent_model_provider import SentinelAction


def _make_protocol():
    return DslProtocol()


def test_valid_dsl_action_parses():
    proto = _make_protocol()
    raw = "ACTION read_file path=solution.py"
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert action.action_type == "read_file"
    assert diag.format_parse_ok
    assert diag.schema_valid


def test_heredoc_multiline_handled():
    proto = _make_protocol()
    raw = (
        "ACTION propose_patch file_path=solution.py\n"
        "old_text <<EOF\n"
        "def old():\n"
        "    pass\n"
        "EOF\n"
        "new_text <<EOF\n"
        "def new():\n"
        "    return 42\n"
        "EOF\n"
    )
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert action.action_type == "propose_patch"
    assert "def old()" in action.arguments.old_text
    assert "def new()" in action.arguments.new_text
    assert diag.schema_valid


def test_malformed_dsl_fails():
    proto = _make_protocol()
    raw = "I think we should read the file."
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert not diag.format_parse_ok
    assert diag.failure_class == "FORMAT_PARSE_FAIL"


def test_missing_value_fails():
    proto = _make_protocol()
    raw = "ACTION read_file path="
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert diag.format_parse_ok  # DSL format is recognized
    assert not diag.schema_valid  # but empty path fails validation


def test_unknown_action_type_fails():
    proto = _make_protocol()
    raw = "ACTION run_terminal command=rm"
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert not diag.action_type_valid
    assert diag.failure_class == "UNKNOWN_ACTION_TYPE"


def test_forbidden_path_fails():
    proto = _make_protocol()
    raw = "ACTION read_file path=../etc/passwd"
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert diag.action_type_valid
    assert not diag.schema_valid
    assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"


def test_empty_output_fails():
    proto = _make_protocol()
    action, diag = proto.parse_output("")
    assert isinstance(action, SentinelAction)
    assert not diag.format_parse_ok
    assert diag.failure_class == "FORMAT_PARSE_FAIL"


def test_build_system_prompt_contains_format_instructions():
    proto = _make_protocol()
    prompt = proto.build_system_prompt("Fix the bug in solution.py")
    assert "ACTION" in prompt
    assert "read_file" in prompt
    assert "Fix the bug" in prompt


def test_minimal_dsl_parses_with_defaults():
    """DSL protocol should fill in defaults for missing fields."""
    proto = _make_protocol()
    raw = "ACTION finish summary=done"
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert action.action_type == "finish"
    assert diag.schema_valid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_protocol_dsl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.protocols.dsl_protocol'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protocols/dsl_protocol.py
"""Protocol C — One-line DSL action protocol.

Minimum-entropy format: model outputs a single ACTION line with key=value
pairs. Multiline content uses heredoc syntax (<<EOF...EOF).

Format:
    ACTION read_file path=solution.py

Heredoc for multiline:
    ACTION propose_patch file_path=solution.py
    old_text <<EOF
    def old():
        pass
    EOF
    new_text <<EOF
    def new():
        return 42
    EOF

Parser fills in default safety_flags and other required ActionBase fields.
Unknown action_type hard-fails. Missing values hard-fail.
"""
from __future__ import annotations

import re
import time

from src.agent_actions import Action
from src.agent_model_provider import SentinelAction
from src.protocols.base import ProtocolBase, ProtocolDiagnostics

_ACTION_LINE_RE = re.compile(r"^ACTION\s+(\w+)\s*(.*)$", re.MULTILINE)
_HEREDOC_START_RE = re.compile(r"^(\w+)\s*<<EOF\s*$", re.MULTILINE)
_KV_RE = re.compile(r'(\w+)=(".*?"|\S+)')

_TOOL_LIST = (
    "list_files, read_file, search_text, inspect_task, "
    "propose_patch, apply_patch, rollback_patch, run_tests, "
    "inspect_error, write_memory, finish"
)

_ARGUMENT_KEYS = frozenset({
    "path", "file_path", "query", "file_glob", "max_results",
    "old_text", "new_text", "action_id_ref", "test_path", "timeout_s",
    "error_source", "notes", "hypothesis",
    "success_criterion", "tests_passed", "identification_verified",
    "summary", "pattern", "start_line", "end_line",
    "expected_before_sha256",
})


def _default_safety_flags(action_type: str) -> dict:
    """Infer safe safety_flags from action_type."""
    return {
        "modifies_workspace": action_type in ("apply_patch", "rollback_patch"),
        "executes_code": action_type == "run_tests",
        "network_required": False,
        "reads_sensitive_path": False,
        "is_terminal": action_type == "finish",
    }


class DslProtocol(ProtocolBase):
    """One-line DSL protocol — minimum-entropy format."""

    @property
    def name(self) -> str:
        return "dsl"

    def build_system_prompt(self, task_context: str) -> str:
        return "\n".join([
            f"Task: {task_context}",
            "",
            f"Choose ONE action from: {_TOOL_LIST}",
            "",
            "Format your action as a single line:",
            "ACTION <action_type> <key>=<value> <key>=<value>...",
            "",
            "For multiline content, use heredoc:",
            "ACTION <action_type> <key>=<value>",
            "<multiline_key> <<EOF",
            "<content line 1>",
            "<content line 2>",
            "EOF",
            "",
            "Examples:",
            "ACTION read_file path=solution.py",
            "",
            "ACTION propose_patch file_path=solution.py",
            "old_text <<EOF",
            "def old():",
            "    pass",
            "EOF",
            "new_text <<EOF",
            "def new():",
            "    return 42",
            "EOF",
        ])

    def parse_output(self, raw: str) -> tuple[Action | SentinelAction, ProtocolDiagnostics]:
        t0 = time.monotonic()
        latency_ms = int((time.monotonic() - t0) * 1000)

        m = _ACTION_LINE_RE.search(raw)
        format_parse_ok = m is not None

        diag = ProtocolDiagnostics(
            raw_output=raw,
            format_parse_ok=format_parse_ok,
            schema_valid=False,
            safety_valid=False,
            action_type_valid=False,
            arguments_valid=False,
            repair_attempted=False,
            repair_success=False,
            latency_ms=latency_ms,
        )

        if not format_parse_ok:
            diag.failure_class = "FORMAT_PARSE_FAIL"
            return SentinelAction(reason="no ACTION line found"), diag

        action_type = m.group(1)
        rest_of_line = m.group(2)
        diag.action_type_valid = self.is_valid_action_type(action_type)

        if not diag.action_type_valid:
            diag.failure_class = "UNKNOWN_ACTION_TYPE"
            return SentinelAction(reason=f"unknown action type: {action_type}"), diag

        # Parse key=value pairs from the ACTION line
        arguments: dict = {}
        for km in _KV_RE.finditer(rest_of_line):
            key = km.group(1)
            value = km.group(2)
            # Strip quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            arguments[key] = value

        # Parse heredoc blocks from the rest of the output
        # Find the position after the ACTION line
        action_line_end = m.end()
        remaining = raw[action_line_end:]

        heredoc_matches = list(_HEREDOC_START_RE.finditer(remaining))
        for hm in heredoc_matches:
            key = hm.group(1)
            content_start = hm.end()
            # Find the EOF marker
            eof_pos = remaining.find("\nEOF", content_start)
            if eof_pos == -1:
                # Unclosed heredoc
                diag.failure_class = "FORMAT_PARSE_FAIL"
                return SentinelAction(reason=f"unclosed heredoc for {key}"), diag
            content = remaining[content_start + 1:eof_pos]  # +1 to skip newline
            arguments[key] = content

        # Build data dict for Pydantic validation
        data: dict = {
            "action_type": action_type,
            "action_id": arguments.pop("action_id", f"dsl_{action_type}"),
            "reason_short": arguments.pop("reason", "no reason provided"),
            "expected_observation": arguments.pop("expected", "unknown"),
            "safety_flags": _default_safety_flags(action_type),
        }

        # Fill in default argument values for finish (reduces model entropy)
        if action_type == "finish":
            arguments.setdefault("success_criterion", "test_pass")
            arguments.setdefault("tests_passed", "false")
            arguments.setdefault("identification_verified", "false")

        # Coerce argument types
        arguments = self._coerce_arguments(action_type, arguments)
        if arguments:
            data["arguments"] = arguments

        # Validate against Action schema
        action = self.validate_action(data)
        if action is not None:
            diag.schema_valid = True
            diag.safety_valid = True
            diag.arguments_valid = True
            return action, diag

        diag.failure_class = "SCHEMA_VALIDATION_FAIL"
        return SentinelAction(reason="dsl action failed schema validation"), diag

    @staticmethod
    def _coerce_arguments(action_type: str, arguments: dict) -> dict:
        """Coerce string values to correct types for known argument fields."""
        coerced = dict(arguments)
        for bool_key in ("tests_passed", "identification_verified"):
            if bool_key in coerced:
                coerced[bool_key] = coerced[bool_key].lower() in ("true", "yes", "1")
        for int_key in ("max_results", "start_line", "end_line"):
            if int_key in coerced:
                try:
                    coerced[int_key] = int(coerced[int_key])
                except ValueError:
                    pass
        for float_key in ("timeout_s",):
            if float_key in coerced:
                try:
                    coerced[float_key] = float(coerced[float_key])
                except ValueError:
                    pass
        return coerced
```

```python
# src/protocols/__init__.py (updated)
"""P4.1b Protocol abstraction layer."""
from src.protocols.base import ProtocolBase, ProtocolDiagnostics
from src.protocols.json_protocol import JsonProtocol
from src.protocols.tag_protocol import TagProtocol
from src.protocols.dsl_protocol import DslProtocol

__all__ = [
    "ProtocolBase", "ProtocolDiagnostics",
    "JsonProtocol", "TagProtocol", "DslProtocol",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_protocol_dsl.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/protocols/dsl_protocol.py src/protocols/__init__.py tests/test_protocol_dsl.py
git commit -m "feat(protocols): add DslProtocol with heredoc support (P4.1b T4)"
```
