# P4.1b Protocol Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare 3 action protocols (JSON baseline, XML Tag, One-line DSL) on Qwen3-0.6B across 40 micro-tasks × 2 configs (240 runs total) to determine which protocol achieves the highest schema validity and safety for P4.2 Agent SFT training.

**Architecture:** New `src/protocols/` module with `ProtocolBase` ABC and 3 protocol implementations. `ModelActionProvider` gains an optional `protocol` parameter — when set, it delegates prompt building and output parsing to the protocol; when `None`, existing code path is unchanged (backward-compatible). A new `scripts/run_protocol_ablation.py` script orchestrates the 6-protocol×config combinations, writes trajectories, and generates a comparison report with a final verdict.

**Tech Stack:** Python 3.11, Pydantic v2 (existing), regex (existing), pytest (existing). No new dependencies.

## Global Constraints

- Same safe tool layer: 11 actions only (no shell/network/credential/Git push)
- RTX 3050 Laptop 4GB VRAM: model loaded with `torch.float16`, `device_map={"": "cuda:0"}`
- Deterministic generation: `temperature=0.0, do_sample=False` for all runs
- No training, no external data, no model weights committed
- No new dependencies (pydantic + regex only)
- 40 micro-tasks from `data/p4-agent/micro-tasks-v0/manifest.json`
- All 3 protocols map to the same Pydantic `Action` schema (Action definition unchanged)
- Parsers fill in default `safety_flags` for Tag/DSL protocols (reduces model entropy)
- Unknown or malformed actions hard-fail to `SentinelAction` (no silent no-op)

---

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

---

### Task 2: JsonProtocol (Baseline)

**Files:**
- Create: `src/protocols/json_protocol.py`
- Modify: `src/protocols/__init__.py`
- Test: `tests/test_protocol_json.py`

**Interfaces:**
- Consumes: `ProtocolBase`, `ProtocolDiagnostics` from Task 1; `SentinelAction`, `extract_json`, `repair_json` from `src.agent_model_provider`
- Produces: `JsonProtocol` class with `name="json"`, `build_system_prompt(task_context)`, `parse_output(raw)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_protocol_json.py
"""Tests for JsonProtocol — baseline JSON action protocol."""
import json
import pytest
from src.protocols.json_protocol import JsonProtocol
from src.agent_model_provider import SentinelAction


VALID_ACTION = {
    "action_type": "read_file",
    "action_id": "a1",
    "reason_short": "inspect failing file",
    "expected_observation": "file contents",
    "safety_flags": {
        "modifies_workspace": False,
        "executes_code": False,
        "network_required": False,
        "reads_sensitive_path": False,
        "is_terminal": False,
    },
    "arguments": {"path": "solution.py"},
}


def _make_protocol():
    return JsonProtocol()


def test_valid_action_parses():
    proto = _make_protocol()
    raw = json.dumps(VALID_ACTION)
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert action.action_type == "read_file"
    assert diag.format_parse_ok
    assert diag.schema_valid
    assert diag.safety_valid
    assert diag.action_type_valid
    assert diag.arguments_valid
    assert diag.failure_class is None


def test_fenced_json_parses():
    proto = _make_protocol()
    raw = f"Here is the action:\n```json\n{json.dumps(VALID_ACTION)}\n```\nDone."
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert diag.format_parse_ok
    assert diag.schema_valid


def test_malformed_json_fails():
    proto = _make_protocol()
    raw = '{"action_type": "read_file", "path": "solution.py"'
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert diag.failure_class is not None


def test_unknown_action_type_fails():
    proto = _make_protocol()
    data = {**VALID_ACTION, "action_type": "run_terminal"}
    action, diag = proto.parse_output(json.dumps(data))
    assert isinstance(action, SentinelAction)
    assert not diag.action_type_valid
    assert diag.failure_class == "UNKNOWN_ACTION_TYPE"


def test_forbidden_path_fails():
    proto = _make_protocol()
    data = json.loads(json.dumps(VALID_ACTION))
    data["arguments"]["path"] = "../etc/passwd"
    action, diag = proto.parse_output(json.dumps(data))
    assert isinstance(action, SentinelAction)
    assert diag.action_type_valid  # action_type is valid
    assert not diag.schema_valid   # but schema fails (path validation)
    assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"


def test_missing_required_field_fails():
    proto = _make_protocol()
    raw = json.dumps({"action_type": "read_file"})  # missing required fields
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert diag.action_type_valid  # action_type is valid
    assert not diag.schema_valid   # missing other required fields


def test_repair_fixes_trailing_comma():
    proto = _make_protocol()
    raw = '{"action_type": "read_file", "action_id": "a1", "reason_short": "x", "expected_observation": "y", "safety_flags": {"modifies_workspace": false, "executes_code": false, "network_required": false, "reads_sensitive_path": false, "is_terminal": false}, "arguments": {"path": "x.py",},}'
    action, diag = proto.parse_output(raw)
    assert diag.repair_attempted
    assert diag.repair_success
    assert diag.schema_valid
    assert not isinstance(action, SentinelAction)


def test_repair_does_not_change_semantics():
    proto = _make_protocol()
    raw = '{"action_type": "???"}'
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    # repair must not have substituted a valid action_type
    assert not diag.action_type_valid


def test_empty_output_fails():
    proto = _make_protocol()
    action, diag = proto.parse_output("")
    assert isinstance(action, SentinelAction)
    assert not diag.format_parse_ok
    assert diag.failure_class == "FORMAT_PARSE_FAIL"


def test_repair_path_returns_action():
    """Regression: P4.1 bug where repair validated but never returned action."""
    proto = _make_protocol()
    # Construct JSON that needs repair (trailing comma) but is otherwise valid
    raw = json.dumps(VALID_ACTION)[:-1] + ",}"
    action, diag = proto.parse_output(raw)
    assert diag.repair_attempted
    assert diag.repair_success
    assert not isinstance(action, SentinelAction), "repaired action must be returned"


def test_build_system_prompt_contains_format_instructions():
    proto = _make_protocol()
    prompt = proto.build_system_prompt("Fix the bug in solution.py")
    assert "JSON" in prompt
    assert "action_type" in prompt
    assert "read_file" in prompt
    assert "Fix the bug" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_protocol_json.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.protocols.json_protocol'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protocols/json_protocol.py
"""Protocol A — Action JSON (baseline).

Extracts existing parsing logic from ModelActionProvider into a protocol
class. Computes diagnostics dimensions independently (fixes P4.1 issue
where schema_valid/safety_valid/action_type_valid/arguments_valid were
all set to True together).

Also fixes the repair-path bug (P4.1 agent_model_provider.py lines 220-230):
when repair succeeds, the validated action is now returned instead of
falling through to SentinelAction.
"""
from __future__ import annotations

import json
import time

from src.agent_actions import Action
from src.agent_model_provider import SentinelAction, extract_json, repair_json
from src.protocols.base import ProtocolBase, ProtocolDiagnostics

_TOOL_LIST = (
    "list_files, read_file, search_text, inspect_task, "
    "propose_patch, apply_patch, rollback_patch, run_tests, "
    "inspect_error, write_memory, finish"
)


class JsonProtocol(ProtocolBase):
    """Baseline JSON protocol — same format as P4.1 ModelActionProvider."""

    @property
    def name(self) -> str:
        return "json"

    def build_system_prompt(self, task_context: str) -> str:
        return "\n".join([
            f"Task: {task_context}",
            "",
            f"Choose ONE action from: {_TOOL_LIST}",
            "",
            "Respond with a single JSON object with these fields:",
            '- "action_type": one of the 11 action types above',
            '- "action_id": a short unique identifier',
            '- "reason_short": why you chose this action (max 120 chars)',
            '- "expected_observation": what you expect to see',
            '- "safety_flags": {"modifies_workspace": bool, "executes_code": bool,',
            '  "network_required": false, "reads_sensitive_path": false, "is_terminal": bool}',
            '- "arguments": action-specific arguments',
            "",
            "Example:",
            '{"action_type": "read_file", "action_id": "a1", "reason_short": "inspect file",',
            ' "expected_observation": "file contents", "safety_flags": {"modifies_workspace": false,',
            ' "executes_code": false, "network_required": false, "reads_sensitive_path": false,',
            ' "is_terminal": false}, "arguments": {"path": "solution.py"}}',
        ])

    def parse_output(self, raw: str) -> tuple[Action | SentinelAction, ProtocolDiagnostics]:
        t0 = time.monotonic()
        json_str = extract_json(raw)
        latency_ms = int((time.monotonic() - t0) * 1000)

        format_parse_ok = json_str is not None

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
            if not raw.strip():
                diag.failure_class = "FORMAT_PARSE_FAIL"
            else:
                diag.failure_class = "FORMAT_PARSE_FAIL"
            return SentinelAction(reason="json parse failed"), diag

        # Parse JSON and check action_type independently
        try:
            data = json.loads(json_str)
            action_type = data.get("action_type", "") if isinstance(data, dict) else ""
            diag.action_type_valid = self.is_valid_action_type(action_type)
        except (json.JSONDecodeError, Exception):
            diag.failure_class = "FORMAT_PARSE_FAIL"
            return SentinelAction(reason="json decode failed"), diag

        # Try direct validation
        action = self.validate_action(data)
        if action is not None:
            diag.schema_valid = True
            diag.safety_valid = True
            diag.arguments_valid = True
            return action, diag

        # Attempt format-only repair
        diag.repair_attempted = True
        repaired = repair_json(json_str)
        try:
            data = json.loads(repaired)
            action = self.validate_action(data)
            if action is not None:
                # FIX: P4.1 bug — was missing `return action` here
                diag.repair_success = True
                diag.schema_valid = True
                diag.safety_valid = True
                diag.arguments_valid = True
                return action, diag
        except (json.JSONDecodeError, Exception):
            pass

        # Classify failure
        if not diag.action_type_valid:
            diag.failure_class = "UNKNOWN_ACTION_TYPE"
        else:
            diag.failure_class = "SCHEMA_VALIDATION_FAIL"
        return SentinelAction(reason="schema validation failed after repair"), diag
```

```python
# src/protocols/__init__.py (updated)
"""P4.1b Protocol abstraction layer.

Exports ProtocolBase, ProtocolDiagnostics, and protocol implementations.
"""
from src.protocols.base import ProtocolBase, ProtocolDiagnostics
from src.protocols.json_protocol import JsonProtocol

__all__ = ["ProtocolBase", "ProtocolDiagnostics", "JsonProtocol"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_protocol_json.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/protocols/json_protocol.py src/protocols/__init__.py tests/test_protocol_json.py
git commit -m "feat(protocols): add JsonProtocol baseline with independent diagnostics (P4.1b T2)"
```

---

### Task 3: TagProtocol

**Files:**
- Create: `src/protocols/tag_protocol.py`
- Modify: `src/protocols/__init__.py`
- Test: `tests/test_protocol_tag.py`

**Interfaces:**
- Consumes: `ProtocolBase`, `ProtocolDiagnostics` from Task 1; `SentinelAction` from `src.agent_model_provider`
- Produces: `TagProtocol` class with `name="tag"`, `build_system_prompt(task_context)`, `parse_output(raw)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_protocol_tag.py
"""Tests for TagProtocol — XML Tag action protocol."""
import pytest
from src.protocols.tag_protocol import TagProtocol
from src.agent_model_provider import SentinelAction


def _make_protocol():
    return TagProtocol()


def test_valid_tag_action_parses():
    proto = _make_protocol()
    raw = "<action>\ntool: read_file\npath: solution.py\n</action>"
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert action.action_type == "read_file"
    assert diag.format_parse_ok
    assert diag.schema_valid
    assert diag.action_type_valid


def test_think_block_ignored():
    proto = _make_protocol()
    raw = "<think>I should read the file first.</think>\n<action>\ntool: read_file\npath: solution.py\n</action>"
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert diag.format_parse_ok


def test_content_subtag_handled():
    proto = _make_protocol()
    raw = (
        "<action>\n"
        "tool: propose_patch\n"
        "file_path: solution.py\n"
        "old_text: <content>def old():\n    pass</content>\n"
        "new_text: <content>def new():\n    return 42</content>\n"
        "</action>"
    )
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert action.action_type == "propose_patch"
    assert "def old()" in action.arguments.old_text
    assert "def new()" in action.arguments.new_text
    assert diag.schema_valid


def test_unclosed_tag_fails():
    proto = _make_protocol()
    raw = "<action>\ntool: read_file\npath: solution.py"
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert not diag.format_parse_ok
    assert diag.failure_class == "FORMAT_PARSE_FAIL"


def test_unknown_key_fails():
    proto = _make_protocol()
    raw = "<action>\ntool: read_file\npath: solution.py\nbadkey: value\n</action>"
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert diag.format_parse_ok  # tag format is correct
    assert not diag.schema_valid  # but unknown key fails validation


def test_unknown_action_type_fails():
    proto = _make_protocol()
    raw = "<action>\ntool: run_terminal\ncommand: rm -rf /\n</action>"
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert not diag.action_type_valid
    assert diag.failure_class == "UNKNOWN_ACTION_TYPE"


def test_forbidden_path_fails():
    proto = _make_protocol()
    raw = "<action>\ntool: read_file\npath: ../etc/passwd\n</action>"
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
    assert "<action>" in prompt
    assert "tool:" in prompt
    assert "read_file" in prompt
    assert "Fix the bug" in prompt


def test_minimal_tag_parses_with_defaults():
    """Tag protocol should fill in defaults for missing fields."""
    proto = _make_protocol()
    raw = "<action>\ntool: finish\nsummary: done\n</action>"
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert action.action_type == "finish"
    assert diag.schema_valid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_protocol_tag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.protocols.tag_protocol'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protocols/tag_protocol.py
"""Protocol B — XML Tag action protocol.

Lower-entropy format: model outputs key-value pairs inside <action> tags.
Parser fills in default safety_flags and other required ActionBase fields
to reduce the model's formatting burden.

Format:
    <action>
    tool: read_file
    path: solution.py
    </action>

Multiline content uses <content>...</content> subtags:
    <action>
    tool: propose_patch
    file_path: solution.py
    old_text: <content>def old():
        pass</content>
    new_text: <content>def new():
        return 42</content>
    </action>

<think>...</think> blocks are ignored.
Unknown keys hard-fail (not silently ignored).
"""
from __future__ import annotations

import re
import time

from src.agent_actions import Action
from src.agent_model_provider import SentinelAction
from src.protocols.base import ProtocolBase, ProtocolDiagnostics

_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_CONTENT_RE = re.compile(r"<content>(.*?)</content>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_TOOL_LIST = (
    "list_files, read_file, search_text, inspect_task, "
    "propose_patch, apply_patch, rollback_patch, run_tests, "
    "inspect_error, write_memory, finish"
)

# Keys recognized by the tag protocol.
# "tool" maps to action_type; others map to ActionBase fields or arguments.
_BASE_FIELD_MAP = {
    "action_id": "action_id",
    "reason": "reason_short",
    "expected": "expected_observation",
}

_ARGUMENT_KEYS = frozenset({
    "path", "file_path", "query", "file_glob", "max_results",
    "old_text", "new_text", "action_id_ref", "test_path", "timeout_s",
    "error_source", "notes", "hypothesis",
    "success_criterion", "tests_passed", "identification_verified",
    "summary", "pattern", "start_line", "end_line",
    "expected_before_sha256",
})

_FINISH_KEYS = frozenset({
    "success_criterion", "tests_passed", "identification_verified", "summary",
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


def _parse_kv_pairs(block: str) -> dict[str, str]:
    """Parse key-value pairs from an <action> block.

    Handles <content>...</content> subtags for multiline values.
    """
    # Extract content subtags and replace with placeholders
    contents: list[str] = []

    def _store_content(m):
        contents.append(m.group(1))
        return f"__CONTENT_{len(contents) - 1}__"

    block = _CONTENT_RE.sub(_store_content, block)

    result: dict[str, str] = {}
    for line in block.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Restore content placeholders
        for i, content in enumerate(contents):
            value = value.replace(f"__CONTENT_{i}__", content)
        result[key] = value
    return result


class TagProtocol(ProtocolBase):
    """XML Tag protocol — lower-entropy key-value format."""

    @property
    def name(self) -> str:
        return "tag"

    def build_system_prompt(self, task_context: str) -> str:
        return "\n".join([
            f"Task: {task_context}",
            "",
            f"Choose ONE action from: {_TOOL_LIST}",
            "",
            "Format your action as:",
            "<action>",
            "tool: <action_type>",
            "<argument_key>: <value>",
            "</action>",
            "",
            "Rules:",
            "- 'tool' is the action type (e.g. read_file, run_tests, finish)",
            "- Provide only the arguments needed for that action type",
            "- For multiline content (old_text, new_text), use: <content>...</content>",
            "- You may use <think>...</think> to reason before the action",
            "",
            "Examples:",
            "<action>",
            "tool: read_file",
            "path: solution.py",
            "</action>",
            "",
            "<action>",
            "tool: propose_patch",
            "file_path: solution.py",
            "old_text: <content>def old():\n    pass</content>",
            "new_text: <content>def new():\n    return 42</content>",
            "</action>",
        ])

    def parse_output(self, raw: str) -> tuple[Action | SentinelAction, ProtocolDiagnostics]:
        t0 = time.monotonic()
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Strip <think> blocks
        cleaned = _THINK_RE.sub("", raw)

        m = _ACTION_TAG_RE.search(cleaned)
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
            return SentinelAction(reason="no <action> tag found"), diag

        block = m.group(1)
        kv = _parse_kv_pairs(block)

        # Extract action_type
        action_type = kv.pop("tool", "")
        diag.action_type_valid = self.is_valid_action_type(action_type)

        if not diag.action_type_valid:
            diag.failure_class = "UNKNOWN_ACTION_TYPE"
            return SentinelAction(reason=f"unknown action type: {action_type}"), diag

        # Build data dict for Pydantic validation
        data: dict = {
            "action_type": action_type,
            "action_id": kv.pop("action_id", f"tag_{action_type}"),
            "reason_short": kv.pop("reason", "no reason provided"),
            "expected_observation": kv.pop("expected", "unknown"),
            "safety_flags": _default_safety_flags(action_type),
        }

        # Map remaining keys to arguments
        arguments: dict = {}
        for key, value in kv.items():
            if key in _ARGUMENT_KEYS or key in _FINISH_KEYS:
                arguments[key] = value
            else:
                # Unknown key — will cause schema validation failure
                arguments[key] = value

        # Type coercion for known argument fields
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
        return SentinelAction(reason="tag action failed schema validation"), diag

    @staticmethod
    def _coerce_arguments(action_type: str, arguments: dict) -> dict:
        """Coerce string values to correct types for known argument fields."""
        coerced = dict(arguments)
        # Boolean coercion
        for bool_key in ("tests_passed", "identification_verified"):
            if bool_key in coerced:
                coerced[bool_key] = coerced[bool_key].lower() in ("true", "yes", "1")
        # Numeric coercion
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

__all__ = ["ProtocolBase", "ProtocolDiagnostics", "JsonProtocol", "TagProtocol"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_protocol_tag.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/protocols/tag_protocol.py src/protocols/__init__.py tests/test_protocol_tag.py
git commit -m "feat(protocols): add TagProtocol with content subtag support (P4.1b T3)"
```

---

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

---

### Task 5: ModelActionProvider Protocol Adapter

**Files:**
- Modify: `src/agent_model_provider.py:124-135` (add `protocol` param), `:178-233` (protocol delegation in `next_action`), `:238-240` (diagnostics property)
- Test: `tests/test_agent_model_provider_protocol.py`

**Interfaces:**
- Consumes: `ProtocolBase`, `ProtocolDiagnostics` from Task 1; `JsonProtocol` from Task 2
- Produces: `ModelActionProvider(protocol=...)` — when protocol is set, delegates prompt/parsing to protocol; when `None`, uses existing code path (backward-compatible)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_model_provider_protocol.py
"""Tests for ModelActionProvider protocol adapter.

Verifies that:
1. protocol=None preserves existing behavior (backward compat)
2. protocol set delegates to protocol.build_system_prompt and parse_output
3. ProtocolDiagnostics are recorded when protocol is used
4. Existing ModelStepDiagnostics are recorded when protocol=None
"""
import pytest
from unittest.mock import patch, MagicMock
from src.agent_model_provider import ModelActionProvider, SentinelAction
from src.agent_evaluator import AgentState
from src.agent_state import AgentMemory
from src.protocols import JsonProtocol, TagProtocol
from src.protocols.base import ProtocolDiagnostics


def _make_state():
    return AgentState(
        memory=AgentMemory(),
        step_count=0,
        task_id="task_001",
        workspace_id="test_ws",
    )


def test_protocol_none_uses_legacy_path():
    """When protocol=None, existing code path is used (ModelStepDiagnostics)."""
    provider = ModelActionProvider(model_path="fake", protocol=None)
    # Mock _generate to return invalid JSON
    with patch.object(provider, "_generate", return_value="not json at all"):
        provider._model = MagicMock()  # skip model loading
        action = provider.next_action(_make_state())
        assert isinstance(action, SentinelAction)
        # Legacy path stores ModelStepDiagnostics
        assert len(provider.diagnostics) == 1
        # ModelStepDiagnostics has json_parse_ok, not format_parse_ok
        d = provider.diagnostics[0].model_dump()
        assert "json_parse_ok" in d


def test_protocol_set_uses_protocol_path():
    """When protocol is set, protocol.parse_output is used."""
    proto = JsonProtocol()
    provider = ModelActionProvider(model_path="fake", protocol=proto)
    # Mock _generate to return empty (will fail parse)
    with patch.object(provider, "_generate", return_value=""):
        provider._model = MagicMock()
        action = provider.next_action(_make_state())
        assert isinstance(action, SentinelAction)
        # Protocol path stores ProtocolDiagnostics
        assert len(provider.diagnostics) == 1
        d = provider.diagnostics[0].model_dump()
        assert "format_parse_ok" in d
        assert "json_parse_ok" not in d


def test_protocol_set_builds_system_prompt():
    """When protocol is set, build_system_prompt is called with task context."""
    proto = JsonProtocol()
    provider = ModelActionProvider(model_path="fake", protocol=proto)
    captured_prompt = []

    def fake_generate(prompt):
        captured_prompt.append(prompt)
        return ""

    with patch.object(provider, "_generate", side_effect=fake_generate):
        provider._model = MagicMock()
        provider.next_action(_make_state())
        assert len(captured_prompt) == 1
        # System prompt should contain format instructions from protocol
        assert "JSON" in captured_prompt[0] or "action_type" in captured_prompt[0]
        # Should contain task context
        assert "task_001" in captured_prompt[0]


def test_protocol_set_records_protocol_diagnostics():
    """ProtocolDiagnostics are recorded with independent dimensions."""
    proto = TagProtocol()
    provider = ModelActionProvider(model_path="fake", protocol=proto)
    with patch.object(provider, "_generate", return_value="<action>\ntool: read_file\npath: solution.py\n</action>"):
        provider._model = MagicMock()
        action = provider.next_action(_make_state())
        assert not isinstance(action, SentinelAction)
        assert len(provider.diagnostics) == 1
        diag = provider.diagnostics[0]
        assert isinstance(diag, ProtocolDiagnostics)
        assert diag.format_parse_ok
        assert diag.schema_valid
        assert diag.action_type_valid


def test_reset_clears_protocol_diagnostics():
    proto = JsonProtocol()
    provider = ModelActionProvider(model_path="fake", protocol=proto)
    with patch.object(provider, "_generate", return_value=""):
        provider._model = MagicMock()
        provider.next_action(_make_state())
        assert len(provider.diagnostics) == 1
    provider.reset()
    assert len(provider.diagnostics) == 0


def test_existing_tests_still_pass_with_protocol_none():
    """Smoke check: protocol=None doesn't break existing ModelActionProvider."""
    provider = ModelActionProvider(model_path="fake")
    assert provider._protocol is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_agent_model_provider_protocol.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'protocol'`

- [ ] **Step 3: Modify ModelActionProvider**

Edit `src/agent_model_provider.py`. The changes are:

1. Add `from __future__ import annotations` at top (if not already present — it IS already present at line 8)
2. Add `TYPE_CHECKING` import for protocol types
3. Add `protocol` parameter to `__init__`
4. Add protocol delegation in `next_action`
5. Add `protocol_diagnostics` list and update `diagnostics` property

```python
# src/agent_model_provider.py — modified sections

# At top, after existing imports (line 17-18), add:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.protocols.base import ProtocolBase, ProtocolDiagnostics


# Modified __init__ (lines 124-135 → new version):
class ModelActionProvider(ActionProvider):
    """Loads Qwen3-0.6B and generates actions. GPU required.

    When protocol is None: uses legacy JSON parsing (backward-compatible).
    When protocol is set: delegates prompt building and output parsing
    to the protocol object.
    """

    def __init__(
        self,
        model_path: str = "models/Qwen3-0.6B",
        adapter_path: str | None = None,
        max_new_tokens: int = 128,
        protocol: "ProtocolBase | None" = None,
    ):
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._protocol = protocol
        self._diagnostics: list[ModelStepDiagnostics] = []
        self._protocol_diagnostics: list["ProtocolDiagnostics"] = []


# Add new method before next_action (after _generate, ~line 177):
    def _build_user_prompt(self, state: AgentState) -> str:
        """Build user prompt for protocol path (state info only)."""
        lines = [
            f"Task ID: {state.task_id}",
            f"Step: {state.step_count}",
        ]
        if state.memory.notes:
            lines.append(f"Notes: {state.memory.notes}")
        if state.memory.hypothesis:
            lines.append(f"Hypothesis: {state.memory.hypothesis}")
        lines.append("What is your next action?")
        return "\n".join(lines)


# Modified next_action (lines 178-233 → new version):
    def next_action(self, state: AgentState) -> Action | SentinelAction:
        if self._model is None:
            self._load_model()

        if self._protocol is not None:
            return self._next_action_protocol(state)
        return self._next_action_legacy(state)

    def _next_action_protocol(self, state: AgentState) -> Action | SentinelAction:
        """Protocol path: delegate to protocol for prompt and parsing."""
        system_prompt = self._protocol.build_system_prompt(state.task_id)
        user_prompt = self._build_user_prompt(state)
        prompt = system_prompt + "\n\n" + user_prompt
        raw_output = self._generate(prompt)
        action, diag = self._protocol.parse_output(raw_output)
        self._protocol_diagnostics.append(diag)
        return action

    def _next_action_legacy(self, state: AgentState) -> Action | SentinelAction:
        """Legacy path: existing JSON parsing (backward-compatible)."""
        prompt = build_prompt(state, task_description="", last_observation=None)
        t0 = time.monotonic()
        raw_output = self._generate(prompt)
        latency_ms = int((time.monotonic() - t0) * 1000)

        json_str = extract_json(raw_output)
        diag = ModelStepDiagnostics(
            raw_output=raw_output,
            json_parse_ok=json_str is not None,
            schema_valid=False,
            safety_valid=False,
            action_type_valid=False,
            arguments_valid=False,
            repair_attempted=False,
            repair_success=False,
            latency_ms=latency_ms,
        )

        if json_str is None:
            self._diagnostics.append(diag)
            return SentinelAction(reason="json parse failed")

        # Try direct validation
        try:
            data = json.loads(json_str)
            action = _validate_action(data)
            if action is not None:
                diag.schema_valid = True
                diag.safety_valid = True
                diag.action_type_valid = True
                diag.arguments_valid = True
                self._diagnostics.append(diag)
                return action
        except (json.JSONDecodeError, Exception):
            pass

        # Attempt repair
        diag.repair_attempted = True
        repaired = repair_json(json_str)
        try:
            data = json.loads(repaired)
            action = _validate_action(data)
            if action is not None:
                diag.repair_success = True
                diag.schema_valid = True
                diag.safety_valid = True
                diag.action_type_valid = True
                diag.arguments_valid = True
                self._diagnostics.append(diag)
                return action  # FIX: was missing return (P4.1 bug)
        except (json.JSONDecodeError, Exception):
            pass

        self._diagnostics.append(diag)
        return SentinelAction(reason="schema validation failed after repair")


# Modified reset and diagnostics (lines 235-240 → new version):
    def reset(self) -> None:
        self._diagnostics.clear()
        self._protocol_diagnostics.clear()

    @property
    def diagnostics(self):
        """Returns ProtocolDiagnostics if protocol set, else ModelStepDiagnostics."""
        if self._protocol is not None:
            return list(self._protocol_diagnostics)
        return list(self._diagnostics)
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_model_provider_protocol.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `py -3.11 -m pytest tests/test_agent_model_provider.py -v`
Expected: PASS (all existing tests)

- [ ] **Step 6: Run broader test suite (non-GPU)**

Run: `py -3.11 -m pytest tests/ -p no:warnings --tb=no -q -m "not gpu" --timeout=120 --ignore=tests/test_data_pipeline.py --ignore=tests/test_p3_readiness_gate.py`
Expected: PASS (all non-GPU tests)

- [ ] **Step 7: Commit**

```bash
git add src/agent_model_provider.py tests/test_agent_model_provider_protocol.py
git commit -m "feat(agent-model): add protocol adapter to ModelActionProvider (P4.1b T5)

- protocol=None: legacy JSON path (backward-compatible)
- protocol set: delegates to protocol.build_system_prompt and parse_output
- Fixes P4.1 repair-path bug (validated action was not returned)
- ProtocolDiagnostics computed independently per dimension"
```

---

### Task 6: Baseline Lock + Smoke Run Script

**Files:**
- Create: `scripts/run_protocol_ablation.py`
- Create: `tests/test_protocol_ablation.py`
- Output dir: `reports/p4/protocol-ablation/`

**Interfaces:**
- Consumes: `JsonProtocol`, `TagProtocol`, `DslProtocol` from Tasks 2-4; `ModelActionProvider` from Task 5; `AgentEvaluator`, `MicroTaskWorkspace` from existing code
- Produces: `run_protocol_ablation.py` with `baseline_lock()`, `run_combination()`, `main()`; writes `baseline-lock.json` and trajectory JSONL files

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_protocol_ablation.py
"""Integration tests for protocol ablation script.

Tests use MockProtocol to avoid loading the actual model.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.protocols.base import ProtocolBase, ProtocolDiagnostics
from src.agent_model_provider import SentinelAction


class MockProtocol(ProtocolBase):
    """Protocol that returns a predefined action for testing."""
    def __init__(self, action_type="list_files"):
        self._action_type = action_type

    @property
    def name(self):
        return "mock"

    def build_system_prompt(self, task_context):
        return f"Mock prompt for {task_context}"

    def parse_output(self, raw):
        diag = ProtocolDiagnostics(
            raw_output=raw, format_parse_ok=True, schema_valid=True,
            safety_valid=True, action_type_valid=True, arguments_valid=True,
            repair_attempted=False, repair_success=False, latency_ms=1,
        )
        if self._action_type == "invalid":
            return SentinelAction(reason="mock invalid"), diag
        from src.agent_actions import ListFilesAction, SafetyFlags
        action = ListFilesAction(
            action_id="mock_a1", reason_short="mock",
            expected_observation="mock",
            safety_flags=SafetyFlags(
                modifies_workspace=False, executes_code=False,
                network_required=False, reads_sensitive_path=False,
                is_terminal=False,
            ),
        )
        return action, diag


def test_baseline_lock_records_all_fields():
    from scripts.run_protocol_ablation import baseline_lock
    lock = baseline_lock()
    assert "commit_sha" in lock
    assert "micro_task_manifest_sha256" in lock
    assert "model_path" in lock
    assert "adapter_path" in lock
    assert "generation_config" in lock
    assert lock["generation_config"]["temperature"] == 0.0
    assert lock["generation_config"]["do_sample"] is False
    assert "total_tasks" in lock
    assert lock["total_tasks"] == 40


def test_aggregate_metrics_computes_all_fields():
    from scripts.run_protocol_ablation import aggregate_metrics
    trajectories = [
        {
            "step_diagnostics": [
                {"format_parse_ok": True, "schema_valid": True,
                 "safety_valid": True, "action_type_valid": True,
                 "arguments_valid": True, "failure_class": None},
                {"format_parse_ok": True, "schema_valid": False,
                 "safety_valid": False, "action_type_valid": False,
                 "arguments_valid": False, "failure_class": "SCHEMA_VALIDATION_FAIL"},
            ],
            "metrics": {"forbidden_action_count": 0, "tool_error_rate": 0.0,
                        "max_step_exceeded_count": 1},
            "success": False,
            "finish_claim_mismatch": True,
            "steps_executed": 12,
        },
    ]
    metrics = aggregate_metrics(trajectories, crashes=0)
    # Spec §6.2 requires 12+ metrics
    assert "format_parse_rate" in metrics
    assert "schema_valid_rate" in metrics
    assert "safety_valid_rate" in metrics
    assert "action_type_valid_rate" in metrics
    assert "arguments_valid_rate" in metrics
    assert "forbidden_action_count" in metrics
    assert "unknown_action_count" in metrics
    assert "task_success_rate" in metrics
    assert "finish_without_tests_count" in metrics
    assert "finish_claim_mismatch_count" in metrics
    assert "max_steps_hit_rate" in metrics
    assert "runtime_crash_count" in metrics
    assert metrics["format_parse_rate"] == 1.0
    assert metrics["schema_valid_rate"] == 0.5
    assert metrics["unknown_action_count"] == 1  # one step had action_type_valid=False
    assert metrics["finish_claim_mismatch_count"] == 1


def test_classify_failures_returns_taxonomy():
    from scripts.run_protocol_ablation import classify_failures
    trajectories = [
        {
            "step_diagnostics": [
                {"failure_class": "FORMAT_PARSE_FAIL"},
                {"failure_class": "SCHEMA_VALIDATION_FAIL"},
                {"failure_class": "UNKNOWN_ACTION_TYPE"},
                {"failure_class": None},  # success
            ],
            "actions": [
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
            ],
        },
    ]
    taxonomy = classify_failures(trajectories)
    assert taxonomy["FORMAT_PARSE_FAIL"] == 1
    assert taxonomy["SCHEMA_VALIDATION_FAIL"] == 1
    assert taxonomy["UNKNOWN_ACTION_TYPE"] == 1
    assert taxonomy["REPEATED_ACTION_LOOP"] == 1  # detected from actions


def test_run_combination_with_mock_protocol():
    """Test that run_combination works with a mock protocol (no model loading)."""
    from scripts.run_protocol_ablation import run_combination, _TASKS_DIR
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    proto = MockProtocol(action_type="list_files")
    config = {"name": "mock-base", "model_path": "fake", "adapter_path": None}

    # Use only first 2 tasks for speed
    manifest = json.loads((_TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    task_ids = [t["task_id"] for t in manifest["tasks"][:2]]

    with patch("scripts.run_protocol_ablation.ModelActionProvider") as MockProvider:
        # Create a mock provider that uses our protocol
        def make_provider(*args, **kwargs):
            provider = MagicMock()
            provider._protocol = kwargs.get("protocol")
            provider._model = MagicMock()
            provider.diagnostics = []
            provider.reset = MagicMock()
            # Simulate next_action returning list_files then finish
            actions_iter = iter(["list_files", "finish"])
            def next_action(state):
                at = next(actions_iter)
                if at == "list_files":
                    from src.agent_actions import ListFilesAction, SafetyFlags
                    return ListFilesAction(
                        action_id="a1", reason_short="test",
                        expected_observation="files",
                        safety_flags=SafetyFlags(
                            modifies_workspace=False, executes_code=False,
                            network_required=False, reads_sensitive_path=False,
                            is_terminal=False,
                        ),
                    )
                else:
                    from src.agent_actions import FinishAction, FinishArgs, SafetyFlags, TaskSuccessCriterion
                    return FinishAction(
                        action_id="a2", reason_short="done",
                        expected_observation="finished",
                        safety_flags=SafetyFlags(
                            modifies_workspace=False, executes_code=False,
                            network_required=False, reads_sensitive_path=False,
                            is_terminal=True,
                        ),
                        arguments=FinishArgs(
                            success_criterion=TaskSuccessCriterion.TEST_PASS,
                            tests_passed=False, identification_verified=False,
                            summary="mock finish",
                        ),
                    )
            provider.next_action = next_action
            return provider
        MockProvider.side_effect = make_provider

        result = run_combination(proto, config, task_ids, max_steps=5)
        assert result["config"] == "mock-base"
        assert result["trajectories_written"] == 2
        assert "metrics" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_protocol_ablation.py
"""P4.1b Protocol Ablation: run 3 protocols x 2 configs on 40 micro-tasks.

Produces:
- reports/p4/protocol-ablation/baseline-lock.json
- reports/p4/protocol-ablation/trajectories/{protocol}-{config}.jsonl
- reports/p4/protocol-ablation/comparison-matrix.json
- reports/p4/protocol-ablation/failure-taxonomy.json
- reports/p4/protocol-ablation/comparison-report.md

Usage:
    py -3.11 scripts/run_protocol_ablation.py
    py -3.11 scripts/run_protocol_ablation.py --task-limit 5  # quick smoke
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("P4_ALLOW_NETWORK", "0")

from src.protocols import JsonProtocol, TagProtocol, DslProtocol
from src.agent_model_provider import ModelActionProvider, SentinelAction
from src.agent_evaluator import AgentEvaluator, ActionProvider, AgentState
from src.agent_workspace import MicroTaskWorkspace

_PROTOCOLS = [
    {"name": "json", "class": JsonProtocol},
    {"name": "tag", "class": TagProtocol},
    {"name": "dsl", "class": DslProtocol},
]

_CONFIGS = [
    {"name": "base", "model_path": "models/Qwen3-0.6B", "adapter_path": None},
    {"name": "repair-lora", "model_path": "models/Qwen3-0.6B",
     "adapter_path": "adapters/p3/repair-limited"},
]

_TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
_REPORT_DIR = _ROOT / "reports" / "p4" / "protocol-ablation"
MAX_STEPS = 12

_FAILURE_CLASSES = [
    "FORMAT_PARSE_FAIL", "SCHEMA_VALIDATION_FAIL", "UNKNOWN_ACTION_TYPE",
    "FORBIDDEN_ACTION", "INVALID_PATH", "EMPTY_OR_USELESS_ACTION",
    "MODEL_REFUSAL_OR_CHATTER", "REPEATED_ACTION_LOOP",
]


def _load_task_ids():
    manifest = json.loads((_TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    return [t["task_id"] for t in manifest["tasks"]]


def _git_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(_ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def baseline_lock() -> dict:
    """Record experiment starting state for reproducibility."""
    manifest_path = _TASKS_DIR / "manifest.json"
    return {
        "commit_sha": _git_sha(),
        "micro_task_manifest_path": str(manifest_path.relative_to(_ROOT)),
        "micro_task_manifest_sha256": _file_sha256(manifest_path),
        "model_path": "models/Qwen3-0.6B",
        "adapter_path": "adapters/p3/repair-limited",
        "generation_config": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 128,
            "dtype": "float16",
        },
        "max_steps": MAX_STEPS,
        "total_tasks": 40,
        "protocols": [p["name"] for p in _PROTOCOLS],
        "configs": [c["name"] for c in _CONFIGS],
        "total_combinations": len(_PROTOCOLS) * len(_CONFIGS),
        "total_runs": len(_PROTOCOLS) * len(_CONFIGS) * 40,
    }


class RecordingProvider(ActionProvider):
    """Wraps an ActionProvider and records each action for later replay."""

    def __init__(self, inner: ActionProvider):
        self._inner = inner
        self._recorded: list[dict] = []

    def next_action(self, state: AgentState):
        action = self._inner.next_action(state)
        if isinstance(action, SentinelAction):
            self._recorded.append({
                "__sentinel__": True,
                "is_invalid": True,
                "reason": action.reason,
            })
        else:
            self._recorded.append(action.model_dump())
        return action

    @property
    def recorded_actions(self) -> list[dict]:
        return list(self._recorded)

    @property
    def diagnostics(self):
        return self._inner.diagnostics if hasattr(self._inner, "diagnostics") else []

    def reset(self) -> None:
        self._recorded.clear()
        if hasattr(self._inner, "reset"):
            self._inner.reset()


def run_combination(protocol, config, task_ids, max_steps=MAX_STEPS):
    """Run one protocol x config combination on all task_ids."""
    trajectories = []
    crashes = 0
    model_load_ok = False

    inner_provider = ModelActionProvider(
        model_path=config["model_path"],
        adapter_path=config["adapter_path"],
        protocol=protocol,
    )

    # Try to load the model once
    try:
        inner_provider._load_model()
        model_load_ok = True
    except Exception as e:
        print(f"[{protocol.name}/{config['name']}] model load failed: {e}")
        return {
            "protocol": protocol.name,
            "config": config["name"],
            "total_tasks": len(task_ids),
            "trajectories_written": 0,
            "model_load_ok": False,
            "metrics": {},
            "trajectories": [],
        }

    for i, task_id in enumerate(task_ids):
        task_dir = _TASKS_DIR / task_id
        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            inner_provider.reset()
            provider = RecordingProvider(inner_provider)
            evaluator = AgentEvaluator(ws, provider, task_id, max_steps=max_steps)
            result = evaluator.run()
            trajectories.append({
                "trajectory_id": f"{protocol.name}_{config['name']}_{task_id}",
                "task_id": task_id,
                "protocol": protocol.name,
                "config": config["name"],
                "success": result.success,
                "finish_claim_mismatch": result.finish_claim_mismatch,
                "metrics": result.metrics,
                "steps_executed": result.steps_executed,
                "max_steps_hit": result.max_steps_hit,
                "actions": provider.recorded_actions,
                "step_diagnostics": [d.model_dump() for d in inner_provider.diagnostics],
            })
        except Exception:
            crashes += 1
            traceback.print_exc()
        finally:
            ws.cleanup()
        print(f"\r[{protocol.name}/{config['name']}] {i+1}/{len(task_ids)} {task_id}", end="", flush=True)
    print()

    metrics = aggregate_metrics(trajectories, crashes, model_load_ok)
    return {
        "protocol": protocol.name,
        "config": config["name"],
        "total_tasks": len(task_ids),
        "trajectories_written": len(trajectories),
        "model_load_ok": model_load_ok,
        "metrics": metrics,
        "trajectories": trajectories,
    }


def aggregate_metrics(trajectories: list[dict], crashes: int = 0,
                      model_load_ok: bool = True) -> dict:
    """Compute aggregated metrics from trajectory step_diagnostics.

    Returns 13 metrics per spec §6.2 (12+ required by acceptance criteria).
    """
    all_diags: list[dict] = []
    for traj in trajectories:
        all_diags.extend(traj.get("step_diagnostics", []))

    total_diags = len(all_diags)
    if total_diags > 0:
        format_parse_rate = sum(1 for d in all_diags if d.get("format_parse_ok")) / total_diags
        schema_valid_rate = sum(1 for d in all_diags if d.get("schema_valid")) / total_diags
        safety_valid_rate = sum(1 for d in all_diags if d.get("safety_valid")) / total_diags
        action_type_valid_rate = sum(1 for d in all_diags if d.get("action_type_valid")) / total_diags
        arguments_valid_rate = sum(1 for d in all_diags if d.get("arguments_valid")) / total_diags
    else:
        format_parse_rate = 0.0
        schema_valid_rate = 0.0
        safety_valid_rate = 0.0
        action_type_valid_rate = 0.0
        arguments_valid_rate = 0.0

    forbidden_action_count = sum(
        t.get("metrics", {}).get("forbidden_action_count", 0) for t in trajectories
    )
    unknown_action_count = sum(1 for d in all_diags if not d.get("action_type_valid"))
    total_tasks = len(trajectories)
    task_success_rate = sum(1 for t in trajectories if t.get("success")) / total_tasks if total_tasks else 0.0
    max_steps_hit_rate = sum(1 for t in trajectories if t.get("max_steps_hit")) / total_tasks if total_tasks else 0.0

    # Finish-related metrics (from evaluator)
    finish_without_tests_count = sum(
        1 for t in trajectories
        if t.get("success") and not t.get("metrics", {}).get("tests_passed", False)
    )
    finish_claim_mismatch_count = sum(
        1 for t in trajectories if t.get("finish_claim_mismatch")
    )

    return {
        "format_parse_rate": format_parse_rate,
        "schema_valid_rate": schema_valid_rate,
        "safety_valid_rate": safety_valid_rate,
        "action_type_valid_rate": action_type_valid_rate,
        "arguments_valid_rate": arguments_valid_rate,
        "forbidden_action_count": forbidden_action_count,
        "unknown_action_count": unknown_action_count,
        "task_success_rate": task_success_rate,
        "finish_without_tests_count": finish_without_tests_count,
        "finish_claim_mismatch_count": finish_claim_mismatch_count,
        "max_steps_hit_rate": max_steps_hit_rate,
        "runtime_crash_count": crashes,
        "model_load_ok": model_load_ok,
    }


def classify_failures(trajectories: list[dict]) -> dict:
    """Classify all failed steps into failure taxonomy.

    Also detects REPEATED_ACTION_LOOP at trajectory level (3+ consecutive
    identical actions in a single trajectory).
    """
    taxonomy = {fc: 0 for fc in _FAILURE_CLASSES}
    for traj in trajectories:
        for d in traj.get("step_diagnostics", []):
            fc = d.get("failure_class")
            if fc and fc in taxonomy:
                taxonomy[fc] += 1
        # Trajectory-level: detect repeated action loops
        actions = traj.get("actions", [])
        if _detect_repeated_loop(actions):
            taxonomy["REPEATED_ACTION_LOOP"] += 1
    return taxonomy


def _detect_repeated_loop(actions: list[dict]) -> bool:
    """Return True if 3+ consecutive identical actions found."""
    if len(actions) < 3:
        return False
    for i in range(len(actions) - 2):
        a1 = actions[i]
        a2 = actions[i + 1]
        a3 = actions[i + 2]
        if (a1.get("action_type") == a2.get("action_type") == a3.get("action_type")
                and a1.get("arguments") == a2.get("arguments") == a3.get("arguments")):
            return True
    return False


def main():
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    traj_dir = _REPORT_DIR / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Baseline lock
    print("=== Step 1: Baseline Lock ===")
    lock = baseline_lock()
    (_REPORT_DIR / "baseline-lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(f"Wrote {_REPORT_DIR / 'baseline-lock.json'}")

    # Step 2: Run all 6 combinations
    print("\n=== Step 2: Run Protocol x Config Combinations ===")
    all_task_ids = _load_task_ids()
    task_limit = os.environ.get("P4_1B_TASK_LIMIT")
    if task_limit:
        all_task_ids = all_task_ids[:int(task_limit)]
        print(f"P4_1B_TASK_LIMIT={task_limit}: running first {len(all_task_ids)} tasks")

    all_results = []
    for proto_spec in _PROTOCOLS:
        for config in _CONFIGS:
            print(f"\n--- Protocol: {proto_spec['name']} | Config: {config['name']} ---")
            proto = proto_spec["class"]()
            result = run_combination(proto, config, all_task_ids)
            # Write trajectories JSONL
            out_file = traj_dir / f"{proto_spec['name']}-{config['name']}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for traj in result["trajectories"]:
                    f.write(json.dumps(traj) + "\n")
            # Strip trajectories from summary
            summary = {k: v for k, v in result.items() if k != "trajectories"}
            all_results.append(summary)
            print(f"  schema_valid_rate: {summary['metrics'].get('schema_valid_rate', 0):.2%}")

    # Step 3: Comparison matrix
    print("\n=== Step 3: Comparison Matrix ===")
    (_REPORT_DIR / "comparison-matrix.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # Step 4: Failure taxonomy
    print("\n=== Step 4: Failure Taxonomy ===")
    # Reload trajectories for taxonomy
    all_trajectories = []
    for proto_spec in _PROTOCOLS:
        for config in _CONFIGS:
            traj_file = traj_dir / f"{proto_spec['name']}-{config['name']}.jsonl"
            if traj_file.exists():
                for line in traj_file.read_text(encoding="utf-8").strip().split("\n"):
                    if line:
                        all_trajectories.append(json.loads(line))
    taxonomy = classify_failures(all_trajectories)
    (_REPORT_DIR / "failure-taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2), encoding="utf-8"
    )

    print(f"\nDone. Reports in {_REPORT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_protocol_ablation.py tests/test_protocol_ablation.py
git commit -m "feat(p4-1b): add baseline lock and smoke run script (P4.1b T1+T6)"
```

---

### Task 7: Comparison Report Generation

**Files:**
- Modify: `scripts/run_protocol_ablation.py` (add `generate_report` function)
- Modify: `tests/test_protocol_ablation.py` (add report tests)
- Output: `reports/p4/protocol-ablation/comparison-report.md`

**Interfaces:**
- Consumes: `aggregate_metrics` and `classify_failures` from Task 6
- Produces: `generate_report(results, taxonomy)` → markdown string

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_protocol_ablation.py`:

```python
def test_generate_report_contains_all_protocols():
    from scripts.run_protocol_ablation import generate_report
    results = [
        {"protocol": "json", "config": "base", "metrics": {"schema_valid_rate": 0.0}},
        {"protocol": "json", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.0}},
        {"protocol": "tag", "config": "base", "metrics": {"schema_valid_rate": 0.5}},
        {"protocol": "tag", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.6}},
        {"protocol": "dsl", "config": "base", "metrics": {"schema_valid_rate": 0.3}},
        {"protocol": "dsl", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.4}},
    ]
    taxonomy = {"FORMAT_PARSE_FAIL": 10, "SCHEMA_VALIDATION_FAIL": 20}
    report = generate_report(results, taxonomy)
    assert "json" in report
    assert "tag" in report
    assert "dsl" in report
    assert "schema_valid_rate" in report
    assert "FORMAT_PARSE_FAIL" in report


def test_generate_report_has_markdown_table():
    from scripts.run_protocol_ablation import generate_report
    results = [
        {"protocol": "json", "config": "base",
         "metrics": {"format_parse_rate": 1.0, "schema_valid_rate": 0.0,
                     "safety_valid_rate": 0.0, "action_type_valid_rate": 0.5,
                     "arguments_valid_rate": 0.0, "forbidden_action_count": 0,
                     "task_success_rate": 0.0, "max_steps_hit_rate": 1.0,
                     "runtime_crash_count": 0}},
    ]
    taxonomy = {"FORMAT_PARSE_FAIL": 5}
    report = generate_report(results, taxonomy)
    assert "|" in report  # markdown table
    assert "format_parse_rate" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py::test_generate_report_contains_all_protocols -v`
Expected: FAIL with `ImportError: cannot import name 'generate_report'`

- [ ] **Step 3: Add generate_report to run_protocol_ablation.py**

Add this function to `scripts/run_protocol_ablation.py` (before `main()`):

```python
def generate_report(results: list[dict], taxonomy: dict) -> str:
    """Generate markdown comparison report from ablation results."""
    lines = [
        "# P4.1b Protocol Ablation — Comparison Report",
        "",
        "## Overview",
        "",
        f"- Protocols: {len(set(r['protocol'] for r in results))}",
        f"- Configs: {len(set(r['config'] for r in results))}",
        f"- Total combinations: {len(results)}",
        "",
        "## Metrics by Protocol x Config",
        "",
        "| Protocol | Config | format_parse_rate | schema_valid_rate | safety_valid_rate | action_type_valid_rate | arguments_valid_rate | forbidden_count | unknown_count | task_success_rate | finish_no_tests | finish_mismatch | max_steps_hit_rate | crashes |",
        "|----------|--------|-------------------|-------------------|-------------------|------------------------|----------------------|-----------------|----------------|-------------------|------------------|-----------------|---------------------|---------|",
    ]

    for r in sorted(results, key=lambda x: (x["protocol"], x["config"])):
        m = r.get("metrics", {})
        lines.append(
            f"| {r['protocol']} | {r['config']} "
            f"| {m.get('format_parse_rate', 0):.2%} "
            f"| {m.get('schema_valid_rate', 0):.2%} "
            f"| {m.get('safety_valid_rate', 0):.2%} "
            f"| {m.get('action_type_valid_rate', 0):.2%} "
            f"| {m.get('arguments_valid_rate', 0):.2%} "
            f"| {m.get('forbidden_action_count', 0)} "
            f"| {m.get('unknown_action_count', 0)} "
            f"| {m.get('task_success_rate', 0):.2%} "
            f"| {m.get('finish_without_tests_count', 0)} "
            f"| {m.get('finish_claim_mismatch_count', 0)} "
            f"| {m.get('max_steps_hit_rate', 0):.2%} "
            f"| {m.get('runtime_crash_count', 0)} |"
        )

    lines.extend([
        "",
        "## Failure Taxonomy",
        "",
        "| Failure Class | Count |",
        "|---------------|-------|",
    ])
    for fc, count in sorted(taxonomy.items()):
        lines.append(f"| {fc} | {count} |")

    lines.extend([
        "",
        "## Protocol Comparison Summary",
        "",
    ])

    # Summarize by protocol (average across configs)
    protocols = sorted(set(r["protocol"] for r in results))
    for proto in protocols:
        proto_results = [r for r in results if r["protocol"] == proto]
        avg_schema = sum(r["metrics"].get("schema_valid_rate", 0) for r in proto_results) / len(proto_results)
        lines.append(f"- **{proto}**: avg schema_valid_rate = {avg_schema:.2%}")

    return "\n".join(lines)
```

Also add to `main()` after writing failure-taxonomy.json:

```python
    # Step 5: Generate report
    print("\n=== Step 5: Comparison Report ===")
    report = generate_report(all_results, taxonomy)
    (_REPORT_DIR / "comparison-report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {_REPORT_DIR / 'comparison-report.md'}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_protocol_ablation.py tests/test_protocol_ablation.py
git commit -m "feat(p4-1b): add comparison report generation (P4.1b T7)"
```

---

### Task 8: Verdict Decision Logic

**Files:**
- Modify: `scripts/run_protocol_ablation.py` (add `compute_verdict` function)
- Modify: `tests/test_protocol_ablation.py` (add verdict tests)
- Output: verdict in `comparison-report.md`

**Interfaces:**
- Consumes: `results` list from Task 6-7
- Produces: `compute_verdict(results)` → verdict string from allowed enum

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_protocol_ablation.py`:

```python
def test_verdict_keep_action_json_when_json_best():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "KEEP_ACTION_JSON"


def test_verdict_try_tag_when_tag_significantly_better():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "TRY_TAG_PROTOCOL_FOR_P4_2"


def test_verdict_try_dsl_when_dsl_significantly_better():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.6, "safety_valid_rate": 0.6, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.6, "safety_valid_rate": 0.6, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "TRY_DSL_FOR_P4_2"


def test_verdict_fix_prompt_when_all_below_30pct():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_PROMPT_FIRST"


def test_verdict_fix_evaluator_when_model_load_fails():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_EVALUATOR_FIRST"


def test_verdict_fix_evaluator_when_high_crash():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 30}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_EVALUATOR_FIRST"


def test_verdict_is_valid_enum():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    allowed = {
        "KEEP_ACTION_JSON", "TRY_TAG_PROTOCOL_FOR_P4_2", "TRY_DSL_FOR_P4_2",
        "FIX_PROMPT_FIRST", "FIX_EVALUATOR_FIRST", "STOP_PROTOCOL_CHANGE",
    }
    assert verdict in allowed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py::test_verdict_keep_action_json_when_json_best -v`
Expected: FAIL with `ImportError: cannot import name 'compute_verdict'`

- [ ] **Step 3: Add compute_verdict to run_protocol_ablation.py**

Add this function to `scripts/run_protocol_ablation.py` (after `generate_report`):

```python
_ALLOWED_VERDICTS = {
    "KEEP_ACTION_JSON", "TRY_TAG_PROTOCOL_FOR_P4_2", "TRY_DSL_FOR_P4_2",
    "FIX_PROMPT_FIRST", "FIX_EVALUATOR_FIRST", "STOP_PROTOCOL_CHANGE",
}


def compute_verdict(results: list[dict]) -> str:
    """Apply T8 verdict decision rules.

    Rules (from spec §7.2):
    1. Any alternative protocol's schema_valid_rate >30% better than JSON
       AND safety_valid_rate not degraded → TRY_TAG/TRY_DSL
    2. All protocols schema_valid_rate < 30% → FIX_PROMPT_FIRST
    3. JSON baseline has highest schema_valid_rate → KEEP_ACTION_JSON
    4. Evaluator issues (model load failed or >50% trajectories crashed)
       → FIX_EVALUATOR_FIRST
    5. Fallback → STOP_PROTOCOL_CHANGE
    """
    # Average schema_valid_rate per protocol (across configs)
    proto_rates: dict[str, float] = {}
    proto_safety: dict[str, float] = {}
    for r in results:
        proto = r["protocol"]
        if proto not in proto_rates:
            proto_rates[proto] = []
            proto_safety[proto] = []
        proto_rates[proto].append(r["metrics"].get("schema_valid_rate", 0))
        proto_safety[proto].append(r["metrics"].get("safety_valid_rate", 0))

    avg_schema = {p: sum(v) / len(v) for p, v in proto_rates.items()}
    avg_safety = {p: sum(v) / len(v) for p, v in proto_safety.items()}

    json_rate = avg_schema.get("json", 0.0)
    json_safety = avg_safety.get("json", 0.0)

    # Rule 4: evaluator issues make metrics unreliable
    # - model_load_ok=False for all results, OR
    # - runtime_crash_count > 50% of total_tasks for any combination
    all_model_load_failed = all(
        not r.get("model_load_ok", True) for r in results
    )
    high_crash = any(
        r.get("metrics", {}).get("runtime_crash_count", 0) > r.get("total_tasks", 0) / 2
        for r in results
    )
    if all_model_load_failed or high_crash:
        return "FIX_EVALUATOR_FIRST"

    # Rule 2: all below 30%
    if all(rate < 0.30 for rate in avg_schema.values()):
        return "FIX_PROMPT_FIRST"

    # Rule 1: alternative protocol significantly better (>30% improvement)
    for proto in ("tag", "dsl"):
        if proto in avg_schema:
            improvement = avg_schema[proto] - json_rate
            if improvement > 0.30 and avg_safety[proto] >= json_safety:
                if proto == "tag":
                    return "TRY_TAG_PROTOCOL_FOR_P4_2"
                else:
                    return "TRY_DSL_FOR_P4_2"

    # Rule 3: JSON is best
    if json_rate >= max(avg_schema.values()):
        return "KEEP_ACTION_JSON"

    # Fallback
    return "STOP_PROTOCOL_CHANGE"
```

Also update `main()` to include verdict in the report:

```python
    # Step 6: Verdict
    print("\n=== Step 6: Verdict ===")
    verdict = compute_verdict(all_results)
    print(f"Verdict: {verdict}")
    # Append verdict to report
    report_path = _REPORT_DIR / "comparison-report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        report += f"\n\n## Verdict\n\n**{verdict}**\n"
        report_path.write_text(report, encoding="utf-8")
    # Write verdict as separate file for machine reading
    (_REPORT_DIR / "verdict.json").write_text(
        json.dumps({"verdict": verdict}, indent=2), encoding="utf-8"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
Expected: PASS (13 tests: 4 from Task 6 + 2 from Task 7 + 7 from Task 8)

- [ ] **Step 5: Run full protocol test suite**

Run: `py -3.11 -m pytest tests/test_protocol_base.py tests/test_protocol_json.py tests/test_protocol_tag.py tests/test_protocol_dsl.py tests/test_protocol_ablation.py tests/test_agent_model_provider_protocol.py tests/test_agent_model_provider.py -v`
Expected: PASS (all protocol + regression tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_protocol_ablation.py tests/test_protocol_ablation.py
git commit -m "feat(p4-1b): add verdict decision logic (P4.1b T8)

Verdict rules:
- Alternative protocol >30% better than JSON → TRY_TAG/TRY_DSL
- All protocols <30% schema_valid → FIX_PROMPT_FIRST
- JSON highest → KEEP_ACTION_JSON
- Fallback → STOP_PROTOCOL_CHANGE"
```

---

## Post-Plan: Execution

After all 8 tasks are implemented and tests pass, the actual smoke run is executed:

```bash
# Quick smoke (first 5 tasks, ~15 min)
py -3.11 scripts/run_protocol_ablation.py
# With env var: set P4_1B_TASK_LIMIT=5 && py -3.11 scripts/run_protocol_ablation.py

# Full run (40 tasks, ~2-3 hours)
set P4_1B_TASK_LIMIT= && py -3.11 scripts/run_protocol_ablation.py
```

This produces:
- `reports/p4/protocol-ablation/baseline-lock.json`
- `reports/p4/protocol-ablation/trajectories/{json,tag,dsl}-{base,repair-lora}.jsonl` (6 files)
- `reports/p4/protocol-ablation/comparison-matrix.json`
- `reports/p4/protocol-ablation/failure-taxonomy.json`
- `reports/p4/protocol-ablation/comparison-report.md`
- `reports/p4/protocol-ablation/verdict.json`

The verdict determines the P4.2 training target protocol.
