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


def test_missing_safety_flags_gets_injected():
    """Regression: model omits safety_flags field; JSON protocol injects
    safe defaults consistent with Tag/DSL protocols."""
    proto = _make_protocol()
    data = {
        "action_type": "read_file",
        "action_id": "a1",
        "reason_short": "inspect file",
        "expected_observation": "contents",
        "arguments": {"path": "solution.py"},
    }
    action, diag = proto.parse_output(json.dumps(data))
    assert not isinstance(action, SentinelAction)
    assert diag.safety_valid
    assert diag.schema_valid


def test_missing_safety_flags_terminal_action():
    """Safety flag default for finish must mark is_terminal=True.

    Note: business fields (success_criterion, tests_passed,
    identification_verified) must be provided by the model —
    structural default injection only covers safety_flags.
    """
    proto = _make_protocol()
    data = {
        "action_type": "finish",
        "action_id": "a1",
        "reason_short": "done",
        "expected_observation": "task complete",
        "arguments": {
            "success_criterion": "test_pass",
            "tests_passed": True,
            "identification_verified": True,
            "summary": "fixed the bug",
        },
    }
    action, diag = proto.parse_output(json.dumps(data))
    assert not isinstance(action, SentinelAction)
    assert diag.safety_valid
    assert action.safety_flags.is_terminal is True


def test_explicit_safety_flags_not_overwritten():
    """If model provides safety_flags, defaults must not override them."""
    proto = _make_protocol()
    data = {
        "action_type": "read_file",
        "action_id": "a1",
        "reason_short": "x",
        "expected_observation": "y",
        "safety_flags": {
            "modifies_workspace": False,
            "executes_code": False,
            "network_required": False,
            "reads_sensitive_path": False,
            "is_terminal": False,
        },
        "arguments": {"path": "solution.py"},
    }
    action, diag = proto.parse_output(json.dumps(data))
    assert not isinstance(action, SentinelAction)
    assert diag.safety_valid


def test_missing_safety_flags_run_tests():
    """Default for run_tests must mark executes_code=True."""
    proto = _make_protocol()
    data = {
        "action_type": "run_tests",
        "action_id": "a1",
        "reason_short": "verify fix",
        "expected_observation": "tests pass",
        "arguments": {"test_path": "tests/test_solution.py"},
    }
    action, diag = proto.parse_output(json.dumps(data))
    assert not isinstance(action, SentinelAction)
    assert diag.safety_valid
    assert action.safety_flags.executes_code is True
