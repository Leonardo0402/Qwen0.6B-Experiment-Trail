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


def test_finish_missing_required_fields_fails():
    """Issue #32 Final Trust Repair: finish with only summary must FAIL.

    Previously the parser injected defaults for success_criterion,
    tests_passed, and identification_verified. Now the model must
    explicitly provide all required business parameters.
    """
    proto = _make_protocol()
    raw = "ACTION finish summary=done"
    action, diag = proto.parse_output(raw)
    assert isinstance(action, SentinelAction)
    assert not diag.schema_valid
    assert not diag.arguments_valid


def test_complete_finish_passes():
    """A complete finish action with all required fields must pass."""
    proto = _make_protocol()
    raw = (
        "ACTION finish "
        "success_criterion=test_pass "
        "tests_passed=true "
        "identification_verified=true "
        "summary=all tests pass"
    )
    action, diag = proto.parse_output(raw)
    assert not isinstance(action, SentinelAction)
    assert action.action_type == "finish"
    assert diag.schema_valid
