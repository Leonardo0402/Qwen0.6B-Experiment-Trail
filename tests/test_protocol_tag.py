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
