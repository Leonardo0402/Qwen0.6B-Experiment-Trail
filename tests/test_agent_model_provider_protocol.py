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
