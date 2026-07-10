import pytest
from unittest.mock import patch, MagicMock
from src.agent_model_provider import (
    build_prompt, extract_json, repair_json, ModelStepDiagnostics,
    ModelActionProvider, SentinelAction,
)
from src.agent_evaluator import AgentState
from src.agent_state import AgentMemory


def test_build_prompt_produces_nonempty_string():
    state = AgentState(
        memory=AgentMemory(),
        step_count=0,
        task_id="task_001",
        workspace_id="test_ws",
    )
    prompt = build_prompt(state, task_description="Fix the bug", last_observation=None)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "task_001" in prompt or "Fix the bug" in prompt


def test_extract_json_finds_first_json_block():
    raw = 'Here is the action:\n```json\n{"action_type": "list_files"}\n```\nDone.'
    result = extract_json(raw)
    assert result == '{"action_type": "list_files"}'


def test_extract_json_returns_none_on_no_json():
    raw = "I cannot produce an action."
    assert extract_json(raw) is None


def test_repair_json_strips_markdown_fences():
    raw = '```json\n{"action_type": "read_file"}\n```'
    repaired = repair_json(raw)
    assert "```" not in repaired
    assert '"action_type"' in repaired


def test_repair_json_removes_trailing_commas():
    raw = '{"action_type": "read_file", "arguments": {"path": "x.py",},}'
    repaired = repair_json(raw)
    assert ",}" not in repaired
    assert ",," not in repaired


def test_repair_json_does_not_choose_action_type():
    """Repair must NOT substitute a valid action_type for an invalid one."""
    raw = '{"action_type": "???"}'
    repaired = repair_json(raw)
    assert '"???"' in repaired, "repair must not alter action_type value"


def test_sentinel_action_marks_invalid():
    sa = SentinelAction(reason="json parse failed")
    assert sa.is_invalid
    assert sa.reason == "json parse failed"


def test_model_provider_extracts_valid_json_mocked():
    """Mock _generate to return valid Action JSON → provider returns a
    valid Action, diagnostics record schema_valid=True."""
    provider = ModelActionProvider.__new__(ModelActionProvider)
    provider._model = MagicMock()  # skip _load_model
    provider._tokenizer = MagicMock()
    provider._max_new_tokens = 512
    provider._diagnostics = []
    provider._model_path = "fake"
    provider._adapter_path = None

    valid_json = '{"action_type": "list_files", "action_id": "a1", "reason_short": "list", "expected_observation": "files", "safety_flags": {"modifies_workspace": false, "executes_code": false, "network_required": false, "reads_sensitive_path": false, "is_terminal": false}}'

    with patch.object(provider, "_generate", return_value=valid_json):
        state = AgentState(
            memory=AgentMemory(), step_count=0,
            task_id="t1", workspace_id="w1",
        )
        action = provider.next_action(state)

    assert not isinstance(action, SentinelAction), "expected valid Action, got SentinelAction"
    assert provider.diagnostics[0].json_parse_ok
    assert provider.diagnostics[0].schema_valid


def test_model_provider_records_diagnostics_on_invalid_json():
    """Mock _generate to return garbage → provider returns SentinelAction,
    diagnostics record json_parse_ok=False."""
    provider = ModelActionProvider.__new__(ModelActionProvider)
    provider._model = MagicMock()
    provider._tokenizer = MagicMock()
    provider._max_new_tokens = 512
    provider._diagnostics = []
    provider._model_path = "fake"
    provider._adapter_path = None

    with patch.object(provider, "_generate", return_value="I cannot do that."):
        state = AgentState(
            memory=AgentMemory(), step_count=0,
            task_id="t1", workspace_id="w1",
        )
        action = provider.next_action(state)

    assert isinstance(action, SentinelAction)
    assert not provider.diagnostics[0].json_parse_ok
    assert not provider.diagnostics[0].schema_valid


def test_model_provider_repair_strips_fences_then_validates():
    """Mock _generate to return fenced JSON → repair strips fences,
    validation succeeds."""
    provider = ModelActionProvider.__new__(ModelActionProvider)
    provider._model = MagicMock()
    provider._tokenizer = MagicMock()
    provider._max_new_tokens = 512
    provider._diagnostics = []
    provider._model_path = "fake"
    provider._adapter_path = None

    fenced = '```json\n{"action_type": "list_files", "action_id": "a1", "reason_short": "list", "expected_observation": "files", "safety_flags": {"modifies_workspace": false, "executes_code": false, "network_required": false, "reads_sensitive_path": false, "is_terminal": false}}\n```'

    with patch.object(provider, "_generate", return_value=fenced):
        state = AgentState(
            memory=AgentMemory(), step_count=0,
            task_id="t1", workspace_id="w1",
        )
        action = provider.next_action(state)

    # extract_json should handle fences, but if not, repair kicks in
    assert not isinstance(action, SentinelAction), \
        f"expected valid Action after repair, got SentinelAction; diag: {provider.diagnostics[0]}"
