import pytest
from src.agent_model_provider import (
    build_prompt, extract_json, repair_json, ModelStepDiagnostics,
    SentinelAction,
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
