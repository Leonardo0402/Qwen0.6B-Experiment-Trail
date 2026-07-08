"""Phase B: Action schema tests — Part 1: enums and core schemas."""
import pytest
from pydantic import ValidationError

from src.agent_actions import ActionType, SafetyFlags, TaskSuccessCriterion, EvaluationMode
from src.agent_state import AgentMemory


def test_action_type_has_11_values():
    assert len(ActionType) == 11
    assert ActionType.rollback_patch == "rollback_patch"
    assert ActionType.finish == "finish"


def test_task_success_criterion_values():
    assert TaskSuccessCriterion.TEST_PASS == "test_pass"
    assert TaskSuccessCriterion.IDENTIFY_BUG == "identify_bug"
    assert TaskSuccessCriterion.PATCH_APPLIED == "patch_applied"


def test_evaluation_mode_values():
    assert EvaluationMode.REPLAY == "replay"
    assert EvaluationMode.AGENT_RUN == "agent_run"


def test_safety_flags_frozen():
    flags = SafetyFlags(
        modifies_workspace=False, executes_code=False,
        network_required=False, reads_sensitive_path=False, is_terminal=False,
    )
    with pytest.raises(ValidationError):
        flags.modifies_workspace = True


def test_agent_memory_defaults():
    mem = AgentMemory()
    assert mem.notes == ""
    assert mem.hypothesis == ""
    assert mem.failed_attempts == []
    assert mem.last_test_summary == ""


def test_agent_memory_failed_attempts_list():
    mem = AgentMemory(failed_attempts=["tried x", "tried y"])
    assert len(mem.failed_attempts) == 2


def test_safety_flags_network_field_exists():
    flags = SafetyFlags(
        modifies_workspace=False, executes_code=False,
        network_required=True, reads_sensitive_path=False, is_terminal=False,
    )
    assert flags.network_required is True
