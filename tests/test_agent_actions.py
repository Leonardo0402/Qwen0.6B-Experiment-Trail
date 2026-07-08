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


from src.agent_actions import (
    ActionBase, ListFilesAction, ReadFileAction, ApplyPatchAction,
    RollbackPatchAction, FinishAction, validate_path, PathValidationError,
    P4ForbiddenActionError,
)


def test_validate_path_rejects_absolute():
    with pytest.raises(PathValidationError, match="absolute"):
        validate_path("/etc/passwd")


def test_validate_path_rejects_parent_traversal():
    with pytest.raises(PathValidationError, match="parent traversal"):
        validate_path("../secret")


def test_validate_path_rejects_unc():
    with pytest.raises(PathValidationError, match="UNC"):
        validate_path(r"\\server\share")


def test_validate_path_rejects_url():
    with pytest.raises(PathValidationError, match="URL"):
        validate_path("http://evil.com/x")


def test_validate_path_rejects_secret_basename():
    for name in [".env", ".git", "credentials.json", "secret.key", "token"]:
        with pytest.raises(PathValidationError, match="sensitive"):
            validate_path(name)


def test_validate_path_accepts_normal():
    assert validate_path("src/foo.py") == "src/foo.py"
    assert validate_path("solution.py") == "solution.py"


def test_action_with_network_required_rejected():
    with pytest.raises(P4ForbiddenActionError, match="network_required"):
        ApplyPatchAction(
            action_id="a1", action_type=ActionType.apply_patch,
            reason_short="patch", expected_observation="patch_result",
            safety_flags=SafetyFlags(
                modifies_workspace=True, executes_code=False,
                network_required=True, reads_sensitive_path=False, is_terminal=False,
            ),
            arguments={"file_path": "solution.py", "old_text": "a", "new_text": "b"},
        )


def test_action_with_reads_sensitive_rejected():
    with pytest.raises(P4ForbiddenActionError, match="reads_sensitive_path"):
        ReadFileAction(
            action_id="a1", action_type=ActionType.read_file,
            reason_short="read", expected_observation="file_content",
            safety_flags=SafetyFlags(
                modifies_workspace=False, executes_code=False,
                network_required=False, reads_sensitive_path=True, is_terminal=False,
            ),
            arguments={"path": "solution.py"},
        )


def test_list_files_action_serialization():
    a = ListFilesAction(
        action_id="a1", action_type=ActionType.list_files,
        reason_short="list", expected_observation="file_list",
        safety_flags=SafetyFlags(
            modifies_workspace=False, executes_code=False,
            network_required=False, reads_sensitive_path=False, is_terminal=False,
        ),
        arguments={},
    )
    j = a.model_dump_json()
    a2 = ListFilesAction.model_validate_json(j)
    assert a2.action_id == "a1"


def test_finish_action_with_success_criterion():
    a = FinishAction(
        action_id="a1", action_type=ActionType.finish,
        reason_short="done", expected_observation="finish",
        safety_flags=SafetyFlags(
            modifies_workspace=False, executes_code=False,
            network_required=False, reads_sensitive_path=False, is_terminal=True,
        ),
        arguments={
            "success_criterion": TaskSuccessCriterion.TEST_PASS,
            "tests_passed": True, "identification_verified": False, "summary": "fixed",
        },
    )
    assert a.arguments.success_criterion == TaskSuccessCriterion.TEST_PASS
