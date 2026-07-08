"""Phase D: Trajectory schema tests."""
import json
import tempfile
from pathlib import Path

import pytest

from src.agent_actions import (
    ActionType,
    FinishAction,
    ListFilesAction,
    ReadFileAction,
    SafetyFlags,
    TaskSuccessCriterion,
)
from src.agent_state import AgentMemory
from src.agent_trajectory import (
    TrajectoryStep,
    Trajectory,
    is_mutating_action,
    load_trajectory,
    save_trajectory,
)


def _safety(**overrides):
    defaults = dict(
        modifies_workspace=False, executes_code=False,
        network_required=False, reads_sensitive_path=False, is_terminal=False,
    )
    defaults.update(overrides)
    return SafetyFlags(**defaults)


def _make_finish_step(step_index, tests_passed=True, success_label=True):
    return TrajectoryStep(
        trajectory_id="traj_1", task_id="task_001",
        workspace_id="ws/001", step_index=step_index,
        goal="fix the bug", state_summary="done",
        observation={"kind": "finish"},
        action=FinishAction(
            action_id="f1", action_type=ActionType.finish,
            reason_short="finish", expected_observation="finish",
            safety_flags=_safety(is_terminal=True),
            arguments={
                "success_criterion": TaskSuccessCriterion.TEST_PASS,
                "tests_passed": tests_passed,
                "identification_verified": False,
                "summary": "done",
            },
        ),
        result={"kind": "finish"},
        memory_before=AgentMemory(notes="3 files"),
        memory_after=AgentMemory(),
        success_label=success_label,
        source="scripted",
        verified=True,
    )


def _make_list_files_step(step_index):
    return TrajectoryStep(
        trajectory_id="traj_1", task_id="task_001",
        workspace_id="ws/001", step_index=step_index,
        goal="fix the bug", state_summary="start",
        observation={"kind": "task", "goal": "fix bug"},
        action=ListFilesAction(
            action_id="a1", action_type=ActionType.list_files,
            reason_short="list", expected_observation="file_list",
            safety_flags=_safety(),
            arguments={},
        ),
        result={"kind": "file_list", "paths": ["solution.py"]},
        memory_before=AgentMemory(),
        memory_after=AgentMemory(notes="3 files"),
        success_label=False,
        source="scripted",
        verified=True,
    )


def test_step_index_monotonic():
    """Reject non-monotonic step_index."""
    steps = [_make_list_files_step(0), _make_finish_step(2)]
    with pytest.raises(ValueError, match="step_index not monotonic"):
        Trajectory(
            trajectory_id="traj_1", task_id="task_001",
            workspace_id="ws/001", goal="fix bug",
            steps=steps, source="scripted",
        )


def test_must_end_with_finish():
    """Reject trajectory without terminal finish action."""
    step = _make_list_files_step(0)
    with pytest.raises(ValueError, match="must end with finish"):
        Trajectory(
            trajectory_id="traj_1", task_id="task_001",
            workspace_id="ws/001", goal="fix bug",
            steps=[step], source="scripted",
        )


def test_success_requires_verified():
    """Reject success_label=True without verified=True."""
    step = _make_finish_step(0, success_label=True)
    step = step.model_copy(update={"verified": False})
    with pytest.raises(ValueError, match="success_label=True requires verified=True"):
        Trajectory(
            trajectory_id="traj_1", task_id="task_001",
            workspace_id="ws/001", goal="fix bug",
            steps=[step], source="scripted",
        )


def test_memory_chain():
    """Reject broken memory_before/memory_after chain."""
    step0 = _make_list_files_step(0)
    step1 = _make_finish_step(1)
    # break the chain: step1.memory_before != step0.memory_after
    step1 = step1.model_copy(update={
        "memory_before": AgentMemory(notes="different"),
    })
    with pytest.raises(ValueError, match="memory chain broken"):
        Trajectory(
            trajectory_id="traj_1", task_id="task_001",
            workspace_id="ws/001", goal="fix bug",
            steps=[step0, step1], source="scripted",
        )


def test_round_trip():
    """Trajectory → JSONL → Trajectory equality."""
    traj = Trajectory(
        trajectory_id="traj_1", task_id="task_001",
        workspace_id="ws/001", goal="fix bug",
        steps=[_make_list_files_step(0), _make_finish_step(1)],
        source="scripted",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traj.jsonl"
        save_trajectory(traj, path)
        loaded = load_trajectory(path)
    assert loaded.trajectory_id == traj.trajectory_id
    assert loaded.action_count == traj.action_count
    assert loaded.final_success == traj.final_success
    assert len(loaded.steps) == len(traj.steps)
    assert loaded.steps[0].action.action_type == traj.steps[0].action.action_type
    assert loaded.steps[1].action.action_type == traj.steps[1].action.action_type


def test_tool_distribution():
    """tool_distribution counts each action_type correctly."""
    traj = Trajectory(
        trajectory_id="traj_1", task_id="task_001",
        workspace_id="ws/001", goal="fix bug",
        steps=[_make_list_files_step(0), _make_finish_step(1)],
        source="scripted",
    )
    assert traj.action_count == 2
    assert traj.tool_distribution["list_files"] == 1
    assert traj.tool_distribution["finish"] == 1


def test_is_mutating_action():
    """is_mutating_action returns True for apply_patch and rollback_patch."""
    assert is_mutating_action("apply_patch") is True
    assert is_mutating_action("rollback_patch") is True
    assert is_mutating_action("list_files") is False
    assert is_mutating_action("run_tests") is False
    assert is_mutating_action("finish") is False
    assert is_mutating_action("write_memory") is False
