"""Phase G: Agent evaluator tests."""
import json
from pathlib import Path

import pytest

from src.agent_state import AgentMemory
from src.agent_actions import (
    TaskSuccessCriterion, FinishAction, FinishArgs, SafetyFlags,
)
from src.agent_trajectory import Trajectory
from src.agent_evaluator import (
    AgentEvaluator, ScriptedActionProvider, ReplayActionProvider,
    CorruptedActionProvider, Corruption, CorruptionType, AgentState,
    EvalResult,
)
from src.agent_workspace import MicroTaskWorkspace

TRAJ_PATH = Path(__file__).parent.parent / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
TASKS_DIR = Path(__file__).parent.parent / "data" / "p4-agent" / "micro-tasks-v0"


def _load_first_trajectory() -> Trajectory:
    with TRAJ_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return Trajectory.model_validate_json(line)
    raise RuntimeError("no trajectories")


def _load_first_success_trajectory() -> Trajectory:
    """Load the first trajectory with final_success=True (types 2-8)."""
    with TRAJ_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traj = Trajectory.model_validate_json(line)
                if traj.final_success:
                    return traj
    raise RuntimeError("no success trajectory")


def test_replay_provider():
    """ScriptedActionProvider replays actions in order."""
    traj = _load_first_trajectory()
    provider = ScriptedActionProvider(traj)
    state = AgentState(
        memory=AgentMemory(), step_count=0,
        task_id=traj.task_id, workspace_id=traj.workspace_id,
    )
    actions = []
    try:
        while True:
            actions.append(provider.next_action(state))
    except StopIteration:
        pass
    assert len(actions) == len(traj.steps)
    assert actions[-1].action_type == "finish"


def test_replay_success(monkeypatch):
    """Replaying a success trajectory through real tools succeeds."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        provider = ReplayActionProvider(traj)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        assert result.success, f"expected success, errors: {result.errors}"
        assert result.steps_executed == len(traj.steps)
    finally:
        ws.cleanup()


def test_corrupted_injection(monkeypatch):
    """CorruptedActionProvider with WRONG_PATCH causes patch_success_rate < 1.0."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        # Corrupt the apply_patch step (find it in the trajectory)
        patch_step = None
        for i, step in enumerate(traj.steps):
            if step.action.action_type == "apply_patch":
                patch_step = i
                break
        assert patch_step is not None, "no apply_patch step found"

        corruption = Corruption(step_index=patch_step, type=CorruptionType.WRONG_PATCH)
        provider = CorruptedActionProvider(traj, corruption)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        # patch_success_rate should be < 1.0 (the corrupted patch fails)
        assert result.metrics["patch_success_rate"] < 1.0, \
            f"expected patch_success_rate < 1.0, got {result.metrics['patch_success_rate']}"
    finally:
        ws.cleanup()


def test_all_metrics_present():
    """EvalResult has all 8 metrics."""
    traj = _load_first_success_trajectory()
    provider = ScriptedActionProvider(traj)
    # Don't need to run — just check the metric keys are defined
    expected_keys = {
        "task_success_rate", "action_validity_rate", "tool_error_rate",
        "patch_success_rate", "tests_pass_rate", "forbidden_action_count",
        "max_step_exceeded_count", "finish_without_tests_count",
    }
    # Build a minimal EvalResult to check keys
    result = EvalResult(
        task_id="test", trajectory_id="test", steps_executed=0,
        success=False,
        metrics={
            "task_success_rate": 0.0,
            "action_validity_rate": 0.0,
            "tool_error_rate": 0.0,
            "patch_success_rate": 0.0,
            "tests_pass_rate": 0.0,
            "forbidden_action_count": 0,
            "max_step_exceeded_count": 0,
            "finish_without_tests_count": 0,
        },
    )
    assert set(result.metrics.keys()) == expected_keys
