"""Phase F: Scripted teacher trajectories tests."""
import json
from pathlib import Path

import pytest

from src.agent_trajectory import Trajectory

TRAJ_PATH = Path(__file__).parent.parent / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"


def _load_trajectories():
    trajectories = []
    with TRAJ_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(Trajectory.model_validate_json(line))
    return trajectories


def test_scripted_jsonl_exists():
    """scripted.jsonl exists and is non-empty."""
    assert TRAJ_PATH.exists(), f"missing: {TRAJ_PATH}"
    trajectories = _load_trajectories()
    assert len(trajectories) > 0


def test_40_trajectories():
    """40 trajectories exist and all pass schema validation."""
    trajectories = _load_trajectories()
    assert len(trajectories) == 40
    # All are valid Trajectory objects (schema validation passed on load)
    for traj in trajectories:
        assert traj.task_id.startswith("task_")
        assert traj.source == "scripted"
        assert len(traj.steps) >= 1
        assert traj.steps[-1].action.action_type == "finish"


def test_success_labels():
    """35 trajectories (types 2-8) have final_success=True;
    5 (type 1) have final_success=False with identification_verified=True."""
    trajectories = _load_trajectories()
    success_count = sum(1 for t in trajectories if t.final_success)
    identify_count = sum(
        1 for t in trajectories
        if not t.final_success
        and t.steps[-1].action.action_type == "finish"
        and t.steps[-1].action.arguments.identification_verified
    )
    assert success_count == 35, f"expected 35 success, got {success_count}"
    assert identify_count == 5, f"expected 5 identify, got {identify_count}"


def test_replay_first_5_standard(monkeypatch):
    """For first 5 type-2-8 trajectories, replay apply_patch + run_tests passes."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    from src.agent_workspace import MicroTaskWorkspace
    from src.agent_tools import tool_apply_patch, tool_run_tests

    trajectories = _load_trajectories()
    tasks_dir = Path(__file__).parent.parent / "data" / "p4-agent" / "micro-tasks-v0"

    # Find first 5 trajectories with final_success=True
    success_trajs = [t for t in trajectories if t.final_success][:5]
    assert len(success_trajs) == 5

    for traj in success_trajs:
        task_dir = tasks_dir / traj.task_id
        patch = json.loads((task_dir / "expected_patch.json").read_text())
        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            tool_apply_patch(ws, patch["file_path"], patch["old_text"], patch["new_text"])
            obs = tool_run_tests(ws, timeout_s=10.0)
            assert obs.passed, f"{traj.task_id}: replay tests should pass"
        finally:
            ws.cleanup()
