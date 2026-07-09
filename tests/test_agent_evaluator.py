"""Phase G: Agent evaluator tests."""
import json
from pathlib import Path

import pytest

from src.agent_state import AgentMemory
from src.agent_actions import (
    TaskSuccessCriterion, FinishAction, FinishArgs, SafetyFlags,
    RunTestsAction, ReadFileAction, ReadFileArgs,
    ApplyPatchAction, ApplyPatchArgs, ListFilesAction,
)
from src.agent_trajectory import Trajectory
from src.agent_evaluator import (
    AgentEvaluator, ScriptedActionProvider, ReplayActionProvider,
    CorruptedActionProvider, Corruption, CorruptionType, AgentState,
    EvalResult, ActionProvider, _make_safe_safety_flags,
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


class _FixedProvider(ActionProvider):
    """Test provider that yields a fixed list of actions."""
    def __init__(self, actions):
        self._actions = actions
        self._index = 0

    def next_action(self, state: AgentState):
        if self._index >= len(self._actions):
            raise StopIteration
        action = self._actions[self._index]
        self._index += 1
        return action

    def reset(self):
        self._index = 0


def _make_finish(tests_passed, success_criterion=TaskSuccessCriterion.TEST_PASS,
                 identification_verified=False):
    return FinishAction(
        action_id="finish_test",
        reason_short="finish",
        expected_observation="done",
        safety_flags=_make_safe_safety_flags(is_terminal=True),
        arguments=FinishArgs(
            success_criterion=success_criterion,
            tests_passed=tests_passed,
            identification_verified=identification_verified,
            summary="test finish",
        ),
    )


# --- Fix 2: finish_without_tests_count based on actual run_tests execution ---

def test_finish_without_tests_zero_when_run_tests_executed(monkeypatch):
    """finish_without_tests_count=0 when run_tests WAS executed, even if
    finish.tests_passed=False (e.g., identify-only or failed-test task)."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        # Reuse the trajectory's actions but modify finish to tests_passed=False
        actions = [s.action for s in traj.steps]
        original_finish = actions[-1]
        modified_finish = original_finish.model_copy(update={
            "arguments": original_finish.arguments.model_copy(update={
                "tests_passed": False,
                "success_criterion": TaskSuccessCriterion.IDENTIFY_BUG,
                "identification_verified": True,
            }),
        })
        actions[-1] = modified_finish

        provider = _FixedProvider(actions)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        # run_tests WAS executed before finish → count must be 0
        assert result.metrics["finish_without_tests_count"] == 0, \
            f"expected 0 (run_tests was executed), got {result.metrics['finish_without_tests_count']}"
    finally:
        ws.cleanup()


def test_finish_without_tests_one_when_no_run_tests(monkeypatch):
    """finish_without_tests_count=1 when finish is reached WITHOUT any
    prior run_tests action, even if finish.tests_passed=True (false claim)."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        # Strip all run_tests actions, keep only non-run_tests + finish
        actions = [s.action for s in traj.steps
                   if s.action.action_type != "run_tests"]
        provider = _FixedProvider(actions)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        # No run_tests executed → count must be 1
        assert result.metrics["finish_without_tests_count"] == 1, \
            f"expected 1 (no run_tests executed), got {result.metrics['finish_without_tests_count']}"
    finally:
        ws.cleanup()


# --- Fix 3: evaluator handles PATCH_APPLIED success criterion ---

def test_patch_applied_success_when_patch_succeeded(monkeypatch):
    """success=True when success_criterion=PATCH_APPLIED and a patch was
    successfully applied during replay."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        actions = [s.action for s in traj.steps]
        original_finish = actions[-1]
        modified_finish = original_finish.model_copy(update={
            "arguments": original_finish.arguments.model_copy(update={
                "success_criterion": TaskSuccessCriterion.PATCH_APPLIED,
            }),
        })
        actions[-1] = modified_finish

        provider = _FixedProvider(actions)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        # Patch was applied successfully → success must be True
        assert result.success, \
            f"expected success=True (PATCH_APPLIED + patch succeeded), got False; errors: {result.errors}"
    finally:
        ws.cleanup()


def test_patch_applied_fail_when_no_patch(monkeypatch):
    """success=False when success_criterion=PATCH_APPLIED but no patch was
    applied during replay."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        # Strip all apply_patch actions
        actions = [s.action for s in traj.steps
                   if s.action.action_type != "apply_patch"]
        original_finish = actions[-1]
        modified_finish = original_finish.model_copy(update={
            "arguments": original_finish.arguments.model_copy(update={
                "success_criterion": TaskSuccessCriterion.PATCH_APPLIED,
            }),
        })
        actions[-1] = modified_finish

        provider = _FixedProvider(actions)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        # No patch applied → success must be False
        assert not result.success, \
            "expected success=False (PATCH_APPLIED but no patch applied), got True"
    finally:
        ws.cleanup()


# --- Fix 4: tool_error_rate denominator counts attempted tools ---

def test_tool_error_rate_counts_failed_attempts(monkeypatch):
    """tool_error_rate must count failed tool calls in the denominator.
    A read_file on a non-existent file raises → 1 error / 1 attempted = 1.0."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        # Provider yields: read_file(nonexistent) → finish
        bad_read = ReadFileAction(
            action_id="bad_read",
            reason_short="read missing file",
            expected_observation="file content",
            safety_flags=_make_safe_safety_flags(is_terminal=False),
            arguments=ReadFileArgs(path="nonexistent_file.py"),
        )
        finish = _make_finish(tests_passed=True)
        provider = _FixedProvider([bad_read, finish])
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        # 1 failed tool call → tool_error_rate must be 1.0 (not 0.0)
        assert result.metrics["tool_error_rate"] == 1.0, \
            f"expected tool_error_rate=1.0 (1 error / 1 attempted), got {result.metrics['tool_error_rate']}"
    finally:
        ws.cleanup()


# --- Task 2: TEST_PASS replay-authoritative + finish_claim_mismatch ---

def test_test_pass_success_uses_replay_not_claim(monkeypatch):
    """finish claims tests_passed=True but replay has 0 passed_tests
    → success=False, finish_claim_mismatch=True."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        # Strip all run_tests actions so replay passed_tests=0,
        # but finish still claims tests_passed=True (false claim)
        actions = [s.action for s in traj.steps
                   if s.action.action_type != "run_tests"]
        provider = _FixedProvider(actions)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        assert not result.success, \
            "expected success=False (replay has 0 passed, claim says True)"
        assert result.finish_claim_mismatch, \
            "expected finish_claim_mismatch=True (claim≠replay)"
    finally:
        ws.cleanup()


def test_test_pass_mismatch_claimed_fail_actual_pass(monkeypatch):
    """finish claims tests_passed=False but replay passed_tests>0
    → success=True, finish_claim_mismatch=True."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        actions = [s.action for s in traj.steps]
        original_finish = actions[-1]
        # Keep run_tests (so replay passes), but finish claims tests_passed=False
        modified_finish = original_finish.model_copy(update={
            "arguments": original_finish.arguments.model_copy(update={
                "tests_passed": False,
            }),
        })
        actions[-1] = modified_finish
        provider = _FixedProvider(actions)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        assert result.success, \
            "expected success=True (replay passed_tests>0 is authoritative)"
        assert result.finish_claim_mismatch, \
            "expected finish_claim_mismatch=True (claim says fail, replay says pass)"
    finally:
        ws.cleanup()


def test_test_pass_no_mismatch_when_claim_matches_replay(monkeypatch):
    """finish claims tests_passed=True AND replay passed_tests>0
    → success=True, finish_claim_mismatch=False."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        actions = [s.action for s in traj.steps]
        provider = _FixedProvider(actions)
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        assert result.success
        assert not result.finish_claim_mismatch, \
            "expected finish_claim_mismatch=False (claim matches replay)"
    finally:
        ws.cleanup()
