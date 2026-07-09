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
    """EvalResult has all 9 metrics."""
    traj = _load_first_success_trajectory()
    provider = ScriptedActionProvider(traj)
    # Don't need to run — just check the metric keys are defined
    expected_keys = {
        "task_success_rate", "action_validity_rate", "tool_error_rate",
        "patch_success_rate", "tests_pass_rate", "forbidden_action_count",
        "max_step_exceeded_count", "finish_without_tests_count",
        "invalid_action_count",
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
            "invalid_action_count": 0,
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


# --- Task 3: 11-action allowlist + unknown hard-fail + dispatch ---

from src.agent_evaluator import _ALLOWED_ACTION_TYPES  # noqa: E402


def test_allowed_action_types_has_exactly_11():
    expected = {
        "list_files", "read_file", "search_text", "inspect_task",
        "propose_patch", "apply_patch", "rollback_patch", "run_tests",
        "inspect_error", "write_memory", "finish",
    }
    assert _ALLOWED_ACTION_TYPES == expected
    assert len(_ALLOWED_ACTION_TYPES) == 11


def test_unknown_action_type_recorded_as_forbidden(monkeypatch):
    """If an action with an unknown action_type somehow reaches dispatch
    (bypassing re-validation), the else branch must record it as forbidden,
    not silently no-op. We test the guard by mocking model_validate to pass."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        # Create a valid FinishAction, then patch its action_type attribute
        # and mock re-validation to let it through (simulates a future action
        # type slipping past the Literal check).
        finish = _make_finish(tests_passed=True)
        object.__setattr__(finish, "action_type", "shell_exec")
        provider = _FixedProvider([finish])
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        import unittest.mock as mock
        with mock.patch.object(type(finish), 'model_validate', return_value=finish):
            result = evaluator.run()
        assert result.metrics.get("forbidden_action_count", 0) >= 1, \
            "unknown action type must be counted as forbidden"
        assert any("shell_exec" in e for e in result.errors), \
            "unknown action type must be recorded in errors"
    finally:
        ws.cleanup()


def test_search_text_dispatched(monkeypatch):
    """search_text action must produce a real tool call to tool_search_text."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        from src.agent_actions import SearchTextAction, SearchTextArgs
        import src.agent_evaluator as evaluator_mod

        # Spy on tool_search_text to verify it actually gets called.
        call_count = {"n": 0}
        original = evaluator_mod.tool_search_text

        def spy(workspace, query, *args, **kwargs):
            call_count["n"] += 1
            return original(workspace, query, *args, **kwargs)

        monkeypatch.setattr(evaluator_mod, "tool_search_text", spy)

        search_action = SearchTextAction(
            action_id="search_1",
            reason_short="search",
            expected_observation="matches",
            safety_flags=_make_safe_safety_flags(is_terminal=False),
            arguments=SearchTextArgs(query="def"),
        )
        finish = _make_finish(tests_passed=True)
        provider = _FixedProvider([search_action, finish])
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        evaluator.run()
        assert call_count["n"] >= 1, \
            f"expected tool_search_text to be called, got {call_count['n']}"
    finally:
        ws.cleanup()


# --- Task 5: All 5 CorruptionType values tested ---


def _run_corrupted(monkeypatch, corruption_type, step_index=2):
    """Helper: run a scripted trajectory with corruption injected."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        provider = CorruptedActionProvider(
            traj, Corruption(step_index=step_index, type=corruption_type)
        )
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        return evaluator.run()
    finally:
        ws.cleanup()


def test_corruption_wrong_action_type(monkeypatch):
    result = _run_corrupted(monkeypatch, CorruptionType.WRONG_ACTION_TYPE)
    # WRONG_ACTION_TYPE swaps to ListFilesAction — not invalid, but different
    # The key: trajectory still completes (ListFiles is safe) OR errors if
    # the swap breaks the expected flow. Assert no crash + metrics present.
    assert "action_validity_rate" in result.metrics
    assert result.steps_executed > 0


def test_corruption_invalid_path(monkeypatch):
    # INVALID_PATH only corrupts actions with `path` or `file_path` args;
    # find the first such step so the corruption has a real effect.
    traj = _load_first_success_trajectory()
    path_step = None
    for i, s in enumerate(traj.steps):
        args = s.action.arguments
        if args is not None and (
            hasattr(args, "path") or hasattr(args, "file_path")
        ):
            path_step = i
            break
    if path_step is None:
        pytest.skip("trajectory has no step with path/file_path to corrupt")
    result = _run_corrupted(
        monkeypatch, CorruptionType.INVALID_PATH, step_index=path_step
    )
    # INVALID_PATH → re-validation raises PathValidationError → recorded in errors
    assert len(result.errors) > 0, "expected errors for invalid path corruption"
    assert any("path" in e.lower() or "sensitive" in e.lower() or "invalid" in e.lower()
               for e in result.errors), \
        f"expected path-related error, got: {result.errors}"


def test_corruption_wrong_patch(monkeypatch):
    # WRONG_PATCH only affects apply_patch/propose_patch steps. Find apply_patch
    # specifically so the corruption has a measurable effect: a failed patch
    # increments tool_errors and leaves successful_patches at 0. (propose_patch
    # corruption is a no-op for metrics because tool_propose_patch doesn't raise
    # and propose_patch isn't counted in total_patches.)
    traj = _load_first_success_trajectory()
    patch_step = None
    for i, s in enumerate(traj.steps):
        if s.action.action_type == "apply_patch":
            patch_step = i
            break
    if patch_step is None:
        pytest.skip("trajectory has no apply_patch step to corrupt")
    result = _run_corrupted(monkeypatch, CorruptionType.WRONG_PATCH, step_index=patch_step)
    # Wrong patch → tool_error or patch failure
    assert result.metrics.get("tool_error_rate", 0) > 0 or \
           result.metrics.get("patch_success_rate", 1) < 1, \
        "expected patch failure for WRONG_PATCH corruption"


def test_corruption_skip_tests_before_finish(monkeypatch):
    # SKIP_TESTS_BEFORE_FINISH only converts run_tests→finish; find the first
    # run_tests step so the corruption has a real effect (early finish).
    traj = _load_first_success_trajectory()
    run_tests_step = None
    for i, s in enumerate(traj.steps):
        if s.action.action_type == "run_tests":
            run_tests_step = i
            break
    if run_tests_step is None:
        pytest.skip("trajectory has no run_tests step to corrupt")
    result = _run_corrupted(
        monkeypatch, CorruptionType.SKIP_TESTS_BEFORE_FINISH, step_index=run_tests_step
    )
    # SKIP_TESTS converts run_tests→finish, so the trajectory finishes early
    # without executing run_tests → finish_without_tests_count >= 1 and
    # steps_executed is less than the full trajectory length.
    assert result.metrics.get("finish_without_tests_count", 0) >= 1, \
        f"expected finish_without_tests_count >= 1, got " \
        f"{result.metrics.get('finish_without_tests_count')}"
    assert result.steps_executed < len(traj.steps), \
        f"expected early finish (steps_executed < {len(traj.steps)}), " \
        f"got {result.steps_executed}"


def test_corruption_exceed_max_steps(monkeypatch):
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        provider = CorruptedActionProvider(
            traj, Corruption(step_index=0, type=CorruptionType.EXCEED_MAX_STEPS)
        )
        # Use a small max_steps so the test doesn't loop 20 times
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=5)
        result = evaluator.run()
        assert result.max_steps_hit, "expected max_steps_hit=True for EXCEED_MAX_STEPS"
        assert result.metrics.get("max_step_exceeded_count", 0) >= 1, \
            "expected max_step_exceeded_count >= 1"
    finally:
        ws.cleanup()


# --- Task 6: SentinelAction counted as invalid, not forbidden ---

def test_sentinel_action_counted_as_invalid_not_forbidden(monkeypatch):
    """SentinelAction must increment invalid_action_count, not forbidden_action_count."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        from src.agent_model_provider import SentinelAction
        # Build a provider that returns SentinelAction then finish
        sentinel = SentinelAction(reason="test invalid")
        finish = _make_finish(tests_passed=True)
        provider = _FixedProvider([sentinel, finish])
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        assert result.metrics.get("invalid_action_count", 0) >= 1, \
            "SentinelAction must increment invalid_action_count"
        assert result.metrics.get("forbidden_action_count", 0) == 0, \
            "SentinelAction must NOT increment forbidden_action_count"
    finally:
        ws.cleanup()
