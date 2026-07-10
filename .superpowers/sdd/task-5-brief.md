## Task 5: Phase D — Corruption test expansion (all 5 CorruptionType values)

**Files:**
- Modify: `tests/test_agent_evaluator.py` (+5 tests)

**Interfaces:**
- Consumes: `CorruptedActionProvider`, `Corruption`, `CorruptionType` from `src/agent_evaluator.py`, `CorruptedActionProvider._corrupt_action`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_evaluator.py`:

```python
from src.agent_evaluator import CorruptedActionProvider, Corruption, CorruptionType


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
    result = _run_corrupted(monkeypatch, CorruptionType.INVALID_PATH)
    # INVALID_PATH → re-validation raises PathValidationError → recorded in errors
    assert len(result.errors) > 0, "expected errors for invalid path corruption"
    assert any("path" in e.lower() or "sensitive" in e.lower() or "invalid" in e.lower()
               for e in result.errors), \
        f"expected path-related error, got: {result.errors}"


def test_corruption_wrong_patch(monkeypatch):
    # WRONG_PATCH only affects apply_patch/propose_patch steps; find one
    traj = _load_first_success_trajectory()
    patch_step = None
    for i, s in enumerate(traj.steps):
        if s.action.action_type in ("apply_patch", "propose_patch"):
            patch_step = i
            break
    if patch_step is None:
        pytest.skip("trajectory has no patch step to corrupt")
    result = _run_corrupted(monkeypatch, CorruptionType.WRONG_PATCH, step_index=patch_step)
    # Wrong patch → tool_error or patch failure
    assert result.metrics.get("tool_error_rate", 0) > 0 or \
           result.metrics.get("patch_success_rate", 1) < 1, \
        "expected patch failure for WRONG_PATCH corruption"


def test_corruption_skip_tests_before_finish(monkeypatch):
    result = _run_corrupted(monkeypatch, CorruptionType.SKIP_TESTS_BEFORE_FINISH)
    # SKIP_TESTS converts run_tests→finish, so finish_without_tests may fire
    # if there's only one run_tests, OR the trajectory finishes early.
    # Key: no crash, metrics present.
    assert "finish_without_tests_count" in result.metrics


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.11 -m pytest tests/test_agent_evaluator.py::test_corruption_wrong_action_type tests/test_agent_evaluator.py::test_corruption_invalid_path tests/test_agent_evaluator.py::test_corruption_wrong_patch tests/test_agent_evaluator.py::test_corruption_skip_tests_before_finish tests/test_agent_evaluator.py::test_corruption_exceed_max_steps -v -p no:warnings`
Expected: Some may pass (corruption infrastructure exists from P4.0), but at least the `max_step_exceeded_count` metric and `EXCEED_MAX_STEPS` behavior likely fail because P4.0 didn't test them. Fix the metric name if the evaluator uses a different name — check `_make_result`.

- [ ] **Step 3: Fix any metric gaps (minimal)**

If `max_step_exceeded_count` is not in metrics, check `_make_result` in `src/agent_evaluator.py`. The P4.0 evaluator should already compute it (it's one of the 8 metrics). If the EXCEED_MAX_STEPS path doesn't set `max_steps_hit=True`, fix the loop's max_steps break to set it. This is a minimal fix, not new behavior.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_evaluator.py -v -p no:warnings`
Expected: All tests PASS. No regressions.

- [ ] **Step 5: Commit**

```bash
git add tests/test_agent_evaluator.py
git commit -m "feat(p4-1): Phase D — corruption test expansion (all 5 CorruptionType values)"
```

---

