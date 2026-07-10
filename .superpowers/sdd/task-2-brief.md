## Task 2: Phase B — TEST_PASS replay-authoritative + finish_claim_mismatch

**Files:**
- Modify: `src/agent_evaluator.py` (EvalResult model, run() TEST_PASS branch)
- Modify: `tests/test_agent_evaluator.py` (+3 tests)

**Interfaces:**
- Consumes: `EvalResult` model, `AgentEvaluator.run()`, `passed_tests` counter, `finish.tests_passed` field
- Produces: `EvalResult.finish_claim_mismatch: bool` field; TEST_PASS success now uses `passed_tests > 0`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_evaluator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.11 -m pytest tests/test_agent_evaluator.py::test_test_pass_success_uses_replay_not_claim tests/test_agent_evaluator.py::test_test_pass_mismatch_claimed_fail_actual_pass tests/test_agent_evaluator.py::test_test_pass_no_mismatch_when_claim_matches_replay -v -p no:warnings`
Expected: 2 FAIL (first two — `finish_claim_mismatch` attribute doesn't exist; success still uses claim). Third may pass if existing behavior happens to align, but the `finish_claim_mismatch` attribute access will fail.

- [ ] **Step 3: Add `finish_claim_mismatch` to `EvalResult`**

In `src/agent_evaluator.py`, add the field to `EvalResult`:

```python
class EvalResult(BaseModel):
    """Result of evaluating one task."""
    task_id: str
    trajectory_id: str
    steps_executed: int
    success: bool
    metrics: dict[str, float | int]
    errors: list[str] = Field(default_factory=list)
    max_steps_hit: bool = False
    finish_claim_mismatch: bool = False
```

- [ ] **Step 4: Change TEST_PASS branch to use replay**

In `AgentEvaluator.run()`, in the `finish` dispatch branch, replace:

```python
if fa.success_criterion == TaskSuccessCriterion.TEST_PASS:
    success = fa.tests_passed
```

with:

```python
if fa.success_criterion == TaskSuccessCriterion.TEST_PASS:
    replay_passed = passed_tests > 0
    success = replay_passed
    finish_claim_mismatch = (fa.tests_passed != replay_passed)
```

Then pass `finish_claim_mismatch` to `_make_result`. Update `_make_result` to accept and set it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_evaluator.py -v -p no:warnings`
Expected: All tests PASS (existing + 3 new). No regressions.

- [ ] **Step 6: Commit**

```bash
git add src/agent_evaluator.py tests/test_agent_evaluator.py
git commit -m "feat(p4-1): Phase B — TEST_PASS replay-authoritative + finish_claim_mismatch"
```

---

