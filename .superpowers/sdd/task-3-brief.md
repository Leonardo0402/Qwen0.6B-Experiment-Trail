## Task 3: Phase B — 11-action allowlist + unknown hard-fail + search_text/rollback_patch dispatch

**Files:**
- Modify: `src/agent_evaluator.py` (add `_ALLOWED_ACTION_TYPES`, add search_text/rollback_patch branches, add `else: raise`)
- Modify: `tests/test_agent_evaluator.py` (+3 tests)

**Interfaces:**
- Consumes: `tool_search_text`, `tool_rollback_patch` from `src/agent_tools.py`, `P4ForbiddenActionError` from `src/agent_actions.py`
- Produces: evaluator dispatches all 11 action types; unknown action raises

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_evaluator.py`:

```python
from src.agent_actions import P4ForbiddenActionError
from src.agent_evaluator import _ALLOWED_ACTION_TYPES


# --- Task 3: 11-action allowlist + unknown hard-fail + dispatch ---

class _UnknownAction(FinishAction):
    """A fake action with an action_type not in the allowlist."""
    pass


def test_allowed_action_types_has_exactly_11():
    expected = {
        "list_files", "read_file", "search_text", "inspect_task",
        "propose_patch", "apply_patch", "rollback_patch", "run_tests",
        "inspect_error", "write_memory", "finish",
    }
    assert _ALLOWED_ACTION_TYPES == expected
    assert len(_ALLOWED_ACTION_TYPES) == 11


def test_unknown_action_type_raises(monkeypatch):
    """An action with action_type not in the 11-allowlist must raise
    P4ForbiddenActionError, not silently no-op."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        # Build an action with a bogus action_type by model_construct
        fake_action = FinishAction.model_construct(
            action_id="bad",
            reason_short="unknown type",
            expected_observation="x",
            safety_flags=_make_safe_safety_flags(is_terminal=False),
            arguments=FinishArgs(
                success_criterion=TaskSuccessCriterion.TEST_PASS,
                tests_passed=True,
                identification_verified=False,
                summary="x",
            ),
        )
        # Override action_type to a bogus value
        object.__setattr__(fake_action, "action_type", "shell_exec")
        provider = _FixedProvider([fake_action])
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        # Must NOT be silent: error recorded + forbidden_count incremented
        assert result.metrics["forbidden_action_count"] >= 1, \
            "unknown action type must be counted as forbidden"
        assert any("shell_exec" in e for e in result.errors), \
            "unknown action type must be recorded in errors"
    finally:
        ws.cleanup()


def test_search_text_dispatched(monkeypatch):
    """search_text action must produce a real tool call, incrementing
    total_tools."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    traj = _load_first_success_trajectory()
    task_dir = TASKS_DIR / traj.task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        from src.agent_actions import SearchTextAction, SearchTextArgs
        search_action = SearchTextAction(
            action_id="search_1",
            reason_short="search",
            expected_observation="matches",
            safety_flags=_make_safe_safety_flags(is_terminal=False),
            arguments=SearchTextArgs(pattern="def"),
        )
        finish = _make_finish(tests_passed=True)
        provider = _FixedProvider([search_action, finish])
        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
        result = evaluator.run()
        # search_text was dispatched → total_tools >= 1
        assert result.metrics["total_tools"] >= 1, \
            f"expected total_tools>=1 (search_text dispatched), got {result.metrics['total_tools']}"
    finally:
        ws.cleanup()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.11 -m pytest tests/test_agent_evaluator.py::test_allowed_action_types_has_exactly_11 tests/test_agent_evaluator.py::test_unknown_action_type_raises tests/test_agent_evaluator.py::test_search_text_dispatched -v -p no:warnings`
Expected: FAIL — `_ALLOWED_ACTION_TYPES` doesn't exist; search_text not dispatched; unknown action silently no-ops.

- [ ] **Step 3: Implement the allowlist, dispatch branches, and hard-fail**

In `src/agent_evaluator.py`:

1. Add the import: `from src.agent_tools import tool_search_text, tool_rollback_patch`
2. Add module-level constant:
```python
_ALLOWED_ACTION_TYPES = frozenset({
    "list_files", "read_file", "search_text", "inspect_task",
    "propose_patch", "apply_patch", "rollback_patch", "run_tests",
    "inspect_error", "write_memory", "finish",
})
```
3. Add the two dispatch branches (before the `finish` branch):
```python
elif at == "search_text":
    total_tools += 1
    tool_search_text(self._ws, action.arguments.pattern)
elif at == "rollback_patch":
    total_tools += 1
    tool_rollback_patch(self._ws, action.arguments.action_id)
```
4. Add the `else` hard-fail at the end of the dispatch chain (after `finish`):
```python
else:
    forbidden_count += 1
    errors.append(
        f"step {step}: unknown action type (not in 11-action "
        f"allowlist): {at}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_evaluator.py -v -p no:warnings`
Expected: All tests PASS. No regressions.

- [ ] **Step 5: Commit**

```bash
git add src/agent_evaluator.py tests/test_agent_evaluator.py
git commit -m "feat(p4-1): Phase B — 11-action allowlist + unknown hard-fail + search_text/rollback_patch dispatch"
```

---

