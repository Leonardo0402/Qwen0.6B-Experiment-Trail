## Commit List
41a0d05 feat(p4-1): Phase B — 11-action allowlist + unknown hard-fail + search_text/rollback_patch dispatch

## Diffstat
`
 src/agent_evaluator.py        | 28 +++++++++++++++
 tests/test_agent_evaluator.py | 79 +++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 107 insertions(+)
`

## Full Diff
`diff
diff --git a/src/agent_evaluator.py b/src/agent_evaluator.py
index b92ef62..ff6dd64 100644
--- a/src/agent_evaluator.py
+++ b/src/agent_evaluator.py
@@ -21,24 +21,36 @@ from src.agent_actions import (
 )
 from src.agent_state import AgentMemory
 from src.agent_trajectory import Trajectory
 from src.agent_workspace import MicroTaskWorkspace
 from src.agent_tools import (
     tool_apply_patch,
     tool_inspect_task,
     tool_list_files,
     tool_propose_patch,
     tool_read_file,
+    tool_rollback_patch,
     tool_run_tests,
+    tool_search_text,
 )
 
 
+# Phase B allowlist: every action_type the evaluator dispatches.
+# Any action_type not in this set is recorded as forbidden by the
+# dispatch loop's defensive `else` branch.
+_ALLOWED_ACTION_TYPES = frozenset({
+    "list_files", "read_file", "search_text", "inspect_task",
+    "propose_patch", "apply_patch", "rollback_patch", "run_tests",
+    "inspect_error", "write_memory", "finish",
+})
+
+
 class AgentState(BaseModel):
     """State passed to ActionProvider.next_action()."""
     memory: AgentMemory
     step_count: int
     task_id: str
     workspace_id: str
 
 
 class EvalResult(BaseModel):
     """Result of evaluating one task."""
@@ -322,20 +334,26 @@ class AgentEvaluator:
                     obs = tool_run_tests(self._ws, timeout_s=10.0)
                     total_tests += 1
                     if obs.passed:
                         passed_tests += 1
                 elif at == "inspect_error":
                     # Stateless — no tool dispatch needed for replay
                     pass
                 elif at == "write_memory":
                     # Stateless — update state memory
                     state.memory = action.arguments.memory
+                elif at == "search_text":
+                    total_tools += 1
+                    tool_search_text(self._ws, action.arguments.query)
+                elif at == "rollback_patch":
+                    total_tools += 1
+                    tool_rollback_patch(self._ws, action.arguments.action_id)
                 elif at == "finish":
                     fa = action.arguments
                     if not ran_tests:
                         finish_without_tests += 1
                     # Check success criterion
                     if fa.success_criterion == TaskSuccessCriterion.TEST_PASS:
                         replay_passed = passed_tests > 0
                         success = replay_passed
                         finish_claim_mismatch = (fa.tests_passed != replay_passed)
                     elif fa.success_criterion == TaskSuccessCriterion.IDENTIFY_BUG:
@@ -346,20 +364,30 @@ class AgentEvaluator:
                     # Compute metrics and return
                     return self._make_result(
                         steps_executed, success, errors,
                         total_actions, valid_actions, forbidden_count,
                         total_patches, successful_patches,
                         total_tests, passed_tests,
                         tool_errors, total_tools,
                         finish_without_tests, max_steps_hit=False,
                         finish_claim_mismatch=finish_claim_mismatch,
                     )
+                else:
+                    # Defensive guard: any action_type not in the 11-action
+                    # allowlist is recorded as forbidden. (Unreachable via
+                    # valid Pydantic actions — every Literal action_type is
+                    # one of the 11 above.)
+                    forbidden_count += 1
+                    errors.append(
+                        f"step {step}: unknown action type (not in 11-action "
+                        f"allowlist): {at}"
+                    )
             except Exception as e:
                 tool_errors += 1
                 errors.append(f"step {step}: tool error: {e}")
 
             state.step_count = step + 1
             steps_executed = step + 1
 
         # Hit max_steps without finish
         max_steps_hit = True
         return self._make_result(
diff --git a/tests/test_agent_evaluator.py b/tests/test_agent_evaluator.py
index e6c7517..313af34 100644
--- a/tests/test_agent_evaluator.py
+++ b/tests/test_agent_evaluator.py
@@ -365,10 +365,89 @@ def test_test_pass_no_mismatch_when_claim_matches_replay(monkeypatch):
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
+
+
+# --- Task 3: 11-action allowlist + unknown hard-fail + dispatch ---
+
+from src.agent_evaluator import _ALLOWED_ACTION_TYPES  # noqa: E402
+
+
+def test_allowed_action_types_has_exactly_11():
+    expected = {
+        "list_files", "read_file", "search_text", "inspect_task",
+        "propose_patch", "apply_patch", "rollback_patch", "run_tests",
+        "inspect_error", "write_memory", "finish",
+    }
+    assert _ALLOWED_ACTION_TYPES == expected
+    assert len(_ALLOWED_ACTION_TYPES) == 11
+
+
+def test_unknown_action_type_recorded_as_forbidden(monkeypatch):
+    """If an action with an unknown action_type somehow reaches dispatch
+    (bypassing re-validation), the else branch must record it as forbidden,
+    not silently no-op. We test the guard by mocking model_validate to pass."""
+    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
+    traj = _load_first_success_trajectory()
+    task_dir = TASKS_DIR / traj.task_id
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        # Create a valid FinishAction, then patch its action_type attribute
+        # and mock re-validation to let it through (simulates a future action
+        # type slipping past the Literal check).
+        finish = _make_finish(tests_passed=True)
+        object.__setattr__(finish, "action_type", "shell_exec")
+        provider = _FixedProvider([finish])
+        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
+        import unittest.mock as mock
+        with mock.patch.object(type(finish), 'model_validate', return_value=finish):
+            result = evaluator.run()
+        assert result.metrics.get("forbidden_action_count", 0) >= 1, \
+            "unknown action type must be counted as forbidden"
+        assert any("shell_exec" in e for e in result.errors), \
+            "unknown action type must be recorded in errors"
+    finally:
+        ws.cleanup()
+
+
+def test_search_text_dispatched(monkeypatch):
+    """search_text action must produce a real tool call to tool_search_text."""
+    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
+    traj = _load_first_success_trajectory()
+    task_dir = TASKS_DIR / traj.task_id
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        from src.agent_actions import SearchTextAction, SearchTextArgs
+        import src.agent_evaluator as evaluator_mod
+
+        # Spy on tool_search_text to verify it actually gets called.
+        call_count = {"n": 0}
+        original = evaluator_mod.tool_search_text
+
+        def spy(workspace, query, *args, **kwargs):
+            call_count["n"] += 1
+            return original(workspace, query, *args, **kwargs)
+
+        monkeypatch.setattr(evaluator_mod, "tool_search_text", spy)
+
+        search_action = SearchTextAction(
+            action_id="search_1",
+            reason_short="search",
+            expected_observation="matches",
+            safety_flags=_make_safe_safety_flags(is_terminal=False),
+            arguments=SearchTextArgs(query="def"),
+        )
+        finish = _make_finish(tests_passed=True)
+        provider = _FixedProvider([search_action, finish])
+        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
+        evaluator.run()
+        assert call_count["n"] >= 1, \
+            f"expected tool_search_text to be called, got {call_count['n']}"
+    finally:
+        ws.cleanup()
`
