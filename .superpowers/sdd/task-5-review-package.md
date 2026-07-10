# Task 5 Review Package

**BASE:** 50ec3af
**HEAD:** 439c097

## Commit list

`
439c097 feat(p4-1): Phase D — corruption test expansion (all 5 CorruptionType values)
`

## Diffstat

`
 tests/test_agent_evaluator.py | 117 ++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 117 insertions(+)
`

## Full diff (with 10 lines context)

`diff
diff --git a/tests/test_agent_evaluator.py b/tests/test_agent_evaluator.py
index 313af34..26a235f 100644
--- a/tests/test_agent_evaluator.py
+++ b/tests/test_agent_evaluator.py
@@ -444,10 +444,127 @@ def test_search_text_dispatched(monkeypatch):
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
+
+
+# --- Task 5: All 5 CorruptionType values tested ---
+
+
+def _run_corrupted(monkeypatch, corruption_type, step_index=2):
+    """Helper: run a scripted trajectory with corruption injected."""
+    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
+    traj = _load_first_success_trajectory()
+    task_dir = TASKS_DIR / traj.task_id
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        provider = CorruptedActionProvider(
+            traj, Corruption(step_index=step_index, type=corruption_type)
+        )
+        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
+        return evaluator.run()
+    finally:
+        ws.cleanup()
+
+
+def test_corruption_wrong_action_type(monkeypatch):
+    result = _run_corrupted(monkeypatch, CorruptionType.WRONG_ACTION_TYPE)
+    # WRONG_ACTION_TYPE swaps to ListFilesAction — not invalid, but different
+    # The key: trajectory still completes (ListFiles is safe) OR errors if
+    # the swap breaks the expected flow. Assert no crash + metrics present.
+    assert "action_validity_rate" in result.metrics
+    assert result.steps_executed > 0
+
+
+def test_corruption_invalid_path(monkeypatch):
+    # INVALID_PATH only corrupts actions with `path` or `file_path` args;
+    # find the first such step so the corruption has a real effect.
+    traj = _load_first_success_trajectory()
+    path_step = None
+    for i, s in enumerate(traj.steps):
+        args = s.action.arguments
+        if args is not None and (
+            hasattr(args, "path") or hasattr(args, "file_path")
+        ):
+            path_step = i
+            break
+    if path_step is None:
+        pytest.skip("trajectory has no step with path/file_path to corrupt")
+    result = _run_corrupted(
+        monkeypatch, CorruptionType.INVALID_PATH, step_index=path_step
+    )
+    # INVALID_PATH → re-validation raises PathValidationError → recorded in errors
+    assert len(result.errors) > 0, "expected errors for invalid path corruption"
+    assert any("path" in e.lower() or "sensitive" in e.lower() or "invalid" in e.lower()
+               for e in result.errors), \
+        f"expected path-related error, got: {result.errors}"
+
+
+def test_corruption_wrong_patch(monkeypatch):
+    # WRONG_PATCH only affects apply_patch/propose_patch steps. Find apply_patch
+    # specifically so the corruption has a measurable effect: a failed patch
+    # increments tool_errors and leaves successful_patches at 0. (propose_patch
+    # corruption is a no-op for metrics because tool_propose_patch doesn't raise
+    # and propose_patch isn't counted in total_patches.)
+    traj = _load_first_success_trajectory()
+    patch_step = None
+    for i, s in enumerate(traj.steps):
+        if s.action.action_type == "apply_patch":
+            patch_step = i
+            break
+    if patch_step is None:
+        pytest.skip("trajectory has no apply_patch step to corrupt")
+    result = _run_corrupted(monkeypatch, CorruptionType.WRONG_PATCH, step_index=patch_step)
+    # Wrong patch → tool_error or patch failure
+    assert result.metrics.get("tool_error_rate", 0) > 0 or \
+           result.metrics.get("patch_success_rate", 1) < 1, \
+        "expected patch failure for WRONG_PATCH corruption"
+
+
+def test_corruption_skip_tests_before_finish(monkeypatch):
+    # SKIP_TESTS_BEFORE_FINISH only converts run_tests→finish; find the first
+    # run_tests step so the corruption has a real effect (early finish).
+    traj = _load_first_success_trajectory()
+    run_tests_step = None
+    for i, s in enumerate(traj.steps):
+        if s.action.action_type == "run_tests":
+            run_tests_step = i
+            break
+    if run_tests_step is None:
+        pytest.skip("trajectory has no run_tests step to corrupt")
+    result = _run_corrupted(
+        monkeypatch, CorruptionType.SKIP_TESTS_BEFORE_FINISH, step_index=run_tests_step
+    )
+    # SKIP_TESTS converts run_tests→finish, so the trajectory finishes early
+    # without executing run_tests → finish_without_tests_count >= 1 and
+    # steps_executed is less than the full trajectory length.
+    assert result.metrics.get("finish_without_tests_count", 0) >= 1, \
+        f"expected finish_without_tests_count >= 1, got " \
+        f"{result.metrics.get('finish_without_tests_count')}"
+    assert result.steps_executed < len(traj.steps), \
+        f"expected early finish (steps_executed < {len(traj.steps)}), " \
+        f"got {result.steps_executed}"
+
+
+def test_corruption_exceed_max_steps(monkeypatch):
+    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
+    traj = _load_first_success_trajectory()
+    task_dir = TASKS_DIR / traj.task_id
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        provider = CorruptedActionProvider(
+            traj, Corruption(step_index=0, type=CorruptionType.EXCEED_MAX_STEPS)
+        )
+        # Use a small max_steps so the test doesn't loop 20 times
+        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=5)
+        result = evaluator.run()
+        assert result.max_steps_hit, "expected max_steps_hit=True for EXCEED_MAX_STEPS"
+        assert result.metrics.get("max_step_exceeded_count", 0) >= 1, \
+            "expected max_step_exceeded_count >= 1"
+    finally:
+        ws.cleanup()
`
