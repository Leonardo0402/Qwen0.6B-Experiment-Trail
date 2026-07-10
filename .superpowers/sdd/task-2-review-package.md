## Commit List
f3a9a7e feat(p4-1): Phase B — TEST_PASS replay-authoritative + finish_claim_mismatch

## Diffstat
`
 src/agent_evaluator.py        | 10 +++++-
 tests/test_agent_evaluator.py | 72 +++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 81 insertions(+), 1 deletion(-)
`

## Full Diff
`diff
diff --git a/src/agent_evaluator.py b/src/agent_evaluator.py
index a5c77b5..b92ef62 100644
--- a/src/agent_evaluator.py
+++ b/src/agent_evaluator.py
@@ -42,20 +42,21 @@ class AgentState(BaseModel):
 
 class EvalResult(BaseModel):
     """Result of evaluating one task."""
     task_id: str
     trajectory_id: str
     steps_executed: int
     success: bool              # success_criterion was met
     metrics: dict[str, float | int]
     errors: list[str] = Field(default_factory=list)
     max_steps_hit: bool = False
+    finish_claim_mismatch: bool = False
 
 
 class CorruptionType(str, Enum):
     WRONG_ACTION_TYPE = "wrong_action_type"        # replace action_type with a different one
     INVALID_PATH = "invalid_path"                   # use a forbidden path (e.g., ../etc/passwd)
     WRONG_PATCH = "wrong_patch"                     # use old_text that doesn't exist
     SKIP_TESTS_BEFORE_FINISH = "skip_tests_before_finish"  # finish without prior run_tests
     EXCEED_MAX_STEPS = "exceed_max_steps"           # provider never returns finish
 
 
@@ -227,20 +228,21 @@ class AgentEvaluator:
         self._task_id = task_id
         self._max_steps = max_steps
 
     def run(self) -> EvalResult:
         """Execute actions until finish or max_steps."""
         self._provider.reset()
         errors: list[str] = []
         steps_executed = 0
         success = False
         max_steps_hit = False
+        finish_claim_mismatch = False
 
         # Metric counters
         total_actions = 0
         valid_actions = 0
         forbidden_count = 0
         total_patches = 0
         successful_patches = 0
         total_tests = 0
         passed_tests = 0
         tool_errors = 0
@@ -326,79 +328,85 @@ class AgentEvaluator:
                     pass
                 elif at == "write_memory":
                     # Stateless — update state memory
                     state.memory = action.arguments.memory
                 elif at == "finish":
                     fa = action.arguments
                     if not ran_tests:
                         finish_without_tests += 1
                     # Check success criterion
                     if fa.success_criterion == TaskSuccessCriterion.TEST_PASS:
-                        success = fa.tests_passed
+                        replay_passed = passed_tests > 0
+                        success = replay_passed
+                        finish_claim_mismatch = (fa.tests_passed != replay_passed)
                     elif fa.success_criterion == TaskSuccessCriterion.IDENTIFY_BUG:
                         success = fa.identification_verified
                     elif fa.success_criterion == TaskSuccessCriterion.PATCH_APPLIED:
                         success = successful_patches > 0
                     steps_executed = step + 1
                     # Compute metrics and return
                     return self._make_result(
                         steps_executed, success, errors,
                         total_actions, valid_actions, forbidden_count,
                         total_patches, successful_patches,
                         total_tests, passed_tests,
                         tool_errors, total_tools,
                         finish_without_tests, max_steps_hit=False,
+                        finish_claim_mismatch=finish_claim_mismatch,
                     )
             except Exception as e:
                 tool_errors += 1
                 errors.append(f"step {step}: tool error: {e}")
 
             state.step_count = step + 1
             steps_executed = step + 1
 
         # Hit max_steps without finish
         max_steps_hit = True
         return self._make_result(
             steps_executed, success, errors,
             total_actions, valid_actions, forbidden_count,
             total_patches, successful_patches,
             total_tests, passed_tests,
             tool_errors, total_tools,
             finish_without_tests, max_steps_hit=True,
+            finish_claim_mismatch=finish_claim_mismatch,
         )
 
     def _make_result(
         self,
         steps_executed: int,
         success: bool,
         errors: list[str],
         total_actions: int,
         valid_actions: int,
         forbidden_count: int,
         total_patches: int,
         successful_patches: int,
         total_tests: int,
         passed_tests: int,
         tool_errors: int,
         total_tools: int,
         finish_without_tests: int,
         max_steps_hit: bool,
+        finish_claim_mismatch: bool = False,
     ) -> EvalResult:
         metrics = {
             "task_success_rate": 1.0 if success else 0.0,
             "action_validity_rate": valid_actions / total_actions if total_actions else 0.0,
             "tool_error_rate": tool_errors / total_tools if total_tools else 0.0,
             "patch_success_rate": successful_patches / total_patches if total_patches else 0.0,
             "tests_pass_rate": passed_tests / total_tests if total_tests else 0.0,
             "forbidden_action_count": forbidden_count,
             "max_step_exceeded_count": 1 if max_steps_hit else 0,
             "finish_without_tests_count": finish_without_tests,
         }
         return EvalResult(
             task_id=self._task_id,
             trajectory_id="",  # filled by caller
             steps_executed=steps_executed,
             success=success,
             metrics=metrics,
             errors=errors,
             max_steps_hit=max_steps_hit,
+            finish_claim_mismatch=finish_claim_mismatch,
         )
diff --git a/tests/test_agent_evaluator.py b/tests/test_agent_evaluator.py
index 56f9813..e6c7517 100644
--- a/tests/test_agent_evaluator.py
+++ b/tests/test_agent_evaluator.py
@@ -293,10 +293,82 @@ def test_tool_error_rate_counts_failed_attempts(monkeypatch):
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
+
+
+# --- Task 2: TEST_PASS replay-authoritative + finish_claim_mismatch ---
+
+def test_test_pass_success_uses_replay_not_claim(monkeypatch):
+    """finish claims tests_passed=True but replay has 0 passed_tests
+    → success=False, finish_claim_mismatch=True."""
+    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
+    traj = _load_first_success_trajectory()
+    task_dir = TASKS_DIR / traj.task_id
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        # Strip all run_tests actions so replay passed_tests=0,
+        # but finish still claims tests_passed=True (false claim)
+        actions = [s.action for s in traj.steps
+                   if s.action.action_type != "run_tests"]
+        provider = _FixedProvider(actions)
+        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
+        result = evaluator.run()
+        assert not result.success, \
+            "expected success=False (replay has 0 passed, claim says True)"
+        assert result.finish_claim_mismatch, \
+            "expected finish_claim_mismatch=True (claim≠replay)"
+    finally:
+        ws.cleanup()
+
+
+def test_test_pass_mismatch_claimed_fail_actual_pass(monkeypatch):
+    """finish claims tests_passed=False but replay passed_tests>0
+    → success=True, finish_claim_mismatch=True."""
+    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
+    traj = _load_first_success_trajectory()
+    task_dir = TASKS_DIR / traj.task_id
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        actions = [s.action for s in traj.steps]
+        original_finish = actions[-1]
+        # Keep run_tests (so replay passes), but finish claims tests_passed=False
+        modified_finish = original_finish.model_copy(update={
+            "arguments": original_finish.arguments.model_copy(update={
+                "tests_passed": False,
+            }),
+        })
+        actions[-1] = modified_finish
+        provider = _FixedProvider(actions)
+        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
+        result = evaluator.run()
+        assert result.success, \
+            "expected success=True (replay passed_tests>0 is authoritative)"
+        assert result.finish_claim_mismatch, \
+            "expected finish_claim_mismatch=True (claim says fail, replay says pass)"
+    finally:
+        ws.cleanup()
+
+
+def test_test_pass_no_mismatch_when_claim_matches_replay(monkeypatch):
+    """finish claims tests_passed=True AND replay passed_tests>0
+    → success=True, finish_claim_mismatch=False."""
+    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
+    traj = _load_first_success_trajectory()
+    task_dir = TASKS_DIR / traj.task_id
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        actions = [s.action for s in traj.steps]
+        provider = _FixedProvider(actions)
+        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
+        result = evaluator.run()
+        assert result.success
+        assert not result.finish_claim_mismatch, \
+            "expected finish_claim_mismatch=False (claim matches replay)"
+    finally:
+        ws.cleanup()
`
