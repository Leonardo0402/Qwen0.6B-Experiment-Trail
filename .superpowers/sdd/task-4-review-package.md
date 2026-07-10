# Task 4 Review Package

**BASE:** 41a0d05
**HEAD:** 50ec3af

## Commit list

`
50ec3af feat(p4-1): Phase C — inspect_error returns stdout+stderr capped 8KB
`

## Diffstat

`
 src/agent_tools.py        |  7 +++----
 tests/test_agent_tools.py | 39 +++++++++++++++++++++++++++++++++++++++
 2 files changed, 42 insertions(+), 4 deletions(-)
`

## Full diff (with 10 lines context)

`diff
diff --git a/src/agent_tools.py b/src/agent_tools.py
index 6b24620..1027779 100644
--- a/src/agent_tools.py
+++ b/src/agent_tools.py
@@ -601,24 +601,23 @@ def tool_inspect_error(
     last_patch_observation: "PatchObservation | None",
 ) -> ErrorObservation:
     """Return the error content from the last test or patch observation.
 
     Stateless: the caller passes the last observations; the tool layer
     does not track history.
     """
     if error_source == "last_test":
         if last_test_observation is None:
             raise ValueError("no prior run_tests observation")
-        return ErrorObservation(
-            source="last_test",
-            content=last_test_observation.stderr,
-        )
+        raw = last_test_observation.stdout + "\n" + last_test_observation.stderr
+        capped = raw[:8192]
+        return ErrorObservation(source="last_test", content=capped)
     elif error_source == "last_patch":
         if last_patch_observation is None:
             raise ValueError("no prior patch observation")
         content = last_patch_observation.error or ""
         return ErrorObservation(source="last_patch", content=content)
     else:
         raise ValueError(
             f"error_source must be 'last_test' or 'last_patch': {error_source}"
         )
 
diff --git a/tests/test_agent_tools.py b/tests/test_agent_tools.py
index 01a3512..8b5173d 100644
--- a/tests/test_agent_tools.py
+++ b/tests/test_agent_tools.py
@@ -511,10 +511,49 @@ def test_finish():
     obs = tool_finish(
         success_criterion=TaskSuccessCriterion.TEST_PASS,
         tests_passed=True,
         identification_verified=False,
         summary="Fixed the add() function by changing subtraction to addition.",
     )
     assert obs.success_criterion == TaskSuccessCriterion.TEST_PASS
     assert obs.tests_passed is True
     assert obs.identification_verified is False
     assert "add()" in obs.summary
+
+
+# --- Task 4: inspect_error returns stdout+stderr capped 8KB ---
+
+def test_inspect_error_returns_stdout_on_test_failure():
+    """Failed run_tests writes traceback to stdout; inspect_error must
+    surface it, not return empty content."""
+    test_obs = TestObservation(
+        passed=False, num_collected=1, num_passed=0, num_failed=1,
+        timed_out=False,
+        stdout="AssertionError: expected 5 but got 4",
+        stderr="",
+        duration_s=0.1,
+    )
+    result = tool_inspect_error(
+        error_source="last_test",
+        last_test_observation=test_obs,
+        last_patch_observation=None,
+    )
+    assert result.content != "", "inspect_error returned empty content for stdout-only failure"
+    assert "AssertionError" in result.content
+
+
+def test_inspect_error_caps_at_8kb():
+    """stdout+stderr > 8KB -> content truncated to exactly 8192 chars."""
+    big_stdout = "x" * 10000
+    test_obs = TestObservation(
+        passed=False, num_collected=1, num_passed=0, num_failed=1,
+        timed_out=False,
+        stdout=big_stdout, stderr="",
+        duration_s=0.1,
+    )
+    result = tool_inspect_error(
+        error_source="last_test",
+        last_test_observation=test_obs,
+        last_patch_observation=None,
+    )
+    assert len(result.content) == 8192, \
+        f"expected 8192 chars (8KB cap), got {len(result.content)}"
`
