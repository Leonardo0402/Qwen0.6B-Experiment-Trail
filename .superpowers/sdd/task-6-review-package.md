# Task 6 Review Package

**BASE:** 439c097
**HEAD:** 37c4ef2

## Commit list

`
37c4ef2 feat(p4-1): Phase E — ModelActionProvider prompt builder + JSON extraction + repair + diagnostics + SentinelAction dispatch
`

## Diffstat

`
 src/agent_evaluator.py             |  15 ++-
 src/agent_model_provider.py        | 246 +++++++++++++++++++++++++++++++++++++
 tests/test_agent_evaluator.py      |  28 ++++-
 tests/test_agent_model_provider.py |  58 +++++++++
 4 files changed, 344 insertions(+), 3 deletions(-)
`

## Full diff (with 10 lines context)

`diff
diff --git a/src/agent_evaluator.py b/src/agent_evaluator.py
index ff6dd64..846bdf5 100644
--- a/src/agent_evaluator.py
+++ b/src/agent_evaluator.py
@@ -253,39 +253,46 @@ class AgentEvaluator:
         total_actions = 0
         valid_actions = 0
         forbidden_count = 0
         total_patches = 0
         successful_patches = 0
         total_tests = 0
         passed_tests = 0
         tool_errors = 0
         total_tools = 0
         finish_without_tests = 0
+        invalid_action_count = 0
         ran_tests = False
 
         state = AgentState(
             memory=AgentMemory(),
             step_count=0,
             task_id=self._task_id,
             # MicroTaskWorkspace has no .workspace_id attribute; use the
             # temp workspace root path as a unique identifier.
             workspace_id=str(self._ws.workspace_root),
         )
 
         for step in range(self._max_steps):
             try:
                 action = self._provider.next_action(state)
             except StopIteration:
                 break
 
             total_actions += 1
 
+            # Handle SentinelAction (from ModelActionProvider) — invalid, not forbidden
+            if hasattr(action, 'is_invalid') and getattr(action, 'is_invalid', False):
+                invalid_action_count += 1
+                errors.append(f"step {step}: invalid action (sentinel: {getattr(action, 'reason', 'unknown')})")
+                continue
+
             # Validate action (schema + safety)
             try:
                 # Re-validate by constructing a new instance of the same type
                 action.__class__.model_validate(action.model_dump())
                 valid_actions += 1
             except Exception as e:
                 errors.append(f"step {step}: invalid action: {e}")
                 continue
 
             # Check safety flags
@@ -361,21 +368,22 @@ class AgentEvaluator:
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
-                        finish_without_tests, max_steps_hit=False,
+                        finish_without_tests, invalid_action_count,
+                        max_steps_hit=False,
                         finish_claim_mismatch=finish_claim_mismatch,
                     )
                 else:
                     # Defensive guard: any action_type not in the 11-action
                     # allowlist is recorded as forbidden. (Unreachable via
                     # valid Pydantic actions — every Literal action_type is
                     # one of the 11 above.)
                     forbidden_count += 1
                     errors.append(
                         f"step {step}: unknown action type (not in 11-action "
@@ -389,51 +397,54 @@ class AgentEvaluator:
             steps_executed = step + 1
 
         # Hit max_steps without finish
         max_steps_hit = True
         return self._make_result(
             steps_executed, success, errors,
             total_actions, valid_actions, forbidden_count,
             total_patches, successful_patches,
             total_tests, passed_tests,
             tool_errors, total_tools,
-            finish_without_tests, max_steps_hit=True,
+            finish_without_tests, invalid_action_count,
+            max_steps_hit=True,
             finish_claim_mismatch=finish_claim_mismatch,
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
+        invalid_action_count: int,
         max_steps_hit: bool,
         finish_claim_mismatch: bool = False,
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
+            "invalid_action_count": invalid_action_count,
         }
         return EvalResult(
             task_id=self._task_id,
             trajectory_id="",  # filled by caller
             steps_executed=steps_executed,
             success=success,
             metrics=metrics,
             errors=errors,
             max_steps_hit=max_steps_hit,
             finish_claim_mismatch=finish_claim_mismatch,
diff --git a/src/agent_model_provider.py b/src/agent_model_provider.py
new file mode 100644
index 0000000..181fba4
--- /dev/null
+++ b/src/agent_model_provider.py
@@ -0,0 +1,246 @@
+"""P4.1 Phase E — ModelActionProvider: prompt builder, JSON extraction,
+format-only repair, and structured diagnostics.
+
+This module does NOT load the model (that's in the GPU tests / collection
+script). It provides the building blocks that the ModelActionProvider class
+composes.
+"""
+from __future__ import annotations
+
+import json
+import re
+import time
+from typing import Any
+
+from pydantic import BaseModel, Field
+
+from src.agent_actions import Action, SafetyFlags
+from src.agent_evaluator import AgentState, ActionProvider
+
+
+class ModelStepDiagnostics(BaseModel):
+    """Diagnostics recorded for each model.generate() call."""
+    raw_output: str
+    json_parse_ok: bool
+    schema_valid: bool
+    safety_valid: bool
+    action_type_valid: bool
+    arguments_valid: bool
+    repair_attempted: bool
+    repair_success: bool
+    latency_ms: int
+
+
+class SentinelAction(BaseModel):
+    """Marker returned when the model output cannot be parsed into a valid
+    Action. Not a real Action — the evaluator records it as action_invalid."""
+    is_invalid: bool = True
+    reason: str = ""
+
+    @property
+    def action_type(self) -> str:
+        return "invalid"
+
+    @property
+    def safety_flags(self) -> SafetyFlags:
+        return SafetyFlags(
+            modifies_workspace=False,
+            executes_code=False,
+            network_required=False,
+            reads_sensitive_path=False,
+            is_terminal=False,
+        )
+
+
+def build_prompt(
+    state: AgentState,
+    task_description: str,
+    last_observation: dict | None,
+) -> str:
+    """Build the prompt for model.generate()."""
+    lines = [
+        f"Task ID: {state.task_id}",
+        f"Step: {state.step_count}",
+        f"Task: {task_description}",
+    ]
+    if state.memory.notes:
+        lines.append(f"Notes: {state.memory.notes}")
+    if state.memory.hypothesis:
+        lines.append(f"Hypothesis: {state.memory.hypothesis}")
+    if last_observation:
+        lines.append(f"Last observation: {last_observation}")
+    lines.append(
+        "Choose ONE action from: list_files, read_file, search_text, "
+        "inspect_task, propose_patch, apply_patch, rollback_patch, run_tests, "
+        "inspect_error, write_memory, finish."
+    )
+    lines.append("Respond with a single JSON object.")
+    return "\n".join(lines)
+
+
+_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
+_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)
+
+
+def extract_json(raw: str) -> str | None:
+    """Extract the first JSON object from raw model output. Returns the
+    JSON string or None if no JSON found."""
+    m = _JSON_FENCE_RE.search(raw)
+    if m:
+        return m.group(1)
+    m = _BARE_JSON_RE.search(raw)
+    if m:
+        return m.group(1)
+    return None
+
+
+def repair_json(raw: str) -> str:
+    """Format-only repair. NEVER alters action semantics (action_type,
+    arguments values). Only fixes: markdown fences, trailing commas,
+    unbalanced braces (best-effort)."""
+    result = raw
+    # Strip markdown fences
+    result = re.sub(r"```(?:json)?\s*", "", result)
+    result = result.replace("```", "")
+    # Remove trailing commas before } or ]
+    result = re.sub(r",\s*([}\]])", r"\1", result)
+    # Best-effort brace balancing (append missing closing braces)
+    opens = result.count("{")
+    closes = result.count("}")
+    if opens > closes:
+        result = result + ("}" * (opens - closes))
+    return result.strip()
+
+
+class ModelActionProvider(ActionProvider):
+    """Loads Qwen3-0.6B and generates actions. GPU required.
+
+    The actual model loading happens in __init__ (lazy torch import).
+    Non-GPU tests mock the _generate method.
+    """
+
+    def __init__(
+        self,
+        model_path: str = "models/Qwen3-0.6B",
+        adapter_path: str | None = None,
+        max_new_tokens: int = 512,
+    ):
+        self._model_path = model_path
+        self._adapter_path = adapter_path
+        self._max_new_tokens = max_new_tokens
+        self._model = None
+        self._tokenizer = None
+        self._diagnostics: list[ModelStepDiagnostics] = []
+
+    def _load_model(self):
+        """Lazy-load the model. Called on first next_action."""
+        import torch
+        from transformers import AutoModelForCausalLM, AutoTokenizer
+
+        self._tokenizer = AutoTokenizer.from_pretrained(
+            self._model_path, trust_remote_code=True
+        )
+        if self._tokenizer.pad_token is None:
+            self._tokenizer.pad_token = self._tokenizer.eos_token
+
+        self._model = AutoModelForCausalLM.from_pretrained(
+            self._model_path,
+            dtype=torch.float16,
+            device_map={"": "cuda:0"},
+            trust_remote_code=True,
+        )
+        if self._adapter_path:
+            from peft import PeftModel
+            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
+            self._model = self._model.merge_and_unload()
+        self._model.config.use_cache = True
+        self._model.eval()
+
+    def _generate(self, prompt: str) -> str:
+        """Generate raw text from the model. Override in tests."""
+        import torch
+        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda:0")
+        with torch.no_grad():
+            outputs = self._model.generate(
+                **inputs,
+                max_new_tokens=self._max_new_tokens,
+                temperature=0.0,
+                do_sample=False,
+                pad_token_id=self._tokenizer.eos_token_id,
+            )
+        return self._tokenizer.decode(
+            outputs[0][inputs["input_ids"].shape[1]:],
+            skip_special_tokens=True,
+        )
+
+    def next_action(self, state: AgentState) -> Action | SentinelAction:
+        if self._model is None:
+            self._load_model()
+        prompt = build_prompt(state, task_description="", last_observation=None)
+        t0 = time.monotonic()
+        raw_output = self._generate(prompt)
+        latency_ms = int((time.monotonic() - t0) * 1000)
+
+        json_str = extract_json(raw_output)
+        diag = ModelStepDiagnostics(
+            raw_output=raw_output,
+            json_parse_ok=json_str is not None,
+            schema_valid=False,
+            safety_valid=False,
+            action_type_valid=False,
+            arguments_valid=False,
+            repair_attempted=False,
+            repair_success=False,
+            latency_ms=latency_ms,
+        )
+
+        if json_str is None:
+            self._diagnostics.append(diag)
+            return SentinelAction(reason="json parse failed")
+
+        # Try direct validation
+        try:
+            data = json.loads(json_str)
+            action = _validate_action(data)
+            if action is not None:
+                diag.schema_valid = True
+                diag.safety_valid = True
+                diag.action_type_valid = True
+                diag.arguments_valid = True
+                self._diagnostics.append(diag)
+                return action
+        except (json.JSONDecodeError, Exception):
+            pass
+
+        # Attempt repair
+        diag.repair_attempted = True
+        repaired = repair_json(json_str)
+        try:
+            data = json.loads(repaired)
+            action = _validate_action(data)
+            if action is not None:
+                diag.repair_success = True
+                diag.schema_valid = True
+                diag.safety_valid = True
+                diag.action_type_valid = True
+                diag.arguments_valid = True
+        except (json.JSONDecodeError, Exception):
+            pass
+
+        self._diagnostics.append(diag)
+        return SentinelAction(reason="schema validation failed after repair")
+
+    def reset(self) -> None:
+        self._diagnostics.clear()
+
+    @property
+    def diagnostics(self) -> list[ModelStepDiagnostics]:
+        return list(self._diagnostics)
+
+
+def _validate_action(data: dict) -> Action | None:
+    """Validate a dict against the Action union. Returns the Action or None."""
+    try:
+        return Action.model_validate(data)
+    except Exception:
+        return None
diff --git a/tests/test_agent_evaluator.py b/tests/test_agent_evaluator.py
index 26a235f..8ccd386 100644
--- a/tests/test_agent_evaluator.py
+++ b/tests/test_agent_evaluator.py
@@ -95,42 +95,44 @@ def test_corrupted_injection(monkeypatch):
         evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
         result = evaluator.run()
         # patch_success_rate should be < 1.0 (the corrupted patch fails)
         assert result.metrics["patch_success_rate"] < 1.0, \
             f"expected patch_success_rate < 1.0, got {result.metrics['patch_success_rate']}"
     finally:
         ws.cleanup()
 
 
 def test_all_metrics_present():
-    """EvalResult has all 8 metrics."""
+    """EvalResult has all 9 metrics."""
     traj = _load_first_success_trajectory()
     provider = ScriptedActionProvider(traj)
     # Don't need to run — just check the metric keys are defined
     expected_keys = {
         "task_success_rate", "action_validity_rate", "tool_error_rate",
         "patch_success_rate", "tests_pass_rate", "forbidden_action_count",
         "max_step_exceeded_count", "finish_without_tests_count",
+        "invalid_action_count",
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
+            "invalid_action_count": 0,
         },
     )
     assert set(result.metrics.keys()) == expected_keys
 
 
 class _FixedProvider(ActionProvider):
     """Test provider that yields a fixed list of actions."""
     def __init__(self, actions):
         self._actions = actions
         self._index = 0
@@ -561,10 +563,34 @@ def test_corruption_exceed_max_steps(monkeypatch):
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
+
+
+# --- Task 6: SentinelAction counted as invalid, not forbidden ---
+
+def test_sentinel_action_counted_as_invalid_not_forbidden(monkeypatch):
+    """SentinelAction must increment invalid_action_count, not forbidden_action_count."""
+    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
+    traj = _load_first_success_trajectory()
+    task_dir = TASKS_DIR / traj.task_id
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        from src.agent_model_provider import SentinelAction
+        # Build a provider that returns SentinelAction then finish
+        sentinel = SentinelAction(reason="test invalid")
+        finish = _make_finish(tests_passed=True)
+        provider = _FixedProvider([sentinel, finish])
+        evaluator = AgentEvaluator(ws, provider, traj.task_id, max_steps=20)
+        result = evaluator.run()
+        assert result.metrics.get("invalid_action_count", 0) >= 1, \
+            "SentinelAction must increment invalid_action_count"
+        assert result.metrics.get("forbidden_action_count", 0) == 0, \
+            "SentinelAction must NOT increment forbidden_action_count"
+    finally:
+        ws.cleanup()
diff --git a/tests/test_agent_model_provider.py b/tests/test_agent_model_provider.py
new file mode 100644
index 0000000..6c54e2b
--- /dev/null
+++ b/tests/test_agent_model_provider.py
@@ -0,0 +1,58 @@
+import pytest
+from src.agent_model_provider import (
+    build_prompt, extract_json, repair_json, ModelStepDiagnostics,
+    SentinelAction,
+)
+from src.agent_evaluator import AgentState
+from src.agent_state import AgentMemory
+
+
+def test_build_prompt_produces_nonempty_string():
+    state = AgentState(
+        memory=AgentMemory(),
+        step_count=0,
+        task_id="task_001",
+        workspace_id="test_ws",
+    )
+    prompt = build_prompt(state, task_description="Fix the bug", last_observation=None)
+    assert isinstance(prompt, str)
+    assert len(prompt) > 0
+    assert "task_001" in prompt or "Fix the bug" in prompt
+
+
+def test_extract_json_finds_first_json_block():
+    raw = 'Here is the action:\n```json\n{"action_type": "list_files"}\n```\nDone.'
+    result = extract_json(raw)
+    assert result == '{"action_type": "list_files"}'
+
+
+def test_extract_json_returns_none_on_no_json():
+    raw = "I cannot produce an action."
+    assert extract_json(raw) is None
+
+
+def test_repair_json_strips_markdown_fences():
+    raw = '```json\n{"action_type": "read_file"}\n```'
+    repaired = repair_json(raw)
+    assert "```" not in repaired
+    assert '"action_type"' in repaired
+
+
+def test_repair_json_removes_trailing_commas():
+    raw = '{"action_type": "read_file", "arguments": {"path": "x.py",},}'
+    repaired = repair_json(raw)
+    assert ",}" not in repaired
+    assert ",," not in repaired
+
+
+def test_repair_json_does_not_choose_action_type():
+    """Repair must NOT substitute a valid action_type for an invalid one."""
+    raw = '{"action_type": "???"}'
+    repaired = repair_json(raw)
+    assert '"???"' in repaired, "repair must not alter action_type value"
+
+
+def test_sentinel_action_marks_invalid():
+    sa = SentinelAction(reason="json parse failed")
+    assert sa.is_invalid
+    assert sa.reason == "json parse failed"
`
