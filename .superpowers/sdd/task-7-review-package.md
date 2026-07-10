# Task 7 Review Package

**BASE:** 37c4ef2
**HEAD:** 2f244cd

## Commit list

`
2f244cd feat(p4-1): Phase E — ModelActionProvider mocked generation + diagnostics tests
`

## Diffstat

`
 src/agent_model_provider.py        |  6 ++-
 tests/test_agent_model_provider.py | 76 +++++++++++++++++++++++++++++++++++++-
 2 files changed, 79 insertions(+), 3 deletions(-)
`

## Full diff (with 10 lines context)

`diff
diff --git a/src/agent_model_provider.py b/src/agent_model_provider.py
index 181fba4..323de54 100644
--- a/src/agent_model_provider.py
+++ b/src/agent_model_provider.py
@@ -5,25 +5,27 @@ This module does NOT load the model (that's in the GPU tests / collection
 script). It provides the building blocks that the ModelActionProvider class
 composes.
 """
 from __future__ import annotations
 
 import json
 import re
 import time
 from typing import Any
 
-from pydantic import BaseModel, Field
+from pydantic import BaseModel, Field, TypeAdapter
 
 from src.agent_actions import Action, SafetyFlags
 from src.agent_evaluator import AgentState, ActionProvider
 
+_ACTION_ADAPTER = TypeAdapter(Action)
+
 
 class ModelStepDiagnostics(BaseModel):
     """Diagnostics recorded for each model.generate() call."""
     raw_output: str
     json_parse_ok: bool
     schema_valid: bool
     safety_valid: bool
     action_type_valid: bool
     arguments_valid: bool
     repair_attempted: bool
@@ -234,13 +236,13 @@ class ModelActionProvider(ActionProvider):
         self._diagnostics.clear()
 
     @property
     def diagnostics(self) -> list[ModelStepDiagnostics]:
         return list(self._diagnostics)
 
 
 def _validate_action(data: dict) -> Action | None:
     """Validate a dict against the Action union. Returns the Action or None."""
     try:
-        return Action.model_validate(data)
+        return _ACTION_ADAPTER.validate_python(data)
     except Exception:
         return None
diff --git a/tests/test_agent_model_provider.py b/tests/test_agent_model_provider.py
index 6c54e2b..9c214dc 100644
--- a/tests/test_agent_model_provider.py
+++ b/tests/test_agent_model_provider.py
@@ -1,14 +1,15 @@
 import pytest
+from unittest.mock import patch, MagicMock
 from src.agent_model_provider import (
     build_prompt, extract_json, repair_json, ModelStepDiagnostics,
-    SentinelAction,
+    ModelActionProvider, SentinelAction,
 )
 from src.agent_evaluator import AgentState
 from src.agent_state import AgentMemory
 
 
 def test_build_prompt_produces_nonempty_string():
     state = AgentState(
         memory=AgentMemory(),
         step_count=0,
         task_id="task_001",
@@ -49,10 +50,83 @@ def test_repair_json_does_not_choose_action_type():
     """Repair must NOT substitute a valid action_type for an invalid one."""
     raw = '{"action_type": "???"}'
     repaired = repair_json(raw)
     assert '"???"' in repaired, "repair must not alter action_type value"
 
 
 def test_sentinel_action_marks_invalid():
     sa = SentinelAction(reason="json parse failed")
     assert sa.is_invalid
     assert sa.reason == "json parse failed"
+
+
+def test_model_provider_extracts_valid_json_mocked():
+    """Mock _generate to return valid Action JSON → provider returns a
+    valid Action, diagnostics record schema_valid=True."""
+    provider = ModelActionProvider.__new__(ModelActionProvider)
+    provider._model = MagicMock()  # skip _load_model
+    provider._tokenizer = MagicMock()
+    provider._max_new_tokens = 512
+    provider._diagnostics = []
+    provider._model_path = "fake"
+    provider._adapter_path = None
+
+    valid_json = '{"action_type": "list_files", "action_id": "a1", "reason_short": "list", "expected_observation": "files", "safety_flags": {"modifies_workspace": false, "executes_code": false, "network_required": false, "reads_sensitive_path": false, "is_terminal": false}}'
+
+    with patch.object(provider, "_generate", return_value=valid_json):
+        state = AgentState(
+            memory=AgentMemory(), step_count=0,
+            task_id="t1", workspace_id="w1",
+        )
+        action = provider.next_action(state)
+
+    assert not isinstance(action, SentinelAction), "expected valid Action, got SentinelAction"
+    assert provider.diagnostics[0].json_parse_ok
+    assert provider.diagnostics[0].schema_valid
+
+
+def test_model_provider_records_diagnostics_on_invalid_json():
+    """Mock _generate to return garbage → provider returns SentinelAction,
+    diagnostics record json_parse_ok=False."""
+    provider = ModelActionProvider.__new__(ModelActionProvider)
+    provider._model = MagicMock()
+    provider._tokenizer = MagicMock()
+    provider._max_new_tokens = 512
+    provider._diagnostics = []
+    provider._model_path = "fake"
+    provider._adapter_path = None
+
+    with patch.object(provider, "_generate", return_value="I cannot do that."):
+        state = AgentState(
+            memory=AgentMemory(), step_count=0,
+            task_id="t1", workspace_id="w1",
+        )
+        action = provider.next_action(state)
+
+    assert isinstance(action, SentinelAction)
+    assert not provider.diagnostics[0].json_parse_ok
+    assert not provider.diagnostics[0].schema_valid
+
+
+def test_model_provider_repair_strips_fences_then_validates():
+    """Mock _generate to return fenced JSON → repair strips fences,
+    validation succeeds."""
+    provider = ModelActionProvider.__new__(ModelActionProvider)
+    provider._model = MagicMock()
+    provider._tokenizer = MagicMock()
+    provider._max_new_tokens = 512
+    provider._diagnostics = []
+    provider._model_path = "fake"
+    provider._adapter_path = None
+
+    fenced = '```json\n{"action_type": "list_files", "action_id": "a1", "reason_short": "list", "expected_observation": "files", "safety_flags": {"modifies_workspace": false, "executes_code": false, "network_required": false, "reads_sensitive_path": false, "is_terminal": false}}\n```'
+
+    with patch.object(provider, "_generate", return_value=fenced):
+        state = AgentState(
+            memory=AgentMemory(), step_count=0,
+            task_id="t1", workspace_id="w1",
+        )
+        action = provider.next_action(state)
+
+    # extract_json should handle fences, but if not, repair kicks in
+    assert not isinstance(action, SentinelAction), \
+        f"expected valid Action after repair, got SentinelAction; diag: {provider.diagnostics[0]}"
`
