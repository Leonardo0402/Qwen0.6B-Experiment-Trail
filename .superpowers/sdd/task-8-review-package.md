# Task 8 Review Package

**BASE:** 2f244cd
**HEAD:** 7069daf

## Commit list

`
7069daf feat(p4-1): Phase E — GPU smoke tests (base + repair-lora)
`

## Diffstat

`
 pyproject.toml                         |  3 ++
 tests/test_agent_model_provider_gpu.py | 78 ++++++++++++++++++++++++++++++++++
 2 files changed, 81 insertions(+)
`

## Full diff (with 10 lines context)

`diff
diff --git a/pyproject.toml b/pyproject.toml
index c286c63..bd246a0 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,17 +1,20 @@
 [project]
 name = "qwen3-code-lab"
 version = "0.1.0"
 description = "Qwen3-0.6B Python Code Recovery Lab — LoRA + curriculum fine-tuning on RTX 3050 4GB"
 requires-python = ">=3.10,<3.11"
 
 [tool.pytest.ini_options]
 testpaths = ["tests"]
 addopts = "-q"
+markers = [
+    "gpu: tests requiring a physical GPU (skipped in CI, run manually on RTX 3050 before PR merge)",
+]
 
 [tool.ruff]
 line-length = 100
 target-version = "py310"
 
 [tool.mypy]
 python_version = "3.10"
 ignore_missing_imports = true
diff --git a/tests/test_agent_model_provider_gpu.py b/tests/test_agent_model_provider_gpu.py
new file mode 100644
index 0000000..946fea7
--- /dev/null
+++ b/tests/test_agent_model_provider_gpu.py
@@ -0,0 +1,78 @@
+# tests/test_agent_model_provider_gpu.py
+"""GPU smoke tests for ModelActionProvider.
+
+Marked @pytest.mark.gpu — skipped in CI (CI uses -m "not gpu").
+Run manually on the RTX 3050 before PR merge.
+"""
+import pytest
+from pathlib import Path
+
+pytestmark = pytest.mark.gpu
+
+_ROOT = Path(__file__).resolve().parent.parent
+_TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
+
+
+def test_model_provider_smoke_base():
+    """Load base Qwen3-0.6B, run 1 micro-task, assert:
+    - no runtime crash
+    - forbidden_action_count == 0
+    - at least 1 schema-valid action OR structured diagnostics recorded
+
+    Note: invalid_action_count > 0 is acceptable (model may produce invalid
+    JSON → SentinelAction → invalid_action_count). forbidden_action_count == 0
+    is still required (no unknown action types should slip through).
+    """
+    from src.agent_model_provider import ModelActionProvider, SentinelAction
+    from src.agent_evaluator import AgentEvaluator, AgentState
+    from src.agent_workspace import MicroTaskWorkspace
+    import os
+    os.environ.setdefault("P4_ALLOW_NETWORK", "0")
+
+    task_dir = _TASKS_DIR / "task_001"
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        provider = ModelActionProvider(
+            model_path="models/Qwen3-0.6B",
+            adapter_path=None,
+        )
+        evaluator = AgentEvaluator(ws, provider, "task_001", max_steps=12)
+        result = evaluator.run()
+
+        # Minimum bar (user decision #3)
+        # forbidden_action_count must be 0 — unknown action types are not
+        # acceptable. invalid_action_count > 0 is OK (model may emit bad JSON).
+        assert result.metrics.get("forbidden_action_count", 0) == 0, \
+            f"forbidden_action_count must be 0, got {result.metrics.get('forbidden_action_count')}"
+        # At least one diagnostic recorded (even if all invalid)
+        assert len(provider.diagnostics) > 0, "no diagnostics recorded"
+    finally:
+        ws.cleanup()
+
+
+def test_model_provider_smoke_repair_lora():
+    """Load Qwen3-0.6B + Repair-Limited LoRA, run 1 micro-task, same bar.
+
+    Note: invalid_action_count > 0 is acceptable (model may produce invalid
+    JSON). forbidden_action_count == 0 is still required.
+    """
+    from src.agent_model_provider import ModelActionProvider
+    from src.agent_evaluator import AgentEvaluator
+    from src.agent_workspace import MicroTaskWorkspace
+    import os
+    os.environ.setdefault("P4_ALLOW_NETWORK", "0")
+
+    task_dir = _TASKS_DIR / "task_001"
+    ws = MicroTaskWorkspace.from_task(task_dir)
+    try:
+        provider = ModelActionProvider(
+            model_path="models/Qwen3-0.6B",
+            adapter_path="adapters/p3/repair-limited",
+        )
+        evaluator = AgentEvaluator(ws, provider, "task_001", max_steps=12)
+        result = evaluator.run()
+
+        assert result.metrics.get("forbidden_action_count", 0) == 0
+        assert len(provider.diagnostics) > 0
+    finally:
+        ws.cleanup()
`
