## Commit Log (e70d339..HEAD)
824a5c2 feat(p4-1): Phase A — P4.0 baseline lock

## Diff Stat
 reports/p4/p4-0-baseline-lock.json |  9 ++++++
 scripts/lock_p4_0_baseline.py      | 45 ++++++++++++++++++++++++++++++
 tests/test_p4_0_baseline_lock.py   | 57 ++++++++++++++++++++++++++++++++++++++
 3 files changed, 111 insertions(+)

## Full Diff (-U10)
diff --git a/reports/p4/p4-0-baseline-lock.json b/reports/p4/p4-0-baseline-lock.json
new file mode 100644
index 0000000..9044c6c
--- /dev/null
+++ b/reports/p4/p4-0-baseline-lock.json
@@ -0,0 +1,9 @@
+{
+  "p4_0_merge_commit": "7ccd06c4d479b269f7708a6a430b9965af5f17e6",
+  "micro_tasks_manifest_sha256": "bdcc2eaa268b8965ff764ac6c710c97ba90298e11b7c05d0133cdb7103f692bc",
+  "scripted_trajectories_sha256": "50485b6df9a6a7eaf6fd2d00b718c32cb52545b402379cc964cd66ee350ec18f",
+  "agent_evaluator_sha256": "9ee5742d535bed14ae7434417ecffa3436e975b96fd48687497f1a010877fe05",
+  "readiness_report_sha256": "e4500cdc46132776cc3730363a671197cff7ca2f7b0ebed4384023337ce5cd11",
+  "p4_0_verdict": "GO_FOR_P4_AGENT_SFT_DATA",
+  "p4_0_test_count": 81
+}
\ No newline at end of file
diff --git a/scripts/lock_p4_0_baseline.py b/scripts/lock_p4_0_baseline.py
new file mode 100644
index 0000000..f54507c
--- /dev/null
+++ b/scripts/lock_p4_0_baseline.py
@@ -0,0 +1,45 @@
+"""Phase A: lock P4.0 baseline SHAs into a JSON file.
+
+Idempotent: re-running produces the same JSON.
+"""
+from __future__ import annotations
+
+import hashlib
+import json
+from pathlib import Path
+
+_ROOT = Path(__file__).resolve().parent.parent
+_OUT = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
+
+_P4_0_MERGE_COMMIT = "7ccd06c4d479b269f7708a6a430b9965af5f17e6"
+_P4_0_VERDICT = "GO_FOR_P4_AGENT_SFT_DATA"
+_P4_0_TEST_COUNT = 81
+
+
+def _sha256(path: Path) -> str:
+    return hashlib.sha256(path.read_bytes()).hexdigest()
+
+
+def main() -> None:
+    lock = {
+        "p4_0_merge_commit": _P4_0_MERGE_COMMIT,
+        "micro_tasks_manifest_sha256": _sha256(
+            _ROOT / "data" / "p4-agent" / "micro-tasks-v0" / "manifest.json"
+        ),
+        "scripted_trajectories_sha256": _sha256(
+            _ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
+        ),
+        "agent_evaluator_sha256": _sha256(_ROOT / "src" / "agent_evaluator.py"),
+        "readiness_report_sha256": _sha256(
+            _ROOT / "reports" / "p4" / "p4-agent-foundation-readiness.md"
+        ),
+        "p4_0_verdict": _P4_0_VERDICT,
+        "p4_0_test_count": _P4_0_TEST_COUNT,
+    }
+    _OUT.parent.mkdir(parents=True, exist_ok=True)
+    _OUT.write_text(json.dumps(lock, indent=2), encoding="utf-8")
+    print(f"wrote {_OUT}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/tests/test_p4_0_baseline_lock.py b/tests/test_p4_0_baseline_lock.py
new file mode 100644
index 0000000..2b72699
--- /dev/null
+++ b/tests/test_p4_0_baseline_lock.py
@@ -0,0 +1,57 @@
+# tests/test_p4_0_baseline_lock.py
+import hashlib
+import json
+from pathlib import Path
+
+_ROOT = Path(__file__).resolve().parent.parent
+
+
+def test_baseline_lock_exists():
+    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
+    assert lock_path.exists(), "p4-0-baseline-lock.json not found"
+
+
+def test_baseline_lock_has_required_fields():
+    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
+    data = json.loads(lock_path.read_text(encoding="utf-8"))
+    required = {
+        "p4_0_merge_commit", "micro_tasks_manifest_sha256",
+        "scripted_trajectories_sha256", "agent_evaluator_sha256",
+        "readiness_report_sha256", "p4_0_verdict", "p4_0_test_count",
+    }
+    assert required.issubset(data.keys()), f"missing: {required - data.keys()}"
+
+
+def test_baseline_lock_shas_match_files():
+    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
+    data = json.loads(lock_path.read_text(encoding="utf-8"))
+
+    def sha256(path):
+        return hashlib.sha256(path.read_bytes()).hexdigest()
+
+    assert data["micro_tasks_manifest_sha256"] == sha256(
+        _ROOT / "data" / "p4-agent" / "micro-tasks-v0" / "manifest.json"
+    )
+    assert data["scripted_trajectories_sha256"] == sha256(
+        _ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
+    )
+    assert data["agent_evaluator_sha256"] == sha256(
+        _ROOT / "src" / "agent_evaluator.py"
+    )
+    assert data["readiness_report_sha256"] == sha256(
+        _ROOT / "reports" / "p4" / "p4-agent-foundation-readiness.md"
+    )
+
+
+def test_baseline_lock_p4_0_merge_commit_is_7ccd06c():
+    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
+    data = json.loads(lock_path.read_text(encoding="utf-8"))
+    assert data["p4_0_merge_commit"].startswith("7ccd06c"), \
+        f"expected 7ccd06c..., got {data['p4_0_merge_commit']}"
+
+
+def test_baseline_lock_verdict_is_go():
+    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
+    data = json.loads(lock_path.read_text(encoding="utf-8"))
+    assert data["p4_0_verdict"] == "GO_FOR_P4_AGENT_SFT_DATA"
+    assert data["p4_0_test_count"] == 81
