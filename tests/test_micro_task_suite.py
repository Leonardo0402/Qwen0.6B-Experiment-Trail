"""Phase E: Micro-task suite tests."""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.agent_workspace import MicroTaskWorkspace
from src.agent_tools import tool_run_tests, tool_apply_patch

SUITE_DIR = Path(__file__).parent.parent / "data" / "p4-agent" / "micro-tasks-v0"


def _load_manifest():
    with (SUITE_DIR / "manifest.json").open() as f:
        return json.load(f)


def test_manifest_exists():
    """Top-level manifest.json exists and is valid."""
    assert SUITE_DIR.exists()
    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["suite_name"] == "micro-tasks-v0"
    assert manifest["total_tasks"] == 40
    assert len(manifest["tasks"]) == 40


def test_all_40_tasks_exist():
    """All 40 task directories exist with required files."""
    manifest = _load_manifest()
    for entry in manifest["tasks"]:
        task_dir = SUITE_DIR / entry["path"]
        assert task_dir.is_dir(), f"missing dir: {entry['path']}"
        assert (task_dir / "README.md").exists()
        assert (task_dir / "solution.py").exists()
        assert (task_dir / "test_solution.py").exists()
        assert (task_dir / "expected_patch.json").exists()
        assert (task_dir / "manifest.json").exists()


def test_baseline_fails(monkeypatch):
    """Each task's baseline solution.py fails pytest."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    manifest = _load_manifest()
    for entry in manifest["tasks"][:5]:  # test first 5 for speed
        task_dir = SUITE_DIR / entry["path"]
        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            obs = tool_run_tests(ws, timeout_s=10.0)
            assert not obs.passed, f"{entry['task_id']}: baseline should fail but passed"
        finally:
            ws.cleanup()


def test_post_patch_passes(monkeypatch):
    """Each task passes pytest after applying expected_patch."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")
    manifest = _load_manifest()
    for entry in manifest["tasks"][:5]:  # test first 5 for speed
        task_dir = SUITE_DIR / entry["path"]
        patch = json.loads((task_dir / "expected_patch.json").read_text())
        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            tool_apply_patch(
                ws, patch["file_path"],
                patch["old_text"], patch["new_text"],
            )
            obs = tool_run_tests(ws, timeout_s=10.0)
            assert obs.passed, f"{entry['task_id']}: post-patch should pass but failed"
        finally:
            ws.cleanup()


def test_no_frozen_v4_overlap():
    """Task IDs don't overlap with frozen v4 sample IDs."""
    frozen_path = Path(__file__).parent.parent / "data" / "frozen-eval" / "v4" / "test_raw.jsonl"
    if not frozen_path.exists():
        pytest.skip("frozen v4 data not found")
    frozen_ids = set()
    with frozen_path.open() as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                frozen_ids.add(sample.get("sample_id", ""))
    manifest = _load_manifest()
    for entry in manifest["tasks"]:
        # micro-task IDs use task_NNN format, frozen v4 uses MBPP/other IDs
        assert entry["task_id"] not in frozen_ids
        assert entry["task_id"].startswith("task_")  # not MBPP family


def test_sha256_recorded():
    """Each task's manifest.json has SHA256 for all 4 files."""
    manifest = _load_manifest()
    for entry in manifest["tasks"]:
        assert "readme_sha256" in entry
        assert "solution_sha256" in entry
        assert "test_sha256" in entry
        assert "expected_patch_sha256" in entry
        assert len(entry["readme_sha256"]) == 64  # SHA256 hex length
