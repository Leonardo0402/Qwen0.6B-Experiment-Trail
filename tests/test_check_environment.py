"""
tests/test_check_environment.py — Unit tests for scripts/check_environment.py.

Coverage strategy:
- Pure-ish functions (python version, disk space, HF cache, model dir, summary)
  are fully unit-tested without a GPU via argument injection / monkeypatching.
- CUDA-dependent checks get a light smoke test only: assert the function
  returns a CheckResult and does not raise, regardless of whether CUDA is
  available in the test environment.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make `scripts` importable from the project root.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.check_environment import (  # noqa: E402
    CheckResult,
    check_cuda_available,
    check_cuda_smoke,
    check_disk_space,
    check_hf_cache,
    check_model_dir,
    check_python_version,
    summarize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FakeDiskUsage = collections.namedtuple("usage", ["total", "used", "free"])


def _fake_disk_usage(free_gb: float, total_gb: float = 500.0):
    """Return a monkeypatch-compatible disk_usage callable."""
    free_bytes = int(free_gb * 1024 ** 3)
    total_bytes = int(total_gb * 1024 ** 3)
    used_bytes = total_bytes - free_bytes
    stub = _FakeDiskUsage(total=total_bytes, used=used_bytes, free=free_bytes)
    return lambda _path: stub


# ---------------------------------------------------------------------------
# check_python_version
# ---------------------------------------------------------------------------


class TestCheckPythonVersion:
    def test_passes_for_310(self):
        result = check_python_version((3, 10, 12, "final", 0))
        assert result.status == "PASS"
        assert "3.10.12" in result.value

    def test_fails_for_38(self):
        result = check_python_version((3, 8, 18, "final", 0))
        assert result.status == "FAIL"
        assert "3.8" in result.value

    def test_fails_for_39(self):
        result = check_python_version((3, 9, 7, "final", 0))
        assert result.status == "FAIL"

    def test_fails_for_311(self):
        result = check_python_version((3, 11, 0, "final", 0))
        assert result.status == "FAIL"

    def test_fails_for_312(self):
        result = check_python_version((3, 12, 1, "final", 0))
        assert result.status == "FAIL"

    def test_returns_check_result_instance(self):
        result = check_python_version((3, 10, 5, "final", 0))
        assert isinstance(result, CheckResult)
        assert result.name == "Python version"

    def test_detail_contains_hint_on_fail(self):
        result = check_python_version((3, 8, 0, "final", 0))
        assert result.detail  # non-empty hint


# ---------------------------------------------------------------------------
# check_disk_space
# ---------------------------------------------------------------------------


class TestCheckDiskSpace:
    def test_pass_when_plenty_of_space(self, monkeypatch, tmp_path):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "disk_usage", _fake_disk_usage(free_gb=100.0))
        result = check_disk_space(project_root=tmp_path)
        assert result.status == "PASS"
        assert "100.0 GB" in result.value

    def test_warn_when_below_threshold(self, monkeypatch, tmp_path):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "disk_usage", _fake_disk_usage(free_gb=10.0))
        result = check_disk_space(project_root=tmp_path)
        assert result.status == "WARN"
        assert "10.0 GB" in result.value

    def test_warn_exactly_at_boundary(self, monkeypatch, tmp_path):
        # default threshold is 20 GB; 19.9 GB should warn
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "disk_usage", _fake_disk_usage(free_gb=19.9))
        result = check_disk_space(project_root=tmp_path)
        assert result.status == "WARN"

    def test_pass_just_above_threshold(self, monkeypatch, tmp_path):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "disk_usage", _fake_disk_usage(free_gb=20.1))
        result = check_disk_space(project_root=tmp_path)
        assert result.status == "PASS"

    def test_custom_warn_threshold(self, monkeypatch, tmp_path):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "disk_usage", _fake_disk_usage(free_gb=5.0))
        result = check_disk_space(project_root=tmp_path, warn_free_gb=3.0)
        assert result.status == "PASS"

    def test_returns_check_result_instance(self, monkeypatch, tmp_path):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "disk_usage", _fake_disk_usage(free_gb=50.0))
        result = check_disk_space(project_root=tmp_path)
        assert isinstance(result, CheckResult)
        assert result.name == "Disk space"

    def test_value_shows_total_as_well(self, monkeypatch, tmp_path):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "disk_usage", _fake_disk_usage(free_gb=50.0, total_gb=200.0))
        result = check_disk_space(project_root=tmp_path)
        assert "200.0 GB" in result.value


# ---------------------------------------------------------------------------
# check_hf_cache
# ---------------------------------------------------------------------------


class TestCheckHfCache:
    def test_warn_when_unset(self, tmp_path):
        result = check_hf_cache(project_root=tmp_path, env={})
        assert result.status == "WARN"
        assert result.value == "unset"

    def test_warn_when_both_vars_absent(self, tmp_path):
        env = {"PATH": "/usr/bin", "HOME": "/home/user"}
        result = check_hf_cache(project_root=tmp_path, env=env)
        assert result.status == "WARN"

    def test_pass_when_hf_home_under_project_root(self, tmp_path):
        cache = tmp_path / ".hf_cache"
        result = check_hf_cache(project_root=tmp_path, env={"HF_HOME": str(cache)})
        assert result.status == "PASS"

    def test_pass_when_hub_cache_under_project_root(self, tmp_path):
        cache = tmp_path / ".hf_cache"
        result = check_hf_cache(
            project_root=tmp_path,
            env={"HUGGINGFACE_HUB_CACHE": str(cache)},
        )
        assert result.status == "PASS"

    def test_hf_home_takes_priority_over_hub_cache(self, tmp_path):
        cache = tmp_path / ".hf_cache"
        # HF_HOME points under project → PASS; HUGGINGFACE_HUB_CACHE points away
        result = check_hf_cache(
            project_root=tmp_path,
            env={
                "HF_HOME": str(cache),
                "HUGGINGFACE_HUB_CACHE": r"C:\some\other\path",
            },
        )
        assert result.status == "PASS"

    def test_warn_when_hf_home_outside_project_root(self, tmp_path):
        # Point to a sibling directory (outside tmp_path)
        outside = tmp_path.parent / "some_other_cache"
        result = check_hf_cache(
            project_root=tmp_path,
            env={"HF_HOME": str(outside)},
        )
        assert result.status == "WARN"

    def test_warn_contains_recommended_path(self, tmp_path):
        result = check_hf_cache(project_root=tmp_path, env={})
        assert ".hf_cache" in result.detail

    def test_pass_value_shows_cache_path(self, tmp_path):
        cache = tmp_path / ".hf_cache"
        result = check_hf_cache(project_root=tmp_path, env={"HF_HOME": str(cache)})
        assert str(cache) in result.value


# ---------------------------------------------------------------------------
# check_model_dir
# ---------------------------------------------------------------------------


class TestCheckModelDir:
    def test_missing_when_model_dir_absent(self, tmp_path):
        result = check_model_dir(project_root=tmp_path)
        assert result.status == "WARN"
        assert "MISSING" in result.value

    def test_present_when_config_and_safetensors_exist(self, tmp_path):
        model_dir = tmp_path / "models" / "Qwen3-0.6B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 16)
        result = check_model_dir(project_root=tmp_path)
        assert result.status == "PASS"
        assert "PRESENT" in result.value
        assert "1 safetensors" in result.value

    def test_multiple_safetensors_reported(self, tmp_path):
        model_dir = tmp_path / "models" / "Qwen3-0.6B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        for i in range(3):
            (model_dir / f"model-{i:05d}.safetensors").write_bytes(b"\x00" * 8)
        result = check_model_dir(project_root=tmp_path)
        assert result.status == "PASS"
        assert "3 safetensors" in result.value

    def test_incomplete_when_only_config_exists(self, tmp_path):
        model_dir = tmp_path / "models" / "Qwen3-0.6B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        result = check_model_dir(project_root=tmp_path)
        assert result.status == "WARN"
        assert "INCOMPLETE" in result.value
        assert "safetensors" in result.detail

    def test_incomplete_when_only_safetensors_exists(self, tmp_path):
        model_dir = tmp_path / "models" / "Qwen3-0.6B"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 16)
        result = check_model_dir(project_root=tmp_path)
        assert result.status == "WARN"
        assert "config.json" in result.detail

    def test_returns_check_result_instance(self, tmp_path):
        result = check_model_dir(project_root=tmp_path)
        assert isinstance(result, CheckResult)
        assert result.name == "Model directory"


# ---------------------------------------------------------------------------
# summarize — overall READY / NOT READY logic
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_ready_when_all_pass(self):
        results = [
            CheckResult("A", "PASS", "ok"),
            CheckResult("B", "PASS", "ok"),
        ]
        status, failed = summarize(results)
        assert status == "READY"
        assert failed == []

    def test_ready_when_only_warnings(self):
        results = [
            CheckResult("A", "PASS", "ok"),
            CheckResult("B", "WARN", "low"),
            CheckResult("C", "WARN", "missing"),
        ]
        status, failed = summarize(results)
        assert status == "READY"
        assert failed == []

    def test_not_ready_when_single_fail(self):
        results = [
            CheckResult("A", "PASS", "ok"),
            CheckResult("B", "FAIL", "broken"),
        ]
        status, failed = summarize(results)
        assert status == "NOT READY"
        assert "B" in failed

    def test_not_ready_lists_all_failed_checks(self):
        results = [
            CheckResult("X", "FAIL", "bad"),
            CheckResult("Y", "PASS", "ok"),
            CheckResult("Z", "FAIL", "also bad"),
        ]
        status, failed = summarize(results)
        assert status == "NOT READY"
        assert set(failed) == {"X", "Z"}

    def test_pass_names_not_in_failed_list(self):
        results = [
            CheckResult("Good", "PASS", "ok"),
            CheckResult("Bad", "FAIL", "no"),
        ]
        _, failed = summarize(results)
        assert "Good" not in failed

    def test_empty_results_is_ready(self):
        status, failed = summarize([])
        assert status == "READY"
        assert failed == []


# ---------------------------------------------------------------------------
# CUDA-dependent checks — light coverage: returns CheckResult, does not raise
# ---------------------------------------------------------------------------


class TestCudaChecksReturnCheckResult:
    def test_cuda_available_returns_check_result(self):
        result = check_cuda_available()
        assert isinstance(result, CheckResult)
        assert result.name == "CUDA available"
        assert result.status in ("PASS", "FAIL")

    def test_cuda_smoke_returns_check_result(self):
        result = check_cuda_smoke()
        assert isinstance(result, CheckResult)
        assert result.name == "CUDA smoke test"
        assert result.status in ("PASS", "FAIL")

    def test_cuda_available_value_is_bool_like(self):
        result = check_cuda_available()
        assert result.value in ("True", "False", "torch not installed")
