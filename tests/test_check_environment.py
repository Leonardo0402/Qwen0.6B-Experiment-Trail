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

import scripts.check_environment as ce  # noqa: E402
from scripts.check_environment import (  # noqa: E402
    CheckResult,
    check_cuda_available,
    check_cuda_smoke,
    check_disk_space,
    check_gpu_name,
    check_hf_cache,
    check_library_versions,
    check_model_dir,
    check_pytorch,
    check_python_version,
    check_vram,
    main,
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
        # WARN is possible when CUDA is unavailable (skipped, not a hard fail).
        assert result.status in ("PASS", "WARN", "FAIL")

    def test_cuda_available_value_is_bool_like(self):
        result = check_cuda_available()
        assert result.value in ("True", "False", "torch not installed")


# ---------------------------------------------------------------------------
# I-1: torch / library import-failure handling (simulate missing DLL etc.)
# ---------------------------------------------------------------------------


class TestTorchImportFailure:
    """When torch failed to import (e.g. missing CUDA DLL → OSError), the
    GPU-dependent checks must report a clean FAIL rather than raising."""

    def test_pytorch_check_fail_when_torch_missing(self, monkeypatch):
        monkeypatch.setattr(ce, "_torch", None)
        monkeypatch.setattr(ce, "_TORCH_IMPORT_ERROR", "OSError: [WinError 126] DLL not found")
        monkeypatch.setattr(ce, "_TORCH_VERSION", None)
        result = check_pytorch()
        assert result.status == "FAIL"
        assert "WinError 126" in result.detail

    def test_cuda_available_fail_when_torch_missing(self, monkeypatch):
        monkeypatch.setattr(ce, "_torch", None)
        result = check_cuda_available()
        assert result.status == "FAIL"
        assert "torch not installed" in result.value

    def test_gpu_name_fail_when_torch_missing(self, monkeypatch):
        monkeypatch.setattr(ce, "_torch", None)
        result = check_gpu_name()
        assert result.status == "FAIL"

    def test_vram_fail_when_torch_missing(self, monkeypatch):
        monkeypatch.setattr(ce, "_torch", None)
        result = check_vram()
        assert result.status == "FAIL"

    def test_cuda_smoke_fail_when_torch_missing(self, monkeypatch):
        monkeypatch.setattr(ce, "_torch", None)
        result = check_cuda_smoke()
        assert result.status == "FAIL"

    def test_no_check_raises_when_torch_missing(self, monkeypatch):
        monkeypatch.setattr(ce, "_torch", None)
        # None of these should raise even with torch unavailable.
        for fn in (check_pytorch, check_cuda_available, check_gpu_name,
                   check_vram, check_cuda_smoke):
            assert isinstance(fn(), CheckResult)


class TestCudaUnavailableIsWarn:
    """M-2: when CUDA is unavailable (torch present), the dependent checks
    WARN/skip rather than FAIL so only the CUDA-available check fails."""

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    class _FakeTorch:
        cuda = None  # set in fixture

    @pytest.fixture
    def fake_torch_no_cuda(self, monkeypatch):
        fake = self._FakeTorch()
        fake.cuda = self._FakeCuda()
        monkeypatch.setattr(ce, "_torch", fake)
        return fake

    def test_cuda_available_fails(self, fake_torch_no_cuda):
        assert check_cuda_available().status == "FAIL"

    def test_gpu_name_warns(self, fake_torch_no_cuda):
        result = check_gpu_name()
        assert result.status == "WARN"
        assert result.value == "skipped"

    def test_vram_warns(self, fake_torch_no_cuda):
        result = check_vram()
        assert result.status == "WARN"
        assert result.value == "skipped"

    def test_cuda_smoke_warns(self, fake_torch_no_cuda):
        result = check_cuda_smoke()
        assert result.status == "WARN"
        assert result.value == "skipped"


# ---------------------------------------------------------------------------
# I-2: check_library_versions — PASS and FAIL branches
# ---------------------------------------------------------------------------


class TestCheckLibraryVersions:
    def test_pass_when_all_present(self, monkeypatch):
        monkeypatch.setattr(
            ce, "_LIB_VERSIONS",
            {"transformers": "5.0", "peft": "0.1", "datasets": "3.0", "accelerate": "1.0"},
        )
        monkeypatch.setattr(ce, "_LIB_MISSING", {})
        result = check_library_versions()
        assert result.status == "PASS"
        assert "transformers=5.0" in result.value

    def test_fail_when_a_lib_missing(self, monkeypatch):
        monkeypatch.setattr(
            ce, "_LIB_VERSIONS",
            {"transformers": "5.0", "peft": "0.1", "accelerate": "1.0"},
        )
        monkeypatch.setattr(
            ce, "_LIB_MISSING",
            {"datasets": "OSError: [WinError 126] The specified module could not be found"},
        )
        result = check_library_versions()
        assert result.status == "FAIL"
        assert "datasets" in result.value
        # The captured reason is surfaced in the detail.
        assert "WinError 126" in result.detail

    def test_fail_lists_all_missing(self, monkeypatch):
        monkeypatch.setattr(ce, "_LIB_VERSIONS", {"transformers": "5.0"})
        monkeypatch.setattr(
            ce, "_LIB_MISSING",
            {"peft": "ImportError: no peft", "datasets": "ImportError: no datasets"},
        )
        # accelerate is neither in versions nor missing dict -> also reported missing
        result = check_library_versions(
            libs=["transformers", "peft", "datasets", "accelerate"]
        )
        assert result.status == "FAIL"
        for lib in ("peft", "datasets", "accelerate"):
            assert lib in result.value

    def test_returns_check_result_instance(self, monkeypatch):
        monkeypatch.setattr(ce, "_LIB_VERSIONS", {"transformers": "5.0"})
        monkeypatch.setattr(ce, "_LIB_MISSING", {})
        result = check_library_versions(libs=["transformers"])
        assert isinstance(result, CheckResult)
        assert result.name == "Library versions"


# ---------------------------------------------------------------------------
# M-4: check_disk_space exception path
# ---------------------------------------------------------------------------


class TestCheckDiskSpaceErrors:
    def test_fail_when_disk_usage_raises(self, monkeypatch, tmp_path):
        import shutil as _shutil

        def _boom(_path):
            raise OSError("disk gone")

        monkeypatch.setattr(_shutil, "disk_usage", _boom)
        result = check_disk_space(project_root=tmp_path)
        assert result.status == "FAIL"
        assert "disk gone" in result.detail


# ---------------------------------------------------------------------------
# M-3: check_model_dir filesystem-error path
# ---------------------------------------------------------------------------


class TestCheckModelDirErrors:
    def test_warn_when_exists_raises(self, monkeypatch, tmp_path):
        # Simulate a PermissionError while probing the filesystem.
        import pathlib

        def _boom(self):
            raise PermissionError("access denied")

        monkeypatch.setattr(pathlib.Path, "exists", _boom)
        result = check_model_dir(project_root=tmp_path)
        assert result.status == "WARN"
        assert "access denied" in result.detail


# ---------------------------------------------------------------------------
# I-3: main() exit-code contract
# ---------------------------------------------------------------------------


class TestMainExitCode:
    def test_returns_0_when_no_fail(self, monkeypatch, capsys):
        stubs = [
            lambda: CheckResult("A", "PASS", "ok"),
            lambda: CheckResult("B", "WARN", "meh"),
        ]
        monkeypatch.setattr(ce, "ALL_CHECKS", stubs)
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Status: READY" in out

    def test_returns_1_when_fail_present(self, monkeypatch, capsys):
        stubs = [
            lambda: CheckResult("A", "PASS", "ok"),
            lambda: CheckResult("B", "FAIL", "broken"),
        ]
        monkeypatch.setattr(ce, "ALL_CHECKS", stubs)
        rc = main()
        assert rc == 1
        out = capsys.readouterr().out
        assert "Status: NOT READY" in out
        assert "B" in out

    def test_warnings_alone_are_ready(self, monkeypatch):
        stubs = [
            lambda: CheckResult("A", "WARN", "x"),
            lambda: CheckResult("B", "WARN", "y"),
        ]
        monkeypatch.setattr(ce, "ALL_CHECKS", stubs)
        assert main() == 0
