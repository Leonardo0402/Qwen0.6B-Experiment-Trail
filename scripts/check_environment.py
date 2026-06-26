"""
scripts/check_environment.py — Preflight environment checker for qwen3-code-lab.

Checks Python version, PyTorch/CUDA setup, GPU VRAM, key library versions,
disk space, HuggingFace cache location, model directory, and a CUDA smoke test.

Usage:
    python scripts/check_environment.py

Exit codes:
    0  READY     — all checks passed or only warnings
    1  NOT READY — one or more checks failed

Windows DLL note
----------------
On Windows, importing ``datasets`` (which carries C extensions) *after*
``torch.cuda`` has been initialised causes an access-violation crash.  All
heavy library imports are therefore performed eagerly at **module-load time**,
before any CUDA initialisation happens.  The check functions then read the
pre-computed results rather than importing inside the function body.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Eager library imports — order matters on Windows.
# datasets MUST be imported before torch.cuda is initialised; we import all
# required libs here so they are all available before any check function runs.
# ---------------------------------------------------------------------------

_REQUIRED_LIBS: List[str] = ["datasets", "transformers", "peft", "accelerate"]
_LIB_VERSIONS: Dict[str, str] = {}
_LIB_MISSING: List[str] = []

for _lib_name in _REQUIRED_LIBS:
    try:
        _mod = importlib.import_module(_lib_name)
        _LIB_VERSIONS[_lib_name] = getattr(_mod, "__version__", "unknown")
    except ImportError:
        _LIB_MISSING.append(_lib_name)

# torch import — done after datasets/transformers/peft/accelerate.
try:
    import torch as _torch

    _TORCH_VERSION: Optional[str] = _torch.__version__
    _TORCH_IMPORT_ERROR: Optional[str] = None
except ImportError as _exc:
    _torch = None  # type: ignore[assignment]
    _TORCH_VERSION = None
    _TORCH_IMPORT_ERROR = str(_exc)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    status: str       # "PASS", "WARN", "FAIL"
    value: str        # short human-readable value string
    detail: str = ""  # optional longer detail / hint


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version(version_info: Optional[Sequence] = None) -> CheckResult:
    """Check that Python is 3.10.x.

    Parameters
    ----------
    version_info:
        A sequence of at least 3 ints (major, minor, micro).  Defaults to
        ``sys.version_info`` so callers can inject a fake version for tests.
    """
    if version_info is None:
        version_info = sys.version_info
    major, minor, micro = version_info[0], version_info[1], version_info[2]
    ver_str = f"{major}.{minor}.{micro}"
    if major == 3 and minor == 10:
        return CheckResult("Python version", "PASS", ver_str)
    return CheckResult(
        "Python version",
        "FAIL",
        ver_str,
        "Must be Python 3.10.x — activate the qwen3-code-lab conda env.",
    )


def check_pytorch() -> CheckResult:
    """Report PyTorch version (includes +cuXXX build tag if present)."""
    if _TORCH_IMPORT_ERROR is not None:
        return CheckResult("PyTorch", "FAIL", "not installed", _TORCH_IMPORT_ERROR)
    return CheckResult("PyTorch", "PASS", _TORCH_VERSION or "unknown")


def check_cuda_available() -> CheckResult:
    """Verify that CUDA is reachable via torch.cuda.is_available()."""
    if _torch is None:
        return CheckResult("CUDA available", "FAIL", "torch not installed", "")
    available = _torch.cuda.is_available()
    if available:
        return CheckResult("CUDA available", "PASS", "True")
    return CheckResult(
        "CUDA available",
        "FAIL",
        "False",
        "CUDA not available — GPU training is not possible in this environment.",
    )


def check_gpu_name() -> CheckResult:
    """Report the name of CUDA device 0."""
    if _torch is None:
        return CheckResult("GPU name", "FAIL", "N/A", "torch not installed")
    if not _torch.cuda.is_available():
        return CheckResult("GPU name", "FAIL", "N/A", "CUDA not available")
    try:
        name = _torch.cuda.get_device_name(0)
        return CheckResult("GPU name", "PASS", name)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("GPU name", "FAIL", "error", str(exc))


def check_vram(warn_free_gb: float = 2.5) -> CheckResult:
    """Report total / free VRAM.  Warn when free < *warn_free_gb* GB."""
    if _torch is None:
        return CheckResult("VRAM", "FAIL", "N/A", "torch not installed")
    if not _torch.cuda.is_available():
        return CheckResult("VRAM", "FAIL", "N/A", "CUDA not available")
    try:
        props = _torch.cuda.get_device_properties(0)
        total_bytes = props.total_memory
        free_bytes, _ = _torch.cuda.mem_get_info(0)
        total_gb = total_bytes / (1024 ** 3)
        free_gb = free_bytes / (1024 ** 3)
        value = f"total={total_gb:.1f} GB, free={free_gb:.1f} GB"
        if free_gb < warn_free_gb:
            return CheckResult(
                "VRAM",
                "WARN",
                value,
                f"Free VRAM ({free_gb:.1f} GB) < {warn_free_gb} GB — OOM risk during training.",
            )
        return CheckResult("VRAM", "PASS", value)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("VRAM", "FAIL", "error", str(exc))


# Public name for callers who want to check a specific subset.
REQUIRED_LIBS: List[str] = ["transformers", "peft", "datasets", "accelerate"]


def check_library_versions(libs: Optional[List[str]] = None) -> CheckResult:
    """Report versions of required ML libraries (uses eagerly-loaded results)."""
    if libs is None:
        libs = REQUIRED_LIBS
    missing = [lib for lib in libs if lib in _LIB_MISSING or lib not in _LIB_VERSIONS]
    if missing:
        return CheckResult(
            "Library versions",
            "FAIL",
            f"MISSING: {', '.join(missing)}",
            "pip install " + " ".join(missing),
        )
    ver_str = ", ".join(
        f"{lib}={_LIB_VERSIONS[lib]}" for lib in libs if lib in _LIB_VERSIONS
    )
    return CheckResult("Library versions", "PASS", ver_str)


def check_disk_space(
    project_root: Optional[Path] = None,
    warn_free_gb: float = 20.0,
) -> CheckResult:
    """Check free disk space on the drive hosting *project_root*."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    try:
        usage = shutil.disk_usage(str(project_root))
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        value = f"free={free_gb:.1f} GB / {total_gb:.1f} GB total"
        if free_gb < warn_free_gb:
            return CheckResult(
                "Disk space",
                "WARN",
                value,
                f"Less than {warn_free_gb:.0f} GB free on project drive — may run short during training.",
            )
        return CheckResult("Disk space", "PASS", value)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Disk space", "FAIL", "error", str(exc))


def check_hf_cache(
    project_root: Optional[Path] = None,
    env: Optional[dict] = None,
) -> CheckResult:
    """Check where the HuggingFace model cache is pointed.

    PASS when the cache lives under *project_root*.
    WARN when the cache is unset or points somewhere outside the project root
    (e.g. C: drive, which has limited free space on this machine).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    if env is None:
        env = dict(os.environ)

    # HF_HOME takes priority; fall back to HUGGINGFACE_HUB_CACHE.
    raw = env.get("HF_HOME") or env.get("HUGGINGFACE_HUB_CACHE")

    if not raw:
        recommended = project_root / ".hf_cache"
        return CheckResult(
            "HF cache",
            "WARN",
            "unset",
            f"Set HF_HOME={recommended} before downloading the model "
            "(keeps large files off the C: drive).",
        )

    cache_path = Path(raw)
    recommended = project_root / ".hf_cache"

    # PASS if cache is nested under project root.
    try:
        cache_path.relative_to(project_root)
        return CheckResult("HF cache", "PASS", str(cache_path))
    except ValueError:
        pass

    # Cache exists but is outside the project root — warn.
    return CheckResult(
        "HF cache",
        "WARN",
        str(cache_path),
        f"Cache is outside the project root. Recommended: set HF_HOME={recommended}",
    )


def check_model_dir(project_root: Optional[Path] = None) -> CheckResult:
    """Check whether models/Qwen3-0.6B is present and looks populated."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models" / "Qwen3-0.6B"

    if not model_dir.exists():
        return CheckResult(
            "Model directory",
            "WARN",
            "MISSING",
            f"{model_dir} not found — download the model in a later task step.",
        )

    config_json = model_dir / "config.json"
    safetensors_files = list(model_dir.glob("*.safetensors"))

    if config_json.exists() and safetensors_files:
        n = len(safetensors_files)
        return CheckResult(
            "Model directory",
            "PASS",
            f"PRESENT ({n} safetensors file(s))",
        )

    missing_parts: List[str] = []
    if not config_json.exists():
        missing_parts.append("config.json")
    if not safetensors_files:
        missing_parts.append("*.safetensors")
    return CheckResult(
        "Model directory",
        "WARN",
        "INCOMPLETE",
        f"Directory exists but missing: {', '.join(missing_parts)}",
    )


def check_cuda_smoke() -> CheckResult:
    """Allocate two small tensors on CUDA, multiply, synchronize, verify finite."""
    if _torch is None:
        return CheckResult("CUDA smoke test", "FAIL", "skipped", "torch not installed")
    if not _torch.cuda.is_available():
        return CheckResult("CUDA smoke test", "FAIL", "skipped", "CUDA not available")
    try:
        a = _torch.ones(64, 64, device="cuda")
        b = _torch.ones(64, 64, device="cuda")
        c = _torch.mm(a, b)
        _torch.cuda.synchronize()
        if not _torch.isfinite(c).all().item():
            return CheckResult(
                "CUDA smoke test",
                "FAIL",
                "non-finite result",
                "matmul produced non-finite values — possible hardware issue.",
            )
        return CheckResult("CUDA smoke test", "PASS", "OK (64x64 matmul, all finite)")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("CUDA smoke test", "FAIL", "error", str(exc))


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_STATUS_ICON = {"PASS": "v", "WARN": "!", "FAIL": "X"}
_STATUS_COLOR = {
    "PASS": "\033[32m",
    "WARN": "\033[33m",
    "FAIL": "\033[31m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def render_report(results: List[CheckResult]) -> None:
    """Print a formatted report to stdout."""
    max_name = max(len(r.name) for r in results) if results else 0
    print()
    print("=" * 76)
    print(f"  {_BOLD}qwen3-code-lab -- Environment Check{_RESET}")
    print("=" * 76)
    for r in results:
        icon = _STATUS_ICON[r.status]
        color = _STATUS_COLOR[r.status]
        name_col = r.name.ljust(max_name)
        status_col = f"{color}{icon} {r.status:<4}{_RESET}"
        print(f"  {status_col}  {name_col}  {r.value}")
        if r.detail:
            indent = " " * (2 + 2 + 1 + 4 + 2 + max_name + 2)
            print(f"{indent}-> {r.detail}")
    print("-" * 76)


def summarize(results: List[CheckResult]) -> Tuple[str, List[str]]:
    """Return ``(status_str, [failed_check_names])``.

    *status_str* is ``"READY"`` when there are no FAILs; warnings do not
    affect readiness.
    """
    failed = [r.name for r in results if r.status == "FAIL"]
    if failed:
        return "NOT READY", failed
    return "READY", []


# ---------------------------------------------------------------------------
# Check registry and entry point
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_python_version,
    check_pytorch,
    check_cuda_available,
    check_gpu_name,
    check_vram,
    check_library_versions,
    check_disk_space,
    check_hf_cache,
    check_model_dir,
    check_cuda_smoke,
]


def main() -> int:
    """Run all checks, render the report, return exit code (0=READY, 1=NOT READY)."""
    results: List[CheckResult] = []
    for fn in ALL_CHECKS:
        results.append(fn())

    render_report(results)
    status, failed = summarize(results)

    if status == "READY":
        print(f"\n  {_STATUS_COLOR['PASS']}{_BOLD}Status: READY{_RESET}\n")
        return 0

    failed_str = ", ".join(failed)
    print(
        f"\n  {_STATUS_COLOR['FAIL']}{_BOLD}Status: NOT READY{_RESET}"
        f" -- failed checks: {failed_str}\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
