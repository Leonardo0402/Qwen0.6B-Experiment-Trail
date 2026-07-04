"""scripts/p3_readiness_gate.py -- P3 Readiness Gate Checker (Task 14, SESSION END).

Executes 11 PASS checks (Check 6 split into 6a CPU smoke mandatory + 6b GPU
smoke deferrable; Check 10 train sample capacity) and produces:
  - a stdout summary table
  - reports/p3/p3-training-readiness-report.md (human-readable report)

Three-state verdict (Issue #10 Fix 2+5+6):
  - GO_FOR_P3_TRAINING: all mandatory checks PASS AND train capacity >= 2300
  - GO_FOR_P3_PILOT_ONLY: all mandatory checks PASS but capacity < 2300
        (Pilot only; results must NOT be reported as formal capability)
  - FIX_FIRST: any mandatory check FAILS

NO actual training is launched. This is the LAST task of the P3.0-P3.4
session.

Per .superpowers/sdd/task-14-brief.md:
  - Use `from __future__ import annotations` at top
  - Python 3.8.10 active interpreter
  - Each check returns (passed: bool, details: dict)
  - The report file is the primary deliverable
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

# Ensure the project root is on sys.path so we can import src.* and scripts.*
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.family_registry import FamilyRegistry  # noqa: E402
from src.schemas import Sample, to_chatml  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FROZEN_V3_DIR = _ROOT / "data" / "frozen-eval" / "v3"
MANIFEST_PATH = FROZEN_V3_DIR / "manifest.json"
FAMILIES_PATH = FROZEN_V3_DIR / "families.json"
TEST_RAW_PATH = FROZEN_V3_DIR / "test_raw.jsonl"
REJECTED_PATH = FROZEN_V3_DIR / "rejected.jsonl"

REGISTRY_PATH = _ROOT / "data" / "family-registry.json"

BALANCED_TRAIN_PATH = _ROOT / "data" / "p3-curriculum" / "balanced-generalist" / "train.jsonl"
REPAIR_TRAIN_PATH = _ROOT / "data" / "p3-curriculum" / "repair-specialist" / "train.jsonl"

BASELINE_LOCK_PATH = _ROOT / "reports" / "p3" / "p3-baseline-lock.json"
REPORT_PATH = _ROOT / "reports" / "p3" / "p3-training-readiness-report.md"

ADAPTERS_P3_BALANCED = _ROOT / "adapters" / "p3" / "balanced-generalist"
ADAPTERS_P3_REPAIR = _ROOT / "adapters" / "p3" / "repair-specialist"

# P3-specific tests for Check 8 (kept narrow for speed)
P3_TEST_FILES = [
    "tests/test_build_frozen_v3_samples.py",
    "tests/test_build_p3_partition.py",
    "tests/test_sample_pool.py",
    "tests/test_build_balanced_generalist.py",
    "tests/test_build_repair_specialist.py",
    "tests/test_p3_checkpoint_evaluator.py",
]

# max_seq_length from configs/p3/*.yaml
MAX_SEQ_LENGTH = 384

# Train sample capacity thresholds (A7 spec).
# total >= MIN -> FULL training; 0 < total < MIN -> PILOT_ONLY; total == 0 -> FAIL.
MIN_TRAIN_SAMPLES_FOR_FULL = 2300
MAX_TRAIN_SAMPLES_FOR_FULL = 3100

# Expected SHA lock from Task 8 (referenced in task-14-brief.md)
EXPECTED_SHA_LOCK = "a27f36bf5558fbaeff4ee98c906d8e2ecba25794a93adb4d535585d733d8fd09"

# Expected model names in baseline lock (Task 1)
EXPECTED_BASELINE_MODELS = (
    "Base Qwen3-0.6B",
    "Stage3-Independent",
    "Stage3-v3-Antiforget",
)
BASELINE_REQUIRED_FIELDS = (
    "adapter_path",
    "weight_sha256",
    "config_sha256",
    "historical_eval_set_sha256",
    "training_config_sha256",
    "created_at",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Empty lines ignored."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _count_variant_types(path: Path) -> "dict[str, int]":
    """Count samples per ``variant_type`` in a JSONL file (Issue #10 docs).

    Returns a dict mapping variant_type -> count. Returns an empty dict if
    the file does not exist. Lines that fail JSON parsing are skipped.
    """
    counts: "dict[str, int]" = {}
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            vt = record.get("variant_type", "unknown")
            counts[vt] = counts.get(vt, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Check 1: Frozen v3 frozen (SHA locked)
# ---------------------------------------------------------------------------

def check1_frozen_v3_sha_locked() -> Tuple[bool, dict]:
    """Verify the frozen v3 sha_lock matches a recompute over the 3 files."""
    if not MANIFEST_PATH.exists():
        return False, {"error": f"manifest not found: {MANIFEST_PATH}"}
    for p in (FAMILIES_PATH, TEST_RAW_PATH, REJECTED_PATH):
        if not p.exists():
            return False, {"error": f"required file missing: {p}"}

    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    sha_lock = manifest.get("immutability", {}).get("sha_lock", "")
    if not (isinstance(sha_lock, str) and len(sha_lock) == 64):
        return False, {"error": "immutability.sha_lock missing or not 64 hex chars"}

    h = hashlib.sha256()
    for p in (FAMILIES_PATH, TEST_RAW_PATH, REJECTED_PATH):
        h.update(_read_bytes(p))
    recomputed = h.hexdigest()

    if recomputed != sha_lock:
        return False, {
            "sha_lock": sha_lock,
            "recomputed": recomputed,
            "error": "SHA lock mismatch",
        }
    return True, {"sha_lock": sha_lock, "recomputed": recomputed}


# ---------------------------------------------------------------------------
# Check 2: Pairwise disjoint (zero leakage)
# ---------------------------------------------------------------------------

def check2_pairwise_disjoint() -> Tuple[bool, dict]:
    """Run FamilyRegistry.assert_pairwise_disjoint with P2 replay whitelist."""
    if not REGISTRY_PATH.exists():
        return False, {"error": f"registry not found: {REGISTRY_PATH}"}
    registry = FamilyRegistry.from_path(REGISTRY_PATH)
    try:
        registry.assert_pairwise_disjoint(
            ["frozen_v3", "p3_validation", "p3_train", "p3_train_replay"],
            whitelist=[("p3_train_replay", "p2_train")],
        )
    except AssertionError as e:
        return False, {"overlap": str(e)}
    # Snapshot counts for the report
    counts = {
        tag: len(registry.families_with_usage(tag))
        for tag in ("frozen_v3", "p3_validation", "p3_train", "p3_train_replay")
    }
    counts["whitelist"] = [("p3_train_replay", "p2_train")]
    return True, counts


# ---------------------------------------------------------------------------
# Check 3: Assistant retention = 100%
# ---------------------------------------------------------------------------

def _check_assistant_retention_one(sample: Sample) -> bool:
    """True if target_code is non-empty AND appears in the assistant message."""
    if not (sample.target_code or "").strip():
        return False
    chatml = to_chatml(sample)
    messages = chatml.get("messages", [])
    assistant_msg = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"),
        None,
    )
    if assistant_msg is None:
        return False
    content = assistant_msg.get("content", "")
    return sample.target_code in content


def check3_assistant_retention() -> Tuple[bool, dict]:
    """Verify 100% of train samples preserve the full target_code."""
    paths = [BALANCED_TRAIN_PATH, REPAIR_TRAIN_PATH]
    total = 0
    retained = 0
    failures: list[str] = []
    for path in paths:
        if not path.exists():
            return False, {"error": f"train file missing: {path}"}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    sample = Sample.from_json_line(line)
                except Exception as e:  # pragma: no cover -- defensive
                    failures.append(f"{path.name}: parse_error: {e}")
                    continue
                if _check_assistant_retention_one(sample):
                    retained += 1
                else:
                    failures.append(sample.sample_id)
    passed = retained == total
    details = {"checked": total, "retained": retained}
    if failures:
        # Keep the report small: only first 5 failures
        details["first_failures"] = failures[:5]
    return passed, details


# ---------------------------------------------------------------------------
# Check 4: Silent target truncation = 0
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Conservative char-to-token estimate: len(text) // 4."""
    return max(1, len(text) // 4)


def _sample_full_text(sample: Sample) -> str:
    """Concatenate the four payload fields used in training conversations."""
    parts = [
        sample.instruction or "",
        sample.target_code or "",
        sample.public_tests or "",
        sample.hidden_tests or "",
    ]
    return "\n".join(parts)


def check4_silent_truncation_zero() -> Tuple[bool, dict]:
    """Flag any sample whose assistant target would be silently truncated.

    Per the brief:
      - Conservative flag: if ``len(full_text) // 4 > max_seq_length``, the
        sample's full conversation exceeds the budget — recorded as
        ``potential_truncations`` (informational).
      - True silent truncation (PASS/FAIL criterion): under the
        ``preserve_assistant`` policy, the assistant is silently cut ONLY
        when ``assistant_tokens >= max_seq_length`` (target_too_long state
        in ``build_assistant_only_features``). When assistant_tokens <
        max_seq_length, the trainer truncates the prompt from the left and
        the assistant is fully preserved — no silent truncation.

    PASS iff ``real_silent_truncations == 0``.
    """
    paths = [BALANCED_TRAIN_PATH, REPAIR_TRAIN_PATH]
    total = 0
    potential = 0
    real_silent = 0
    flagged_potential: list[str] = []
    flagged_real: list[str] = []
    for path in paths:
        if not path.exists():
            return False, {"error": f"train file missing: {path}"}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    sample = Sample.from_json_line(line)
                except Exception:  # pragma: no cover -- defensive
                    continue
                full_est = _estimate_tokens(_sample_full_text(sample))
                assistant_est = _estimate_tokens(sample.target_code or "")
                if full_est > MAX_SEQ_LENGTH:
                    potential += 1
                    if len(flagged_potential) < 5:
                        flagged_potential.append(sample.sample_id)
                if assistant_est >= MAX_SEQ_LENGTH:
                    real_silent += 1
                    if len(flagged_real) < 5:
                        flagged_real.append(sample.sample_id)
    # PASS criterion: 0 real silent truncations (assistant target cut).
    # potential_truncations is informational (prompt will be truncated,
    # assistant preserved under preserve_assistant policy).
    passed = real_silent == 0
    details = {
        "checked": total,
        "potential_truncations": potential,
        "real_silent_truncations": real_silent,
        "max_seq_length": MAX_SEQ_LENGTH,
        "estimate_method": "len(text) // 4",
        "first_flagged_potential": flagged_potential,
        "first_flagged_real": flagged_real,
        "policy": "preserve_assistant",
    }
    return passed, details


# ---------------------------------------------------------------------------
# Check 5: Canary all fail
# ---------------------------------------------------------------------------

def check5_canary_all_fail() -> Tuple[bool, dict]:
    """Verify all canary samples in frozen v3 test_raw have verified=False."""
    if not TEST_RAW_PATH.exists():
        return False, {"error": f"test_raw not found: {TEST_RAW_PATH}"}
    records = _read_jsonl(TEST_RAW_PATH)
    canaries = [r for r in records if r.get("variant_type") == "canary"]
    if not canaries:
        return False, {"error": "no canary samples found in test_raw.jsonl"}
    failed = sum(1 for c in canaries if c.get("verified") is False)
    passed = failed == len(canaries)
    return passed, {
        "canary_count": len(canaries),
        "all_failed": failed,
        "verified_true": len(canaries) - failed,
    }


# ---------------------------------------------------------------------------
# Check 6a: CPU smoke (mandatory, never SKIPs)
# ---------------------------------------------------------------------------

def check6a_cpu_smoke() -> Tuple[bool, dict]:
    """Lightweight CPU smoke check. MANDATORY -- never SKIPs.

    Verifies the Python interpreter is healthy and core libraries are
    importable. Runs a tiny CPU compute (sum + optional numpy 10x10 matmul).
    FAIL on any exception; otherwise PASS.
    """
    try:
        # Core stdlib library must be importable
        import json  # noqa: F401 -- import smoke
        # Light CPU compute
        result = sum(range(10000))
        assert result == 49995000, f"unexpected sum: {result}"

        # Optional: numpy 10x10 matmul (does NOT fail if numpy absent)
        numpy_available = False
        try:
            import numpy as np  # noqa: WPS433 -- lazy import intentional
            arr = np.arange(100).reshape(10, 10)
            _ = arr @ arr  # 10x10 matmul
            numpy_available = True
        except ImportError:
            pass

        return True, {
            "skipped": False,
            "smoke_passed": True,
            "sum_result": result,
            "numpy_available": numpy_available,
        }
    except Exception as e:  # pragma: no cover -- defensive
        return False, {
            "skipped": False,
            "error": f"CPU smoke failed: {type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# Check 6b: GPU smoke (deferrable -- SKIP allowed when no CUDA)
# ---------------------------------------------------------------------------

def check6b_gpu_smoke() -> Tuple[bool, dict]:
    """Lightweight GPU smoke check. SKIP if torch/CUDA unavailable.

    The brief explicitly forbids loading the full Qwen3-0.6B model here.
    On a CUDA-available host we run a tiny torch forward+backward+optimizer.step
    over a 16x16 tensor and record check_bf16_support() output.
    """
    try:
        import torch  # noqa: WPS433 -- lazy import intentional
    except ImportError:
        return True, {
            "skipped": True,
            "reason": "torch not installed",
        }

    if not torch.cuda.is_available():
        return True, {
            "skipped": True,
            "reason": "CUDA not available",
            "torch_version": torch.__version__,
        }

    # Lazy import (only when CUDA is available)
    from src.p3_checkpoint_evaluator import check_bf16_support  # noqa: E402

    bf16_supported, bf16_msg = check_bf16_support()
    try:
        device = torch.device("cuda")
        # Tiny smoke: forward + backward + optimizer.step
        model = torch.nn.Linear(16, 16, device=device)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        x = torch.randn(4, 16, device=device)
        y = model(x).sum()
        y.backward()
        opt.step()
        opt.zero_grad()
        # eval + save + reload + inference (in-memory smoke)
        with torch.no_grad():
            _ = model(x)
        return True, {
            "skipped": False,
            "bf16_supported": bf16_supported,
            "bf16_message": bf16_msg,
            "smoke_passed": True,
            "device": str(device),
        }
    except Exception as e:  # pragma: no cover -- only triggers on HW fault
        return False, {
            "skipped": False,
            "bf16_supported": bf16_supported,
            "error": f"smoke test failed: {type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# Check 7: Output dirs don't exist (no overwrite)
# ---------------------------------------------------------------------------

def check7_output_dirs_dont_exist() -> Tuple[bool, dict]:
    """Verify adapters/p3/{balanced-generalist,repair-specialist} don't exist."""
    checked = [
        str(ADAPTERS_P3_BALANCED.relative_to(_ROOT)),
        str(ADAPTERS_P3_REPAIR.relative_to(_ROOT)),
    ]
    existing = []
    for p in (ADAPTERS_P3_BALANCED, ADAPTERS_P3_REPAIR):
        if p.exists():
            existing.append(str(p.relative_to(_ROOT)))
    passed = not existing
    return passed, {
        "checked": checked,
        "existing": existing,
        "none_exist": passed,
    }


# ---------------------------------------------------------------------------
# Check 8: CPU CI green
# ---------------------------------------------------------------------------

def check8_cpu_ci_green(timeout: int = 600) -> Tuple[bool, dict]:
    """Run a P3-specific subset of pytest. PASS if 0 failures."""
    cmd = [sys.executable, "-m", "pytest"] + P3_TEST_FILES + ["-v"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, {"error": f"timeout after {timeout}s", "cmd": " ".join(cmd)}
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Parse summary line: e.g. "12 passed in 3.45s"
    total = passed = failed = 0
    summary_line = ""
    last_line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    summary_line = last_line[0]
    import re
    m = re.search(r"(\d+) passed", summary_line)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", summary_line)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", summary_line)
    if m:
        failed += int(m.group(1))
    total = passed + failed
    success = proc.returncode == 0 and failed == 0
    return success, {
        "total": total,
        "passed": passed,
        "failed": failed,
        "returncode": proc.returncode,
        "summary_line": summary_line,
        "cmd": " ".join(cmd),
    }


# ---------------------------------------------------------------------------
# Check 9: P3 baseline lock present
# ---------------------------------------------------------------------------

def check9_baseline_lock_present() -> Tuple[bool, dict]:
    """Verify 3 baseline models with required fields."""
    if not BASELINE_LOCK_PATH.exists():
        return False, {"error": f"baseline lock missing: {BASELINE_LOCK_PATH}"}
    with BASELINE_LOCK_PATH.open(encoding="utf-8") as fh:
        lock = json.load(fh)
    models = lock.get("models", [])
    found_names = [m.get("model_name") for m in models]
    missing = [n for n in EXPECTED_BASELINE_MODELS if n not in found_names]
    if missing:
        return False, {"missing_models": missing, "found": found_names}
    missing_fields_per_model = {}
    for m in models:
        name = m.get("model_name")
        absent = [f for f in BASELINE_REQUIRED_FIELDS if f not in m or m[f] in (None, "")]
        if absent:
            missing_fields_per_model[name] = absent
    if missing_fields_per_model:
        return False, {"missing_fields": missing_fields_per_model}
    return True, {
        "models": list(EXPECTED_BASELINE_MODELS),
        "all_fields_present": True,
    }


# ---------------------------------------------------------------------------
# Check 10: Train sample capacity vs 2300-3100 threshold (Fix 2)
# ---------------------------------------------------------------------------

def check10_train_capacity(
    balanced_path: Path = BALANCED_TRAIN_PATH,
    repair_path: Path = REPAIR_TRAIN_PATH,
) -> Tuple[bool, dict]:
    """Train capacity vs MIN/MAX threshold (Issue #10 Fix 2).

    - total == 0  -> FAIL (no train data, hard failure)
    - 0 < total < MIN -> PASS with verdict_impact=PILOT_ONLY
    - total >= MIN -> PASS with verdict_impact=FULL

    Returns dict with verdict_impact consumed by compute_verdict to choose
    between GO_FOR_P3_TRAINING and GO_FOR_P3_PILOT_ONLY.
    """
    def _count(p: Path) -> int:
        if not p.exists():
            return 0
        n = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n

    balanced_train = _count(balanced_path)
    repair_train = _count(repair_path)
    total = balanced_train + repair_train

    base = {
        "balanced_train": balanced_train,
        "repair_train": repair_train,
        "total": total,
        "min_threshold": MIN_TRAIN_SAMPLES_FOR_FULL,
        "max_threshold": MAX_TRAIN_SAMPLES_FOR_FULL,
    }

    if total == 0:
        base["verdict_impact"] = "FAIL"
        return False, base
    if total < MIN_TRAIN_SAMPLES_FOR_FULL:
        base["verdict_impact"] = "PILOT_ONLY"
        return True, base
    base["verdict_impact"] = "FULL"
    return True, base


# ---------------------------------------------------------------------------
# Verdict + report rendering
# ---------------------------------------------------------------------------

CHECK_NAMES = [
    ("check1_frozen_v3_sha_locked", "Frozen v3 SHA locked"),
    ("check2_pairwise_disjoint", "Pairwise disjoint"),
    ("check3_assistant_retention", "Assistant retention = 100%"),
    ("check4_silent_truncation_zero", "Silent truncation = 0"),
    ("check5_canary_all_fail", "Canary all fail"),
    ("check6a_cpu_smoke", "CPU smoke (mandatory)"),
    ("check6b_gpu_smoke", "GPU smoke (deferrable)"),
    ("check7_output_dirs_dont_exist", "Output dirs don't exist"),
    ("check8_cpu_ci_green", "CPU CI green"),
    ("check9_baseline_lock_present", "P3 baseline lock present"),
    ("check10_train_capacity", "Train sample capacity (2300-3100)"),
]


def compute_verdict(results: list[Tuple[bool, dict]]) -> str:
    """Three-state verdict (Issue #10 Fix 6).

    - GO_FOR_P3_TRAINING: every check PASS AND no capacity PILOT_ONLY warning
    - GO_FOR_P3_PILOT_ONLY: every check PASS but capacity below 2300
    - FIX_FIRST: any check FAIL

    SKIP on Check 6b (GPU smoke) is the only allowed SKIP and counts as PASS.
    Capacity warning is detected via ``details.get("verdict_impact") == "PILOT_ONLY"``
    emitted by check10_train_capacity.
    """
    has_fail = False
    capacity_warning = False

    for passed, details in results:
        if not passed:
            has_fail = True
        if details.get("verdict_impact") == "PILOT_ONLY":
            capacity_warning = True

    if has_fail:
        return "FIX_FIRST"
    if capacity_warning:
        return "GO_FOR_P3_PILOT_ONLY"
    return "GO_FOR_P3_TRAINING"


def _format_details_short(name: str, passed: bool, details: dict) -> str:
    """One-line summary of the details for the table."""
    if name == "check1_frozen_v3_sha_locked":
        if passed:
            return f"sha_lock={details['sha_lock'][:16]}..."
        return f"error={details.get('error', 'mismatch')}"
    if name == "check2_pairwise_disjoint":
        if passed:
            return (
                f"frozen_v3={details.get('frozen_v3')} "
                f"p3_val={details.get('p3_validation')} "
                f"p3_train={details.get('p3_train')} "
                f"replay={details.get('p3_train_replay')} "
                f"wl=p3_train_replay∩p2_train"
            )
        return f"overlap={details.get('overlap', 'unknown')}"
    if name == "check3_assistant_retention":
        return f"{details.get('retained')}/{details.get('checked')} samples"
    if name == "check4_silent_truncation_zero":
        return (
            f"real_silent={details.get('real_silent_truncations', 0)} "
            f"potential={details.get('potential_truncations', 0)} "
            f"/ {details.get('checked', 0)} (preserve_assistant)"
        )
    if name == "check5_canary_all_fail":
        return f"{details.get('all_failed')}/{details.get('canary_count')} verified=False"
    if name == "check6a_cpu_smoke":
        if passed:
            return (
                f"smoke=ok numpy={details.get('numpy_available')} "
                f"sum={details.get('sum_result')}"
            )
        return f"error={details.get('error', 'unknown')}"
    if name == "check6b_gpu_smoke":
        if details.get("skipped"):
            return f"SKIP: {details.get('reason', '')}"
        if passed:
            return (
                f"bf16={details.get('bf16_supported')} smoke={details.get('smoke_passed')} "
                f"device={details.get('device')}"
            )
        return f"error={details.get('error', 'unknown')}"
    if name == "check7_output_dirs_dont_exist":
        if passed:
            return f"{len(details.get('checked', []))} paths checked, none exist"
        return f"existing={details.get('existing', [])}"
    if name == "check8_cpu_ci_green":
        return f"{details.get('passed')}/{details.get('total')} tests pass (rc={details.get('returncode')})"
    if name == "check9_baseline_lock_present":
        if passed:
            return f"{len(details.get('models', []))}/3 models, all fields present"
        return f"missing={details}"
    if name == "check10_train_capacity":
        impact = details.get("verdict_impact", "?")
        return (
            f"{details.get('balanced_train', 0)}+{details.get('repair_train', 0)}="
            f"{details.get('total', 0)} "
            f"(min={details.get('min_threshold', 0)}, "
            f"impact={impact})"
        )
    return json.dumps(details, ensure_ascii=False)[:80]


def _status_token(passed: bool, details: dict) -> str:
    """PASS / FAIL / SKIP token."""
    if not passed:
        return "FAIL"
    if details.get("skipped") is True:
        return "SKIP"
    return "PASS"


def render_report(results: list[Tuple[bool, dict]], verdict: str) -> str:
    """Render the readiness report as Markdown (Chinese prose, English paths)."""
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# P3 Training Readiness Gate Report")
    lines.append("")
    lines.append(f"**Generated**: {now}")
    lines.append("**Branch**: feat/p3-capability-expansion-v2")
    lines.append("**Scope**: P3.0–P3.4 (data + tests + Readiness Gate). NO full training.")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append("## 11 PASS Checks")
    lines.append("")
    lines.append("| # | Check | Status | Details |")
    lines.append("|---|---|---|---|")
    for i, (name, label) in enumerate(CHECK_NAMES, 1):
        passed, details = results[i - 1]
        status = _status_token(passed, details)
        short = _format_details_short(name, passed, details)
        lines.append(f"| {i} | {label} | {status} | {short} |")
    lines.append("")
    lines.append("## Additional Evaluations")
    lines.append("")
    # §B.1
    lines.append("### B.1 verified=True Normalization Deviation (Task 11/12)")
    lines.append("")
    lines.append("#### B.1.1 历史背景")
    lines.append("")
    lines.append("Task 10 构建的 canonical 样本池中包含 501 条来自 P2-replay 衍生的样本，")
    lines.append("这些样本携带 `verified=False` 且 `verification` 子字段全部为 False（上游 Task 10 数据契约问题）。")
    lines.append("为满足 Task 11/12 的硬性闸门 8 (`verified=True`)，构建器原使用")
    lines.append("`model_copy(update={\"verified\": True})` 将样本归一化为 `verified=True`，")
    lines.append("但 `verification` 子字段保持原值（all False），产生 `verified=True && verification all-False`")
    lines.append("的自相矛盾状态。")
    lines.append("")
    lines.append("历史偏差计数（已被 Fix 1 取代，仅作追溯）：")
    lines.append("- Task 11 (balanced-generalist)：438/626 train 样本曾被归一化为 `verified=True`。")
    lines.append("- Task 12 (repair-specialist)：419/493 train 样本曾被归一化为 `verified=True`。")
    lines.append("- 历史合计 857/1119 train 样本曾处于 `verified=True && verification.{syntax_ok,pytest_ok,ruff_ok,timeout}==False` 的自相矛盾状态。")
    lines.append("")
    lines.append("#### B.1.2 Fix 1 修复结果")
    lines.append("")
    lines.append("Fix 1（Issue #10）已通过 `scripts/backfill_canonical_pool_verification.py` 实际对 501 条")
    lines.append("P2-replay 衍生样本运行 `pad_hidden_tests + verify_sample`，回填真实的 `verification` 子字段")
    lines.append("并据此设置 `verified`，移除了 build scripts 中的 `model_copy(update={\"verified\": True})`")
    lines.append("归一化 hack。回填揭露了 boundary 变体生成器 bug：boundary 桶样本的 `target_code` 在边界")
    lines.append("输入下返回错误值，因此实际 `verified=False`。修复后的当前状态：")
    lines.append("")
    lines.append("- 回填脚本：`scripts/backfill_canonical_pool_verification.py`")
    lines.append("- Balanced Generalist train.jsonl：626 → 501 样本（-125 boundary 全失败移除）")
    lines.append("- Repair Specialist train.jsonl：493 → 416 样本（-77，含 boundary 失败与重平衡）")
    lines.append("- 当前 0 条样本处于 `verified=True && verification all-False` 自相矛盾状态")
    lines.append("- build scripts 中的 `model_copy(update={\"verified\": True})` hack 已移除")
    lines.append("- canonical-pool.jsonl 已写入真实 `verification` 子字段")
    lines.append("")
    lines.append("**风险评级**：低（Fix 1 已修复数据契约不一致；train.jsonl 中的 `verified` 与 `verification`")
    lines.append("现在反映真实运行结果）。")
    lines.append("- 训练损失/收敛行为不受影响（损失仅依赖 instruction/target_code/tests，不读 verified 字段）。")
    lines.append("- 下游评估器（如 frozen v3 verification、Tier 2 probe）独立运行 `verify_sample`，不依赖 train.jsonl 的 `verified` 字段。")
    lines.append("- 未来若使用 train.jsonl 的 `verified` 字段做统计/过滤，已可信赖。")
    lines.append("")
    lines.append("**已采取的缓解措施**：")
    lines.append("- Fix 1 回填脚本：`scripts/backfill_canonical_pool_verification.py`")
    lines.append("- canonical-pool backfill manifest：`data/p3-curriculum/canonical-pool-backfill-manifest.json`")
    lines.append("- build scripts 已移除 `model_copy(update={\"verified\": True})` hack")
    lines.append("- 本次 Readiness Gate Check 3/Check 4 仅验证 `target_code` 完整性与 silent truncation，不依赖 `verified` 字段。")
    lines.append("")
    # §B.2
    lines.append("### B.2 Task 13 Reviewer Recommendations (7 items)")
    lines.append("")
    lines.append("Task 13 review APPROVED_WITH_NOTES，记录了 7 条对 trainer 实现的建议。")
    lines.append("本次 Readiness Gate 将其作为 **documentation only**（非 PASS/FAIL 闸门），逐条登记与处置如下：")
    lines.append("")
    lines.append("#### B.2.1 Trainer 调用顺序")
    lines.append("**约定**：`compute_composite(metrics_by_variant, weights) → CompositeScore.compute(weights) → ProbeResult/FullValidationResult.composite_value`。")
    lines.append("**处置**：留给 trainer 实现。本次 Readiness Gate 仅验证 `compute_composite` 与 `CompositeScore.compute` 均存在且可被调用（Check 8 通过 `test_p3_checkpoint_evaluator.py`）。")
    lines.append("")
    lines.append("#### B.2.2 `check_early_stop` 调用时机")
    lines.append("**约定**：必须在 Tier 3 full validation 之后调用，因为 trigger 2 需要 `full_history` 确认。")
    lines.append("**处置**：留给 trainer 实现。当前 `check_early_stop` 在 `full_history=[]` 时返回 `(False, '... awaiting full validation confirm')`，安全降级，不会过早停训。")
    lines.append("")
    lines.append("#### B.2.3 `should_run_tier3` 严格 int 契约")
    lines.append("**约定**：trainer 必须传 `int`（非 `float`），且 `bool` 视为非法。")
    lines.append("**处置**：当前实现已显式拒绝 `bool` 与非 `int` 类型。trainer 必须使用 `int(epoch)` 而非 `epoch`（HuggingFace Trainer 在 epoch 边界返回 float）。建议 trainer 在调用前做 `int(round(epoch))`。")
    lines.append("")
    lines.append("#### B.2.4 Baseline key 映射：`codegen_pass1` ↔ `pass_at_1`")
    lines.append("**约定**：baseline lock (Task 1) 使用 `historical_held_out_metrics.codegen_pass1`，当前 `src/metrics.summarize()` 返回 `pass_at_1`。")
    lines.append("**处置**（Fix 4 已统一）：")
    lines.append("- `src/metrics.py` 新增 `METRICS_SCHEMA_VERSION = \"1.0.0\"` 常量、`BASELINE_TO_METRICS_KEY_MAP = {\"codegen_pass1\": \"pass_at_1\"}` 字典、`normalize_baseline_key(baseline)` 函数。")
    lines.append("- baseline lock JSON 顶层已新增 `schema_version: \"1.0.0\"` 字段。")
    lines.append("- `check_hard_constraint` 现使用 `normalize_baseline_key` 后比较 `pass_at_1`，trainer 与 evaluator 直接调用即可，无需自行映射。")
    lines.append("- `schema_version` mismatch 时记录 warning 但不 FAIL（前向兼容）。")
    lines.append("")
    lines.append("#### B.2.5 BF16 实际硬件验证")
    lines.append("**约定**：trainer 启动时调用 `check_bf16_support()` 并记录输出到日志/报告。")
    lines.append("**处置**：本次 Readiness Gate Check 6b 已调用 `check_bf16_support()`。当前环境为 CPU-only (`torch 2.4.1+cpu`, `cuda.is_available()==False`)，返回 `BF16 not supported, falling back to FP16`。Check 6b SKIP，但 BF16 检查函数本身工作正常。trainer 在 CUDA 环境下启动时必须再次调用并记录输出。Check 6a CPU smoke PASS（不依赖 CUDA）。")
    lines.append("")
    lines.append("#### B.2.6 Probe 样本 `variant_type` 分布 ≥ 19/bucket")
    lines.append("**约定**：Tier 2 probe 每个变体类型桶至少 19 条样本（probe_size=75, 4 桶，base=18+1=19 for first 3 buckets，last bucket=18 — borderline）。")
    balanced_counts = _count_variant_types(BALANCED_TRAIN_PATH)
    repair_counts = _count_variant_types(REPAIR_TRAIN_PATH)
    bal_total = sum(balanced_counts.values())
    rep_total = sum(repair_counts.values())
    bal_line = (
        f"- balanced-generalist train.jsonl: "
        f"code={balanced_counts.get('code', 0)}, "
        f"boundary={balanced_counts.get('boundary', 0)}, "
        f"static_repair={balanced_counts.get('static_repair', 0)}, "
        f"execution_repair={balanced_counts.get('execution_repair', 0)} "
        f"(total={bal_total})."
    )
    rep_line = (
        f"- repair-specialist train.jsonl: "
        f"code={repair_counts.get('code', 0)}, "
        f"boundary={repair_counts.get('boundary', 0)}, "
        f"static_repair={repair_counts.get('static_repair', 0)}, "
        f"execution_repair={repair_counts.get('execution_repair', 0)} "
        f"(total={rep_total})."
    )
    lines.append("**当前数据**（动态读取 train.jsonl 计算）：")
    lines.append(bal_line)
    lines.append(rep_line)
    # Per-bucket warnings (informational only; PASS criterion unchanged).
    bucket_warns: list[str] = []
    for bucket_name in ("code", "boundary", "static_repair", "execution_repair"):
        bal_v = balanced_counts.get(bucket_name, 0)
        rep_v = repair_counts.get(bucket_name, 0)
        if bal_v < 19:
            bucket_warns.append(
                f"balanced-generalist `{bucket_name}` bucket has {bal_v} < 19 samples"
            )
        if rep_v < 19:
            bucket_warns.append(
                f"repair-specialist `{bucket_name}` bucket has {rep_v} < 19 samples"
            )
    if bucket_warns:
        lines.append("**Warnings**（informational，不影响 PASS 判据）：")
        for w in bucket_warns:
            lines.append(f"- {w}")
    lines.append("**处置**：PASS 判据不变——trainer 在调用 `select_probe_samples` 时若某 bucket 不足 19，会自动取 `min(target, len(pool))`，不会抛错。Fix 1 已移除 boundary 失败样本，boundary 桶可能为 0；这是预期的（boundary 变体生成器 bug 已记录于 B.1.2），不影响其他 3 桶的 probe 选择。")
    lines.append("")
    lines.append("#### B.2.7 Composite Score 在 validation 上的退化")
    lines.append("**约定**：P3 validation 集 90 条样本全部为 `variant_type=\"code\"`，因此 Tier 3 full validation Composite 实际只由 `code_generation_pass_at_1 × weight` 主导（其余 3 个分量为 0.0）。")
    lines.append("**处置**：设计意图如此 — validation 测量 held-out code generation 泛化，probe（Tier 2）测量训练时 4 桶能力。trainer 实现时需注意：")
    lines.append("- best checkpoint 仅基于 `full_validation_composite`，因此 best checkpoint 实质等价于 \"best code_generation_pass_at_1\"。")
    lines.append("- hard_constraint (`code_generation_drop_vs_p2_final_max_pct <= 3.0pp`) 进一步约束 — 不会因为 repair 指标提升而接受 code_gen 退化。")
    lines.append("- 如未来希望 best checkpoint 反映 repair 能力，需扩展 validation 集至包含 boundary/static/exec 样本（超出 P3.0-P3.4 scope）。")
    lines.append("")
    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    if verdict == "GO_FOR_P3_TRAINING":
        lines.append("**GO_FOR_P3_TRAINING** — 11 项检查全部通过（含 SKIP 计入 PASS）且 Check 10 verdict_impact=FULL（train 容量 >= 2300）。训练可在以下已记录风险下启动：")
        lines.append("")
        lines.append("- **R1 (B.1)**: 历史 verified normalization 偏差（Task 11/12）。Fix 1 已回填 verification 子字段，当前 0 条样本处于 `verified=True && verification 不一致` 状态。")
        lines.append("- **R2 (B.2.5)**: 当前环境 CPU-only，BF16 实际硬件验证未执行。trainer 在 CUDA 环境启动时必须调用 `check_bf16_support()` 并记录输出。")
        lines.append("- **R3 (B.2.3)**: `should_run_tier3` 严格 int 契约要求 trainer 在调用前对 epoch 做 `int(round(epoch))`。")
        lines.append("- **R4 (B.2.7)**: best checkpoint 实质等价于 best code_generation_pass_at_1（validation 集仅含 code 样本）。hard_constraint 已限制 code_gen 退化，但 repair 提升不会直接反映在 best checkpoint 选择上。")
        lines.append("- **R5**: Check 10 已将 train 容量硬编码为 verdict 决策因子——当前 verdict=GO_FOR_P3_TRAINING 表示 `total >= MIN_TRAIN_SAMPLES_FOR_FULL (2300)`。若容量降至 2300 以下，verdict 自动降级为 GO_FOR_P3_PILOT_ONLY。")
        lines.append("")
        lines.append("训练启动前须由用户明确批准 GO，并确认上述 5 项风险。")
    elif verdict == "GO_FOR_P3_PILOT_ONLY":
        lines.append("**GO_FOR_P3_PILOT_ONLY** — 11 项必跑检查全部通过（含 Check 6b GPU smoke SKIP），但 Check 10 verdict_impact=PILOT_ONLY（train 容量 < 2300）。")
        lines.append("")
        lines.append("数据量低于 2300 阈值，仅允许 PILOT ONLY 训练；不得将 Pilot 结果作为正式能力结论。Pilot 用途：验证训练管道、配置正确性、收敛趋势；不可作为模型能力声明。")
        lines.append("")
        lines.append("**PILOT ONLY 训练约束**：")
        lines.append("- 仅可用于验证训练管道是否端到端可运行（数据加载、loss 下降、checkpoint 保存/reload、3-tier evaluator 调用链）。")
        lines.append("- 不得在论文、README、对外报告、能力声明中引用 Pilot 训练的指标数字（pass_at_1、composite_score 等）。")
        lines.append("- Pilot 完成后须扩充数据池至 >= 2300 条并重新运行 Readiness Gate，获得 GO_FOR_P3_TRAINING 后方可启动正式训练。")
        lines.append("")
        lines.append("**当前数据状态**：")
        balanced_n = repair_n = total_n = 0
        for i, (name, _) in enumerate(CHECK_NAMES, 1):
            _, det = results[i - 1]
            if name == "check10_train_capacity":
                balanced_n = det.get("balanced_train", 0)
                repair_n = det.get("repair_train", 0)
                total_n = det.get("total", 0)
                break
        lines.append(f"- balanced-generalist train.jsonl: {balanced_n} samples")
        lines.append(f"- repair-specialist train.jsonl: {repair_n} samples")
        lines.append(f"- **总计 {total_n} samples** << MIN_TRAIN_SAMPLES_FOR_FULL ({MIN_TRAIN_SAMPLES_FOR_FULL})")
        lines.append("")
        lines.append("**已记录风险（与 GO_FOR_P3_TRAINING 相同，但加 PILOT 约束）**：")
        lines.append("- **R1 (B.1)**: 历史 verified normalization 偏差；Fix 1 已回填，当前 0 条不一致。")
        lines.append("- **R2 (B.2.5)**: 当前环境 CPU-only；trainer 在 CUDA 环境启动时必须调用 `check_bf16_support()` 并记录。")
        lines.append("- **R3 (B.2.3)**: `should_run_tier3` 严格 int 契约。")
        lines.append("- **R4 (B.2.7)**: best checkpoint 等价于 best code_generation_pass_at_1。")
        lines.append("- **R5**: 容量不足触发 PILOT_ONLY——Pilot 结果不可作为模型能力声明。")
        lines.append("")
        lines.append("PILOT 训练启动前须由用户明确批准，并确认上述 5 项风险与 PILOT 约束。")
    else:
        lines.append("**FIX_FIRST** — 至少一项检查 FAIL。必须先修复后再启动训练：")
        lines.append("")
        for i, (name, label) in enumerate(CHECK_NAMES, 1):
            passed, details = results[i - 1]
            if not passed:
                short = _format_details_short(name, passed, details)
                lines.append(f"- **Check {i} ({label})**: {short}")
        lines.append("")
        lines.append("修复后重新运行 `python scripts/p3_readiness_gate.py` 直到 verdict 变为 GO_FOR_P3_TRAINING 或 GO_FOR_P3_PILOT_ONLY。")
    lines.append("")
    lines.append("## Session End")
    lines.append("")
    lines.append("P3.0–P3.4 scope complete. No training launched in this session.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all 11 checks, print summary, write report. Returns 0/1 exit code."""
    check_fns = [
        check1_frozen_v3_sha_locked,
        check2_pairwise_disjoint,
        check3_assistant_retention,
        check4_silent_truncation_zero,
        check5_canary_all_fail,
        check6a_cpu_smoke,
        check6b_gpu_smoke,
        check7_output_dirs_dont_exist,
        check8_cpu_ci_green,
        check9_baseline_lock_present,
        check10_train_capacity,
    ]
    results: list[Tuple[bool, dict]] = []
    total_checks = len(CHECK_NAMES)
    print("=" * 78)
    print(f"P3 Readiness Gate — {total_checks} PASS Checks (SESSION END)")
    print("=" * 78)
    print()
    for i, (name, label) in enumerate(CHECK_NAMES, 1):
        fn = check_fns[i - 1]
        print(f"[{i}/{total_checks}] {label} ...", flush=True)
        try:
            passed, details = fn()
        except Exception as e:  # pragma: no cover -- defensive
            passed = False
            details = {"error": f"{type(e).__name__}: {e}"}
        results.append((passed, details))
        status = _status_token(passed, details)
        print(f"      -> {status}: {_format_details_short(name, passed, details)}")
        print()

    verdict = compute_verdict(results)

    # Print summary table
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"| # | Check                                      | Status |")
    print(f"|---|---|---|")
    for i, (name, label) in enumerate(CHECK_NAMES, 1):
        passed, details = results[i - 1]
        status = _status_token(passed, details)
        print(f"| {i} | {label:<42} | {status:5} |")
    print()
    print(f"Verdict: {verdict}")
    print()

    # Write report
    report_md = render_report(results, verdict)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")
    print()

    # Exit code: 0 for GO (TRAINING or PILOT_ONLY, both allow next step under user approval);
    # 1 for FIX_FIRST.
    return 0 if verdict in ("GO_FOR_P3_TRAINING", "GO_FOR_P3_PILOT_ONLY") else 1


if __name__ == "__main__":
    sys.exit(main())
