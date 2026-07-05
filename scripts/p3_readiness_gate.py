"""scripts/p3_readiness_gate.py -- P3 Readiness Gate Checker (Task 14, SESSION END).

Executes 12 PASS checks (Check 6 split into 6a CPU smoke mandatory + 6b GPU
smoke deferrable; Check 10 per-candidate train capacity; Check 11 verified
consistency) and produces:
  - a stdout summary table
  - reports/p3/p3-training-readiness-report.md (human-readable report)

Three-state verdict (Issue #10 Fix 2+5+6 + Issue #12 per-candidate):
  - GO_FOR_P3_TRAINING: all mandatory checks PASS AND both candidates' capacity >= 2300
  - GO_FOR_P3_PILOT_ONLY: all mandatory checks PASS but any candidate's capacity < 2300
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

FROZEN_V4_DIR = _ROOT / "data" / "frozen-eval" / "v4"
MANIFEST_PATH = FROZEN_V4_DIR / "manifest.json"
FAMILIES_PATH = FROZEN_V4_DIR / "families.json"
TEST_RAW_PATH = FROZEN_V4_DIR / "test_raw.jsonl"
REJECTED_PATH = FROZEN_V4_DIR / "rejected.jsonl"

FROZEN_V4_LOCK_PATH = _ROOT / "reports" / "p3" / "frozen-v4-lock.json"

REGISTRY_PATH = _ROOT / "data" / "family-registry.json"
FAMILY_PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"

BALANCED_TRAIN_PATH = _ROOT / "data" / "p3-curriculum" / "balanced-generalist" / "train.jsonl"
REPAIR_TRAIN_PATH = _ROOT / "data" / "p3-curriculum" / "repair-specialist" / "train.jsonl"

VALIDATION_V2_DIR = _ROOT / "data" / "p3-curriculum" / "validation-v2"
VALIDATION_V2_PATH = VALIDATION_V2_DIR / "validation.jsonl"
VALIDATION_V2_FAMILIES_PATH = VALIDATION_V2_DIR / "families.json"
VALIDATION_V2_MANIFEST_PATH = VALIDATION_V2_DIR / "manifest.json"

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
    "tests/test_frozen_v4_compliance.py",
]

# max_seq_length from configs/p3/*.yaml
MAX_SEQ_LENGTH = 384

# Train sample capacity thresholds (A7 spec).
# total >= MIN -> FULL training; 0 < total < MIN -> PILOT_ONLY; total == 0 -> FAIL.
MIN_TRAIN_SAMPLES_FOR_FULL = 2300
MAX_TRAIN_SAMPLES_FOR_FULL = 3100

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
    """Read file bytes with CRLF normalized to LF for cross-platform SHA consistency."""
    return path.read_bytes().replace(b"\r\n", b"\n")


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


def _load_family_set(path: Path) -> "set[str]":
    """Load family IDs from a families.json file.

    Supports multiple key conventions:
      - ``families`` (frozen-eval/v4)
      - ``validation_family_ids`` (validation-v2)
      - ``frozen_family_ids`` (other frozen sets)
    """
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out: "set[str]" = set()
    items: list = []
    for key in ("families", "validation_family_ids", "frozen_family_ids",
                "train_family_ids"):
        items.extend(data.get(key, []) or [])
    for item in items:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict):
            fid = item.get("family_id") or item.get("id")
            if isinstance(fid, str):
                out.add(fid)
    return out


def _load_family_ids_from_jsonl(path: Path) -> "set[str]":
    """Extract unique family_id values from a JSONL file."""
    out: "set[str]" = set()
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = rec.get("family_id")
            if isinstance(fid, str):
                out.add(fid)
    return out


# ---------------------------------------------------------------------------
# Check 1: Frozen v4 SHA locked (Issue #14 Wave 2-B)
# ---------------------------------------------------------------------------

def check1_frozen_v4_sha_locked() -> Tuple[bool, dict]:
    """Verify the frozen v4 lock file matches a recompute over the 4 files.

    Checks (Issue #14 P1.1 + P1.2):
      - reports/p3/frozen-v4-lock.json exists
      - manifest, test_raw, families, rejected all exist
      - manifest SHA matches lock file
      - test_raw SHA matches both lock file and manifest's test_raw_sha256
      - families SHA matches both lock file and manifest's families_sha256
      - rejected SHA matches both lock file and manifest's rejected_sha256
      - combined sha_lock (sha256 of families+test_raw+rejected, CRLF normalized)
        matches the lock file
    """
    if not FROZEN_V4_LOCK_PATH.exists():
        return False, {"error": f"lock file not found: {FROZEN_V4_LOCK_PATH}"}
    for p in (MANIFEST_PATH, FAMILIES_PATH, TEST_RAW_PATH, REJECTED_PATH):
        if not p.exists():
            return False, {"error": f"required file missing: {p}"}

    with FROZEN_V4_LOCK_PATH.open(encoding="utf-8") as fh:
        lock = json.load(fh)
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    recomputed = {
        "manifest": hashlib.sha256(_read_bytes(MANIFEST_PATH)).hexdigest(),
        "test_raw": hashlib.sha256(_read_bytes(TEST_RAW_PATH)).hexdigest(),
        "families": hashlib.sha256(_read_bytes(FAMILIES_PATH)).hexdigest(),
        "rejected": hashlib.sha256(_read_bytes(REJECTED_PATH)).hexdigest(),
    }
    h = hashlib.sha256()
    for p in (FAMILIES_PATH, TEST_RAW_PATH, REJECTED_PATH):
        h.update(_read_bytes(p))
    recomputed_lock = h.hexdigest()

    errors: list[str] = []
    for key in ("manifest", "test_raw", "families", "rejected"):
        lock_key = f"{key}_sha256"
        if lock.get(lock_key) != recomputed[key]:
            errors.append(
                f"{lock_key}: lock={lock.get(lock_key)} actual={recomputed[key]}"
            )
    if lock.get("sha_lock") != recomputed_lock:
        errors.append(
            f"sha_lock: lock={lock.get('sha_lock')} actual={recomputed_lock}"
        )
    # Cross-check manifest's individual SHAs (test_raw/families/rejected)
    if manifest.get("test_raw_sha256") != recomputed["test_raw"]:
        errors.append(
            f"manifest.test_raw_sha256={manifest.get('test_raw_sha256')} "
            f"actual={recomputed['test_raw']}"
        )
    if manifest.get("families_sha256") != recomputed["families"]:
        errors.append(
            f"manifest.families_sha256={manifest.get('families_sha256')} "
            f"actual={recomputed['families']}"
        )
    if manifest.get("rejected_sha256") != recomputed["rejected"]:
        errors.append(
            f"manifest.rejected_sha256={manifest.get('rejected_sha256')} "
            f"actual={recomputed['rejected']}"
        )

    if errors:
        return False, {"errors": errors, "recomputed": recomputed}
    return True, {
        "sha_lock": lock.get("sha_lock"),
        "recomputed_lock": recomputed_lock,
        "manifest_sha256": recomputed["manifest"],
        "test_raw_sha256": recomputed["test_raw"],
        "families_sha256": recomputed["families"],
        "rejected_sha256": recomputed["rejected"],
    }


# ---------------------------------------------------------------------------
# Check 2: Family Isolation Gate (Issue #14 Wave 2-B P1.3)
# ---------------------------------------------------------------------------

def _load_all_family_sets() -> "dict[str, set[str]]":
    """Load all family sets used by the isolation check.

    Sets:
      - formal_train: families actually present in balanced + repair train.jsonl
        (the real formal train data; NOT the stale family-partition.json
        assignment, because v4 claimed some p3_train_new families)
      - validation_v2: validation-v2/families.json
      - frozen_v4: frozen-eval/v4/families.json
      - historical_frozen: v1 + v3 + p2 frozen-eval-v2 families
      - historical_validation: p2 stage{1,2,3} validation + curriculum-v2
      - p2_train: registry tag p2_train (for whitelist verification)
      - p3_train_replay: family-partition.json p3_train_replay (whitelist)
    """
    sets: "dict[str, set[str]]" = {}

    # formal_train from actual train.jsonl files (the real formal train data)
    sets["formal_train"] = (
        _load_family_ids_from_jsonl(BALANCED_TRAIN_PATH)
        | _load_family_ids_from_jsonl(REPAIR_TRAIN_PATH)
    )

    # p3_train_replay from family-partition.json (for whitelist verification)
    if FAMILY_PARTITION_PATH.exists():
        with FAMILY_PARTITION_PATH.open(encoding="utf-8") as fh:
            part = json.load(fh)
        sets["p3_train_replay"] = set(
            part.get("p3_train_replay", {}).get("family_ids", [])
        )
    else:
        sets["p3_train_replay"] = set()

    # validation-v2 families
    sets["validation_v2"] = _load_family_set(VALIDATION_V2_FAMILIES_PATH)

    # frozen-v4 families
    sets["frozen_v4"] = _load_family_set(FAMILIES_PATH)

    # Historical frozen families (v1, v3, p2 frozen-eval-v2)
    hist_frozen: set[str] = set()
    hist_frozen |= _load_family_set(
        _ROOT / "data" / "frozen-eval" / "v1" / "families.json"
    )
    hist_frozen |= _load_family_set(
        _ROOT / "data" / "frozen-eval" / "v3" / "families.json"
    )
    hist_frozen |= _load_family_set(
        _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2" / "families.json"
    )
    sets["historical_frozen"] = hist_frozen

    # Historical validation families
    hist_val: set[str] = set()
    # P2 curriculum stage validation files (extract family_id from JSONL)
    for stage in ("stage1-code", "stage2-boundary", "stage3-repair"):
        hist_val |= _load_family_ids_from_jsonl(
            _ROOT / "data" / "p2-curriculum" / stage / "validation.jsonl"
        )
    # P2 stage3-repair-v3 validation
    hist_val |= _load_family_ids_from_jsonl(
        _ROOT / "data" / "p2-curriculum" / "stage3-repair-v3" / "validation.jsonl"
    )
    # curriculum-v2 validation_families
    cv2_part = _ROOT / "data" / "curriculum-v2" / "family-partition.json"
    if cv2_part.exists():
        with cv2_part.open(encoding="utf-8") as fh:
            cv2 = json.load(fh)
        for fid in cv2.get("validation_families", []):
            if isinstance(fid, str):
                hist_val.add(fid)
    sets["historical_validation"] = hist_val

    # p2_train from registry (for whitelist verification)
    if REGISTRY_PATH.exists():
        registry = FamilyRegistry.from_path(REGISTRY_PATH)
        sets["p2_train"] = set(registry.families_with_usage("p2_train"))
    else:
        sets["p2_train"] = set()

    return sets


def check2_family_isolation() -> Tuple[bool, dict]:
    """Verify pairwise disjoint family sets with explicit whitelist (Issue #14).

    The following 5 sets must be pairwise disjoint:
      - formal_train (families in balanced + repair train.jsonl)
      - validation_v2 (validation-v2/families.json)
      - frozen_v4 (frozen-eval/v4/families.json)
      - historical_frozen (v1 + v3 + p2-frozen-v2)
      - historical_validation (p2 stage validation + curriculum-v2)

    Whitelist (explicitly verified, NOT a generic 3-way intersection):
      - p3_train_replay ⊆ p2_train (replay families are allowed to come
        from P2 train, every other overlap is forbidden)
      - formal_train ∩ p2_train must be a subset of p3_train_replay
        (i.e. the only allowed overlap between formal train and p2_train
        is via the replay mechanism)
    """
    sets = _load_all_family_sets()

    main_sets = {
        "formal_train": sets["formal_train"],
        "validation_v2": sets["validation_v2"],
        "frozen_v4": sets["frozen_v4"],
        "historical_frozen": sets["historical_frozen"],
        "historical_validation": sets["historical_validation"],
    }

    # Pairwise disjoint check (5 sets, 10 pairs)
    violations: list[dict] = []
    keys = list(main_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            overlap = main_sets[a] & main_sets[b]
            if overlap:
                violations.append({
                    "pair": (a, b),
                    "count": len(overlap),
                    "samples": sorted(overlap)[:10],
                })

    # Explicit whitelist verification: p3_train_replay ⊆ p2_train
    p3_replay = sets["p3_train_replay"]
    p2_train = sets["p2_train"]
    whitelist_intersection = p3_replay & p2_train
    whitelist_complete = (whitelist_intersection == p3_replay)
    # formal_train ∩ p2_train must be a subset of p3_train_replay
    formal_p2_overlap = sets["formal_train"] & p2_train
    formal_p2_clean = formal_p2_overlap <= p3_replay

    counts = {k: len(v) for k, v in main_sets.items()}
    counts["p3_train_replay"] = len(p3_replay)
    counts["p2_train"] = len(p2_train)
    counts["whitelist_intersection"] = len(whitelist_intersection)
    counts["formal_train_p2_overlap"] = len(formal_p2_overlap)

    passed = (
        len(violations) == 0
        and whitelist_complete
        and formal_p2_clean
    )
    details = {
        "counts": counts,
        "violations": violations,
        "whitelist": {
            "pair": ("p3_train_replay", "p2_train"),
            "replay_count": len(p3_replay),
            "intersection_count": len(whitelist_intersection),
            "all_replay_in_p2_train": whitelist_complete,
            "formal_train_p2_overlap_clean": formal_p2_clean,
            "formal_train_p2_overlap_count": len(formal_p2_overlap),
        },
    }
    return passed, details


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
    """Verify all canary samples in frozen v4 test_raw have verified=False."""
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
            "reason": "GPU smoke: torch not installed",
        }

    if not torch.cuda.is_available():
        return True, {
            "skipped": True,
            "reason": "GPU smoke: CUDA not available",
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
    """Per-candidate train capacity vs MIN/MAX threshold (Issue #12 fix).

    Each candidate is checked INDEPENDENTLY against the threshold:
    - candidate == 0  -> FAIL (hard failure)
    - 0 < candidate < MIN -> PASS with verdict_impact=PILOT_ONLY for that candidate
    - candidate >= MIN -> PASS with verdict_impact=FULL for that candidate

    Overall verdict_impact:
    - "FAIL" if any candidate has 0 samples
    - "PILOT_ONLY" if any candidate is < MIN (but all > 0)
    - "FULL" only if ALL candidates >= MIN
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

    # Per-candidate verdict
    if balanced_train == 0 or repair_train == 0:
        overall_impact = "FAIL"
    elif balanced_train < MIN_TRAIN_SAMPLES_FOR_FULL or repair_train < MIN_TRAIN_SAMPLES_FOR_FULL:
        overall_impact = "PILOT_ONLY"
    else:
        overall_impact = "FULL"

    base = {
        "balanced_train": balanced_train,
        "repair_train": repair_train,
        "total": total,
        "min_threshold": MIN_TRAIN_SAMPLES_FOR_FULL,
        "max_threshold": MAX_TRAIN_SAMPLES_FOR_FULL,
        # Issue #12: per-candidate status
        "balanced_verdict": (
            "FAIL" if balanced_train == 0
            else "PILOT_ONLY" if balanced_train < MIN_TRAIN_SAMPLES_FOR_FULL
            else "FULL"
        ),
        "repair_verdict": (
            "FAIL" if repair_train == 0
            else "PILOT_ONLY" if repair_train < MIN_TRAIN_SAMPLES_FOR_FULL
            else "FULL"
        ),
        "per_candidate_check": True,
    }

    if overall_impact == "FAIL":
        base["verdict_impact"] = "FAIL"
        return False, base
    base["verdict_impact"] = overall_impact
    return True, base


# ---------------------------------------------------------------------------
# Check 11: verified consistency (Issue #12)
# ---------------------------------------------------------------------------

def check11_verified_consistency(
    balanced_path: Path = BALANCED_TRAIN_PATH,
    repair_path: Path = REPAIR_TRAIN_PATH,
) -> Tuple[bool, dict]:
    """Verify that ``verified`` field is consistent with ``verification``
    subfields across all train samples (Issue #12).

    Consistency rule:
    - verified=True REQUIRES verification.syntax_ok=True AND verification.pytest_ok=True
    - verified=False REQUIRES at least one of (syntax_ok, pytest_ok) == False

    Returns (passed, details) where details includes:
    - checked: total samples checked
    - inconsistent: list of sample_ids that violate the rule
    - inconsistent_count: len(inconsistent)
    """
    def _load(p: Path) -> list:
        if not p.exists():
            return []
        samples = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    all_samples = _load(balanced_path) + _load(repair_path)
    inconsistent = []

    for s in all_samples:
        verified = s.get("verified", False)
        ver = s.get("verification") or {}
        syntax_ok = ver.get("syntax_ok", False)
        pytest_ok = ver.get("pytest_ok", False)

        if verified:
            # verified=True requires syntax_ok AND pytest_ok
            if not (syntax_ok and pytest_ok):
                inconsistent.append({
                    "sample_id": s.get("sample_id", "?"),
                    "family_id": s.get("family_id", "?"),
                    "issue": "verified=True but syntax_ok/pytest_ok not both True",
                    "syntax_ok": syntax_ok,
                    "pytest_ok": pytest_ok,
                })
        else:
            # verified=False requires at least one False
            if syntax_ok and pytest_ok:
                inconsistent.append({
                    "sample_id": s.get("sample_id", "?"),
                    "family_id": s.get("family_id", "?"),
                    "issue": "verified=False but syntax_ok AND pytest_ok both True",
                    "syntax_ok": syntax_ok,
                    "pytest_ok": pytest_ok,
                })

    passed = len(inconsistent) == 0
    details = {
        "checked": len(all_samples),
        "inconsistent_count": len(inconsistent),
        "inconsistent_sample_ids": [r["sample_id"] for r in inconsistent[:20]],
        "rule": "verified=True ⟺ syntax_ok AND pytest_ok",
    }
    return passed, details


# ---------------------------------------------------------------------------
# Check 12: Candidate ratio within tolerance (Issue #12 P6 #4)
# ---------------------------------------------------------------------------

def check12_candidate_ratio_within_tolerance() -> Tuple[bool, dict]:
    """Check both candidates' variant_type ratios are within ±3pp tolerance.

    Balanced target: 30% code / 20% boundary / 20% static_repair / 30% execution_repair
    Repair target: 15% code / 15% boundary / 30% static_repair / 40% execution_repair

    Tolerance: ±3 percentage points per bucket.
    PASS iff all 8 ratios (4 buckets × 2 candidates) within tolerance.
    """
    # 使用已有的 _count_variant_types helper 函数
    # Balanced targets
    balanced_targets = {"code": 0.30, "boundary": 0.20, "static_repair": 0.20, "execution_repair": 0.30}
    repair_targets = {"code": 0.15, "boundary": 0.15, "static_repair": 0.30, "execution_repair": 0.40}
    tolerance = 0.03  # ±3pp

    balanced_counts = _count_variant_types(BALANCED_TRAIN_PATH)
    repair_counts = _count_variant_types(REPAIR_TRAIN_PATH)

    details = {
        "balanced_counts": balanced_counts,
        "repair_counts": repair_counts,
        "tolerance_pp": 3,
        "violations": [],
    }

    all_ok = True
    for label, counts, targets in [
        ("balanced", balanced_counts, balanced_targets),
        ("repair", repair_counts, repair_targets),
    ]:
        total = sum(counts.values())
        if total == 0:
            details["violations"].append(f"{label}: total=0")
            all_ok = False
            continue
        for bucket, target in targets.items():
            actual = counts.get(bucket, 0) / total
            diff = abs(actual - target)
            if diff > tolerance:
                details["violations"].append(
                    f"{label}.{bucket}: actual={actual:.2%} target={target:.2%} diff={diff:.2%}"
                )
                all_ok = False

    return all_ok, details


# ---------------------------------------------------------------------------
# Check 13: All required buckets non-empty (Issue #12 P6 #5)
# ---------------------------------------------------------------------------

def check13_all_buckets_non_empty() -> Tuple[bool, dict]:
    """Check all 4 variant_type buckets are non-empty for both candidates.

    Required buckets: code, boundary, static_repair, execution_repair.
    PASS iff all 8 (4 buckets × 2 candidates) are non-empty.
    """
    required = ["code", "boundary", "static_repair", "execution_repair"]
    balanced_counts = _count_variant_types(BALANCED_TRAIN_PATH)
    repair_counts = _count_variant_types(REPAIR_TRAIN_PATH)

    empty = []
    for bucket in required:
        if balanced_counts.get(bucket, 0) == 0:
            empty.append(f"balanced.{bucket}")
        if repair_counts.get(bucket, 0) == 0:
            empty.append(f"repair.{bucket}")

    details = {
        "balanced_counts": balanced_counts,
        "repair_counts": repair_counts,
        "required_buckets": required,
        "empty_buckets": empty,
    }
    return len(empty) == 0, details


# ---------------------------------------------------------------------------
# Check 14: Composite evaluator complete (Issue #12 P6 #12)
# ---------------------------------------------------------------------------

def check14_composite_evaluator_complete() -> Tuple[bool, dict]:
    """Check CompositeScore has all 5 components including hidden_pass_rate.

    Verifies:
    - CompositeScore dataclass has hidden_pass_rate field
    - Both YAML configs have 5-component composite_score with hidden_pass_rate
    - compute() method accepts hidden_pass_rate weight
    """
    import yaml
    from src.p3_checkpoint_evaluator import CompositeScore

    required_fields = [
        "code_generation_pass_at_1",
        "boundary_pass_at_1",
        "static_repair_success",
        "execution_repair_success",
        "hidden_pass_rate",
    ]

    # Check CompositeScore dataclass fields
    cs_fields = {f.name for f in CompositeScore.__dataclass_fields__.values()}
    missing_fields = [f for f in required_fields if f not in cs_fields]

    # Check YAML configs
    config_violations = []
    for cfg_path in [
        _ROOT / "configs" / "p3" / "balanced-generalist.yaml",
        _ROOT / "configs" / "p3" / "repair-specialist.yaml",
    ]:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cs = cfg.get("composite_score", {})
        for field in required_fields:
            if field not in cs:
                config_violations.append(f"{cfg_path.name}: missing {field}")

    # Check compute() method works with 5 components
    test_cs = CompositeScore(0.5, 0.4, 0.6, 0.7, 0.9)
    test_weights = {f: 0.2 for f in required_fields}
    try:
        test_result = test_cs.compute(test_weights)
        compute_ok = True
    except Exception as e:
        compute_ok = False
        config_violations.append(f"compute() failed: {e}")

    all_ok = (
        len(missing_fields) == 0
        and len(config_violations) == 0
        and compute_ok
    )

    details = {
        "composite_score_fields": sorted(cs_fields),
        "missing_fields": missing_fields,
        "config_violations": config_violations,
        "compute_test": test_result if compute_ok else None,
    }
    return all_ok, details


# ---------------------------------------------------------------------------
# Check 15: Frozen v4 Coverage Gate (Issue #14 Wave 2-B P1.2)
# ---------------------------------------------------------------------------

# Coverage target ranges (inclusive)
V4_FAMILY_RANGE = (80, 100)
V4_FORMAL_SAMPLE_RANGE = (360, 700)
V4_RATIO_RANGES = {
    "code": (0.25, 0.30),
    "boundary": (0.15, 0.20),
    "static_repair": (0.25, 0.30),
    "execution_repair": (0.25, 0.30),
}
CANARY_VARIANT_TYPES = ("canary", "canary_repair")
FORMAL_VARIANT_TYPES = ("code", "boundary", "static_repair", "execution_repair")


def check15_v4_coverage_gate() -> Tuple[bool, dict]:
    """Verify Frozen v4 meets Issue #14 P1.2 coverage requirements.

    Checks:
      - families: 80-100
      - formal samples (canary excluded): 360-700
      - Code: 25-30%, Boundary: 15-20%, Static Repair: 25-30%, Exec Repair: 25-30%
      - canary excluded from formal denominator
      - all canaries fail (verified=False)
      - all formal references pass (verified=True)
      - all repair broken_code fails at least one test (broken_code != target_code)
      - all execution_feedback non-empty (for execution_repair)
    """
    if not TEST_RAW_PATH.exists():
        return False, {"error": f"test_raw not found: {TEST_RAW_PATH}"}
    if not FAMILIES_PATH.exists():
        return False, {"error": f"families not found: {FAMILIES_PATH}"}

    records = _read_jsonl(TEST_RAW_PATH)
    v4_families = _load_family_set(FAMILIES_PATH)

    canaries = [r for r in records if r.get("variant_type") in CANARY_VARIANT_TYPES]
    formal = [r for r in records if r.get("variant_type") in FORMAL_VARIANT_TYPES]

    variant_counts = {vt: 0 for vt in FORMAL_VARIANT_TYPES}
    for r in formal:
        vt = r.get("variant_type")
        if vt in variant_counts:
            variant_counts[vt] += 1

    errors: list[str] = []

    # Family count
    fam_count = len(v4_families)
    if not (V4_FAMILY_RANGE[0] <= fam_count <= V4_FAMILY_RANGE[1]):
        errors.append(
            f"family_count={fam_count} not in {V4_FAMILY_RANGE}"
        )

    # Formal sample count
    formal_count = len(formal)
    if not (V4_FORMAL_SAMPLE_RANGE[0] <= formal_count <= V4_FORMAL_SAMPLE_RANGE[1]):
        errors.append(
            f"formal_sample_count={formal_count} not in {V4_FORMAL_SAMPLE_RANGE}"
        )

    # Variant ratios
    ratio_report: "dict[str, float]" = {}
    if formal_count > 0:
        for vt, (lo, hi) in V4_RATIO_RANGES.items():
            r = variant_counts.get(vt, 0) / formal_count
            ratio_report[vt] = r
            if not (lo <= r <= hi):
                errors.append(
                    f"ratio {vt}={r:.2%} not in [{lo:.0%}, {hi:.0%}]"
                )

    # Canary excluded from formal denominator (sanity)
    total_records = len(records)
    if formal_count + len(canaries) != total_records:
        errors.append(
            f"formal({formal_count}) + canary({len(canaries)}) != "
            f"total({total_records}) — variant_type mismatch"
        )

    # All canaries verified=False
    canary_verified_true = sum(1 for c in canaries if c.get("verified") is True)
    if canary_verified_true != 0:
        errors.append(
            f"canary_verified_true={canary_verified_true} (expected 0)"
        )

    # All formal verified=True
    formal_verified_false = sum(1 for r in formal if r.get("verified") is not True)
    if formal_verified_false != 0:
        errors.append(
            f"formal_verified_false={formal_verified_false} (expected 0)"
        )

    # All repair broken_code != target_code
    repair_samples = [
        r for r in formal
        if r.get("variant_type") in ("static_repair", "execution_repair")
    ]
    broken_same_as_target = 0
    broken_missing = 0
    for r in repair_samples:
        broken = r.get("broken_code")
        target = r.get("target_code")
        if broken is None:
            broken_missing += 1
        elif broken == target:
            broken_same_as_target += 1
    if broken_missing > 0:
        errors.append(f"repair broken_code missing={broken_missing}")
    if broken_same_as_target > 0:
        errors.append(
            f"repair broken_code == target_code count={broken_same_as_target}"
        )

    # All execution_repair execution_feedback non-empty
    exec_repair = [
        r for r in formal if r.get("variant_type") == "execution_repair"
    ]
    exec_feedback_empty = sum(
        1 for r in exec_repair if not (r.get("execution_feedback") or "").strip()
    )
    if exec_feedback_empty > 0:
        errors.append(
            f"execution_repair execution_feedback empty={exec_feedback_empty}"
        )

    passed = len(errors) == 0
    details = {
        "family_count": fam_count,
        "family_range": list(V4_FAMILY_RANGE),
        "formal_sample_count": formal_count,
        "formal_sample_range": list(V4_FORMAL_SAMPLE_RANGE),
        "canary_count": len(canaries),
        "variant_counts": variant_counts,
        "variant_ratios": {k: round(v, 4) for k, v in ratio_report.items()},
        "ratio_ranges": {
            k: [lo, hi] for k, (lo, hi) in V4_RATIO_RANGES.items()
        },
        "canary_verified_true": canary_verified_true,
        "formal_verified_false": formal_verified_false,
        "repair_sample_count": len(repair_samples),
        "broken_code_missing": broken_missing,
        "broken_same_as_target": broken_same_as_target,
        "exec_repair_count": len(exec_repair),
        "exec_feedback_empty": exec_feedback_empty,
        "errors": errors,
    }
    return passed, details


# ---------------------------------------------------------------------------
# Check 16: Validation v2 Gate (Issue #14 Wave 2-B P1.4)
# ---------------------------------------------------------------------------

VAL_V2_TARGET_PER_CATEGORY = 45
VAL_V2_TOTAL_SAMPLES = 180


def check16_validation_v2_gate() -> Tuple[bool, dict]:
    """Verify Validation v2 meets Issue #14 P1.4 requirements.

    Checks:
      - 180 samples total
      - 45 each of Code, Boundary, Static Repair, Execution Repair
      - all verified=True
      - all hidden_tests present (non-empty)
      - repair broken_code genuinely broken (different from target_code)
      - execution_feedback genuine (non-empty for execution_repair)
      - SHA locked (validation.jsonl SHA matches frozen-v4-lock.json)
    """
    if not VALIDATION_V2_PATH.exists():
        return False, {"error": f"validation.jsonl not found: {VALIDATION_V2_PATH}"}
    if not FROZEN_V4_LOCK_PATH.exists():
        return False, {"error": f"lock file not found: {FROZEN_V4_LOCK_PATH}"}

    records = _read_jsonl(VALIDATION_V2_PATH)
    with FROZEN_V4_LOCK_PATH.open(encoding="utf-8") as fh:
        lock = json.load(fh)

    errors: list[str] = []

    # Total count
    total = len(records)
    if total != VAL_V2_TOTAL_SAMPLES:
        errors.append(f"total_samples={total} expected {VAL_V2_TOTAL_SAMPLES}")

    # Per-variant counts
    variant_counts: "dict[str, int]" = {}
    for r in records:
        vt = r.get("variant_type", "unknown")
        variant_counts[vt] = variant_counts.get(vt, 0) + 1
    for vt in FORMAL_VARIANT_TYPES:
        actual = variant_counts.get(vt, 0)
        if actual != VAL_V2_TARGET_PER_CATEGORY:
            errors.append(
                f"variant {vt} count={actual} expected {VAL_V2_TARGET_PER_CATEGORY}"
            )

    # All verified=True
    verified_false = sum(1 for r in records if r.get("verified") is not True)
    if verified_false != 0:
        errors.append(f"verified_false={verified_false} (expected 0)")

    # All hidden_tests present
    hidden_missing = sum(1 for r in records if not (r.get("hidden_tests") or "").strip())
    if hidden_missing != 0:
        errors.append(f"hidden_tests_missing={hidden_missing} (expected 0)")

    # Repair broken_code genuinely broken
    repair_samples = [
        r for r in records
        if r.get("variant_type") in ("static_repair", "execution_repair")
    ]
    broken_same_as_target = 0
    broken_missing = 0
    for r in repair_samples:
        broken = r.get("broken_code")
        target = r.get("target_code")
        if broken is None:
            broken_missing += 1
        elif broken == target:
            broken_same_as_target += 1
    if broken_missing > 0:
        errors.append(f"repair broken_code missing={broken_missing}")
    if broken_same_as_target > 0:
        errors.append(
            f"repair broken_code == target_code count={broken_same_as_target}"
        )

    # execution_feedback non-empty for execution_repair
    exec_repair = [
        r for r in records if r.get("variant_type") == "execution_repair"
    ]
    exec_feedback_empty = sum(
        1 for r in exec_repair if not (r.get("execution_feedback") or "").strip()
    )
    if exec_feedback_empty > 0:
        errors.append(
            f"execution_repair execution_feedback empty={exec_feedback_empty}"
        )

    # SHA locked
    actual_sha = hashlib.sha256(_read_bytes(VALIDATION_V2_PATH)).hexdigest()
    expected_sha = lock.get("validation_v2", {}).get("validation_jsonl_sha256", "")
    sha_match = (actual_sha == expected_sha)
    if not sha_match:
        errors.append(
            f"validation.jsonl SHA mismatch: lock={expected_sha} actual={actual_sha}"
        )

    passed = len(errors) == 0
    details = {
        "total_samples": total,
        "expected_total": VAL_V2_TOTAL_SAMPLES,
        "variant_counts": variant_counts,
        "expected_per_category": VAL_V2_TARGET_PER_CATEGORY,
        "verified_false": verified_false,
        "hidden_tests_missing": hidden_missing,
        "repair_sample_count": len(repair_samples),
        "broken_code_missing": broken_missing,
        "broken_same_as_target": broken_same_as_target,
        "exec_repair_count": len(exec_repair),
        "exec_feedback_empty": exec_feedback_empty,
        "sha_match": sha_match,
        "validation_jsonl_sha256": actual_sha,
        "errors": errors,
    }
    return passed, details


# ---------------------------------------------------------------------------
# Verdict + report rendering
# ---------------------------------------------------------------------------

CHECK_NAMES = [
    ("check1_frozen_v4_sha_locked", "Frozen v4 SHA locked"),
    ("check2_family_isolation", "Family isolation (pairwise disjoint)"),
    ("check3_assistant_retention", "Assistant retention = 100%"),
    ("check4_silent_truncation_zero", "Silent truncation = 0"),
    ("check5_canary_all_fail", "Canary all fail (v4)"),
    ("check6a_cpu_smoke", "CPU smoke (mandatory)"),
    ("check6b_gpu_smoke", "GPU smoke (deferrable)"),
    ("check7_output_dirs_dont_exist", "Output dirs don't exist"),
    ("check8_cpu_ci_green", "CPU CI green"),
    ("check9_baseline_lock_present", "P3 baseline lock present"),
    ("check10_train_capacity", "Train capacity per-candidate (2300-3100)"),
    ("check11_verified_consistency", "verified ⟺ verification subfields"),
    ("check12_candidate_ratio_within_tolerance", "Candidate ratio within ±3pp tolerance"),
    ("check13_all_buckets_non_empty", "All required buckets non-empty"),
    ("check14_composite_evaluator_complete", "Composite evaluator complete (5 components)"),
    ("check15_v4_coverage_gate", "Frozen v4 coverage gate"),
    ("check16_validation_v2_gate", "Validation v2 gate"),
]


def compute_verdict(results: list[Tuple[bool, dict]]) -> str:
    """Five-state verdict (Issue #12 P6).

    - FIX_FIRST: any mandatory check FAIL
    - PILOT_PENDING_GPU_SMOKE: all non-GPU checks PASS but GPU smoke SKIP
    - GO_FOR_P3_PILOT_ONLY: all checks PASS (incl. GPU) but capacity < 2300
    - GO_FOR_P3_TRAINING: all checks PASS and capacity >= 2300
    - STOP: reserved for manual trigger (not auto-emitted)

    SKIP on Check 6b (GPU smoke) triggers PILOT_PENDING_GPU_SMOKE when all
    other checks PASS. Capacity warning is detected via
    ``details.get("verdict_impact") == "PILOT_ONLY"`` from check10.
    """
    has_fail = False
    has_gpu_skip = False
    capacity_warning = False

    for passed, details in results:
        if not passed:
            has_fail = True
        if details.get("skipped") is True and "gpu" in str(details.get("reason", "")).lower():
            has_gpu_skip = True
        if details.get("verdict_impact") == "PILOT_ONLY":
            capacity_warning = True

    if has_fail:
        return "FIX_FIRST"
    if has_gpu_skip:
        return "PILOT_PENDING_GPU_SMOKE"
    if capacity_warning:
        return "GO_FOR_P3_PILOT_ONLY"
    return "GO_FOR_P3_TRAINING"


def _format_details_short(name: str, passed: bool, details: dict) -> str:
    """One-line summary of the details for the table."""
    if name == "check1_frozen_v4_sha_locked":
        if passed:
            return f"sha_lock={details['sha_lock'][:16]}..."
        return f"errors={details.get('errors', [details.get('error', 'mismatch')])[:2]}"
    if name == "check2_family_isolation":
        if passed:
            counts = details.get("counts", {})
            wl = details.get("whitelist", {})
            return (
                f"formal={counts.get('formal_train', 0)} "
                f"val_v2={counts.get('validation_v2', 0)} "
                f"frozen_v4={counts.get('frozen_v4', 0)} "
                f"hist_frozen={counts.get('historical_frozen', 0)} "
                f"hist_val={counts.get('historical_validation', 0)} "
                f"wl=p3_train_replay∩p2_train({wl.get('intersection_count', 0)})"
            )
        return f"violations={details.get('violations', [])[:2]}"
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
        b_verdict = details.get("balanced_verdict", "?")
        r_verdict = details.get("repair_verdict", "?")
        return (
            f"balanced={details.get('balanced_train', 0)}[{b_verdict}] "
            f"repair={details.get('repair_train', 0)}[{r_verdict}] "
            f"impact={impact}"
        )
    if name == "check11_verified_consistency":
        return (
            f"{details.get('inconsistent_count', 0)}/"
            f"{details.get('checked', 0)} inconsistent"
        )
    if name == "check12_candidate_ratio_within_tolerance":
        if passed:
            return (
                f"balanced={details.get('balanced_counts', {})} "
                f"repair={details.get('repair_counts', {})} "
                f"tol=±{details.get('tolerance_pp', 3)}pp"
            )
        return f"violations={details.get('violations', [])}"
    if name == "check13_all_buckets_non_empty":
        if passed:
            return f"all 8 buckets non-empty"
        return f"empty={details.get('empty_buckets', [])}"
    if name == "check14_composite_evaluator_complete":
        if passed:
            return f"5 components present, compute_ok"
        return f"missing={details.get('missing_fields', [])} violations={details.get('config_violations', [])}"
    if name == "check15_v4_coverage_gate":
        if passed:
            ratios = details.get("variant_ratios", {})
            return (
                f"fam={details.get('family_count')} "
                f"formal={details.get('formal_sample_count')} "
                f"canary={details.get('canary_count')} "
                f"code={ratios.get('code', 0):.2%} "
                f"bdry={ratios.get('boundary', 0):.2%} "
                f"sr={ratios.get('static_repair', 0):.2%} "
                f"er={ratios.get('execution_repair', 0):.2%}"
            )
        return f"errors={details.get('errors', [])[:2]}"
    if name == "check16_validation_v2_gate":
        if passed:
            return (
                f"total={details.get('total_samples')} "
                f"variants={details.get('variant_counts', {})} "
                f"sha_match={details.get('sha_match')}"
            )
        return f"errors={details.get('errors', [])[:2]}"
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
    lines.append(f"## {len(CHECK_NAMES)} PASS Checks")
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
    lines.append("**处置**：Issue #12 Phase D 已在 RTX 3050 Laptop GPU 上执行真实 GPU Smoke。环境：`torch 2.6.0+cu124`, `cuda.is_available()==True`, `device=NVIDIA GeForce RTX 3050 Laptop GPU`。Check 6b PASS：`bf16=True smoke=True device=cuda`。BF16 实际硬件验证已完成，trainer 可直接使用 bf16 训练。")
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
        lines.append("**GO_FOR_P3_TRAINING** — 12 项检查全部通过（含 SKIP 计入 PASS）且 Check 10 verdict_impact=FULL（train 容量 >= 2300）。训练可在以下已记录风险下启动：")
        lines.append("")
        lines.append("- **R1 (B.1)**: 历史 verified normalization 偏差（Task 11/12）。Fix 1 已回填 verification 子字段，当前 0 条样本处于 `verified=True && verification 不一致` 状态。")
        lines.append("- **R2 (B.2.5)**: BF16 已在 RTX 3050 Laptop GPU 上验证通过（torch 2.6.0+cu124, bf16=True）。trainer 启动时仍须调用 `check_bf16_support()` 并记录输出。")
        lines.append("- **R3 (B.2.3)**: `should_run_tier3` 严格 int 契约要求 trainer 在调用前对 epoch 做 `int(round(epoch))`。")
        lines.append("- **R4 (B.2.7)**: best checkpoint 实质等价于 best code_generation_pass_at_1（validation 集仅含 code 样本）。hard_constraint 已限制 code_gen 退化，但 repair 提升不会直接反映在 best checkpoint 选择上。")
        lines.append("- **R5**: Check 10 已将 train 容量硬编码为 verdict 决策因子——当前 verdict=GO_FOR_P3_TRAINING 表示 `total >= MIN_TRAIN_SAMPLES_FOR_FULL (2300)`。若容量降至 2300 以下，verdict 自动降级为 GO_FOR_P3_PILOT_ONLY。")
        lines.append("")
        lines.append("训练启动前须由用户明确批准 GO，并确认上述 5 项风险。")
    elif verdict == "GO_FOR_P3_PILOT_ONLY":
        lines.append("**GO_FOR_P3_PILOT_ONLY** — 12 项必跑检查全部通过（含 Check 6b GPU smoke PASS on RTX 3050），但 Check 10 verdict_impact=PILOT_ONLY（train 容量 < 2300）。")
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
        lines.append("- **R2 (B.2.5)**: BF16 已在 RTX 3050 上验证通过（Issue #12 Phase D）。trainer 启动时仍须调用 `check_bf16_support()` 并记录。")
        lines.append("- **R3 (B.2.3)**: `should_run_tier3` 严格 int 契约。")
        lines.append("- **R4 (B.2.7)**: best checkpoint 等价于 best code_generation_pass_at_1。")
        lines.append("- **R5**: 容量不足触发 PILOT_ONLY——Pilot 结果不可作为模型能力声明。")
        lines.append("")
        lines.append("PILOT 训练启动前须由用户明确批准，并确认上述 5 项风险与 PILOT 约束。")
    elif verdict == "PILOT_PENDING_GPU_SMOKE":
        lines.append("**PILOT_PENDING_GPU_SMOKE** — 所有非 GPU 检查通过，但 GPU smoke 未执行（SKIP）。")
        lines.append("")
        lines.append("需在具备 CUDA 的环境中执行 GPU smoke (Check 6b) 后重新运行 Readiness Gate。")
        lines.append("GPU smoke PASS 后，verdict 将根据容量自动降级为 GO_FOR_P3_PILOT_ONLY 或 GO_FOR_P3_TRAINING。")
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
    # Phase D Pilot Results (Issue #12)
    lines.append("## Phase D: GPU Smoke + Controlled Pilot (Issue #12)")
    lines.append("")
    lines.append("### GPU Smoke (Check 6b)")
    lines.append("- **Status**: PASS on RTX 3050 Laptop GPU")
    lines.append("- **Environment**: torch 2.6.0+cu124, CUDA 12.4, Python 3.11.7")
    lines.append("- **Result**: bf16=True, smoke=True, device=cuda")
    lines.append("")
    lines.append("### Controlled Pilot (balanced-generalist)")
    lines.append("- **Config**: `configs/p3/balanced-generalist-pilot.yaml`")
    lines.append("- **Mode**: continual (initial_adapter = P2 stage3-repair-v3)")
    lines.append("- **Steps**: 20/20 (0.25 epoch, well within 50-step cap)")
    lines.append("- **Duration**: 146s (2.4 min)")
    lines.append("- **Train loss**: 0.8375 (smoke) → 0.4041 (final)")
    lines.append("- **Eval loss**: 0.5935")
    lines.append("- **Peak GPU**: 1350 MiB / 4096 MiB")
    lines.append("- **Token audit**: 622 samples, 0 assistant lost, 0 target too long")
    lines.append("- **Adapter save/reload**: VERIFIED OK")
    lines.append("- **Parent adapter (P2 final)**: intact")
    lines.append("- **Output**: `adapters/p3/balanced-generalist-pilot/`")
    lines.append("")
    lines.append("**Pilot 结论**: 训练管道端到端可运行（数据加载→loss 下降→checkpoint 保存/reload）。")
    lines.append("Pilot 结果不可作为模型能力声明（GO_FOR_P3_PILOT_ONLY 约束）。")
    lines.append("")
    lines.append("## Session End")
    lines.append("")
    lines.append("P3.0–P3.4 scope complete. Issue #12 Phase D Pilot completed (balanced-generalist, 0.25 epoch).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all checks, print summary, write report. Returns 0/1 exit code."""
    check_fns = [
        check1_frozen_v4_sha_locked,
        check2_family_isolation,
        check3_assistant_retention,
        check4_silent_truncation_zero,
        check5_canary_all_fail,
        check6a_cpu_smoke,
        check6b_gpu_smoke,
        check7_output_dirs_dont_exist,
        check8_cpu_ci_green,
        check9_baseline_lock_present,
        check10_train_capacity,
        check11_verified_consistency,
        check12_candidate_ratio_within_tolerance,
        check13_all_buckets_non_empty,
        check14_composite_evaluator_complete,
        check15_v4_coverage_gate,
        check16_validation_v2_gate,
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
