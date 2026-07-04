"""tests/test_p3_readiness_gate.py -- Task 14 Readiness Gate tests.

11 tests covering the 9 PASS checks + verdict logic:
  1. test_check1_frozen_v3_sha_locked
  2. test_check2_pairwise_disjoint
  3. test_check3_assistant_retention
  4. test_check4_silent_truncation_zero
  5. test_check5_canary_all_fail
  6. test_check6_gpu_smoke
  7. test_check7_output_dirs_dont_exist
  8. test_check8_cpu_ci_green
  9. test_check9_baseline_lock_present
 10. test_compute_verdict_go
 11. test_compute_verdict_fix_first

Per .superpowers/sdd/task-14-brief.md Part E.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# Make scripts/ importable as a package path
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Import the gate module (file named p3_readiness_gate.py in scripts/)
import p3_readiness_gate as gate  # noqa: E402


# ---------------------------------------------------------------------------
# Check 1: Frozen v3 SHA locked
# ---------------------------------------------------------------------------

def test_check1_frozen_v3_sha_locked():
    passed, details = gate.check1_frozen_v3_sha_locked()
    assert passed is True, f"frozen v3 SHA lock mismatch: {details}"
    assert details["sha_lock"] == gate.EXPECTED_SHA_LOCK
    assert details["recomputed"] == details["sha_lock"]
    assert len(details["sha_lock"]) == 64


# ---------------------------------------------------------------------------
# Check 2: Pairwise disjoint
# ---------------------------------------------------------------------------

def test_check2_pairwise_disjoint():
    passed, details = gate.check2_pairwise_disjoint()
    assert passed is True, f"pairwise disjoint failed: {details}"
    # Snapshot counts from progress.md / Task 9
    assert details["frozen_v3"] == 100
    assert details["p3_validation"] == 90
    assert details["p3_train"] == 219
    assert details["p3_train_replay"] == 206
    assert ("p3_train_replay", "p2_train") in details["whitelist"]


# ---------------------------------------------------------------------------
# Check 3: Assistant retention = 100%
# ---------------------------------------------------------------------------

def test_check3_assistant_retention():
    passed, details = gate.check3_assistant_retention()
    assert passed is True, f"assistant retention failed: {details}"
    # 626 + 493 = 1119 total
    assert details["checked"] == 1119
    assert details["retained"] == 1119


# ---------------------------------------------------------------------------
# Check 4: Silent truncation = 0
# ---------------------------------------------------------------------------

def test_check4_silent_truncation_zero():
    passed, details = gate.check4_silent_truncation_zero()
    assert passed is True, f"silent truncation check failed: {details}"
    # PASS criterion: 0 real silent truncations (assistant target cut)
    assert details["real_silent_truncations"] == 0
    assert details["checked"] == 1119
    # potential_truncations is informational; we expect a small number of
    # boundary samples to exceed 384 chars//4 = 384 token estimate, but the
    # assistant target itself is never cut under preserve_assistant policy.
    assert details["policy"] == "preserve_assistant"
    assert details["max_seq_length"] == 384


# ---------------------------------------------------------------------------
# Check 5: Canary all fail
# ---------------------------------------------------------------------------

def test_check5_canary_all_fail():
    passed, details = gate.check5_canary_all_fail()
    assert passed is True, f"canary check failed: {details}"
    assert details["canary_count"] == 100
    assert details["all_failed"] == 100
    assert details["verified_true"] == 0


# ---------------------------------------------------------------------------
# Check 6: GPU smoke (SKIP if no GPU -- passed must still be True)
# ---------------------------------------------------------------------------

def test_check6_gpu_smoke():
    passed, details = gate.check6_gpu_smoke()
    # SKIP counts as PASS per brief Part D
    assert passed is True, f"GPU smoke check failed: {details}"
    if details.get("skipped"):
        assert "reason" in details
    else:
        # If GPU was actually available, the smoke test must have passed
        assert details.get("smoke_passed") is True
        assert "bf16_supported" in details


# ---------------------------------------------------------------------------
# Check 7: Output dirs don't exist
# ---------------------------------------------------------------------------

def test_check7_output_dirs_dont_exist():
    passed, details = gate.check7_output_dirs_dont_exist()
    assert passed is True, f"output dirs check failed: {details}"
    assert details["none_exist"] is True
    assert details["existing"] == []
    # Both expected paths must be in the checked list
    checked_str = " ".join(details["checked"])
    assert "balanced-generalist" in checked_str
    assert "repair-specialist" in checked_str


# ---------------------------------------------------------------------------
# Check 8: CPU CI green (P3 subset)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_check8_cpu_ci_green():
    """Runs the P3-specific pytest subset. May take 1-2 minutes."""
    passed, details = gate.check8_cpu_ci_green(timeout=600)
    assert passed is True, f"CPU CI failed: {details}"
    assert details["failed"] == 0
    assert details["passed"] > 0
    assert details["returncode"] == 0


# ---------------------------------------------------------------------------
# Check 9: Baseline lock present
# ---------------------------------------------------------------------------

def test_check9_baseline_lock_present():
    passed, details = gate.check9_baseline_lock_present()
    assert passed is True, f"baseline lock check failed: {details}"
    assert details["all_fields_present"] is True
    assert set(details["models"]) == set(gate.EXPECTED_BASELINE_MODELS)
    assert len(details["models"]) == 3


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def test_compute_verdict_go():
    """All PASS (with one SKIP) -> GO_FOR_P3_TRAINING."""
    results = [
        (True, {"sha_lock": "x" * 64}),
        (True, {"frozen_v3": 100}),
        (True, {"retained": 1119}),
        (True, {"real_silent_truncations": 0}),
        (True, {"all_failed": 100}),
        (True, {"skipped": True, "reason": "CUDA not available"}),
        (True, {"none_exist": True}),
        (True, {"passed": 70, "failed": 0}),
        (True, {"models": list(gate.EXPECTED_BASELINE_MODELS)}),
    ]
    assert gate.compute_verdict(results) == "GO_FOR_P3_TRAINING"


def test_compute_verdict_fix_first():
    """Any FAIL -> FIX_FIRST."""
    results = [
        (True, {}),
        (False, {"overlap": "frozen_v3 ∩ p3_train"}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {"skipped": True}),
        (True, {}),
        (True, {}),
        (True, {}),
    ]
    assert gate.compute_verdict(results) == "FIX_FIRST"
