"""tests/test_p3_readiness_gate.py -- Readiness Gate tests (Fix 2+5+6).

Tests cover:
  - 9 original PASS checks (Check 6 split into 6a CPU + 6b GPU)
  - Check 10: train sample capacity vs 2300-3100 threshold (Fix 2)
  - Three-state compute_verdict: GO_FOR_P3_TRAINING / GO_FOR_P3_PILOT_ONLY / FIX_FIRST (Fix 6)

Per .superpowers/sdd/task-14-brief.md Part E + Issue #10 Fix 2+5+6.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    # Issue #12 rebuild: 622 (balanced) + 490 (repair) = 1112 total
    assert details["checked"] == 1112
    assert details["retained"] == 1112


# ---------------------------------------------------------------------------
# Check 4: Silent truncation = 0
# ---------------------------------------------------------------------------

def test_check4_silent_truncation_zero():
    passed, details = gate.check4_silent_truncation_zero()
    assert passed is True, f"silent truncation check failed: {details}"
    # PASS criterion: 0 real silent truncations (assistant target cut)
    assert details["real_silent_truncations"] == 0
    assert details["checked"] == 1112
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
# Check 6a: CPU smoke (mandatory, no SKIP)
# ---------------------------------------------------------------------------

def test_check6a_cpu_smoke_pass_when_python_healthy():
    """CPU smoke must PASS in a healthy Python environment."""
    passed, details = gate.check6a_cpu_smoke()
    assert passed is True, f"CPU smoke failed: {details}"
    assert details.get("skipped") is False
    assert details.get("smoke_passed") is True


def test_check6a_cpu_smoke_fail_on_exception():
    """When an exception is raised during CPU smoke, FAIL."""
    with patch("builtins.sum", side_effect=RuntimeError("simulated failure")):
        passed, details = gate.check6a_cpu_smoke()
    assert passed is False
    assert "error" in details
    assert "simulated failure" in details["error"]


def test_check6a_skip_not_allowed():
    """6a must never set skipped=True — CPU smoke is mandatory."""
    passed, details = gate.check6a_cpu_smoke()
    assert details.get("skipped") is False, "check6a must not skip"


# ---------------------------------------------------------------------------
# Check 6b: GPU smoke (SKIP allowed when no CUDA)
# ---------------------------------------------------------------------------

def test_check6b_gpu_smoke_skip_when_no_cuda():
    """SKIP when CUDA not available (default CPU-only env)."""
    passed, details = gate.check6b_gpu_smoke()
    if details.get("skipped"):
        assert passed is True
        assert "reason" in details
    else:
        # If CUDA is actually available, smoke must pass
        assert passed is True
        assert details.get("smoke_passed") is True


def test_check6b_gpu_smoke_pass_when_cuda_available_and_smoke_ok():
    """Mock CUDA available and verify smoke passes with bf16 recorded."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.__version__ = "2.0.0+mock"
    fake_torch.device.return_value = MagicMock()

    with patch.dict("sys.modules", {"torch": fake_torch}), \
         patch(
             "src.p3_checkpoint_evaluator.check_bf16_support",
             return_value=(True, "BF16 supported"),
         ):
        passed, details = gate.check6b_gpu_smoke()

    assert passed is True
    assert details.get("skipped") is False
    assert details.get("smoke_passed") is True
    assert details.get("bf16_supported") is True
    assert details.get("bf16_message") == "BF16 supported"


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
# Check 10: Train sample capacity vs 2300-3100 threshold (Fix 2)
# ---------------------------------------------------------------------------

def test_check10_capacity_pass_when_above_min(tmp_path):
    """Issue #12: BOTH candidates >= MIN -> PASS, verdict_impact=FULL."""
    balanced = tmp_path / "balanced.jsonl"
    repair = tmp_path / "repair.jsonl"
    balanced.write_text("\n".join('{"x":1}' for _ in range(2500)) + "\n", encoding="utf-8")
    repair.write_text("\n".join('{"x":1}' for _ in range(2400)) + "\n", encoding="utf-8")

    passed, details = gate.check10_train_capacity(balanced, repair)
    assert passed is True
    assert details["balanced_train"] == 2500
    assert details["repair_train"] == 2400
    assert details["total"] == 4900
    assert details["verdict_impact"] == "FULL"
    assert details["balanced_verdict"] == "FULL"
    assert details["repair_verdict"] == "FULL"
    assert details["per_candidate_check"] is True


def test_check10_capacity_warn_when_below_min(tmp_path):
    """Issue #12: BOTH candidates < MIN (but >0) -> PASS, verdict_impact=PILOT_ONLY."""
    balanced = tmp_path / "balanced.jsonl"
    repair = tmp_path / "repair.jsonl"
    balanced.write_text("\n".join('{"x":1}' for _ in range(500)) + "\n", encoding="utf-8")
    repair.write_text("\n".join('{"x":1}' for _ in range(400)) + "\n", encoding="utf-8")

    passed, details = gate.check10_train_capacity(balanced, repair)
    assert passed is True
    assert details["total"] == 900
    assert details["verdict_impact"] == "PILOT_ONLY"
    assert details["balanced_verdict"] == "PILOT_ONLY"
    assert details["repair_verdict"] == "PILOT_ONLY"


def test_check10_capacity_mixed_per_candidate(tmp_path):
    """Issue #12: one candidate >= MIN, other < MIN -> PILOT_ONLY."""
    balanced = tmp_path / "balanced.jsonl"
    repair = tmp_path / "repair.jsonl"
    balanced.write_text("\n".join('{"x":1}' for _ in range(2500)) + "\n", encoding="utf-8")
    repair.write_text("\n".join('{"x":1}' for _ in range(400)) + "\n", encoding="utf-8")

    passed, details = gate.check10_train_capacity(balanced, repair)
    assert passed is True
    assert details["verdict_impact"] == "PILOT_ONLY"
    assert details["balanced_verdict"] == "FULL"
    assert details["repair_verdict"] == "PILOT_ONLY"


def test_check10_capacity_fail_when_zero(tmp_path):
    """Any candidate == 0 -> FAIL, verdict_impact=FAIL."""
    balanced = tmp_path / "balanced_empty.jsonl"
    repair = tmp_path / "repair_empty.jsonl"
    balanced.write_text("", encoding="utf-8")
    repair.write_text("", encoding="utf-8")

    passed, details = gate.check10_train_capacity(balanced, repair)
    assert passed is False
    assert details["total"] == 0
    assert details["verdict_impact"] == "FAIL"


# ---------------------------------------------------------------------------
# Check 11: verified consistency (Issue #12)
# ---------------------------------------------------------------------------

def test_check11_verified_consistency_pass(tmp_path):
    """All samples consistent (verified=True with syntax_ok+pytest_ok)."""
    balanced = tmp_path / "balanced.jsonl"
    repair = tmp_path / "repair.jsonl"
    samples = [
        {"sample_id": "s1", "verified": True, "verification": {"syntax_ok": True, "pytest_ok": True}},
        {"sample_id": "s2", "verified": True, "verification": {"syntax_ok": True, "pytest_ok": True}},
    ]
    balanced.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    repair.write_text("", encoding="utf-8")

    passed, details = gate.check11_verified_consistency(balanced, repair)
    assert passed is True
    assert details["checked"] == 2
    assert details["inconsistent_count"] == 0


def test_check11_verified_consistency_fail(tmp_path):
    """verified=True but pytest_ok=False -> inconsistent."""
    balanced = tmp_path / "balanced.jsonl"
    repair = tmp_path / "repair.jsonl"
    samples = [
        {"sample_id": "s1", "verified": True, "verification": {"syntax_ok": True, "pytest_ok": True}},
        {"sample_id": "s2", "verified": True, "verification": {"syntax_ok": True, "pytest_ok": False}},
    ]
    balanced.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    repair.write_text("", encoding="utf-8")

    passed, details = gate.check11_verified_consistency(balanced, repair)
    assert passed is False
    assert details["checked"] == 2
    assert details["inconsistent_count"] == 1
    assert "s2" in details["inconsistent_sample_ids"]


# ---------------------------------------------------------------------------
# Check 12: Candidate ratio within tolerance (Issue #12 P6)
# ---------------------------------------------------------------------------

def test_check12_candidate_ratio_within_tolerance():
    """Both candidates' variant_type ratios must be within ±3pp tolerance."""
    passed, details = gate.check12_candidate_ratio_within_tolerance()
    assert passed is True, f"ratio tolerance check failed: {details.get('violations', [])}"
    assert details["tolerance_pp"] == 3
    assert "balanced_counts" in details
    assert "repair_counts" in details


# ---------------------------------------------------------------------------
# Check 13: All required buckets non-empty (Issue #12 P6)
# ---------------------------------------------------------------------------

def test_check13_all_buckets_non_empty():
    """All 4 variant_type buckets must be non-empty for both candidates."""
    passed, details = gate.check13_all_buckets_non_empty()
    assert passed is True, f"empty buckets: {details.get('empty_buckets', [])}"
    assert len(details["empty_buckets"]) == 0
    assert set(details["required_buckets"]) == {"code", "boundary", "static_repair", "execution_repair"}


# ---------------------------------------------------------------------------
# Check 14: Composite evaluator complete (Issue #12 P6)
# ---------------------------------------------------------------------------

def test_check14_composite_evaluator_complete():
    """CompositeScore must have all 5 components including hidden_pass_rate."""
    passed, details = gate.check14_composite_evaluator_complete()
    assert passed is True, f"composite evaluator incomplete: {details}"
    assert len(details["missing_fields"]) == 0
    assert len(details["config_violations"]) == 0
    assert details["compute_test"] is not None


# ---------------------------------------------------------------------------
# Verdict logic (Fix 6: three-state)
# ---------------------------------------------------------------------------

def test_compute_verdict_go():
    """All PASS (with one SKIP) and capacity FULL -> GO_FOR_P3_TRAINING."""
    results = [
        (True, {"sha_lock": "x" * 64}),
        (True, {"frozen_v3": 100}),
        (True, {"retained": 917}),
        (True, {"real_silent_truncations": 0}),
        (True, {"all_failed": 100}),
        (True, {"smoke_passed": True}),  # 6a
        (True, {"smoke_passed": True, "bf16_supported": True}),  # 6b PASS
        (True, {"none_exist": True}),
        (True, {"passed": 70, "failed": 0}),
        (True, {"models": list(gate.EXPECTED_BASELINE_MODELS)}),
        (True, {"total": 2400, "verdict_impact": "FULL"}),  # check10
    ]
    assert gate.compute_verdict(results) == "GO_FOR_P3_TRAINING"


def test_compute_verdict_full_when_no_fail_no_capacity_warning():
    """All PASS, no verdict_impact=PILOT_ONLY -> GO_FOR_P3_TRAINING."""
    results = [
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),  # 6a
        (True, {"smoke_passed": True, "bf16_supported": True}),  # 6b PASS
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {"total": 2400, "verdict_impact": "FULL"}),
    ]
    assert gate.compute_verdict(results) == "GO_FOR_P3_TRAINING"


def test_compute_verdict_pilot_when_capacity_warning():
    """All PASS but capacity warning -> GO_FOR_P3_PILOT_ONLY."""
    results = [
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),  # 6a
        (True, {"smoke_passed": True, "bf16_supported": True}),  # 6b PASS
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {"total": 917, "verdict_impact": "PILOT_ONLY"}),  # check10
    ]
    assert gate.compute_verdict(results) == "GO_FOR_P3_PILOT_ONLY"


def test_compute_verdict_fix_first():
    """Any FAIL -> FIX_FIRST (regression test)."""
    results = [
        (True, {}),
        (False, {"overlap": "frozen_v3 \u2229 p3_train"}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {"skipped": True}),
        (True, {}),
        (True, {}),
        (True, {}),
    ]
    assert gate.compute_verdict(results) == "FIX_FIRST"


def test_compute_verdict_pilot_pending_gpu_smoke():
    """All non-GPU checks PASS but GPU smoke SKIP -> PILOT_PENDING_GPU_SMOKE."""
    results = [
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),  # 6a
        (True, {"skipped": True, "reason": "GPU smoke: CUDA not available"}),  # 6b SKIP
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {"total": 917, "verdict_impact": "PILOT_ONLY"}),
    ]
    assert gate.compute_verdict(results) == "PILOT_PENDING_GPU_SMOKE"


def test_compute_verdict_pilot_pending_overrides_capacity_warning():
    """GPU smoke SKIP takes precedence over capacity warning."""
    results = [
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {"skipped": True, "reason": "GPU smoke: CUDA not available"}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {"total": 917, "verdict_impact": "PILOT_ONLY"}),
    ]
    # PILOT_PENDING_GPU_SMOKE, not GO_FOR_P3_PILOT_ONLY
    assert gate.compute_verdict(results) == "PILOT_PENDING_GPU_SMOKE"


def test_compute_verdict_fix_first_when_any_fail():
    """Any mandatory check FAIL -> FIX_FIRST (overrides capacity warning)."""
    results = [
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {}),
        (False, {"error": "CPU smoke failed"}),  # 6a FAIL (mandatory)
        (True, {"skipped": True}),  # 6b
        (True, {}),
        (True, {}),
        (True, {}),
        (True, {"total": 917, "verdict_impact": "PILOT_ONLY"}),
    ]
    assert gate.compute_verdict(results) == "FIX_FIRST"
