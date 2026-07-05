"""tests/test_p3_readiness_v4.py -- Readiness Gate v4 tests (Issue #14 Wave 2-B).

Covers the new v4-specific checks added to scripts/p3_readiness_gate.py:
  - check1_frozen_v4_sha_locked: lock file format + SHA consistency
  - check2_family_isolation: pairwise disjoint with explicit whitelist
  - check15_v4_coverage_gate: families/samples/ratios/canary/repair coverage
  - check16_validation_v2_gate: 180 samples, 45/45/45/45, verified, SHA locked

Also covers the frozen-v4-lock.json format and SHA consistency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import p3_readiness_gate as gate  # noqa: E402


LOCK_PATH = _ROOT / "reports" / "p3" / "frozen-v4-lock.json"
V4_DIR = _ROOT / "data" / "frozen-eval" / "v4"
VAL_V2_DIR = _ROOT / "data" / "p3-curriculum" / "validation-v2"


# ---------------------------------------------------------------------------
# Skip if v4 / validation-v2 / lock not present
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not V4_DIR.exists() or not VAL_V2_DIR.exists() or not LOCK_PATH.exists(),
    reason="Frozen v4 / validation-v2 / lock file not present",
)


# ---------------------------------------------------------------------------
# P1.1: frozen-v4-lock.json format
# ---------------------------------------------------------------------------

class TestFrozenV4LockFormat:
    """Issue #14 P1.1: lock file must record all required fields."""

    def _load_lock(self) -> dict:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def test_lock_file_exists(self) -> None:
        assert LOCK_PATH.exists(), f"lock file missing: {LOCK_PATH}"

    def test_lock_has_required_top_level_fields(self) -> None:
        lock = self._load_lock()
        required = [
            "schema_version", "frozen_version",
            "manifest_sha256", "test_raw_sha256", "families_sha256",
            "rejected_sha256", "sha_lock",
            "formal_sample_count", "canary_count", "family_count",
            "variant_distribution", "source_commit", "created_at",
        ]
        for f in required:
            assert f in lock, f"missing field: {f}"

    def test_frozen_version_is_v4(self) -> None:
        lock = self._load_lock()
        assert lock["frozen_version"] == "v4"

    def test_all_shas_are_64_hex(self) -> None:
        lock = self._load_lock()
        for key in ("manifest_sha256", "test_raw_sha256", "families_sha256",
                    "rejected_sha256", "sha_lock"):
            val = lock[key]
            assert isinstance(val, str) and len(val) == 64, (
                f"{key} not 64 hex chars: {val}"
            )
            int(val, 16)  # raises if not hex

    def test_counts_match_distribution(self) -> None:
        lock = self._load_lock()
        vd = lock["variant_distribution"]
        formal = sum(v for k, v in vd.items()
                     if k not in ("canary", "canary_repair"))
        canary = vd.get("canary", 0) + vd.get("canary_repair", 0)
        assert lock["formal_sample_count"] == formal
        assert lock["canary_count"] == canary

    def test_source_commit_is_sha1(self) -> None:
        lock = self._load_lock()
        commit = lock["source_commit"]
        assert isinstance(commit, str) and len(commit) == 40, (
            f"source_commit not 40 chars: {commit}"
        )
        int(commit, 16)

    def test_validation_v2_section_present(self) -> None:
        lock = self._load_lock()
        v2 = lock.get("validation_v2", {})
        assert "validation_jsonl_sha256" in v2
        assert "sample_count" in v2
        assert v2["sample_count"] == 180
        assert len(v2["validation_jsonl_sha256"]) == 64


# ---------------------------------------------------------------------------
# P1.2: SHA consistency (lock file vs actual file recomputation)
# ---------------------------------------------------------------------------

class TestShaConsistency:
    """Issue #14 P1.1: SHA values in lock file must match actual files."""

    def _sha(self, path: Path) -> str:
        import hashlib
        return hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()

    def test_manifest_sha_matches(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        actual = self._sha(V4_DIR / "manifest.json")
        assert lock["manifest_sha256"] == actual

    def test_test_raw_sha_matches(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        actual = self._sha(V4_DIR / "test_raw.jsonl")
        assert lock["test_raw_sha256"] == actual

    def test_families_sha_matches(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        actual = self._sha(V4_DIR / "families.json")
        assert lock["families_sha256"] == actual

    def test_rejected_sha_matches(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        actual = self._sha(V4_DIR / "rejected.jsonl")
        assert lock["rejected_sha256"] == actual

    def test_combined_sha_lock_matches(self) -> None:
        import hashlib
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        h = hashlib.sha256()
        for name in ("families.json", "test_raw.jsonl", "rejected.jsonl"):
            h.update((V4_DIR / name).read_bytes().replace(b"\r\n", b"\n"))
        assert lock["sha_lock"] == h.hexdigest()

    def test_validation_v2_sha_matches(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        actual = self._sha(VAL_V2_DIR / "validation.jsonl")
        assert lock["validation_v2"]["validation_jsonl_sha256"] == actual

    def test_manifest_shas_match_actual_manifest(self) -> None:
        """Manifest's test_raw/families/rejected SHAs must match actual files."""
        manifest = json.loads((V4_DIR / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["test_raw_sha256"] == self._sha(V4_DIR / "test_raw.jsonl")
        assert manifest["families_sha256"] == self._sha(V4_DIR / "families.json")
        assert manifest["rejected_sha256"] == self._sha(V4_DIR / "rejected.jsonl")


# ---------------------------------------------------------------------------
# Check 1: check1_frozen_v4_sha_locked
# ---------------------------------------------------------------------------

class TestCheck1FrozenV4ShaLocked:
    """Verify the gate's check1 function returns PASS with consistent SHAs."""

    def test_check1_passes(self) -> None:
        passed, details = gate.check1_frozen_v4_sha_locked()
        assert passed is True, f"check1 failed: {details}"

    def test_check1_returns_sha_lock(self) -> None:
        passed, details = gate.check1_frozen_v4_sha_locked()
        assert passed is True
        assert "sha_lock" in details
        assert len(details["sha_lock"]) == 64

    def test_check1_returns_all_four_shas(self) -> None:
        passed, details = gate.check1_frozen_v4_sha_locked()
        assert passed is True
        for key in ("manifest_sha256", "test_raw_sha256",
                    "families_sha256", "rejected_sha256"):
            assert key in details
            assert len(details[key]) == 64

    def test_check1_sha_lock_matches_recomputed(self) -> None:
        passed, details = gate.check1_frozen_v4_sha_locked()
        assert passed is True
        assert details["sha_lock"] == details["recomputed_lock"]


# ---------------------------------------------------------------------------
# Check 2: check2_family_isolation (pairwise disjoint + whitelist)
# ---------------------------------------------------------------------------

class TestCheck2FamilyIsolation:
    """Verify the family isolation gate enforces pairwise disjointness."""

    def test_check2_passes(self) -> None:
        passed, details = gate.check2_family_isolation()
        assert passed is True, f"check2 failed: {details}"

    def test_no_pairwise_violations(self) -> None:
        passed, details = gate.check2_family_isolation()
        assert passed is True
        assert details["violations"] == []

    def test_main_set_counts_present(self) -> None:
        passed, details = gate.check2_family_isolation()
        assert passed is True
        counts = details["counts"]
        for key in ("formal_train", "validation_v2", "frozen_v4",
                    "historical_frozen", "historical_validation"):
            assert key in counts
            assert counts[key] > 0

    def test_frozen_v4_count_is_100(self) -> None:
        passed, details = gate.check2_family_isolation()
        assert passed is True
        assert details["counts"]["frozen_v4"] == 100

    def test_validation_v2_count_is_45(self) -> None:
        passed, details = gate.check2_family_isolation()
        assert passed is True
        assert details["counts"]["validation_v2"] == 45

    def test_whitelist_explicit_p3_replay_p2_train(self) -> None:
        """Whitelist must be exactly (p3_train_replay, p2_train)."""
        passed, details = gate.check2_family_isolation()
        assert passed is True
        wl = details["whitelist"]
        assert wl["pair"] == ("p3_train_replay", "p2_train")
        assert wl["all_replay_in_p2_train"] is True
        assert wl["formal_train_p2_overlap_clean"] is True

    def test_whitelist_intersection_equals_replay_count(self) -> None:
        """All p3_train_replay families must be in p2_train (full whitelist)."""
        passed, details = gate.check2_family_isolation()
        assert passed is True
        wl = details["whitelist"]
        assert wl["replay_count"] == wl["intersection_count"]
        assert wl["replay_count"] > 0

    def test_frozen_v4_disjoint_from_historical_frozen(self) -> None:
        """v4 must not share families with v1/v3/p2-frozen-v2."""
        sets = gate._load_all_family_sets()
        assert not (sets["frozen_v4"] & sets["historical_frozen"])

    def test_frozen_v4_disjoint_from_formal_train(self) -> None:
        sets = gate._load_all_family_sets()
        assert not (sets["frozen_v4"] & sets["formal_train"])

    def test_frozen_v4_disjoint_from_validation_v2(self) -> None:
        sets = gate._load_all_family_sets()
        assert not (sets["frozen_v4"] & sets["validation_v2"])

    def test_validation_v2_disjoint_from_formal_train(self) -> None:
        sets = gate._load_all_family_sets()
        assert not (sets["validation_v2"] & sets["formal_train"])

    def test_validation_v2_disjoint_from_historical_frozen(self) -> None:
        sets = gate._load_all_family_sets()
        assert not (sets["validation_v2"] & sets["historical_frozen"])

    def test_formal_train_disjoint_from_historical_frozen(self) -> None:
        """formal_train (p3_train_new + p3_train_replay) must not overlap
        with any historical frozen set — EXCEPT via the p2_train whitelist
        (p3_train_replay comes from p2_train which is separate)."""
        sets = gate._load_all_family_sets()
        assert not (sets["formal_train"] & sets["historical_frozen"])


# ---------------------------------------------------------------------------
# Check 15: check15_v4_coverage_gate
# ---------------------------------------------------------------------------

class TestCheck15V4CoverageGate:
    """Verify the v4 coverage gate enforces all P1.2 requirements."""

    def test_check15_passes(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True, f"check15 failed: {details}"

    def test_family_count_in_range(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        assert 80 <= details["family_count"] <= 100

    def test_formal_sample_count_in_range(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        assert 360 <= details["formal_sample_count"] <= 700

    def test_code_ratio_in_range(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        r = details["variant_ratios"]["code"]
        assert 0.25 <= r <= 0.30

    def test_boundary_ratio_in_range(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        r = details["variant_ratios"]["boundary"]
        assert 0.15 <= r <= 0.20

    def test_static_repair_ratio_in_range(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        r = details["variant_ratios"]["static_repair"]
        assert 0.25 <= r <= 0.30

    def test_execution_repair_ratio_in_range(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        r = details["variant_ratios"]["execution_repair"]
        assert 0.25 <= r <= 0.30

    def test_canary_excluded_from_formal_denominator(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        # formal + canary must equal total records
        total = details["formal_sample_count"] + details["canary_count"]
        # variant_counts sum equals formal_sample_count
        assert sum(details["variant_counts"].values()) == details["formal_sample_count"]

    def test_all_canaries_verified_false(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        assert details["canary_verified_true"] == 0

    def test_all_formal_verified_true(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        assert details["formal_verified_false"] == 0

    def test_repair_broken_code_differs_from_target(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        assert details["broken_same_as_target"] == 0
        assert details["broken_code_missing"] == 0

    def test_execution_feedback_non_empty(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        assert details["exec_feedback_empty"] == 0

    def test_no_errors(self) -> None:
        passed, details = gate.check15_v4_coverage_gate()
        assert passed is True
        assert details["errors"] == []


# ---------------------------------------------------------------------------
# Check 16: check16_validation_v2_gate
# ---------------------------------------------------------------------------

class TestCheck16ValidationV2Gate:
    """Verify the validation-v2 gate enforces all P1.4 requirements."""

    def test_check16_passes(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True, f"check16 failed: {details}"

    def test_total_samples_is_180(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True
        assert details["total_samples"] == 180

    def test_each_variant_has_45(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True
        vc = details["variant_counts"]
        assert vc.get("code") == 45
        assert vc.get("boundary") == 45
        assert vc.get("static_repair") == 45
        assert vc.get("execution_repair") == 45

    def test_all_verified_true(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True
        assert details["verified_false"] == 0

    def test_all_hidden_tests_present(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True
        assert details["hidden_tests_missing"] == 0

    def test_repair_broken_code_genuinely_broken(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True
        assert details["broken_same_as_target"] == 0
        assert details["broken_code_missing"] == 0

    def test_execution_feedback_genuine(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True
        assert details["exec_feedback_empty"] == 0

    def test_sha_locked(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True
        assert details["sha_match"] is True
        assert len(details["validation_jsonl_sha256"]) == 64

    def test_no_errors(self) -> None:
        passed, details = gate.check16_validation_v2_gate()
        assert passed is True
        assert details["errors"] == []


# ---------------------------------------------------------------------------
# Integration: compute_verdict with new checks
# ---------------------------------------------------------------------------

class TestVerdictIntegration:
    """Verify compute_verdict still works with the expanded check list."""

    def test_all_pass_returns_go_for_training(self) -> None:
        """17 PASS results (GPU smoke PASS, no SKIP) + FULL capacity -> GO_FOR_P3_TRAINING."""
        results = [(True, {}) for _ in range(17)]
        # Check 6b GPU smoke must PASS (not SKIP) to avoid PILOT_PENDING_GPU_SMOKE
        results[6] = (True, {"smoke_passed": True, "bf16_supported": True})
        results[10] = (True, {"verdict_impact": "FULL"})
        assert gate.compute_verdict(results) == "GO_FOR_P3_TRAINING"

    def test_any_new_check_fail_returns_fix_first(self) -> None:
        """If check15 or check16 fails, verdict must be FIX_FIRST."""
        results = [(True, {}) for _ in range(17)]
        results[14] = (False, {"errors": ["v4 coverage failure"]})
        assert gate.compute_verdict(results) == "FIX_FIRST"

        results2 = [(True, {}) for _ in range(17)]
        results2[15] = (False, {"errors": ["validation v2 failure"]})
        assert gate.compute_verdict(results2) == "FIX_FIRST"
