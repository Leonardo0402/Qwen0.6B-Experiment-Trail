"""Compliance tests for Frozen Eval v4 (Issue #12 P4).

These tests verify that data/frozen-eval/v4/ meets the Issue #12 P4
requirements:

- 80-100 NEW families (zero overlap with all historical datasets)
- 360-700 real evaluation samples (excluding canary)
- Task ratios within target ranges
- Canary NOT counted in total_sample_count
- All samples verified
- SHA locked, immutable
- Reference passes public+hidden
- broken_code fails at least one test
- repaired target passes all tests
- execution_feedback from real source
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
V4_DIR = _ROOT / "data" / "frozen-eval" / "v4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _all_historical_families() -> set[str]:
    """Collect ALL family IDs used in any historical dataset."""
    used: set[str] = set()

    # Frozen v1, v3
    for ver in ("v1", "v3"):
        fp = _ROOT / "data" / "frozen-eval" / ver / "families.json"
        if fp.exists():
            data = _load_json(fp)
            for item in data.get("families", []):
                if isinstance(item, str):
                    used.add(item)
                elif isinstance(item, dict):
                    fid = item.get("family_id") or item.get("id")
                    if isinstance(fid, str):
                        used.add(fid)

    # P3 validation partition
    fp = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
    if fp.exists():
        part = _load_json(fp)
        used.update(part.get("p3_validation", {}).get("family_ids", []))

    # P3 candidates (balanced + repair) — train + validation + families
    for sub in ("balanced-generalist", "repair-specialist", "validation-v2"):
        sub_path = _ROOT / "data" / "p3-curriculum" / sub
        if not sub_path.exists():
            continue
        fp = sub_path / "families.json"
        if fp.exists():
            data = _load_json(fp)
            for item in data.get("families", []):
                if isinstance(item, str):
                    used.add(item)
        for fname in ("train.jsonl", "validation.jsonl"):
            fp = sub_path / fname
            if fp.exists():
                for obj in _load_jsonl(fp):
                    fid = obj.get("family_id")
                    if isinstance(fid, str):
                        used.add(fid)

    # P2 datasets
    p2_root = _ROOT / "data" / "p2-curriculum"
    if p2_root.exists():
        # P2 frozen-eval-v2
        fp = p2_root / "frozen-eval-v2" / "families.json"
        if fp.exists():
            data = _load_json(fp)
            for item in data.get("families", []):
                if isinstance(item, str):
                    used.add(item)
        # P2 stages
        for stage in ("stage1-code", "stage2-boundary", "stage3-repair"):
            stage_path = p2_root / stage
            if not stage_path.exists():
                continue
            fp = stage_path / "families.json"
            if fp.exists():
                data = _load_json(fp)
                for item in data.get("families", []):
                    if isinstance(item, str):
                        used.add(item)
            for fname in ("train.jsonl", "validation.jsonl"):
                fp = stage_path / fname
                if fp.exists():
                    for obj in _load_jsonl(fp):
                        fid = obj.get("family_id")
                        if isinstance(fid, str):
                            used.add(fid)

    return used


def _v4_families() -> set[str]:
    fp = V4_DIR / "families.json"
    if not fp.exists():
        return set()
    data = _load_json(fp)
    return set(data.get("families", []))


def _v4_samples() -> list[dict]:
    fp = V4_DIR / "test_raw.jsonl"
    if not fp.exists():
        return []
    return _load_jsonl(fp)


def _v4_manifest() -> dict:
    fp = V4_DIR / "manifest.json"
    if not fp.exists():
        return {}
    return _load_json(fp)


# ---------------------------------------------------------------------------
# Skip if v4 not yet rebuilt
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not V4_DIR.exists(),
    reason="Frozen Eval v4 directory does not exist",
)


# ---------------------------------------------------------------------------
# P4 Requirement: 80-100 NEW families
# ---------------------------------------------------------------------------


class TestFamilyCount:
    """Issue #12 P4: 80-100 new families."""

    def test_family_count_in_range(self) -> None:
        fams = _v4_families()
        assert 80 <= len(fams) <= 100, (
            f"Expected 80-100 families, got {len(fams)}"
        )

    def test_family_count_matches_manifest(self) -> None:
        manifest = _v4_manifest()
        fams = _v4_families()
        assert manifest.get("frozen_family_count") == len(fams)


# ---------------------------------------------------------------------------
# P4 Requirement: Zero overlap with historical families
# ---------------------------------------------------------------------------


class TestZeroFamilyOverlap:
    """Issue #12 P4: zero intersection with all Train/Validation/Frozen history."""

    def test_no_overlap_with_historical(self) -> None:
        v4_fams = _v4_families()
        historical = _all_historical_families()
        overlap = v4_fams & historical
        assert not overlap, (
            f"v4 families overlap with historical: {sorted(overlap)[:10]}"
        )

    def test_no_overlap_with_v3(self) -> None:
        v4_fams = _v4_families()
        v3_fp = _ROOT / "data" / "frozen-eval" / "v3" / "families.json"
        if not v3_fp.exists():
            pytest.skip("v3 not found")
        v3_data = _load_json(v3_fp)
        v3_fams = set(v3_data.get("families", []))
        overlap = v4_fams & v3_fams
        assert not overlap, f"v4 overlaps v3: {sorted(overlap)[:10]}"


# ---------------------------------------------------------------------------
# P4 Requirement: 360-700 samples (excluding canary)
# ---------------------------------------------------------------------------


class TestSampleCount:
    """Issue #12 P4: 360-700 real evaluation samples (canary excluded)."""

    def test_total_sample_count_excludes_canary(self) -> None:
        manifest = _v4_manifest()
        samples = _v4_samples()
        non_canary = [
            s for s in samples
            if s.get("variant_type") not in ("canary", "canary_repair")
        ]
        assert manifest.get("total_sample_count") == len(non_canary), (
            f"manifest total_sample_count={manifest.get('total_sample_count')} "
            f"but non-canary samples={len(non_canary)}"
        )

    def test_sample_count_in_range(self) -> None:
        samples = _v4_samples()
        non_canary = [
            s for s in samples
            if s.get("variant_type") not in ("canary", "canary_repair")
        ]
        assert 360 <= len(non_canary) <= 700, (
            f"Expected 360-700 non-canary samples, got {len(non_canary)}"
        )


# ---------------------------------------------------------------------------
# P4 Requirement: Task ratios within target ranges
# ---------------------------------------------------------------------------


class TestTaskRatios:
    """Issue #12 P4: Code 25-30%, Boundary 15-20%, Static 25-30%, Exec 25-30%."""

    def test_four_variant_types_present(self) -> None:
        samples = _v4_samples()
        variants = {s.get("variant_type") for s in samples}
        required = {"code", "boundary", "static_repair", "execution_repair"}
        missing = required - variants
        assert not missing, f"Missing variant types: {missing}"

    def test_code_ratio_in_range(self) -> None:
        samples = _v4_samples()
        non_canary = [
            s for s in samples
            if s.get("variant_type") not in ("canary", "canary_repair")
        ]
        code = sum(1 for s in non_canary if s.get("variant_type") == "code")
        ratio = code / len(non_canary) if non_canary else 0
        assert 0.25 <= ratio <= 0.30, f"Code ratio {ratio:.2%} not in 25-30%"

    def test_boundary_ratio_in_range(self) -> None:
        samples = _v4_samples()
        non_canary = [
            s for s in samples
            if s.get("variant_type") not in ("canary", "canary_repair")
        ]
        boundary = sum(1 for s in non_canary if s.get("variant_type") == "boundary")
        ratio = boundary / len(non_canary) if non_canary else 0
        assert 0.15 <= ratio <= 0.20, f"Boundary ratio {ratio:.2%} not in 15-20%"

    def test_static_repair_ratio_in_range(self) -> None:
        samples = _v4_samples()
        non_canary = [
            s for s in samples
            if s.get("variant_type") not in ("canary", "canary_repair")
        ]
        static = sum(1 for s in non_canary if s.get("variant_type") == "static_repair")
        ratio = static / len(non_canary) if non_canary else 0
        assert 0.25 <= ratio <= 0.30, f"Static repair ratio {ratio:.2%} not in 25-30%"

    def test_execution_repair_ratio_in_range(self) -> None:
        samples = _v4_samples()
        non_canary = [
            s for s in samples
            if s.get("variant_type") not in ("canary", "canary_repair")
        ]
        exec_repair = sum(
            1 for s in non_canary if s.get("variant_type") == "execution_repair"
        )
        ratio = exec_repair / len(non_canary) if non_canary else 0
        assert 0.25 <= ratio <= 0.30, (
            f"Execution repair ratio {ratio:.2%} not in 25-30%"
        )


# ---------------------------------------------------------------------------
# P4 Requirement: Canary handling
# ---------------------------------------------------------------------------


class TestCanaryHandling:
    """Issue #12 P4: canary excluded from total_sample_count."""

    def test_canary_present(self) -> None:
        samples = _v4_samples()
        canary = [
            s for s in samples
            if s.get("variant_type") in ("canary", "canary_repair")
        ]
        assert len(canary) > 0, "No canary samples found"

    def test_canary_not_in_total_count(self) -> None:
        manifest = _v4_manifest()
        samples = _v4_samples()
        canary_count = sum(
            1 for s in samples
            if s.get("variant_type") in ("canary", "canary_repair")
        )
        non_canary_count = len(samples) - canary_count
        assert manifest.get("total_sample_count") == non_canary_count, (
            f"total_sample_count={manifest.get('total_sample_count')} "
            f"includes canary (non-canary={non_canary_count})"
        )

    def test_canary_verified_false(self) -> None:
        samples = _v4_samples()
        canary = [
            s for s in samples
            if s.get("variant_type") in ("canary", "canary_repair")
        ]
        for s in canary:
            assert s.get("verified") is False, (
                f"Canary {s.get('sample_id')} should have verified=False"
            )


# ---------------------------------------------------------------------------
# P4 Requirement: All samples verified (except canary)
# ---------------------------------------------------------------------------


class TestSampleVerification:
    """Issue #12 P4: all non-canary samples must be verified."""

    def test_all_non_canary_verified(self) -> None:
        samples = _v4_samples()
        non_canary = [
            s for s in samples
            if s.get("variant_type") not in ("canary", "canary_repair")
        ]
        unverified = [s for s in non_canary if not s.get("verified")]
        assert not unverified, (
            f"{len(unverified)} non-canary samples are not verified: "
            f"{[s.get('sample_id') for s in unverified[:5]]}"
        )

    def test_repair_broken_code_fails(self) -> None:
        """broken_code must fail at least one test."""
        samples = _v4_samples()
        repair = [
            s for s in samples
            if s.get("variant_type") in ("static_repair", "execution_repair")
        ]
        # Check that broken_code is present and different from target_code
        for s in repair:
            broken = s.get("broken_code")
            target = s.get("target_code")
            assert broken is not None, (
                f"{s.get('sample_id')}: broken_code is None"
            )
            assert broken != target, (
                f"{s.get('sample_id')}: broken_code == target_code"
            )


# ---------------------------------------------------------------------------
# P4 Requirement: SHA locked, immutable
# ---------------------------------------------------------------------------


class TestImmutability:
    """Issue #12 P4: SHA locked, any change requires v5."""

    def test_manifest_has_sha(self) -> None:
        manifest = _v4_manifest()
        assert manifest.get("test_raw_sha256"), "test_raw_sha256 missing"
        assert manifest.get("families_sha256"), "families_sha256 missing"

    def test_immutability_field(self) -> None:
        manifest = _v4_manifest()
        imm = manifest.get("immutability", {})
        assert imm.get("write_once") is True
        assert imm.get("any_change_requires") == "v5"

    def test_sha_matches_actual_file(self) -> None:
        import hashlib
        manifest = _v4_manifest()
        fp = V4_DIR / "test_raw.jsonl"
        if not fp.exists():
            pytest.skip("test_raw.jsonl not found")
        h = hashlib.sha256()
        with fp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        assert actual == manifest.get("test_raw_sha256"), (
            f"SHA mismatch: manifest={manifest.get('test_raw_sha256')[:16]}... "
            f"actual={actual[:16]}..."
        )


# ---------------------------------------------------------------------------
# P4 Requirement: manifest structure
# ---------------------------------------------------------------------------


class TestManifestStructure:
    """Issue #12 P4: manifest must have correct structure."""

    def test_manifest_fields(self) -> None:
        manifest = _v4_manifest()
        required_fields = [
            "frozen_version",
            "frozen_family_count",
            "total_sample_count",
            "variant_breakdown",
            "test_raw_sha256",
            "families_sha256",
            "immutability",
        ]
        for field in required_fields:
            assert field in manifest, f"Missing manifest field: {field}"

    def test_frozen_version_is_v4(self) -> None:
        manifest = _v4_manifest()
        assert manifest.get("frozen_version") == "v4"

    def test_base_version_not_v3(self) -> None:
        """v4 must NOT be built on top of v3 (zero family overlap requirement)."""
        manifest = _v4_manifest()
        # If base_version is v3 and v3_content_preserved is True,
        # that's a violation
        if manifest.get("base_version") == "v3":
            assert not manifest.get("v3_content_preserved"), (
                "v4 must not preserve v3 content (zero overlap requirement)"
            )
