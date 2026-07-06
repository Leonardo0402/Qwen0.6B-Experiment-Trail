"""tests/test_build_validation_v2.py -- Tests for Validation v2 builder.

Verifies the 4-category validation set (code/boundary/static_repair/
execution_repair, 45 each = 180 total) produced by
``scripts/build_validation_v2.py``.

Tests:
  1.  test_total_count -- validation.jsonl has exactly 180 samples
  2.  test_category_counts -- 4 categories x 45 each
  3.  test_all_verified -- all samples have verified=True
  4.  test_no_duplicate_sample_ids -- no duplicate sample_ids
  5.  test_disjoint_from_p3_train -- family_ids disjoint from p3_train
  6.  test_disjoint_from_frozen_v3 -- family_ids disjoint from frozen_v3
  7.  test_variant_type_set -- no sample with variant_type=None
  8.  test_manifest_consistency -- manifest counts + SHA256 match files
  9.  test_families_json_consistency -- families.json matches validation.jsonl
  10. test_repair_samples_have_broken_code -- static/exec repair samples
      have non-empty broken_code and execution_feedback (exec only)

These tests use the REAL data files (data/p3-curriculum/validation-v2/*)
produced by ``scripts/build_validation_v2.py``. If the output files do
not exist yet, the tests are SKIPPED with a clear instruction to run
the builder first.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = _ROOT / "data" / "p3-curriculum" / "validation-v2"
VALIDATION_PATH = OUTPUT_DIR / "validation.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
FAMILIES_PATH = OUTPUT_DIR / "families.json"
REJECTED_PATH = OUTPUT_DIR / "rejected.jsonl"

PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
FROZEN_V3_FAMILIES_PATH = _ROOT / "data" / "frozen-eval" / "v3" / "families.json"
SCRIPT_PATH = _ROOT / "scripts" / "build_validation_v2.py"

TOTAL_TARGET = 180  # 4 categories x 45
TARGET_PER_CATEGORY = 45
CATEGORIES = ("code", "boundary", "static_repair", "execution_repair")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_samples() -> list:
    """Ensure validation.jsonl exists and return a list of Sample objects."""
    if not VALIDATION_PATH.exists():
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        if result.returncode != 0:
            pytest.fail(
                f"build_validation_v2.py failed (exit "
                f"{result.returncode}):\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    if not VALIDATION_PATH.exists():
        pytest.skip("validation.jsonl not produced by build_validation_v2.py")
    samples: list = []
    with VALIDATION_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(Sample.from_json_line(line))
    return samples


@pytest.fixture(scope="module")
def manifest() -> dict:
    """Load the manifest.json produced by the builder."""
    if not MANIFEST_PATH.exists():
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        if result.returncode != 0:
            pytest.fail(
                f"build_validation_v2.py failed (exit "
                f"{result.returncode}):\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest.json not produced by build_validation_v2.py")
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def families_json() -> dict:
    """Load the families.json produced by the builder."""
    if not FAMILIES_PATH.exists():
        pytest.skip("families.json not produced by build_validation_v2.py")
    with FAMILIES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Test 1: total count
# ---------------------------------------------------------------------------

def test_total_count(generated_samples):
    """validation.jsonl has exactly 180 samples."""
    count = len(generated_samples)
    assert count == TOTAL_TARGET, (
        f"validation count {count} != {TOTAL_TARGET}"
    )


# ---------------------------------------------------------------------------
# Test 2: category counts (4 x 45)
# ---------------------------------------------------------------------------

def test_category_counts(generated_samples):
    """Each of the 4 categories has exactly 45 samples."""
    counts = {c: 0 for c in CATEGORIES}
    for s in generated_samples:
        vt = s.variant_type
        if vt in counts:
            counts[vt] += 1
    for c in CATEGORIES:
        assert counts[c] == TARGET_PER_CATEGORY, (
            f"category {c} count {counts[c]} != {TARGET_PER_CATEGORY}"
        )


# ---------------------------------------------------------------------------
# Test 3: all verified=True
# ---------------------------------------------------------------------------

def test_all_verified(generated_samples):
    """All samples have verified=True."""
    unverified = [s.sample_id for s in generated_samples if not s.verified]
    assert not unverified, (
        f"{len(unverified)} samples with verified=False "
        f"(first 5: {unverified[:5]})"
    )


# ---------------------------------------------------------------------------
# Test 4: no duplicate sample_ids
# ---------------------------------------------------------------------------

def test_no_duplicate_sample_ids(generated_samples):
    """No duplicate sample_ids in validation.jsonl."""
    ids = [s.sample_id for s in generated_samples]
    seen: set = set()
    duplicates: list = []
    for sid in ids:
        if sid in seen and sid not in duplicates:
            duplicates.append(sid)
        seen.add(sid)
    assert not duplicates, (
        f"{len(duplicates)} duplicate sample_ids "
        f"(first 5: {duplicates[:5]})"
    )


# ---------------------------------------------------------------------------
# Test 5: disjoint from p3_train
# ---------------------------------------------------------------------------

def test_disjoint_from_p3_train(generated_samples):
    """Validation family_ids are disjoint from p3_train family_ids."""
    if not PARTITION_PATH.exists():
        pytest.skip("family-partition.json missing")
    with PARTITION_PATH.open(encoding="utf-8") as fh:
        partition = json.load(fh)
    p3_train_fids = set(partition["p3_train_new"]["family_ids"]) | set(
        partition["p3_train_replay"]["family_ids"]
    )
    val_fids = {s.family_id for s in generated_samples}
    overlap = val_fids & p3_train_fids
    assert not overlap, (
        f"{len(overlap)} family_ids in both validation and p3_train "
        f"(first 5: {sorted(overlap)[:5]})"
    )


# ---------------------------------------------------------------------------
# Test 6: disjoint from frozen_v3
# ---------------------------------------------------------------------------

def test_disjoint_from_frozen_v3(generated_samples):
    """Validation family_ids are disjoint from frozen_v3 family_ids."""
    if not FROZEN_V3_FAMILIES_PATH.exists():
        pytest.skip("frozen-eval/v3/families.json missing")
    with FROZEN_V3_FAMILIES_PATH.open(encoding="utf-8") as fh:
        frozen_v3_data = json.load(fh)
    frozen_v3_fids = set(frozen_v3_data["families"])
    val_fids = {s.family_id for s in generated_samples}
    overlap = val_fids & frozen_v3_fids
    assert not overlap, (
        f"{len(overlap)} frozen_v3 family_ids in validation "
        f"(first 5: {sorted(overlap)[:5]})"
    )


# ---------------------------------------------------------------------------
# Test 7: all variant_type set (not None)
# ---------------------------------------------------------------------------

def test_variant_type_set(generated_samples):
    """No sample has variant_type=None."""
    none_variants = [s.sample_id for s in generated_samples if s.variant_type is None]
    assert not none_variants, (
        f"{len(none_variants)} samples with variant_type=None "
        f"(first 5: {none_variants[:5]})"
    )


# ---------------------------------------------------------------------------
# Test 8: manifest consistency
# ---------------------------------------------------------------------------

def _compute_sha256(path: Path) -> str:
    """Read entire file as bytes and compute SHA256 hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_consistency(generated_samples, manifest):
    """Manifest counts and SHA256 match the actual files."""
    # Total sample count matches
    assert manifest["total_samples"] == len(generated_samples), (
        f"manifest total_samples {manifest['total_samples']} != "
        f"actual {len(generated_samples)}"
    )
    # Validation count matches
    assert manifest["validation"]["count"] == len(generated_samples), (
        f"manifest validation count {manifest['validation']['count']} != "
        f"actual {len(generated_samples)}"
    )
    # Variant distribution matches actual
    vd = manifest["variant_distribution"]
    actual_vd = {c: 0 for c in CATEGORIES}
    for s in generated_samples:
        vt = s.variant_type
        if vt in actual_vd:
            actual_vd[vt] += 1
    assert vd == actual_vd, (
        f"manifest variant_distribution {vd} != actual {actual_vd}"
    )
    # Variant distribution sums to total
    assert sum(vd.values()) == len(generated_samples), (
        f"variant_distribution sum {sum(vd.values())} != "
        f"total {len(generated_samples)}"
    )
    # SHA256 of validation.jsonl matches recomputed
    actual_sha = _compute_sha256(VALIDATION_PATH)
    assert manifest["validation"]["sha256"] == actual_sha, (
        f"manifest validation sha256 {manifest['validation']['sha256']} != "
        f"actual {actual_sha}"
    )
    # Family count matches actual
    actual_family_count = len({s.family_id for s in generated_samples})
    assert manifest["families"]["validation_family_count"] == actual_family_count, (
        f"manifest validation_family_count "
        f"{manifest['families']['validation_family_count']} != "
        f"actual {actual_family_count}"
    )
    # target_per_category recorded
    assert manifest["target_per_category"] == TARGET_PER_CATEGORY, (
        f"manifest target_per_category {manifest['target_per_category']} != "
        f"{TARGET_PER_CATEGORY}"
    )


# ---------------------------------------------------------------------------
# Test 9: families.json consistency
# ---------------------------------------------------------------------------

def test_families_json_consistency(generated_samples, families_json):
    """families.json matches validation.jsonl family_ids."""
    actual_fids = sorted({s.family_id for s in generated_samples})
    assert families_json["validation_family_count"] == len(actual_fids), (
        f"families.json validation_family_count "
        f"{families_json['validation_family_count']} != "
        f"actual {len(actual_fids)}"
    )
    assert families_json["validation_family_ids"] == actual_fids, (
        f"families.json validation_family_ids do not match actual: "
        f"missing={sorted(set(actual_fids) - set(families_json['validation_family_ids']))[:5]} "
        f"extra={sorted(set(families_json['validation_family_ids']) - set(actual_fids))[:5]}"
    )
    # Shared with both candidates
    shared_with = families_json.get("shared_with", [])
    assert "balanced_generalist" in shared_with, (
        f"balanced_generalist not in shared_with: {shared_with}"
    )
    assert "repair_specialist" in shared_with, (
        f"repair_specialist not in shared_with: {shared_with}"
    )
    # Disjoint_from includes p3_train and frozen_v3
    disjoint_from = families_json.get("disjoint_from", [])
    assert "p3_train" in disjoint_from, (
        f"p3_train not in disjoint_from: {disjoint_from}"
    )
    assert "frozen_v3" in disjoint_from, (
        f"frozen_v3 not in disjoint_from: {disjoint_from}"
    )


# ---------------------------------------------------------------------------
# Test 10: repair samples have broken_code (and execution_feedback for exec)
# ---------------------------------------------------------------------------

def test_repair_samples_have_broken_code(generated_samples):
    """static_repair and execution_repair samples have non-empty broken_code.

    execution_repair samples must also have non-empty execution_feedback.
    """
    for s in generated_samples:
        if s.variant_type == "static_repair":
            assert s.broken_code and s.broken_code.strip(), (
                f"static_repair sample {s.sample_id} has empty broken_code"
            )
        elif s.variant_type == "execution_repair":
            assert s.broken_code and s.broken_code.strip(), (
                f"execution_repair sample {s.sample_id} has empty broken_code"
            )
            assert s.execution_feedback and s.execution_feedback.strip(), (
                f"execution_repair sample {s.sample_id} has empty "
                f"execution_feedback"
            )
