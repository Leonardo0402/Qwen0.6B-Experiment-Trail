"""tests/test_backfill_canonical_pool_verification.py -- Issue #10 Fix 1.

Verifies the canonical-pool verification backfill:

  1. test_backfill_preserves_sample_id_and_target_code -- backfill does not
     modify sample_id/target_code/public_tests (hidden_tests may change).
  2. test_backfill_sets_verification_subfields -- after backfill, no sample
     has the False-preset verification subfields; the backfill manifest exists
     and records the backfilled count.
  3. test_backfill_drops_unverifiable_samples -- canonical-pool.jsonl and
     train.jsonl invariant: verified == (syntax_ok AND pytest_ok AND NOT timeout).
     Samples that fail verification are NOT force-set to verified=True in
     train.jsonl.
  4. test_build_balanced_generalist_no_normalization_hack -- source-level check
     that the model_copy(update={"verified": True}) hack has been removed.
  5. test_build_repair_specialist_no_normalization_hack -- same for repair
     specialist builder.

These tests use the REAL data files produced by the backfill + build scripts.
If the backfill has not been run, the tests FAIL (RED) -- they cannot be
skipped because the backfill is a binding requirement (Issue #10 Fix 1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

POOL_PATH = (
    _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl"
)
POOL_BACKUP_PATH = (
    _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl.pre-backfill.bak"
)
BACKFILL_MANIFEST_PATH = (
    _ROOT / "data" / "p3-curriculum" / "canonical-pool-backfill-manifest.json"
)

BALANCED_GENERALIST_TRAIN = (
    _ROOT / "data" / "p3-curriculum" / "balanced-generalist" / "train.jsonl"
)
REPAIR_SPECIALIST_TRAIN = (
    _ROOT / "data" / "p3-curriculum" / "repair-specialist" / "train.jsonl"
)

BUILD_BALANCED_GENERALIST_SCRIPT = (
    _ROOT / "scripts" / "build_balanced_generalist.py"
)
BUILD_REPAIR_SPECIALIST_SCRIPT = (
    _ROOT / "scripts" / "build_repair_specialist.py"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_samples(path: Path) -> list:
    """Stream a JSONL file into a list of Sample objects."""
    samples: list = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(Sample.from_json_line(line))
    return samples


def _is_preset_verification(v) -> bool:
    """True iff *v* matches the placeholder Verification(syntax_ok=False,
    pytest_ok=False, ruff_ok=False, timeout=False) that P2-replay-derived
    samples shipped with before the backfill."""
    return (
        v.syntax_ok is False
        and v.pytest_ok is False
        and v.ruff_ok is False
        and v.timeout is False
    )


def _is_accepted(v) -> bool:
    """Mirror of SampleVerification.is_accepted:
    syntax_ok AND pytest_ok AND NOT timeout."""
    return bool(v.syntax_ok and v.pytest_ok and not v.timeout)


# ---------------------------------------------------------------------------
# Test 1: backfill preserves sample_id / target_code / public_tests
# ---------------------------------------------------------------------------

def test_backfill_preserves_sample_id_and_target_code():
    """Backfill must not modify sample_id, target_code, or public_tests.

    Compares the post-backfill canonical-pool.jsonl against the pre-backfill
    backup file (canonical-pool.jsonl.pre-backfill.bak) created by the
    backfill script on its first run. hidden_tests is allowed to change
    (pad_hidden_tests may extend it).
    """
    if not POOL_BACKUP_PATH.exists():
        pytest.fail(
            f"Pre-backfill backup not found: {POOL_BACKUP_PATH}. "
            f"Run scripts/backfill_canonical_pool_verification.py first."
        )
    if not POOL_PATH.exists():
        pytest.fail(f"canonical-pool.jsonl not found: {POOL_PATH}")

    before_samples = {s.sample_id: s for s in _load_samples(POOL_BACKUP_PATH)}
    after_samples = {s.sample_id: s for s in _load_samples(POOL_PATH)}

    # Same set of sample_ids
    assert set(before_samples.keys()) == set(after_samples.keys()), (
        f"sample_id set mismatch: before={len(before_samples)} "
        f"after={len(after_samples)}; "
        f"missing_in_after={sorted(set(before_samples) - set(after_samples))[:5]} "
        f"extra_in_after={sorted(set(after_samples) - set(before_samples))[:5]}"
    )

    # For each sample, target_code and public_tests must match exactly
    mismatches: list = []
    for sid, before_s in before_samples.items():
        after_s = after_samples[sid]
        if before_s.target_code != after_s.target_code:
            mismatches.append((sid, "target_code differs"))
        if before_s.public_tests != after_s.public_tests:
            mismatches.append((sid, "public_tests differs"))

    assert not mismatches, (
        f"{len(mismatches)} samples have modified target_code/public_tests: "
        f"first 5={mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 2: backfill sets real verification subfields
# ---------------------------------------------------------------------------

def test_backfill_sets_verification_subfields():
    """After backfill, no sample has the False-preset verification subfields
    when verified=True (the contradictory state introduced by the old hack).

    Also verifies the backfill manifest exists with the expected structure
    and records a non-zero backfilled_count.
    """
    if not BACKFILL_MANIFEST_PATH.exists():
        pytest.fail(
            f"Backfill manifest not found: {BACKFILL_MANIFEST_PATH}. "
            f"Run scripts/backfill_canonical_pool_verification.py first."
        )

    with BACKFILL_MANIFEST_PATH.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    # Manifest must have expected fields
    assert "backfilled_count" in manifest, (
        "manifest missing 'backfilled_count' field"
    )
    assert "by_variant_type" in manifest, (
        "manifest missing 'by_variant_type' field"
    )
    assert "timestamp" in manifest, (
        "manifest missing 'timestamp' field"
    )
    assert manifest["backfilled_count"] > 0, (
        f"backfilled_count should be > 0, got {manifest['backfilled_count']}"
    )

    # No sample in canonical-pool.jsonl should have the preset verification
    # (False-preset) -- after backfill, all 501 P2-replay samples should have
    # real verify_sample results (syntax_ok=True for all of them, since their
    # target_code was already syntax-valid).
    samples = _load_samples(POOL_PATH)
    preset_samples: list = []
    for s in samples:
        if _is_preset_verification(s.verification):
            preset_samples.append(s.sample_id)

    assert not preset_samples, (
        f"{len(preset_samples)} samples in canonical-pool.jsonl still have "
        f"the False-preset verification subfields: first 5="
        f"{preset_samples[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 3: unverifiable samples have verified=False (not force-set True)
# ---------------------------------------------------------------------------

def test_backfill_drops_unverifiable_samples():
    """After backfill, samples that fail verification have verified=False in
    canonical-pool.jsonl AND are not force-set to verified=True in train.jsonl.

    Invariant checked: for every sample, verified == is_accepted(verification)
    where is_accepted = syntax_ok AND pytest_ok AND NOT timeout.
    """
    # Check canonical-pool.jsonl invariant
    pool_samples = _load_samples(POOL_PATH)
    pool_inconsistent: list = []
    for s in pool_samples:
        if s.verified != _is_accepted(s.verification):
            pool_inconsistent.append(
                (s.sample_id, s.verified, s.verification.model_dump())
            )

    assert not pool_inconsistent, (
        f"{len(pool_inconsistent)} samples in canonical-pool.jsonl have "
        f"verified != is_accepted(verification): first 5="
        f"{pool_inconsistent[:5]}"
    )

    # Check train.jsonl invariant (no force-set verified=True via hack)
    for train_path in (BALANCED_GENERALIST_TRAIN, REPAIR_SPECIALIST_TRAIN):
        if not train_path.exists():
            pytest.fail(f"train.jsonl not found: {train_path}")
        train_samples = _load_samples(train_path)
        train_inconsistent: list = []
        for s in train_samples:
            if s.verified != _is_accepted(s.verification):
                train_inconsistent.append(
                    (s.sample_id, s.verified, s.verification.model_dump())
                )
        assert not train_inconsistent, (
            f"{train_path.name}: {len(train_inconsistent)} train samples have "
            f"verified != is_accepted(verification) -- the verified=True hack "
            f"is still active: first 5={train_inconsistent[:5]}"
        )


# ---------------------------------------------------------------------------
# Test 4: build_balanced_generalist.py source has no normalization hack
# ---------------------------------------------------------------------------

def test_build_balanced_generalist_no_normalization_hack():
    """build_balanced_generalist.py source must NOT contain
    model_copy(update={"verified": True}) hack."""
    src = BUILD_BALANCED_GENERALIST_SCRIPT.read_text(encoding="utf-8")
    assert 'model_copy(update={"verified": True})' not in src, (
        "build_balanced_generalist.py still contains the "
        'model_copy(update={"verified": True}) normalization hack'
    )


# ---------------------------------------------------------------------------
# Test 5: build_repair_specialist.py source has no normalization hack
# ---------------------------------------------------------------------------

def test_build_repair_specialist_no_normalization_hack():
    """build_repair_specialist.py source must NOT contain
    model_copy(update={"verified": True}) hack."""
    src = BUILD_REPAIR_SPECIALIST_SCRIPT.read_text(encoding="utf-8")
    assert 'model_copy(update={"verified": True})' not in src, (
        "build_repair_specialist.py still contains the "
        'model_copy(update={"verified": True}) normalization hack'
    )
