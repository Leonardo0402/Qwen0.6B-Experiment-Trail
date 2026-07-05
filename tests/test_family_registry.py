"""tests/test_family_registry.py -- Tests for the Family Registry (Task 6).

Covers the 10 tests specified in ``.superpowers/sdd/task-6-brief.md``:

  1.  test_family_entry_is_used_empty_returns_false
  2.  test_family_entry_is_used_nonempty_returns_true
  3.  test_claim_idempotent
  4.  test_registry_claim_adds_tag
  5.  test_registry_claim_unknown_family_raises
  6.  test_families_with_usage_filters_correctly
  7.  test_assert_pairwise_disjoint_passes
  8.  test_assert_pairwise_disjoint_fails_on_overlap
  9.  test_assert_pairwise_disjoint_whitelist_allows_overlap
  10. test_builder_correctness

Tests 1-9 use synthetic in-memory registries (no file I/O on real data).
Test 10 invokes the builder CLI on small synthetic input files in
``tmp_path`` and asserts the produced registry file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.family_registry import FamilyEntry, FamilyRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    family_id: str = "mbpp_fam_1",
    *,
    usage: list[str] | None = None,
    source_split: str = "train",
) -> FamilyEntry:
    """Build a minimal FamilyEntry for tests."""
    return FamilyEntry(
        family_id=family_id,
        source_task_id=family_id.replace("mbpp_fam_", "mbpp_"),
        source_split=source_split,
        usage=list(usage) if usage is not None else [],
    )


# ---------------------------------------------------------------------------
# Tests 1-3: FamilyEntry
# ---------------------------------------------------------------------------

def test_family_entry_is_used_empty_returns_false():
    """Empty usage -> is_used() is False."""
    e = _entry()
    assert e.is_used() is False


def test_family_entry_is_used_nonempty_returns_true():
    """Non-empty usage -> is_used() is True."""
    e = _entry(usage=["p2_train"])
    assert e.is_used() is True


def test_claim_idempotent():
    """Re-claiming the same tag does not duplicate the entry."""
    e = _entry()
    e.claim("p3_train")
    e.claim("p3_train")
    assert e.usage.count("p3_train") == 1
    assert len(e.usage) == 1


# ---------------------------------------------------------------------------
# Tests 4-5: FamilyRegistry.claim
# ---------------------------------------------------------------------------

def _simple_registry() -> FamilyRegistry:
    """A registry with one known family."""
    reg = FamilyRegistry()
    reg.families["mbpp_fam_42"] = _entry("mbpp_fam_42")
    return reg


def test_registry_claim_adds_tag():
    """registry.claim adds the tag to the family's usage list."""
    reg = _simple_registry()
    reg.claim("mbpp_fam_42", "frozen_v3_candidate")
    assert reg.families["mbpp_fam_42"].has_usage("frozen_v3_candidate")


def test_registry_claim_unknown_family_raises():
    """Claiming a tag for a non-existent family_id raises KeyError."""
    reg = _simple_registry()
    with pytest.raises(KeyError):
        reg.claim("nonexistent", "x")


# ---------------------------------------------------------------------------
# Test 6: families_with_usage
# ---------------------------------------------------------------------------

def test_families_with_usage_filters_correctly():
    """Returns only the families whose usage contains the queried tag."""
    reg = FamilyRegistry()
    reg.families["a"] = _entry("a", usage=["p2_train"])
    reg.families["b"] = _entry("b", usage=["quarantine"], source_split="test")
    reg.families["c"] = _entry("c", usage=[], source_split="validation")
    result = reg.families_with_usage("quarantine")
    assert result == ["b"]


# ---------------------------------------------------------------------------
# Tests 7-9: assert_pairwise_disjoint
# ---------------------------------------------------------------------------

def test_assert_pairwise_disjoint_passes():
    """Three disjoint usage sets -> no exception."""
    reg = FamilyRegistry()
    reg.families["a"] = _entry("a", usage=["p3_train"])
    reg.families["b"] = _entry("b", usage=["p3_validation"], source_split="test")
    reg.families["c"] = _entry("c", usage=["frozen_v3"], source_split="validation")
    # No overlaps across the three sets -> no exception.
    reg.assert_pairwise_disjoint(["p3_train", "p3_validation", "frozen_v3"])


def test_assert_pairwise_disjoint_fails_on_overlap():
    """A family shared between p3_train and p3_validation -> AssertionError."""
    reg = FamilyRegistry()
    reg.families["a"] = _entry("a", usage=["p3_train", "p3_validation"])
    reg.families["b"] = _entry("b", usage=["p3_validation"], source_split="test")
    with pytest.raises(AssertionError) as exc:
        reg.assert_pairwise_disjoint(["p3_train", "p3_validation"])
    # The violating family_id must appear in the error message.
    assert "a" in str(exc.value)


def test_assert_pairwise_disjoint_whitelist_allows_overlap():
    """A family shared between p2_train and p3_train_replay is allowed
    when that pair is whitelisted."""
    reg = FamilyRegistry()
    reg.families["a"] = _entry("a", usage=["p2_train", "p3_train_replay"])
    # Whitelisted overlap -> no exception.
    reg.assert_pairwise_disjoint(
        ["p2_train", "p3_train_replay"],
        whitelist=[("p2_train", "p3_train_replay")],
    )


# ---------------------------------------------------------------------------
# Test 10: builder correctness with synthetic inputs
# ---------------------------------------------------------------------------

def test_builder_correctness(tmp_path):
    """Run the builder on synthetic P2 partition + verified JSONL +
    quarantine list, and assert the produced registry has the expected
    counts and tag assignments.
    """
    # --- Synthetic P2 partition: 3 families ---
    #   train:       mbpp_fam_601, mbpp_fam_602
    #   validation:  mbpp_fam_603
    #   frozen:      (empty)
    p2_path = tmp_path / "p2-partition.json"
    p2_path.write_text(
        json.dumps({
            "train_families": ["mbpp_fam_601", "mbpp_fam_602"],
            "validation_families": ["mbpp_fam_603"],
            "frozen_families": [],
        }),
        encoding="utf-8",
    )

    # --- Synthetic verified JSONL ---
    #   train.jsonl: 2 samples (overlap with P2 train -- sanity-check path)
    #   test.jsonl:  2 samples (NEW families, empty usage)
    verified_dir = tmp_path / "verified"
    verified_dir.mkdir()
    (verified_dir / "train.jsonl").write_text(
        json.dumps({"sample_id": "mbpp_601", "family_id": "mbpp_fam_601",
                    "source_split": "train"}) + "\n"
        + json.dumps({"sample_id": "mbpp_602", "family_id": "mbpp_fam_602",
                      "source_split": "train"}) + "\n",
        encoding="utf-8",
    )
    (verified_dir / "test.jsonl").write_text(
        json.dumps({"sample_id": "mbpp_11", "family_id": "mbpp_fam_11",
                    "source_split": "test"}) + "\n"
        + json.dumps({"sample_id": "mbpp_12", "family_id": "mbpp_fam_12",
                      "source_split": "test"}) + "\n",
        encoding="utf-8",
    )

    # --- Synthetic quarantine: 1 family (a P2 train family) ---
    q_path = tmp_path / "quarantine.json"
    q_path.write_text(
        json.dumps({
            "quarantined_families": ["mbpp_fam_601"],
            "count": 1,
        }),
        encoding="utf-8",
    )

    out_path = tmp_path / "family-registry.json"

    # --- Run the builder via the CLI ---
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "build_family_registry.py"),
            "--p2-partition", str(p2_path),
            "--mbpp-verified-dir", str(verified_dir),
            "--quarantine", str(q_path),
            "--output", str(out_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0, (
        f"builder failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert out_path.exists(), "registry file was not written"

    # --- Load and verify the produced registry ---
    with out_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    # Counts: 3 P2 + 2 new test = 5 total families.
    assert data["total_families"] == 5, data
    # 2 train + 1 validation + 0 frozen = 3 P2 families.
    assert data["total_p2_used"] == 3, data
    # 1 quarantined family.
    assert data["total_quarantined"] == 1, data
    # 2 new test families with empty usage.
    assert data["total_new_available"] == 2, data

    # P2 train family 601 has BOTH p2_train AND quarantine.
    fam_601 = data["families"]["mbpp_fam_601"]
    assert "p2_train" in fam_601["usage"]
    assert "quarantine" in fam_601["usage"]
    assert fam_601["source_split"] == "train"
    assert fam_601["sample_ids"] == ["mbpp_601"]

    # P2 train family 602 (not quarantined) -- only p2_train.
    fam_602 = data["families"]["mbpp_fam_602"]
    assert fam_602["usage"] == ["p2_train"]

    # P2 validation family 603 -- only p2_validation.
    fam_603 = data["families"]["mbpp_fam_603"]
    assert fam_603["usage"] == ["p2_validation"]

    # New test family 11 -- empty usage, source_split=test, sample_id derived.
    fam_11 = data["families"]["mbpp_fam_11"]
    assert fam_11["usage"] == []
    assert fam_11["source_split"] == "test"
    assert fam_11["sample_ids"] == ["mbpp_11"]
    assert fam_11["first_commit"] == "3dce2ce"

    # New test family 12 -- empty usage.
    fam_12 = data["families"]["mbpp_fam_12"]
    assert fam_12["usage"] == []

    # P2 families keep their first_commit.
    assert fam_601["first_commit"] == "515c955"
    assert fam_602["first_commit"] == "515c955"
    assert fam_603["first_commit"] == "515c955"
