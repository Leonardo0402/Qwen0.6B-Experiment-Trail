"""tests/test_sample_pool.py -- Unit tests for SamplePool (Task 10).

Covers the 8 unit tests specified in ``.superpowers/sdd/task-10-brief.md``:

  1. test_add_and_dedup
  2. test_normalize_variant_type_code
  3. test_normalize_variant_type_boundary
  4. test_normalize_variant_type_static_repair
  5. test_normalize_variant_type_execution_repair
  6. test_apply_family_cap
  7. test_filter_families
  8. test_to_jsonl_sorted

All tests use synthetic Sample objects (no I/O on real data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sample_pool import SamplePool  # noqa: E402
from src.schemas import Sample, Verification  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verification() -> Verification:
    return Verification(syntax_ok=True, pytest_ok=True, ruff_ok=True, timeout=False)


def _make_sample(
    sample_id: str,
    *,
    family_id: str = "mbpp_fam_1",
    task_type: str = "code_generation",
    skill_tags: list[str] | None = None,
    broken_code: str | None = None,
    execution_feedback: str | None = None,
    variant_type: str | None = None,
    bug_type: str | None = None,
    source_split: str | None = None,
) -> Sample:
    """Build a minimal valid Sample for testing."""
    return Sample(
        sample_id=sample_id,
        family_id=family_id,
        difficulty=1,
        task_type=task_type,
        language="python",
        skill_tags=list(skill_tags) if skill_tags is not None else ["function"],
        instruction=f"Write code for {sample_id}.",
        broken_code=broken_code,
        execution_feedback=execution_feedback,
        target_code="def answer():\n    return 42",
        public_tests="assert answer() == 42",
        hidden_tests="assert answer() == 42",
        verified=True,
        verification=_verification(),
        generator="test",
        created_at="2026-07-04T00:00:00+00:00",
        dataset_version="test-v1",
        variant_type=variant_type,
        bug_type=bug_type,
        source_split=source_split,
    )


# ---------------------------------------------------------------------------
# Test 1: add + dedup
# ---------------------------------------------------------------------------

def test_add_and_dedup():
    """Adding a duplicate sample_id is rejected; dedup removes dupes that
    slipped through (e.g. via direct list manipulation)."""
    pool = SamplePool()
    s1 = _make_sample("mbpp_1")
    s2 = _make_sample("mbpp_2", family_id="mbpp_fam_2")
    s3_dup = _make_sample("mbpp_1", family_id="mbpp_fam_3")  # same sample_id as s1

    assert pool.add(s1) is True
    assert pool.add(s2) is True
    assert pool.add(s3_dup) is False  # duplicate sample_id rejected
    assert len(pool) == 2

    # Now simulate a pool loaded from JSONL (which bypasses the add() check)
    pool2 = SamplePool()
    # Manually append duplicates to test dedup()
    pool2._samples.extend([s1, s2, s3_dup])
    pool2._reindex()
    assert len(pool2) == 3  # has duplicates
    removed = pool2.dedup()
    assert removed == 1
    assert len(pool2) == 2
    # First occurrence wins
    sample_ids = [s.sample_id for s in pool2]
    assert "mbpp_1" in sample_ids
    assert "mbpp_2" in sample_ids


# ---------------------------------------------------------------------------
# Test 2: normalize variant_type = code
# ---------------------------------------------------------------------------

def test_normalize_variant_type_code():
    """code_generation without boundary -> variant_type='code'."""
    pool = SamplePool()
    s = _make_sample("mbpp_100", task_type="code_generation", skill_tags=["function"])
    pool.add(s)
    # Pre-normalise: variant_type is None
    assert pool._samples[0].variant_type is None
    n = pool.normalize_variant_type()
    assert n == 1
    assert pool._samples[0].variant_type == "code"
    assert pool._samples[0].bug_type is None


# ---------------------------------------------------------------------------
# Test 3: normalize variant_type = boundary
# ---------------------------------------------------------------------------

def test_normalize_variant_type_boundary():
    """code_generation with 'boundary' in skill_tags -> variant_type='boundary'.

    Also covers the sample_id ends with '_boundary' branch.
    """
    pool = SamplePool()
    # Branch A: 'boundary' in skill_tags
    s_a = _make_sample(
        "mbpp_200",
        task_type="code_generation",
        skill_tags=["function", "boundary"],
    )
    # Branch B: sample_id ends with '_boundary' (no boundary tag)
    s_b = _make_sample(
        "mbpp_201_boundary",
        task_type="code_generation",
        skill_tags=["function"],
    )
    pool.add(s_a)
    pool.add(s_b)
    n = pool.normalize_variant_type()
    assert n == 2
    assert pool._samples[0].variant_type == "boundary"
    assert pool._samples[1].variant_type == "boundary"
    # Neither has sr/er in sample_id, so bug_type is None
    assert pool._samples[0].bug_type is None
    assert pool._samples[1].bug_type is None


# ---------------------------------------------------------------------------
# Test 4: normalize variant_type = static_repair
# ---------------------------------------------------------------------------

def test_normalize_variant_type_static_repair():
    """static_repair -> variant_type='static_repair' + bug_type extracted."""
    pool = SamplePool()
    s = _make_sample(
        "mbpp_300_sr_return_value_error",
        task_type="static_repair",
        skill_tags=["function", "return_value_error"],
        broken_code="def f():\n    return None",
    )
    pool.add(s)
    n = pool.normalize_variant_type()
    assert n == 1
    assert pool._samples[0].variant_type == "static_repair"
    # bug_type extracted via .*_(sr|er)_(.+)$ -> "return_value_error"
    assert pool._samples[0].bug_type == "return_value_error"


# ---------------------------------------------------------------------------
# Test 5: normalize variant_type = execution_repair
# ---------------------------------------------------------------------------

def test_normalize_variant_type_execution_repair():
    """execution_repair -> variant_type='execution_repair' + bug_type extracted."""
    pool = SamplePool()
    s = _make_sample(
        "mbpp_400_er_off_by_one",
        task_type="execution_repair",
        skill_tags=["function", "off_by_one"],
        broken_code="def f():\n    return 0",
        execution_feedback="TypeError: returned 0 instead of 1",
    )
    pool.add(s)
    n = pool.normalize_variant_type()
    assert n == 1
    assert pool._samples[0].variant_type == "execution_repair"
    # bug_type extracted via .*_(sr|er)_(.+)$ -> "off_by_one"
    assert pool._samples[0].bug_type == "off_by_one"


# ---------------------------------------------------------------------------
# Test 6: apply_family_cap
# ---------------------------------------------------------------------------

def test_apply_family_cap():
    """A family with 10 samples, cap=7, drops 3 (keeping first 7 by sample_id)."""
    pool = SamplePool()
    # 10 samples in same family, sample_ids mbpp_010..mbpp_019 (sorted)
    for i in range(10, 20):
        sid = f"mbpp_{i:03d}"
        s = _make_sample(sid, family_id="mbpp_fam_X", task_type="code_generation")
        pool.add(s)
    # 2 samples in another family (under cap, untouched)
    for i in range(2):
        sid = f"mbpp_{i:03d}"
        s = _make_sample(sid, family_id="mbpp_fam_Y", task_type="code_generation")
        pool.add(s)
    assert len(pool) == 12

    dropped = pool.apply_family_cap(cap=7)
    assert dropped == 3
    assert len(pool) == 9  # 7 (capped) + 2 (untouched)
    # Surviving sample_ids in family X = first 7 by sample_id ascending
    family_x_samples = [s for s in pool if s.family_id == "mbpp_fam_X"]
    family_x_ids = sorted(s.sample_id for s in family_x_samples)
    assert family_x_ids == [f"mbpp_{i:03d}" for i in range(10, 17)]
    # Family Y untouched
    family_y_samples = [s for s in pool if s.family_id == "mbpp_fam_Y"]
    assert len(family_y_samples) == 2


# ---------------------------------------------------------------------------
# Test 7: filter_families
# ---------------------------------------------------------------------------

def test_filter_families():
    """filter_families keeps only samples whose family_id is in the set."""
    pool = SamplePool()
    s1 = _make_sample("mbpp_1", family_id="mbpp_fam_A")
    s2 = _make_sample("mbpp_2", family_id="mbpp_fam_A")
    s3 = _make_sample("mbpp_3", family_id="mbpp_fam_B")
    s4 = _make_sample("mbpp_4", family_id="mbpp_fam_C")
    for s in (s1, s2, s3, s4):
        pool.add(s)
    assert len(pool) == 4

    kept = pool.filter_families({"mbpp_fam_A", "mbpp_fam_C"})
    assert kept == 3
    assert len(pool) == 3
    surviving = sorted(s.sample_id for s in pool)
    assert surviving == ["mbpp_1", "mbpp_2", "mbpp_4"]


# ---------------------------------------------------------------------------
# Test 8: to_jsonl is sorted by sample_id
# ---------------------------------------------------------------------------

def test_to_jsonl_sorted(tmp_path):
    """Output JSONL is sorted by sample_id ascending."""
    pool = SamplePool()
    # Add samples in non-sorted order
    for sid in ["mbpp_005", "mbpp_001", "mbpp_003", "mbpp_002", "mbpp_004"]:
        s = _make_sample(sid, family_id="mbpp_fam_X", task_type="code_generation")
        pool.add(s)
    out = tmp_path / "pool.jsonl"
    pool.to_jsonl(out)

    # Read back and verify order
    with out.open(encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    sample_ids = [json.loads(l)["sample_id"] for l in lines if l.strip()]
    assert sample_ids == sorted(sample_ids), (
        f"output not sorted: {sample_ids}"
    )
    assert sample_ids == ["mbpp_001", "mbpp_002", "mbpp_003", "mbpp_004", "mbpp_005"]

    # Round-trip via from_jsonl
    pool2 = SamplePool.from_jsonl(out)
    assert len(pool2) == 5
    assert {s.sample_id for s in pool2} == {
        "mbpp_001", "mbpp_002", "mbpp_003", "mbpp_004", "mbpp_005"
    }


# ---------------------------------------------------------------------------
# Bonus: stats() sanity check
# ---------------------------------------------------------------------------

def test_stats_returns_expected_shape():
    """stats() returns a dict with the required keys for the manifest."""
    pool = SamplePool()
    pool.add(_make_sample("mbpp_1", family_id="mbpp_fam_A",
                          task_type="code_generation"))
    pool.add(_make_sample("mbpp_2_sr_x", family_id="mbpp_fam_A",
                          task_type="static_repair",
                          broken_code="def f(): pass"))
    pool.normalize_variant_type()
    stats = pool.stats()
    assert "total_samples" in stats
    assert "family_count" in stats
    assert "variant_distribution" in stats
    assert "bug_type_distribution" in stats
    assert "family_distribution" in stats
    assert stats["total_samples"] == 2
    assert stats["family_count"] == 1
    assert stats["variant_distribution"].get("code") == 1
    assert stats["variant_distribution"].get("static_repair") == 1
    assert stats["bug_type_distribution"].get("x") == 1
