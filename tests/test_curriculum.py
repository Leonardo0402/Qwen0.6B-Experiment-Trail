"""Tests for src/curriculum.py — TDD coverage for curriculum design.

Covers:
- STAGE_MIX fractions each sum to ~1.0; all three stages present.
- level_description valid for 0..3; raises ValueError for out-of-range.
- build_stage_mix: rich pool → produced mix approximates target ratios;
  determinism (same seed → same selection); under-fill returns all available
  without crashing or duplicating; total respected when pool sufficient.
- mix_report counts correctly.
"""
from __future__ import annotations

import pytest

from src.curriculum import (
    STAGE_MIX,
    Stage,
    build_stage_mix,
    get_last_shortfalls,
    level_description,
    mix_report,
)
from src.schemas import Sample, Verification


# ---------------------------------------------------------------------------
# Helpers — self-contained, no cross-test-module imports
# ---------------------------------------------------------------------------

def _verification() -> dict:
    return {"syntax_ok": True, "pytest_ok": True, "ruff_ok": True, "timeout": False}


def _make_sample(difficulty: int, family_id: str = "fam_01", idx: int = 0) -> Sample:
    """Create a minimal code_generation Sample with the given difficulty."""
    return Sample(
        sample_id=f"s_d{difficulty}_{idx}",
        family_id=family_id,
        difficulty=difficulty,
        task_type="code_generation",
        language="python",
        skill_tags=["basics"],
        instruction="写一个函数，返回两数之和。",
        broken_code=None,
        execution_feedback=None,
        target_code="def add(a, b):\n    return a + b",
        public_tests="def test_add():\n    assert add(1, 2) == 3",
        hidden_tests="",
        verified=True,
        verification=_verification(),
        generator="test",
        created_at="2026-01-01T00:00:00Z",
        dataset_version="v1",
    )


def _make_pool(counts: dict[int, int]) -> list[Sample]:
    """Create a pool of samples with given {difficulty: count} mapping."""
    pool: list[Sample] = []
    for diff, count in counts.items():
        for i in range(count):
            pool.append(_make_sample(diff, idx=i))
    return pool


# ---------------------------------------------------------------------------
# STAGE_MIX validation
# ---------------------------------------------------------------------------

class TestStageMix:
    def test_easy_fractions_sum_to_one(self):
        total = sum(STAGE_MIX["easy"].values())
        assert abs(total - 1.0) < 1e-6

    def test_boundary_fractions_sum_to_one(self):
        total = sum(STAGE_MIX["boundary"].values())
        assert abs(total - 1.0) < 1e-6

    def test_repair_fractions_sum_to_one(self):
        total = sum(STAGE_MIX["repair"].values())
        assert abs(total - 1.0) < 1e-6

    def test_all_stages_present(self):
        for stage in Stage:
            assert stage.value in STAGE_MIX, f"Stage {stage.value!r} missing from STAGE_MIX"

    def test_easy_has_exactly_levels_0_and_1(self):
        assert set(STAGE_MIX["easy"].keys()) == {0, 1}

    def test_boundary_has_exactly_levels_0_1_2(self):
        assert set(STAGE_MIX["boundary"].keys()) == {0, 1, 2}

    def test_repair_has_all_levels_0_to_3(self):
        assert set(STAGE_MIX["repair"].keys()) == {0, 1, 2, 3}

    def test_easy_level_0_fraction(self):
        assert abs(STAGE_MIX["easy"][0] - 0.70) < 1e-9

    def test_easy_level_1_fraction(self):
        assert abs(STAGE_MIX["easy"][1] - 0.30) < 1e-9

    def test_repair_level_3_largest(self):
        fracs = STAGE_MIX["repair"]
        assert fracs[3] == max(fracs.values())


# ---------------------------------------------------------------------------
# level_description
# ---------------------------------------------------------------------------

class TestLevelDescription:
    def test_valid_levels_return_non_empty_string(self):
        for level in range(4):  # 0..3
            desc = level_description(level)
            assert isinstance(desc, str)
            assert len(desc.strip()) > 0

    def test_level_0_mentions_syntax(self):
        desc = level_description(0).lower()
        assert "syntax" in desc or "local" in desc

    def test_level_1_mentions_function(self):
        desc = level_description(1).lower()
        assert "function" in desc or "implementation" in desc

    def test_level_2_mentions_boundary(self):
        desc = level_description(2).lower()
        assert "boundary" in desc or "data" in desc

    def test_level_3_mentions_execution(self):
        desc = level_description(3).lower()
        assert "execution" in desc

    def test_raises_for_level_4(self):
        with pytest.raises(ValueError):
            level_description(4)

    def test_raises_for_negative(self):
        with pytest.raises(ValueError):
            level_description(-1)

    def test_raises_for_large_value(self):
        with pytest.raises(ValueError):
            level_description(100)

    def test_descriptions_are_distinct(self):
        descs = [level_description(lvl) for lvl in range(4)]
        assert len(set(descs)) == 4, "Each level must have a unique description"


# ---------------------------------------------------------------------------
# build_stage_mix
# ---------------------------------------------------------------------------

class TestBuildStageMix:
    def test_total_exact_when_fractions_round_cleanly(self):
        """easy stage at total=100: 70+30 rounds exactly → len == total.

        NOTE: total is an APPROXIMATE target; exactness here is incidental to
        easy's fractions dividing 100 cleanly. See the rounding-drift tests
        below for the general (approximate) contract.
        """
        pool = _make_pool({0: 200, 1: 200})
        result = build_stage_mix(pool, "easy", 100, seed=42)
        assert len(result) == 100

    def test_total_is_approximate_boundary_small(self):
        """Per-level round() can overshoot: boundary/total=3 → 1+1+2 = 4."""
        pool = _make_pool({0: 200, 1: 200, 2: 200})
        result = build_stage_mix(pool, "boundary", 3, seed=42)
        # Documented drift: result length need not equal total.
        assert len(result) == 4
        # ...but stays close (within 1 of total).
        assert abs(len(result) - 3) <= 1

    def test_total_is_approximate_repair_small(self):
        """Per-level round() can undershoot: repair/total=10 → 1+2+2+4 = 9."""
        pool = _make_pool({0: 200, 1: 200, 2: 200, 3: 200})
        result = build_stage_mix(pool, "repair", 10, seed=42)
        assert len(result) == 9
        assert abs(len(result) - 10) <= 1

    def test_total_within_tolerance_all_stages(self):
        """Across stages, |len - total| is bounded by the number of levels."""
        pools = {
            "easy":     _make_pool({0: 500, 1: 500}),
            "boundary": _make_pool({0: 500, 1: 500, 2: 500}),
            "repair":   _make_pool({0: 500, 1: 500, 2: 500, 3: 500}),
        }
        for stage, pool in pools.items():
            for total in (3, 7, 10, 33, 100):
                result = build_stage_mix(pool, stage, total, seed=1)
                n_levels = len(STAGE_MIX[stage])
                # Rounding error is at most 0.5 per level.
                assert abs(len(result) - total) <= n_levels

    def test_mix_matches_target_easy(self):
        """Rich pool + round fractions → exact target counts."""
        pool = _make_pool({0: 200, 1: 200})
        result = build_stage_mix(pool, "easy", 100, seed=42)
        report = mix_report(result)
        # round(100 * 0.70) = 70, round(100 * 0.30) = 30
        assert report.get(0, 0) == 70
        assert report.get(1, 0) == 30

    def test_mix_matches_target_boundary(self):
        pool = _make_pool({0: 200, 1: 200, 2: 200})
        result = build_stage_mix(pool, "boundary", 100, seed=42)
        report = mix_report(result)
        # round(100 * 0.20)=20, round(100 * 0.30)=30, round(100 * 0.50)=50
        assert report.get(0, 0) == 20
        assert report.get(1, 0) == 30
        assert report.get(2, 0) == 50

    def test_mix_matches_target_repair(self):
        pool = _make_pool({0: 200, 1: 200, 2: 200, 3: 200})
        result = build_stage_mix(pool, "repair", 100, seed=42)
        report = mix_report(result)
        # round(100 * 0.10)=10, *0.20=20, *0.25=25, *0.45=45
        assert report.get(0, 0) == 10
        assert report.get(1, 0) == 20
        assert report.get(2, 0) == 25
        assert report.get(3, 0) == 45

    def test_determinism_same_seed(self):
        pool = _make_pool({0: 100, 1: 100})
        result1 = build_stage_mix(pool, "easy", 60, seed=42)
        result2 = build_stage_mix(pool, "easy", 60, seed=42)
        assert [s.sample_id for s in result1] == [s.sample_id for s in result2]

    def test_different_seeds_give_different_order(self):
        """Different seeds should produce a different sample ordering."""
        pool = _make_pool({0: 100, 1: 100})
        result1 = build_stage_mix(pool, "easy", 60, seed=42)
        result2 = build_stage_mix(pool, "easy", 60, seed=99)
        # With 100 samples choosing 42/18, the ordering should differ
        assert [s.sample_id for s in result1] != [s.sample_id for s in result2]

    def test_underfill_does_not_crash(self):
        """Pool with fewer level-0 than target must not raise."""
        pool = _make_pool({0: 3, 1: 200})
        result = build_stage_mix(pool, "easy", 100, seed=42)
        assert isinstance(result, list)

    def test_underfill_no_duplicates(self):
        """Under-fill must never duplicate any sample."""
        pool = _make_pool({0: 3, 1: 200})
        result = build_stage_mix(pool, "easy", 100, seed=42)
        ids = [s.sample_id for s in result]
        assert len(ids) == len(set(ids)), "Duplicate sample_ids found in under-fill result"

    def test_underfill_takes_all_available(self):
        """When the pool is short, all available samples of that difficulty are used."""
        pool = _make_pool({0: 3, 1: 200})
        result = build_stage_mix(pool, "easy", 100, seed=42)
        report = mix_report(result)
        # Only 3 level-0 samples exist, so all 3 must appear
        assert report.get(0, 0) == 3

    def test_underfill_reports_shortfall(self):
        """Shortfall is recorded and accessible after an under-fill call."""
        pool = _make_pool({0: 3, 1: 200})
        build_stage_mix(pool, "easy", 100, seed=42)
        shortfalls = get_last_shortfalls()
        assert 0 in shortfalls, "Expected shortfall for difficulty 0"
        # target=70 (round(100*0.70)), available=3 → shortfall=67
        assert shortfalls[0] == 70 - 3

    def test_no_shortfall_when_pool_is_sufficient(self):
        pool = _make_pool({0: 200, 1: 200})
        build_stage_mix(pool, "easy", 100, seed=42)
        assert get_last_shortfalls() == {}

    def test_invalid_stage_raises(self):
        pool = _make_pool({0: 10, 1: 10})
        with pytest.raises((ValueError, KeyError)):
            build_stage_mix(pool, "nonexistent_stage", 10)

    def test_empty_pool_underfill_all_levels(self):
        """Empty pool → no crash; all difficulties report shortfalls."""
        result = build_stage_mix([], "easy", 100, seed=42)
        assert isinstance(result, list)
        assert len(result) == 0
        shortfalls = get_last_shortfalls()
        assert 0 in shortfalls
        assert 1 in shortfalls


# ---------------------------------------------------------------------------
# mix_report
# ---------------------------------------------------------------------------

class TestMixReport:
    def test_single_difficulty(self):
        pool = _make_pool({2: 5})
        assert mix_report(pool) == {2: 5}

    def test_multiple_difficulties(self):
        pool = _make_pool({0: 3, 1: 7, 2: 2})
        assert mix_report(pool) == {0: 3, 1: 7, 2: 2}

    def test_empty_list(self):
        assert mix_report([]) == {}

    def test_missing_levels_omitted(self):
        """Levels with zero count must not appear as keys."""
        pool = _make_pool({1: 4})
        report = mix_report(pool)
        assert 0 not in report
        assert report == {1: 4}

    def test_sum_equals_input_length(self):
        pool = _make_pool({0: 3, 1: 7, 3: 5})
        report = mix_report(pool)
        assert sum(report.values()) == len(pool)
