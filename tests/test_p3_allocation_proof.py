"""Tests for scripts/p3_allocation_proof.py verdict logic (Issue #14 review).

Covers the three overall_verdict branches:
  - FORMAL_CAPACITY_FEASIBLE: both candidates >= 2530 (2300 + 10% reserve)
  - FORMAL_CAPACITY_AT_RISK: both candidates >= 2300 but at least one < 2530
  - MBPP_FAMILY_OR_VARIANT_LIMIT: at least one candidate < 2300
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.p3_allocation_proof import (
    BUCKETS,
    CANDIDATES,
    FEASIBLE_MIN_TOTAL,
    HARD_MIN_TOTAL,
    _compute_lp_feasible_max,
)


class TestVerdictBranches:
    """Test that the verdict logic correctly distinguishes FEASIBLE / AT_RISK / LIMIT."""

    def _make_pool_buckets(
        self,
        code: int,
        boundary: int,
        static_repair: int,
        execution_repair: int,
    ) -> dict:
        return {
            "code": code,
            "boundary": boundary,
            "static_repair": static_repair,
            "execution_repair": execution_repair,
        }

    def test_feasible_both_above_2530(self) -> None:
        """Both candidates max_total >= 2530 → FORMAL_CAPACITY_FEASIBLE."""
        # Provide enough samples for both candidates to exceed 2530.
        # Balanced needs exec >= 0.27 * 2530 = 684; Repair needs exec >= 0.37 * 2530 = 937.
        # So exec >= 937 is the binding constraint for Repair.
        pool = self._make_pool_buckets(900, 900, 900, 950)
        results = {}
        all_feasible = True
        any_limit = False
        for name, cfg in CANDIDATES.items():
            available = {v: pool[v] for v in BUCKETS}
            result = _compute_lp_feasible_max(
                available, cfg["target_ratios"], 0.03
            )
            if result["max_total"] < HARD_MIN_TOTAL:
                result["verdict"] = "MBPP_FAMILY_OR_VARIANT_LIMIT"
                any_limit = True
                all_feasible = False
            elif result["max_total"] < FEASIBLE_MIN_TOTAL:
                result["verdict"] = "FORMAL_CAPACITY_AT_RISK"
                all_feasible = False
            else:
                result["verdict"] = "FORMAL_CAPACITY_FEASIBLE"
            results[name] = result

        if any_limit:
            overall = "MBPP_FAMILY_OR_VARIANT_LIMIT"
        elif not all_feasible:
            overall = "FORMAL_CAPACITY_AT_RISK"
        else:
            overall = "FORMAL_CAPACITY_FEASIBLE"

        assert overall == "FORMAL_CAPACITY_FEASIBLE"
        for name in CANDIDATES:
            assert results[name]["verdict"] == "FORMAL_CAPACITY_FEASIBLE"
            assert results[name]["max_total"] >= FEASIBLE_MIN_TOTAL

    def test_at_risk_both_above_2300_but_below_2530(self) -> None:
        """Both candidates >= 2300 but at least one < 2530 → FORMAL_CAPACITY_AT_RISK."""
        # Carefully craft pool so both candidates land between 2300 and 2530.
        # Balanced: exec/0.27 must be in [2300, 2530) → exec in [621, 683)
        # Repair: exec/0.37 must be in [2300, 2530) → exec in [851, 936)
        # But both candidates share the same exec pool, so we need exec that
        # satisfies both. Let's use exec=650.
        # Balanced: 650/0.27 = 2407 (AT_RISK)
        # Repair: 650/0.37 = 1756 (LIMIT!)
        # That doesn't work. We need exec >= 851 for Repair to be >= 2300.
        # exec=860: Balanced=3185 (FEASIBLE), Repair=2324 (AT_RISK)
        # That gives one FEASIBLE and one AT_RISK → overall AT_RISK.
        pool = self._make_pool_buckets(1000, 1000, 1000, 860)
        results = {}
        all_feasible = True
        any_limit = False
        for name, cfg in CANDIDATES.items():
            available = {v: pool[v] for v in BUCKETS}
            result = _compute_lp_feasible_max(
                available, cfg["target_ratios"], 0.03
            )
            if result["max_total"] < HARD_MIN_TOTAL:
                result["verdict"] = "MBPP_FAMILY_OR_VARIANT_LIMIT"
                any_limit = True
                all_feasible = False
            elif result["max_total"] < FEASIBLE_MIN_TOTAL:
                result["verdict"] = "FORMAL_CAPACITY_AT_RISK"
                all_feasible = False
            else:
                result["verdict"] = "FORMAL_CAPACITY_FEASIBLE"
            results[name] = result

        if any_limit:
            overall = "MBPP_FAMILY_OR_VARIANT_LIMIT"
        elif not all_feasible:
            overall = "FORMAL_CAPACITY_AT_RISK"
        else:
            overall = "FORMAL_CAPACITY_FEASIBLE"

        assert overall == "FORMAL_CAPACITY_AT_RISK"
        # At least one candidate should be AT_RISK
        at_risk_count = sum(
            1 for r in results.values()
            if r["verdict"] == "FORMAL_CAPACITY_AT_RISK"
        )
        assert at_risk_count >= 1
        # No candidate should be LIMIT
        for name in CANDIDATES:
            assert results[name]["verdict"] != "MBPP_FAMILY_OR_VARIANT_LIMIT"

    def test_limit_at_least_one_below_2300(self) -> None:
        """At least one candidate < 2300 → MBPP_FAMILY_OR_VARIANT_LIMIT."""
        # Use the actual pool counts (both below 2300).
        pool = self._make_pool_buckets(612, 754, 589, 589)
        results = {}
        all_feasible = True
        any_limit = False
        for name, cfg in CANDIDATES.items():
            available = {v: pool[v] for v in BUCKETS}
            result = _compute_lp_feasible_max(
                available, cfg["target_ratios"], 0.03
            )
            if result["max_total"] < HARD_MIN_TOTAL:
                result["verdict"] = "MBPP_FAMILY_OR_VARIANT_LIMIT"
                any_limit = True
                all_feasible = False
            elif result["max_total"] < FEASIBLE_MIN_TOTAL:
                result["verdict"] = "FORMAL_CAPACITY_AT_RISK"
                all_feasible = False
            else:
                result["verdict"] = "FORMAL_CAPACITY_FEASIBLE"
            results[name] = result

        if any_limit:
            overall = "MBPP_FAMILY_OR_VARIANT_LIMIT"
        elif not all_feasible:
            overall = "FORMAL_CAPACITY_AT_RISK"
        else:
            overall = "FORMAL_CAPACITY_FEASIBLE"

        assert overall == "MBPP_FAMILY_OR_VARIANT_LIMIT"
        # Both candidates should be LIMIT with the actual pool
        for name in CANDIDATES:
            assert results[name]["verdict"] == "MBPP_FAMILY_OR_VARIANT_LIMIT"
            assert results[name]["max_total"] < HARD_MIN_TOTAL

    def test_limit_only_one_below_2300(self) -> None:
        """One candidate >= 2300, other < 2300 → MBPP_FAMILY_OR_VARIANT_LIMIT."""
        # Craft pool so Balanced passes but Repair fails.
        # Balanced needs exec >= 0.27 * 2300 = 621.
        # Repair needs exec >= 0.37 * 2300 = 851.
        # exec=700: Balanced=2593 (FEASIBLE), Repair=1891 (LIMIT)
        pool = self._make_pool_buckets(1000, 1000, 1000, 700)
        results = {}
        all_feasible = True
        any_limit = False
        for name, cfg in CANDIDATES.items():
            available = {v: pool[v] for v in BUCKETS}
            result = _compute_lp_feasible_max(
                available, cfg["target_ratios"], 0.03
            )
            if result["max_total"] < HARD_MIN_TOTAL:
                result["verdict"] = "MBPP_FAMILY_OR_VARIANT_LIMIT"
                any_limit = True
                all_feasible = False
            elif result["max_total"] < FEASIBLE_MIN_TOTAL:
                result["verdict"] = "FORMAL_CAPACITY_AT_RISK"
                all_feasible = False
            else:
                result["verdict"] = "FORMAL_CAPACITY_FEASIBLE"
            results[name] = result

        if any_limit:
            overall = "MBPP_FAMILY_OR_VARIANT_LIMIT"
        elif not all_feasible:
            overall = "FORMAL_CAPACITY_AT_RISK"
        else:
            overall = "FORMAL_CAPACITY_FEASIBLE"

        assert overall == "MBPP_FAMILY_OR_VARIANT_LIMIT"
        # Repair should be LIMIT, Balanced should be FEASIBLE
        assert results["balanced_generalist"]["verdict"] == "FORMAL_CAPACITY_FEASIBLE"
        assert results["repair_specialist"]["verdict"] == "MBPP_FAMILY_OR_VARIANT_LIMIT"


class TestLpFeasibleMax:
    """Test the _compute_lp_feasible_max function directly."""

    def test_binding_bucket_identified(self) -> None:
        """The binding bucket should be the one with smallest available/lb."""
        available = {"code": 612, "boundary": 754, "static_repair": 589, "execution_repair": 589}
        ratios = CANDIDATES["balanced_generalist"]["target_ratios"]
        result = _compute_lp_feasible_max(available, ratios, 0.03)
        assert result["binding_bucket"] == "execution_repair"
        assert result["max_total"] == 2181  # 589 / 0.27

    def test_repair_binding_bucket(self) -> None:
        """Repair's binding bucket is execution_repair (589/0.37=1591)."""
        available = {"code": 612, "boundary": 754, "static_repair": 589, "execution_repair": 589}
        ratios = CANDIDATES["repair_specialist"]["target_ratios"]
        result = _compute_lp_feasible_max(available, ratios, 0.03)
        assert result["binding_bucket"] == "execution_repair"
        assert result["max_total"] == 1591  # 589 / 0.37

    def test_feasibility_verified_true(self) -> None:
        """At max_total, a feasible allocation should exist."""
        available = {"code": 612, "boundary": 754, "static_repair": 589, "execution_repair": 589}
        for name, cfg in CANDIDATES.items():
            result = _compute_lp_feasible_max(available, cfg["target_ratios"], 0.03)
            assert result["feasibility_verified"] is True, (
                f"{name}: allocation not feasible at max_total={result['max_total']}"
            )
            assert sum(result["per_bucket_selected"].values()) == result["max_total"]
            assert all(result["per_bucket_ratio_ok"].values())

    def test_no_duplicates_in_allocation(self) -> None:
        """Selected counts should not exceed available counts."""
        available = {"code": 612, "boundary": 754, "static_repair": 589, "execution_repair": 589}
        for name, cfg in CANDIDATES.items():
            result = _compute_lp_feasible_max(available, cfg["target_ratios"], 0.03)
            for v in BUCKETS:
                assert result["per_bucket_selected"][v] <= available[v], (
                    f"{name}.{v}: selected {result['per_bucket_selected'][v]} > available {available[v]}"
                )
