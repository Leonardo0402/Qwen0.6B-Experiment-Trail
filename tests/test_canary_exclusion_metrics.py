"""Tests for canary row exclusion from scored metrics (Issue #16 Task 2).

Canary rows (sample_id containing "canary") must be:
- Still evaluated (for harness validation)
- Excluded from scored summary metrics (n_total, n_repair, rates)
- Marked with ``is_canary`` field in the outcomes array

These tests verify the filtering logic in ``evaluate_model.py`` that splits
outcomes into scored (non-canary) and canary groups before computing
summary metrics via ``summarize()``. ``summarize()`` itself stays pure — it
computes metrics over whatever outcomes it receives. The filtering happens
in ``_compute_scored_metrics`` which is the unit under test here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import EvalOutcome, summarize


# ---------------------------------------------------------------------------
# Factories (match test_metrics.py style)
# ---------------------------------------------------------------------------


def _gen_outcome() -> EvalOutcome:
    """Create a passing code_generation EvalOutcome."""
    return EvalOutcome(
        task_type="code_generation",
        syntax_ok=True,
        public_passed=True,
        public_tests_collected=1,
        hidden_passed=True,
        hidden_tests_present=True,
        hidden_tests_collected=1,
        format_ok=True,
        timed_out=False,
        is_repair=False,
        repair_succeeded=None,
        broke_other_tests=None,
    )


def _repair_outcome() -> EvalOutcome:
    """Create a passing static_repair EvalOutcome."""
    return EvalOutcome(
        task_type="static_repair",
        syntax_ok=True,
        public_passed=True,
        public_tests_collected=1,
        hidden_passed=True,
        hidden_tests_present=True,
        hidden_tests_collected=1,
        format_ok=True,
        timed_out=False,
        is_repair=True,
        repair_succeeded=True,
        broke_other_tests=False,
    )


def _detail(sample_id: str, task_type: str = "code_generation") -> dict:
    """Create a minimal detail dict matching evaluate_model.py output shape."""
    return {
        "sample_id": sample_id,
        "family_id": "fam_test",
        "task_type": task_type,
        "difficulty": 0,
        "generated": "",
        "extracted_code": None,
        "format_ok": True,
        "syntax_ok": True,
        "public_tests_present": True,
        "public_tests_collected": 1,
        "public_passed": True,
        "hidden_tests_present": True,
        "hidden_tests_collected": 1,
        "hidden_passed": True,
        "timed_out": False,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Test 1: summarize excludes canary rows from n_total, n_repair, rates
# ---------------------------------------------------------------------------


class TestSummarizeExcludesCanary:
    def test_summarize_excludes_canary_rows(self):
        """summarize() should only count non-canary outcomes.

        We exercise ``_compute_scored_metrics`` which filters canary outcomes
        (sample_id contains 'canary') before calling ``summarize()``. With
        5 total outcomes (2 generation + 1 repair scored, 2 canary repairs),
        metrics should reflect 3 scored outcomes only.
        """
        from scripts.evaluate_model import _compute_scored_metrics

        outcomes = [
            _gen_outcome(),       # mbpp_001 — scored gen
            _gen_outcome(),       # mbpp_002 — scored gen
            _repair_outcome(),    # mbpp_003_repair — scored repair
            _repair_outcome(),    # mbpp_107_canary — canary repair
            _repair_outcome(),    # mbpp_112_canary — canary repair
        ]
        details = [
            _detail("mbpp_001", "code_generation"),
            _detail("mbpp_002", "code_generation"),
            _detail("mbpp_003_repair", "static_repair"),
            _detail("mbpp_107_canary", "static_repair"),
            _detail("mbpp_112_canary", "static_repair"),
        ]

        metrics, _, canary_excluded = _compute_scored_metrics(outcomes, details)

        # n_total should be 3 (5 total - 2 canaries)
        assert metrics["n_total"] == 3.0
        # n_repair should be 1 (3 repairs total - 2 canary repairs)
        assert metrics["n_repair"] == 1.0
        # n_generation should be 2 (canaries are static_repair, unchanged)
        assert metrics["n_generation"] == 2.0
        # canary_excluded count
        assert canary_excluded == 2

    def test_rate_metrics_only_over_scored(self):
        """Rate metrics (syntax_rate, etc.) should be over scored outcomes only.

        Build a list where canary outcomes would skew the rate if included.
        Canary outcomes have syntax_ok=False; scored outcomes have syntax_ok=True.
        If canaries were included, syntax_rate would be 3/5 = 0.6.
        With canaries excluded, syntax_rate should be 1.0.
        """
        from scripts.evaluate_model import _compute_scored_metrics

        canary_outcome = EvalOutcome(
            task_type="static_repair",
            syntax_ok=False,
            public_passed=False,
            public_tests_collected=1,
            hidden_passed=False,
            hidden_tests_present=True,
            hidden_tests_collected=1,
            format_ok=False,
            timed_out=False,
            is_repair=True,
            repair_succeeded=False,
            broke_other_tests=False,
        )

        outcomes = [
            _gen_outcome(),        # scored, syntax_ok=True
            _gen_outcome(),        # scored, syntax_ok=True
            _gen_outcome(),        # scored, syntax_ok=True
            canary_outcome,        # canary, syntax_ok=False
            canary_outcome,        # canary, syntax_ok=False
        ]
        details = [
            _detail("mbpp_001", "code_generation"),
            _detail("mbpp_002", "code_generation"),
            _detail("mbpp_003", "code_generation"),
            _detail("mbpp_107_canary", "static_repair"),
            _detail("mbpp_112_canary", "static_repair"),
        ]

        metrics, _, _ = _compute_scored_metrics(outcomes, details)

        # If canaries were included, syntax_rate would be 3/5 = 0.6
        # With canaries excluded, syntax_rate should be 1.0
        assert metrics["syntax_rate"] == pytest.approx(1.0)
        assert metrics["n_total"] == 3.0


# ---------------------------------------------------------------------------
# Test 2: canary rows marked with is_canary field in outcomes
# ---------------------------------------------------------------------------


class TestCanaryMarkedInOutcomes:
    def test_canary_rows_marked_in_outcomes(self):
        """Each detail dict should have ``is_canary`` based on sample_id."""
        from scripts.evaluate_model import _compute_scored_metrics

        outcomes = [
            _gen_outcome(),
            _repair_outcome(),
            _repair_outcome(),
        ]
        details = [
            _detail("mbpp_001", "code_generation"),
            _detail("mbpp_107_canary", "static_repair"),
            _detail("mbpp_112_canary", "static_repair"),
        ]

        _, enriched_details, _ = _compute_scored_metrics(outcomes, details)

        # Non-canary row should have is_canary: False
        assert enriched_details[0]["is_canary"] is False
        # Canary rows should have is_canary: True
        assert enriched_details[1]["is_canary"] is True
        assert enriched_details[2]["is_canary"] is True

    def test_all_non_canary_marked_false(self):
        """When there are no canary rows, all details have is_canary=False."""
        from scripts.evaluate_model import _compute_scored_metrics

        outcomes = [_gen_outcome(), _repair_outcome()]
        details = [
            _detail("mbpp_001", "code_generation"),
            _detail("mbpp_002_repair", "static_repair"),
        ]

        _, enriched_details, canary_excluded = _compute_scored_metrics(outcomes, details)

        assert all(d["is_canary"] is False for d in enriched_details)
        assert canary_excluded == 0


# ---------------------------------------------------------------------------
# Test 3: metrics.n_total excludes canary
# ---------------------------------------------------------------------------


class TestMetricsNTotalExcludesCanary:
    def test_metrics_n_total_excludes_canary(self):
        """metrics.n_total should equal the number of non-canary samples."""
        from scripts.evaluate_model import _compute_scored_metrics

        # 7 outcomes: 3 scored gen + 1 scored repair + 3 canary repairs
        outcomes = [
            _gen_outcome(),        # scored gen
            _gen_outcome(),        # scored gen
            _gen_outcome(),        # scored gen
            _repair_outcome(),     # scored repair
            _repair_outcome(),     # canary repair
            _repair_outcome(),     # canary repair
            _repair_outcome(),     # canary repair
        ]
        details = [
            _detail("mbpp_001", "code_generation"),
            _detail("mbpp_002", "code_generation"),
            _detail("mbpp_003", "code_generation"),
            _detail("mbpp_004_repair", "static_repair"),
            _detail("mbpp_107_canary", "static_repair"),
            _detail("mbpp_112_canary", "static_repair"),
            _detail("mbpp_122_canary", "static_repair"),
        ]

        metrics, _, canary_excluded = _compute_scored_metrics(outcomes, details)

        # 7 total - 3 canaries = 4 scored
        assert metrics["n_total"] == 4.0
        assert canary_excluded == 3
        # n_repair: 4 repairs total - 3 canary repairs = 1 scored repair
        assert metrics["n_repair"] == 1.0
        # n_generation: 3 (canaries are repairs, so unchanged)
        assert metrics["n_generation"] == 3.0

    def test_metrics_n_total_all_canary(self):
        """When all outcomes are canaries, metrics should be zero-safe."""
        from scripts.evaluate_model import _compute_scored_metrics

        outcomes = [_repair_outcome(), _repair_outcome()]
        details = [
            _detail("mbpp_107_canary", "static_repair"),
            _detail("mbpp_112_canary", "static_repair"),
        ]

        metrics, enriched, canary_excluded = _compute_scored_metrics(outcomes, details)

        # All canaries excluded → summarize([]) returns all zeros
        assert metrics["n_total"] == 0.0
        assert metrics["n_repair"] == 0.0
        assert canary_excluded == 2
        assert all(d["is_canary"] is True for d in enriched)

    def test_metrics_n_total_no_canary(self):
        """When there are no canaries, n_total equals len(outcomes)."""
        from scripts.evaluate_model import _compute_scored_metrics

        outcomes = [_gen_outcome(), _gen_outcome(), _repair_outcome()]
        details = [
            _detail("mbpp_001", "code_generation"),
            _detail("mbpp_002", "code_generation"),
            _detail("mbpp_003_repair", "static_repair"),
        ]

        metrics, _, canary_excluded = _compute_scored_metrics(outcomes, details)

        assert metrics["n_total"] == 3.0
        assert metrics["n_repair"] == 1.0
        assert metrics["n_generation"] == 2.0
        assert canary_excluded == 0
