"""Tests for Issue #14 Wave 2-C: Composite Score hardening.

Covers
------
- P2.2: ``compute_composite`` hard-fails on missing/empty variant buckets
  via ``CompositeCoverageError`` (Readiness=FIX_FIRST).
- P2.3: ``validate_composite_schema`` enforces:
    * required keys exactly match schema (no extra, no missing)
    * sum(weights) == 1.0 ± 1e-9
    * each weight in [0, 1]
    * NaN/Inf hard fail
    * schema_version exact match
- P2.4: ``CANONICAL_METRICS`` is the complete, canonical metric-name list.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import EvalOutcome  # noqa: E402
from src.p3_checkpoint_evaluator import (  # noqa: E402
    CANONICAL_METRICS,
    COMPOSITE_SCHEMA_VERSION,
    CompositeCoverageError,
    CompositeScore,
    validate_composite_schema,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_eval_outcome(
    task_type: str = "code_generation",
    syntax_ok: bool = True,
    public_passed: bool = True,
    public_tests_collected: int = 1,
    hidden_passed: bool = True,
    hidden_tests_present: bool = True,
    hidden_tests_collected: int = 1,
    format_ok: bool = True,
    timed_out: bool = False,
    is_repair: bool = False,
    repair_succeeded: bool | None = None,
    broke_other_tests: bool | None = None,
) -> EvalOutcome:
    return EvalOutcome(
        task_type=task_type,
        syntax_ok=syntax_ok,
        public_passed=public_passed,
        public_tests_collected=public_tests_collected,
        hidden_passed=hidden_passed,
        hidden_tests_present=hidden_tests_present,
        hidden_tests_collected=hidden_tests_collected,
        format_ok=format_ok,
        timed_out=timed_out,
        is_repair=is_repair,
        repair_succeeded=repair_succeeded,
        broke_other_tests=broke_other_tests,
    )


def _make_balanced_weights() -> dict[str, float]:
    """Balanced 30/15/20/25/10 weights (sum = 1.0)."""
    return {
        "code_generation_pass_at_1": 0.30,
        "boundary_pass_at_1": 0.15,
        "static_repair_success": 0.20,
        "execution_repair_success": 0.25,
        "hidden_pass_rate": 0.10,
    }


def _make_repair_weights() -> dict[str, float]:
    """Repair 10/10/30/40/10 weights (sum = 1.0)."""
    return {
        "code_generation_pass_at_1": 0.10,
        "boundary_pass_at_1": 0.10,
        "static_repair_success": 0.30,
        "execution_repair_success": 0.40,
        "hidden_pass_rate": 0.10,
    }


def _make_valid_composite_config(
    weights: dict[str, float] | None = None,
) -> dict:
    """Build a full config dict whose composite_score block passes schema."""
    if weights is None:
        weights = _make_balanced_weights()
    return {
        "composite_score": {
            "code_generation_pass_at_1": weights["code_generation_pass_at_1"],
            "boundary_pass_at_1": weights["boundary_pass_at_1"],
            "static_repair_success": weights["static_repair_success"],
            "execution_repair_success": weights["execution_repair_success"],
            "hidden_pass_rate": weights["hidden_pass_rate"],
            "hard_constraint": {
                "code_generation_drop_vs_p2_final_max_pct": 3.0,
            },
            "schema_version": COMPOSITE_SCHEMA_VERSION,
        },
    }


def _make_full_metrics_by_variant(
    n_per_bucket: int = 2,
) -> dict[str, list[EvalOutcome]]:
    """Build a metrics_by_variant dict with all 4 buckets non-empty."""
    code = [
        _make_eval_outcome(
            task_type="code_generation",
            is_repair=False,
            repair_succeeded=None,
            broke_other_tests=None,
        )
        for _ in range(n_per_bucket)
    ]
    boundary = [
        _make_eval_outcome(
            task_type="code_generation",
            is_repair=False,
            repair_succeeded=None,
            broke_other_tests=None,
        )
        for _ in range(n_per_bucket)
    ]
    static_repair = [
        _make_eval_outcome(
            task_type="static_repair",
            is_repair=True,
            repair_succeeded=True,
            broke_other_tests=False,
        )
        for _ in range(n_per_bucket)
    ]
    execution_repair = [
        _make_eval_outcome(
            task_type="execution_repair",
            is_repair=True,
            repair_succeeded=True,
            broke_other_tests=False,
        )
        for _ in range(n_per_bucket)
    ]
    return {
        "code": code,
        "boundary": boundary,
        "static_repair": static_repair,
        "execution_repair": execution_repair,
    }


# Need CheckpointEvaluator for compute_composite tests — build minimal config.
def _make_evaluator_config(weights: dict[str, float] | None = None) -> dict:
    if weights is None:
        weights = _make_balanced_weights()
    cfg = _make_valid_composite_config(weights)
    cfg["checkpoint_evaluator"] = {
        "tier1": {"interval_steps": 50, "metrics": ["train_loss"]},
        "tier2": {
            "interval_epoch_fraction": 0.25,
            "probe_size": 75,
            "probe_stratify_by": "variant_type",
            "probe_seed": 42,
            "composite_score": True,
        },
        "tier3": {"interval_epochs": 1, "full_validation": True, "composite_score": True},
    }
    cfg["early_stopping"] = {
        "enabled": True,
        "probe_patience": 4,
        "probe_min_delta": 0.005,
        "full_validation_confirm": True,
        "divergence_nan_inf": True,
        "max_epochs": 3,
    }
    cfg["best_checkpoint"] = {
        "selection_metric": "full_validation_composite",
        "never_use": ["frozen_v4", "probe"],
    }
    return cfg


# ---------------------------------------------------------------------------
# P2.2: compute_composite hard fail on missing/empty buckets
# ---------------------------------------------------------------------------


class TestComputeCompositeHardFail:
    """Issue #14 P2.2: missing or empty buckets -> CompositeCoverageError."""

    def _make_evaluator(self):
        from src.p3_checkpoint_evaluator import CheckpointEvaluator

        return CheckpointEvaluator(_make_evaluator_config(), total_train_samples=626)

    def test_missing_code_bucket_raises(self) -> None:
        ev = self._make_evaluator()
        buckets = _make_full_metrics_by_variant()
        del buckets["code"]
        with pytest.raises(CompositeCoverageError) as excinfo:
            ev.compute_composite(buckets, _make_balanced_weights())
        msg = str(excinfo.value)
        assert "code" in msg
        assert "FIX_FIRST" in msg

    def test_missing_boundary_bucket_raises(self) -> None:
        ev = self._make_evaluator()
        buckets = _make_full_metrics_by_variant()
        del buckets["boundary"]
        with pytest.raises(CompositeCoverageError):
            ev.compute_composite(buckets, _make_balanced_weights())

    def test_missing_static_repair_bucket_raises(self) -> None:
        ev = self._make_evaluator()
        buckets = _make_full_metrics_by_variant()
        del buckets["static_repair"]
        with pytest.raises(CompositeCoverageError):
            ev.compute_composite(buckets, _make_balanced_weights())

    def test_missing_execution_repair_bucket_raises(self) -> None:
        ev = self._make_evaluator()
        buckets = _make_full_metrics_by_variant()
        del buckets["execution_repair"]
        with pytest.raises(CompositeCoverageError):
            ev.compute_composite(buckets, _make_balanced_weights())

    def test_empty_code_bucket_raises(self) -> None:
        """Bucket present but empty list -> still hard fail."""
        ev = self._make_evaluator()
        buckets = _make_full_metrics_by_variant()
        buckets["code"] = []
        with pytest.raises(CompositeCoverageError) as excinfo:
            ev.compute_composite(buckets, _make_balanced_weights())
        assert "code" in str(excinfo.value)

    def test_empty_static_repair_bucket_raises(self) -> None:
        ev = self._make_evaluator()
        buckets = _make_full_metrics_by_variant()
        buckets["static_repair"] = []
        with pytest.raises(CompositeCoverageError):
            ev.compute_composite(buckets, _make_balanced_weights())

    def test_all_buckets_missing_raises(self) -> None:
        ev = self._make_evaluator()
        with pytest.raises(CompositeCoverageError) as excinfo:
            ev.compute_composite({}, _make_balanced_weights())
        msg = str(excinfo.value)
        # All four buckets reported missing.
        for bucket in ("code", "boundary", "static_repair", "execution_repair"):
            assert bucket in msg

    def test_extra_bucket_does_not_save_missing_required(self) -> None:
        """Extra bucket 'other' present but 'code' missing -> still raises."""
        ev = self._make_evaluator()
        buckets = _make_full_metrics_by_variant()
        del buckets["code"]
        buckets["other"] = [_make_eval_outcome()]
        with pytest.raises(CompositeCoverageError):
            ev.compute_composite(buckets, _make_balanced_weights())

    def test_composite_coverage_error_readiness_is_fix_first(self) -> None:
        """Readiness policy constant = FIX_FIRST."""
        assert CompositeCoverageError.READINESS == "FIX_FIRST"

    def test_composite_coverage_error_is_exception_subclass(self) -> None:
        assert issubclass(CompositeCoverageError, Exception)

    def test_full_buckets_compute_returns_composite_score(self) -> None:
        """Sanity: all 4 non-empty buckets -> normal CompositeScore."""
        ev = self._make_evaluator()
        buckets = _make_full_metrics_by_variant(n_per_bucket=2)
        cs = ev.compute_composite(buckets, _make_balanced_weights())
        assert isinstance(cs, CompositeScore)
        # All pass_at_1 / repair_success = 1.0 in fixture -> 1.0 components
        assert cs.code_generation_pass_at_1 == 1.0
        assert cs.boundary_pass_at_1 == 1.0
        assert cs.static_repair_success == 1.0
        assert cs.execution_repair_success == 1.0
        # hidden_pass_rate: every fixture outcome has hidden_passed=True
        assert cs.hidden_pass_rate == 1.0


# ---------------------------------------------------------------------------
# P2.3: validate_composite_schema
# ---------------------------------------------------------------------------


class TestValidateCompositeSchema:
    """Issue #14 P2.3: schema/weights/schema_version validation."""

    def test_valid_balanced_config_passes(self) -> None:
        cfg = _make_valid_composite_config(_make_balanced_weights())
        # Should not raise
        validate_composite_schema(cfg)

    def test_valid_repair_config_passes(self) -> None:
        cfg = _make_valid_composite_config(_make_repair_weights())
        validate_composite_schema(cfg)

    # -- required keys exactly match schema -------------------------------

    def test_missing_composite_score_block_raises(self) -> None:
        with pytest.raises(ValueError, match="missing the 'composite_score'"):
            validate_composite_schema({})

    def test_missing_one_weight_key_raises(self) -> None:
        cfg = _make_valid_composite_config()
        del cfg["composite_score"]["hidden_pass_rate"]
        with pytest.raises(ValueError, match="schema keys mismatch"):
            validate_composite_schema(cfg)

    def test_missing_hard_constraint_raises(self) -> None:
        cfg = _make_valid_composite_config()
        del cfg["composite_score"]["hard_constraint"]
        with pytest.raises(ValueError, match="schema keys mismatch"):
            validate_composite_schema(cfg)

    def test_missing_schema_version_raises(self) -> None:
        cfg = _make_valid_composite_config()
        del cfg["composite_score"]["schema_version"]
        with pytest.raises(ValueError, match="schema keys mismatch"):
            validate_composite_schema(cfg)

    def test_extra_key_raises(self) -> None:
        """Schema keys must match EXACTLY (no extras)."""
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["unexpected_key"] = 0.5
        with pytest.raises(ValueError, match="schema keys mismatch"):
            validate_composite_schema(cfg)

    # -- sum(weights) == 1.0 ± 1e-9 ---------------------------------------

    def test_weights_sum_not_one_raises(self) -> None:
        """sum != 1.0 -> ValueError."""
        cfg = _make_valid_composite_config()
        # Bump code weight to 0.40 (now sum = 1.10)
        cfg["composite_score"]["code_generation_pass_at_1"] = 0.40
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_composite_schema(cfg)

    def test_weights_sum_within_tolerance_passes(self) -> None:
        """1e-9 tolerance: tiny float drift passes."""
        cfg = _make_valid_composite_config()
        # Drift code up by 1e-12, hidden down by 1e-12 (well within 1e-9)
        cfg["composite_score"]["code_generation_pass_at_1"] = (
            0.30 + 1e-12
        )
        cfg["composite_score"]["hidden_pass_rate"] = 0.10 - 1e-12
        validate_composite_schema(cfg)  # should not raise

    # -- each metric in [0, 1] --------------------------------------------

    def test_weight_above_one_raises(self) -> None:
        cfg = _make_valid_composite_config()
        # code = 1.10 (out of range, also breaks sum)
        cfg["composite_score"]["code_generation_pass_at_1"] = 1.10
        with pytest.raises(ValueError, match="out of range"):
            validate_composite_schema(cfg)

    def test_weight_negative_raises(self) -> None:
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["boundary_pass_at_1"] = -0.05
        with pytest.raises(ValueError, match="out of range"):
            validate_composite_schema(cfg)

    def test_weight_zero_allowed(self) -> None:
        """0.0 is a valid weight (boundary_pass_at_1 = 0, redistribute)."""
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["boundary_pass_at_1"] = 0.0
        # Move the freed 0.15 to code to keep sum = 1.0
        cfg["composite_score"]["code_generation_pass_at_1"] = 0.45
        validate_composite_schema(cfg)  # should not raise

    def test_weight_one_allowed(self) -> None:
        """1.0 is a valid weight (single-component composite)."""
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["code_generation_pass_at_1"] = 1.0
        cfg["composite_score"]["boundary_pass_at_1"] = 0.0
        cfg["composite_score"]["static_repair_success"] = 0.0
        cfg["composite_score"]["execution_repair_success"] = 0.0
        cfg["composite_score"]["hidden_pass_rate"] = 0.0
        validate_composite_schema(cfg)  # should not raise

    # -- NaN / Inf hard fail ----------------------------------------------

    def test_nan_weight_raises(self) -> None:
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["static_repair_success"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_composite_schema(cfg)

    def test_inf_weight_raises(self) -> None:
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["execution_repair_success"] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            validate_composite_schema(cfg)

    def test_negative_inf_weight_raises(self) -> None:
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["hidden_pass_rate"] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            validate_composite_schema(cfg)

    # -- schema_version exact match ---------------------------------------

    def test_schema_version_mismatch_raises(self) -> None:
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["schema_version"] = "0.0.9"
        with pytest.raises(ValueError, match="schema_version mismatch"):
            validate_composite_schema(cfg)

    def test_schema_version_future_raises(self) -> None:
        """Even a 'newer' version must hard-fail (exact match only)."""
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["schema_version"] = "1.0.1"
        with pytest.raises(ValueError, match="schema_version mismatch"):
            validate_composite_schema(cfg)

    def test_schema_version_none_raises(self) -> None:
        cfg = _make_valid_composite_config()
        cfg["composite_score"]["schema_version"] = None
        with pytest.raises(ValueError, match="schema_version mismatch"):
            validate_composite_schema(cfg)


# ---------------------------------------------------------------------------
# P2.4: CANONICAL_METRICS list completeness
# ---------------------------------------------------------------------------


class TestCanonicalMetrics:
    """Issue #14 P2.4: canonical metric-name list is complete & unambiguous."""

    def test_canonical_metrics_is_list_of_strings(self) -> None:
        assert isinstance(CANONICAL_METRICS, list)
        assert len(CANONICAL_METRICS) > 0
        assert all(isinstance(m, str) for m in CANONICAL_METRICS)

    def test_canonical_metrics_has_expected_members(self) -> None:
        expected = {
            "overall_executable_pass",
            "codegen_pass_at_1",
            "boundary_pass_at_1",
            "static_repair_success",
            "execution_repair_success",
            "hidden_pass_rate",
            "syntax_rate",
            "format_compliance_rate",
            "timeout_rate",
            "strict_family_pass",
        }
        assert set(CANONICAL_METRICS) == expected

    def test_canonical_metrics_has_ten_entries(self) -> None:
        assert len(CANONICAL_METRICS) == 10

    def test_canonical_metrics_no_legacy_aliases(self) -> None:
        """Forbidden aliases must NOT appear in CANONICAL_METRICS."""
        forbidden = {
            "pass_at_1",        # ambiguous: code vs boundary vs overall
            "codegen_pass1",    # legacy baseline key
            "codegen_pass",     # informal
            "overall pass",     # whitespace variant
            "pass1",            # short alias
        }
        for alias in forbidden:
            assert alias not in CANONICAL_METRICS, (
                f"legacy alias {alias!r} must not appear in CANONICAL_METRICS"
            )

    def test_canonical_metrics_no_duplicates(self) -> None:
        assert len(CANONICAL_METRICS) == len(set(CANONICAL_METRICS))

    def test_composite_schema_version_constant(self) -> None:
        """COMPOSITE_SCHEMA_VERSION is the expected string."""
        assert COMPOSITE_SCHEMA_VERSION == "1.0.0"
        assert isinstance(COMPOSITE_SCHEMA_VERSION, str)
