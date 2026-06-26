"""Tests for src/metrics.py.

All tests are pure-function tests — no subprocesses, no I/O.

Coverage
--------
- Each aggregate function with an empty list returns 0.0.
- Each aggregate function with a homogeneous list returns the correct rate.
- Mixed lists (generation + repair outcomes) behave correctly.
- Repair-specific metrics (repair_success_rate, regression_rate) filter out
  generation outcomes.
- pass_at_1 filters out repair outcomes.
- summarize() returns all expected keys with correct values and counts.
"""

from __future__ import annotations

import pytest

from src.metrics import (
    EvalOutcome,
    format_compliance_rate,
    hidden_pass_rate,
    pass_at_1,
    regression_rate,
    repair_success_rate,
    summarize,
    syntax_rate,
    timeout_rate,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _gen(
    *,
    syntax_ok: bool = True,
    public_passed: bool = True,
    hidden_passed: bool = True,
    format_ok: bool = True,
    timed_out: bool = False,
) -> EvalOutcome:
    """Create a code_generation EvalOutcome."""
    return EvalOutcome(
        task_type="code_generation",
        syntax_ok=syntax_ok,
        public_passed=public_passed,
        hidden_passed=hidden_passed,
        format_ok=format_ok,
        timed_out=timed_out,
        is_repair=False,
        repair_succeeded=None,
        broke_other_tests=None,
    )


def _repair(
    *,
    syntax_ok: bool = True,
    public_passed: bool = True,
    hidden_passed: bool = True,
    format_ok: bool = True,
    timed_out: bool = False,
    repair_succeeded: bool | None = True,
    broke_other_tests: bool | None = False,
    task_type: str = "static_repair",
) -> EvalOutcome:
    """Create a repair EvalOutcome."""
    return EvalOutcome(
        task_type=task_type,
        syntax_ok=syntax_ok,
        public_passed=public_passed,
        hidden_passed=hidden_passed,
        format_ok=format_ok,
        timed_out=timed_out,
        is_repair=True,
        repair_succeeded=repair_succeeded,
        broke_other_tests=broke_other_tests,
    )


# ---------------------------------------------------------------------------
# Empty-list guard — every metric returns 0.0
# ---------------------------------------------------------------------------


class TestEmptyList:
    def test_pass_at_1_empty(self):
        assert pass_at_1([]) == 0.0

    def test_syntax_rate_empty(self):
        assert syntax_rate([]) == 0.0

    def test_hidden_pass_rate_empty(self):
        assert hidden_pass_rate([]) == 0.0

    def test_format_compliance_rate_empty(self):
        assert format_compliance_rate([]) == 0.0

    def test_timeout_rate_empty(self):
        assert timeout_rate([]) == 0.0

    def test_repair_success_rate_empty(self):
        assert repair_success_rate([]) == 0.0

    def test_regression_rate_empty(self):
        assert regression_rate([]) == 0.0


# ---------------------------------------------------------------------------
# pass_at_1
# ---------------------------------------------------------------------------


class TestPassAt1:
    def test_all_pass(self):
        outcomes = [_gen(), _gen(), _gen()]
        assert pass_at_1(outcomes) == pytest.approx(1.0)

    def test_none_pass(self):
        outcomes = [_gen(syntax_ok=False), _gen(hidden_passed=False)]
        assert pass_at_1(outcomes) == pytest.approx(0.0)

    def test_half_pass(self):
        outcomes = [_gen(), _gen(public_passed=False)]
        assert pass_at_1(outcomes) == pytest.approx(0.5)

    def test_repair_outcomes_ignored(self):
        """Repair outcomes must not count toward pass_at_1."""
        outcomes = [_repair(repair_succeeded=True)]
        # No generation outcomes → 0.0
        assert pass_at_1(outcomes) == 0.0

    def test_mixed_only_counts_generation(self):
        outcomes = [_gen(), _repair(), _gen(hidden_passed=False)]
        # 1 out of 2 generation outcomes pass
        assert pass_at_1(outcomes) == pytest.approx(0.5)

    def test_no_generation_outcomes_returns_zero(self):
        outcomes = [_repair(), _repair()]
        assert pass_at_1(outcomes) == 0.0


# ---------------------------------------------------------------------------
# syntax_rate
# ---------------------------------------------------------------------------


class TestSyntaxRate:
    def test_all_ok(self):
        assert syntax_rate([_gen(), _repair()]) == pytest.approx(1.0)

    def test_none_ok(self):
        outcomes = [_gen(syntax_ok=False), _repair(syntax_ok=False)]
        assert syntax_rate(outcomes) == pytest.approx(0.0)

    def test_partial(self):
        outcomes = [_gen(syntax_ok=True), _gen(syntax_ok=False), _gen(syntax_ok=False)]
        assert syntax_rate(outcomes) == pytest.approx(1 / 3)

    def test_counts_all_task_types(self):
        outcomes = [_gen(syntax_ok=True), _repair(syntax_ok=False)]
        assert syntax_rate(outcomes) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# hidden_pass_rate
# ---------------------------------------------------------------------------


class TestHiddenPassRate:
    def test_all_hidden_pass(self):
        assert hidden_pass_rate([_gen(), _gen()]) == pytest.approx(1.0)

    def test_none_hidden_pass(self):
        outcomes = [_gen(hidden_passed=False), _gen(hidden_passed=False)]
        assert hidden_pass_rate(outcomes) == pytest.approx(0.0)

    def test_three_quarters(self):
        outcomes = [_gen(), _gen(), _gen(), _gen(hidden_passed=False)]
        assert hidden_pass_rate(outcomes) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# format_compliance_rate
# ---------------------------------------------------------------------------


class TestFormatComplianceRate:
    def test_all_ok(self):
        assert format_compliance_rate([_gen(), _gen()]) == pytest.approx(1.0)

    def test_none_ok(self):
        outcomes = [_gen(format_ok=False), _gen(format_ok=False)]
        assert format_compliance_rate(outcomes) == pytest.approx(0.0)

    def test_partial(self):
        outcomes = [_gen(format_ok=True), _gen(format_ok=False)]
        assert format_compliance_rate(outcomes) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# timeout_rate
# ---------------------------------------------------------------------------


class TestTimeoutRate:
    def test_no_timeouts(self):
        assert timeout_rate([_gen(), _gen()]) == pytest.approx(0.0)

    def test_all_timeout(self):
        outcomes = [_gen(timed_out=True), _gen(timed_out=True)]
        assert timeout_rate(outcomes) == pytest.approx(1.0)

    def test_one_of_four(self):
        outcomes = [_gen(timed_out=True), _gen(), _gen(), _gen()]
        assert timeout_rate(outcomes) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# repair_success_rate
# ---------------------------------------------------------------------------


class TestRepairSuccessRate:
    def test_all_succeeded(self):
        outcomes = [_repair(repair_succeeded=True), _repair(repair_succeeded=True)]
        assert repair_success_rate(outcomes) == pytest.approx(1.0)

    def test_none_succeeded(self):
        outcomes = [_repair(repair_succeeded=False), _repair(repair_succeeded=False)]
        assert repair_success_rate(outcomes) == pytest.approx(0.0)

    def test_half_succeeded(self):
        outcomes = [_repair(repair_succeeded=True), _repair(repair_succeeded=False)]
        assert repair_success_rate(outcomes) == pytest.approx(0.5)

    def test_generation_outcomes_excluded(self):
        """Generation outcomes must not count toward repair_success_rate."""
        outcomes = [_gen(), _repair(repair_succeeded=False)]
        # 0 out of 1 repair succeeded
        assert repair_success_rate(outcomes) == pytest.approx(0.0)

    def test_no_repair_outcomes_returns_zero(self):
        outcomes = [_gen(), _gen()]
        assert repair_success_rate(outcomes) == 0.0

    def test_execution_repair_counted(self):
        outcomes = [
            _repair(task_type="execution_repair", repair_succeeded=True),
            _repair(task_type="static_repair", repair_succeeded=False),
        ]
        assert repair_success_rate(outcomes) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# regression_rate
# ---------------------------------------------------------------------------


class TestRegressionRate:
    def test_no_regressions(self):
        outcomes = [_repair(broke_other_tests=False), _repair(broke_other_tests=False)]
        assert regression_rate(outcomes) == pytest.approx(0.0)

    def test_all_regressed(self):
        outcomes = [_repair(broke_other_tests=True), _repair(broke_other_tests=True)]
        assert regression_rate(outcomes) == pytest.approx(1.0)

    def test_half_regressed(self):
        outcomes = [_repair(broke_other_tests=True), _repair(broke_other_tests=False)]
        assert regression_rate(outcomes) == pytest.approx(0.5)

    def test_generation_outcomes_excluded(self):
        outcomes = [_gen(), _repair(broke_other_tests=True)]
        # 1 out of 1 repair regressed
        assert regression_rate(outcomes) == pytest.approx(1.0)

    def test_no_repair_outcomes_returns_zero(self):
        outcomes = [_gen()]
        assert regression_rate(outcomes) == 0.0

    def test_none_value_not_counted_as_regression(self):
        """broke_other_tests=None (not evaluated) must not count as True."""
        outcomes = [_repair(broke_other_tests=None), _repair(broke_other_tests=False)]
        assert regression_rate(outcomes) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    _EXPECTED_KEYS = {
        "pass_at_1",
        "syntax_rate",
        "hidden_pass_rate",
        "format_compliance_rate",
        "timeout_rate",
        "repair_success_rate",
        "regression_rate",
        "n_total",
        "n_generation",
        "n_repair",
    }

    def test_empty_list_returns_all_zeros(self):
        result = summarize([])
        assert set(result.keys()) == self._EXPECTED_KEYS
        for key, value in result.items():
            assert value == 0.0, f"Expected 0.0 for {key!r}, got {value!r}"

    def test_returns_all_expected_keys(self):
        outcomes = [_gen(), _repair()]
        result = summarize(outcomes)
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_counts_are_correct(self):
        outcomes = [_gen(), _gen(), _repair(), _repair(), _repair()]
        result = summarize(outcomes)
        assert result["n_total"] == 5.0
        assert result["n_generation"] == 2.0
        assert result["n_repair"] == 3.0

    def test_counts_all_zero_for_empty(self):
        result = summarize([])
        assert result["n_total"] == 0.0
        assert result["n_generation"] == 0.0
        assert result["n_repair"] == 0.0

    def test_metric_values_match_individual_functions(self):
        """summarize() must be consistent with the individual metric functions."""
        outcomes = [
            _gen(syntax_ok=True, hidden_passed=True, format_ok=True, timed_out=False),
            _gen(syntax_ok=False, hidden_passed=False, format_ok=False, timed_out=True),
            _repair(repair_succeeded=True, broke_other_tests=False),
            _repair(repair_succeeded=False, broke_other_tests=True),
        ]
        result = summarize(outcomes)
        assert result["pass_at_1"] == pytest.approx(pass_at_1(outcomes))
        assert result["syntax_rate"] == pytest.approx(syntax_rate(outcomes))
        assert result["hidden_pass_rate"] == pytest.approx(hidden_pass_rate(outcomes))
        assert result["format_compliance_rate"] == pytest.approx(format_compliance_rate(outcomes))
        assert result["timeout_rate"] == pytest.approx(timeout_rate(outcomes))
        assert result["repair_success_rate"] == pytest.approx(repair_success_rate(outcomes))
        assert result["regression_rate"] == pytest.approx(regression_rate(outcomes))

    def test_all_values_are_float(self):
        outcomes = [_gen(), _repair()]
        for value in summarize(outcomes).values():
            assert isinstance(value, float)

    def test_metric_rates_in_unit_interval(self):
        outcomes = [_gen(), _gen(syntax_ok=False), _repair()]
        for key, value in summarize(outcomes).items():
            if not key.startswith("n_"):
                assert 0.0 <= value <= 1.0, f"{key}={value} is outside [0, 1]"
