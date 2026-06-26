"""Evaluation metrics for the Qwen3-0.6B Code Recovery Lab (spec §12.1).

All aggregate functions:
- Accept ``list[EvalOutcome]``.
- Return ``float`` values in ``[0.0, 1.0]``.
- Return ``0.0`` on the empty list — division by zero never occurs.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Per-sample outcome record
# ---------------------------------------------------------------------------


@dataclass
class EvalOutcome:
    """Captures all evaluation signals for a single model output.

    Attributes
    ----------
    task_type : str
        One of ``"code_generation"``, ``"static_repair"``,
        ``"execution_repair"``.
    syntax_ok : bool
        True when the model's output compiles without ``SyntaxError``.
    public_passed : bool
        True when the public test suite passes.
    hidden_passed : bool
        True when the hidden test suite passes (or when there is no hidden
        suite — treat as vacuously True).
    format_ok : bool
        True when the model output obeys the required format, e.g. a
        parseable triple-backtick python code block.
    timed_out : bool
        True when any sub-check hit its timeout limit.
    is_repair : bool
        True for ``static_repair`` / ``execution_repair`` tasks; False for
        ``code_generation``.
    repair_succeeded : bool | None
        For repair tasks: True when the repaired code passes all tests.
        ``None`` for generation tasks.
    broke_other_tests : bool | None
        For repair tasks: True when a previously-passing test now fails
        (regression introduced by the repair).  ``None`` for generation tasks.
    """

    task_type: str
    syntax_ok: bool
    public_passed: bool
    hidden_passed: bool
    format_ok: bool
    timed_out: bool
    is_repair: bool
    repair_succeeded: bool | None
    broke_other_tests: bool | None


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, or 0.0 when denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate metric functions
# ---------------------------------------------------------------------------


def pass_at_1(outcomes: list[EvalOutcome]) -> float:
    """Fraction of code_generation outcomes where ALL tests pass.

    Filters to ``task_type == "code_generation"`` outcomes, then counts those
    where ``syntax_ok AND public_passed AND hidden_passed``.

    Returns 0.0 when there are no generation outcomes.
    """
    gen = [o for o in outcomes if o.task_type == "code_generation"]
    passed = sum(
        1 for o in gen if o.syntax_ok and o.public_passed and o.hidden_passed
    )
    return _safe_rate(passed, len(gen))


def syntax_rate(outcomes: list[EvalOutcome]) -> float:
    """Fraction of *all* outcomes where ``syntax_ok`` is True."""
    return _safe_rate(
        sum(1 for o in outcomes if o.syntax_ok),
        len(outcomes),
    )


def hidden_pass_rate(outcomes: list[EvalOutcome]) -> float:
    """Fraction of *all* outcomes where ``hidden_passed`` is True."""
    return _safe_rate(
        sum(1 for o in outcomes if o.hidden_passed),
        len(outcomes),
    )


def format_compliance_rate(outcomes: list[EvalOutcome]) -> float:
    """Fraction of *all* outcomes where ``format_ok`` is True."""
    return _safe_rate(
        sum(1 for o in outcomes if o.format_ok),
        len(outcomes),
    )


def timeout_rate(outcomes: list[EvalOutcome]) -> float:
    """Fraction of *all* outcomes where ``timed_out`` is True."""
    return _safe_rate(
        sum(1 for o in outcomes if o.timed_out),
        len(outcomes),
    )


def repair_success_rate(outcomes: list[EvalOutcome]) -> float:
    """Fraction of repair outcomes where ``repair_succeeded`` is True.

    Filters to ``is_repair == True``; returns 0.0 when there are none.
    """
    repairs = [o for o in outcomes if o.is_repair]
    succeeded = sum(1 for o in repairs if o.repair_succeeded is True)
    return _safe_rate(succeeded, len(repairs))


def regression_rate(outcomes: list[EvalOutcome]) -> float:
    """Fraction of repair outcomes where ``broke_other_tests`` is True.

    Filters to ``is_repair == True``; returns 0.0 when there are none.
    """
    repairs = [o for o in outcomes if o.is_repair]
    regressions = sum(1 for o in repairs if o.broke_other_tests is True)
    return _safe_rate(regressions, len(repairs))


def summarize(outcomes: list[EvalOutcome]) -> dict[str, float]:
    """Return all metrics plus sample counts as a single flat dict.

    Metric keys (floats in ``[0, 1]``)
    ------------------------------------
    ``pass_at_1``, ``syntax_rate``, ``hidden_pass_rate``,
    ``format_compliance_rate``, ``timeout_rate``,
    ``repair_success_rate``, ``regression_rate``

    Count keys (int values stored as float)
    ----------------------------------------
    ``n_total``, ``n_generation``, ``n_repair``
    """
    n_total = len(outcomes)
    n_generation = sum(1 for o in outcomes if o.task_type == "code_generation")
    n_repair = sum(1 for o in outcomes if o.is_repair)

    return {
        "pass_at_1": pass_at_1(outcomes),
        "syntax_rate": syntax_rate(outcomes),
        "hidden_pass_rate": hidden_pass_rate(outcomes),
        "format_compliance_rate": format_compliance_rate(outcomes),
        "timeout_rate": timeout_rate(outcomes),
        "repair_success_rate": repair_success_rate(outcomes),
        "regression_rate": regression_rate(outcomes),
        "n_total": float(n_total),
        "n_generation": float(n_generation),
        "n_repair": float(n_repair),
    }
