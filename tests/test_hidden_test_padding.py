"""tests/test_hidden_test_padding.py -- Unit tests for src/hidden_test_padding.py.

Covers the 7 tests specified in ``.superpowers/sdd/task-3-redo-brief.md``:

  1. test_pad_hidden_tests_already_sufficient
  2. test_pad_hidden_tests_pads_to_3
  3. test_pad_hidden_tests_syntax_error
  4. test_pad_hidden_tests_no_functions
  5. test_pad_hidden_tests_insufficient
  6. test_normalize_public_tests_pure_bare_assert
  7. test_normalize_public_tests_mixed_format

All tests use synthetic Samples built in-memory. Tests that invoke
``pad_hidden_tests`` may spawn one ``run_python_code`` subprocess to
capture expected outputs for boundary candidates (sub-100ms for the
trivial ``add(a, b)`` function used here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hidden_test_padding import (  # noqa: E402
    normalize_public_tests_for_pytest,
    pad_hidden_tests,
)
from src.schemas import Sample, Verification  # noqa: E402
from src.validators import verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)


def _make_sample(
    *,
    target_code: str = "def add(a, b):\n    return a + b\n",
    public_tests: str = (
        "assert add(1, 2) == 3\n"
        "assert add(0, 0) == 0\n"
    ),
    hidden_tests: str = (
        "assert add(-1, 1) == 0\n"
        "assert add(10, -5) == 5\n"
        "assert add(100, 200) == 300\n"
    ),
) -> Sample:
    """Build a minimal but valid Sample for tests."""
    return Sample(
        sample_id="mbpp_42_test",
        family_id="mbpp_fam_42",
        difficulty=1,
        task_type="code_generation",
        language="python",
        skill_tags=["test"],
        instruction="Write a function.",
        broken_code=None,
        execution_feedback=None,
        target_code=target_code,
        public_tests=public_tests,
        hidden_tests=hidden_tests,
        verified=False,
        verification=_PLACEHOLDER_VER,
        generator="test",
        created_at="2026-01-01T00:00:00+00:00",
        dataset_version="mbpp-v1",
    )


# ---------------------------------------------------------------------------
# Tests 1-5: pad_hidden_tests
# ---------------------------------------------------------------------------

def test_pad_hidden_tests_already_sufficient():
    """Sample with hidden_count >= 3 -> returned unchanged."""
    s = _make_sample()  # default has 3 hidden asserts
    padded, reason = pad_hidden_tests(s, target_count=3)
    assert reason is None
    assert padded.hidden_tests == s.hidden_tests


def test_pad_hidden_tests_pads_to_3():
    """Sample with hidden_count=1 -> padded to >= 3, new tests pass."""
    s = _make_sample(
        hidden_tests="assert add(-1, 1) == 0\n",
    )
    assert s.hidden_tests.count("assert ") == 1
    padded, reason = pad_hidden_tests(s, target_count=3)
    assert reason is None, f"unexpected rejection reason: {reason}"
    assert padded.hidden_tests.count("assert ") >= 3
    # Verify the new tests pass against target_code.
    sv = verify_sample(padded, run_ruff=False)
    assert sv.is_accepted, (
        f"padded hidden tests should pass against target_code; "
        f"messages: {sv.messages}"
    )


def test_pad_hidden_tests_syntax_error():
    """Malformed target_code -> returns sample unchanged + syntax_error reason."""
    s = _make_sample(
        target_code="def add(a, b\n    return a + b\n",  # malformed
        hidden_tests="assert add(1, 2) == 3\n",
    )
    padded, reason = pad_hidden_tests(s, target_count=3)
    assert reason == "hidden_padding_failed_syntax_error"
    assert padded.hidden_tests == s.hidden_tests


def test_pad_hidden_tests_no_functions():
    """target_code with no top-level def -> returns sample unchanged +
    no_functions reason."""
    s = _make_sample(
        target_code="x = 1\ny = 2\n",  # valid syntax, no function def
        hidden_tests="assert x == 1\n",
    )
    padded, reason = pad_hidden_tests(s, target_count=3)
    assert reason == "hidden_padding_failed_no_functions"
    assert padded.hidden_tests == s.hidden_tests


def test_pad_hidden_tests_insufficient():
    """No extractable test calls for the function -> returns sample unchanged +
    insufficient reason.

    Here target_code defines ``add`` but public_tests only calls ``other(...)``.
    """
    s = _make_sample(
        target_code="def add(a, b):\n    return a + b\n",
        public_tests="assert other(1, 2) == 3\nassert other(0, 0) == 0\n",
        hidden_tests="assert add(-1, 1) == 0\n",
    )
    padded, reason = pad_hidden_tests(s, target_count=3)
    assert reason == "hidden_padding_insufficient"
    assert padded.hidden_tests == s.hidden_tests


# ---------------------------------------------------------------------------
# Tests 6-7: normalize_public_tests_for_pytest
# ---------------------------------------------------------------------------

def test_normalize_public_tests_pure_bare_assert():
    """public_tests with no ``from solution`` line -> returned unchanged."""
    public_tests = (
        "assert add(1, 2) == 3\n"
        "assert add(0, 0) == 0\n"
    )
    assert normalize_public_tests_for_pytest(public_tests) == public_tests


def test_normalize_public_tests_mixed_format():
    """Bare assert before ``from solution`` line -> prefixed with
    ``from solution import *``."""
    public_tests = (
        "assert add(1, 2) == 3\n"
        "\n"
        "from solution import add\n"
        "\n"
        "def test_add_positive():\n"
        "    assert add(2, 3) == 5\n"
    )
    normalized = normalize_public_tests_for_pytest(public_tests)
    assert normalized.startswith("from solution import *\n\n")
    # The rest of the content is preserved.
    assert "assert add(1, 2) == 3" in normalized
    assert "from solution import add" in normalized
    assert "def test_add_positive" in normalized
