"""Tests for src/validators.py.

All tests use REAL tools — no mocks, no patches.

Coverage
--------
compile_check       valid code → ok; syntax error → not ok + message
ruff_check          clean code → ok; unused import → not ok
mypy_check          smoke test that the function returns (bool, str)
verify_sample       accepts a correct sample; rejects hidden-test failure
verify_broken_is_broken  True for genuinely broken code; False for passing code
"""

from __future__ import annotations

import pytest

from src.schemas import Sample, Verification
from src.validators import (
    SampleVerification,
    compile_check,
    mypy_check,
    ruff_check,
    verify_broken_is_broken,
    verify_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_VERIFICATION = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)


def _make_sample(**overrides) -> Sample:
    """Build a minimal valid Sample for testing."""
    defaults: dict = {
        "sample_id": "test-001",
        "family_id": "fam-001",
        "difficulty": 0,
        "task_type": "code_generation",
        "language": "python",
        "skill_tags": ["arithmetic"],
        "instruction": "Write an add function.",
        "target_code": "def add(a, b):\n    return a + b\n",
        "public_tests": (
            "from solution import add\n\n"
            "def test_add_positive():\n"
            "    assert add(1, 2) == 3\n"
        ),
        "hidden_tests": (
            "from solution import add\n\n"
            "def test_add_negative():\n"
            "    assert add(-1, -2) == -3\n"
        ),
        "verified": False,
        "verification": _BASE_VERIFICATION,
        "generator": "test-harness",
        "created_at": "2026-01-01T00:00:00",
        "dataset_version": "v0.1",
    }
    defaults.update(overrides)
    return Sample(**defaults)


# ---------------------------------------------------------------------------
# compile_check
# ---------------------------------------------------------------------------


class TestCompileCheck:
    def test_valid_code_returns_ok(self):
        ok, msg = compile_check("def foo(x):\n    return x + 1\n")
        assert ok is True
        assert msg == ""

    def test_syntax_error_returns_not_ok(self):
        ok, msg = compile_check("def foo(:\n    pass\n")
        assert ok is False
        assert msg != ""

    def test_empty_string_is_valid(self):
        ok, msg = compile_check("")
        assert ok is True
        assert msg == ""

    def test_multiline_valid_code(self):
        code = "x = 1\ny = 2\nz = x + y\n"
        ok, msg = compile_check(code)
        assert ok is True

    def test_missing_colon_is_syntax_error(self):
        ok, msg = compile_check("if True\n    pass\n")
        assert ok is False
        assert len(msg) > 0

    def test_embedded_null_byte_does_not_crash(self):
        """compile() raises ValueError (not SyntaxError) on null bytes — must
        be caught and reported, never propagated."""
        ok, msg = compile_check("x = 1\x00\n")
        assert ok is False
        assert msg != ""


# ---------------------------------------------------------------------------
# ruff_check
# ---------------------------------------------------------------------------


class TestRuffCheck:
    def test_clean_code_passes(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        ok, output = ruff_check(code)
        assert ok is True

    def test_unused_import_flagged(self):
        # F401 — unused import — is enabled in ruff's default rule set.
        code = "import os\n\ndef greet():\n    return 'hello'\n"
        ok, output = ruff_check(code)
        assert ok is False
        # F401 (unused import) is a stable ruff default rule.
        assert "F401" in output

    def test_returns_tuple_of_bool_and_str(self):
        ok, output = ruff_check("x = 1\n")
        assert isinstance(ok, bool)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# mypy_check  (smoke tests — just confirm the interface works)
# ---------------------------------------------------------------------------


class TestMypyCheck:
    def test_returns_bool_and_str(self):
        # Simple untyped code: mypy may or may not flag it, but the function
        # must return (bool, str) without raising.
        ok, output = mypy_check("def add(a, b):\n    return a + b\n")
        assert isinstance(ok, bool)
        assert isinstance(output, str)

    def test_correct_typed_code_passes(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        ok, output = mypy_check(code)
        assert ok is True


# ---------------------------------------------------------------------------
# verify_sample
# ---------------------------------------------------------------------------


class TestVerifySample:
    def test_correct_sample_is_accepted(self):
        """A sample with correct target_code and matching tests is accepted."""
        sample = _make_sample()
        result = verify_sample(sample, run_ruff=False, pytest_timeout_s=15.0)
        assert isinstance(result, SampleVerification)
        assert result.verification.syntax_ok is True
        assert result.public_ok is True
        assert result.hidden_ok is True
        assert result.verification.pytest_ok is True
        assert result.verification.timeout is False
        assert result.is_accepted is True

    def test_syntax_error_sample_rejected(self):
        """A sample with a syntax error in target_code is rejected."""
        sample = _make_sample(target_code="def add(a, b\n    return a + b\n")
        result = verify_sample(sample, run_ruff=False, pytest_timeout_s=15.0)
        assert result.verification.syntax_ok is False
        assert result.verification.pytest_ok is False
        assert result.is_accepted is False
        # Should have a compile error message
        assert any("compile" in m for m in result.messages)

    def test_hidden_test_failure_rejects_sample(self):
        """target_code that passes public tests but fails hidden tests is rejected."""
        # A function that handles positive but not negative addition
        broken_target = (
            "def add(a, b):\n"
            "    if a < 0 or b < 0:\n"
            "        raise ValueError('negatives not supported')\n"
            "    return a + b\n"
        )
        sample = _make_sample(target_code=broken_target)
        result = verify_sample(sample, run_ruff=False, pytest_timeout_s=15.0)
        assert result.public_ok is True    # public test uses positives
        assert result.hidden_ok is False   # hidden test uses negatives
        assert result.verification.pytest_ok is False
        assert result.is_accepted is False

    def test_ruff_advisory_does_not_reject(self):
        """ruff_ok=False should not cause is_accepted to be False when pytest passes."""
        # Code with an unused import that passes tests
        code_with_lint = (
            "import os  # unused — will be flagged by ruff\n\n"
            "def add(a, b):\n"
            "    return a + b\n"
        )
        sample = _make_sample(target_code=code_with_lint)
        result = verify_sample(sample, run_ruff=True, pytest_timeout_s=15.0)
        # Ruff should flag the unused import
        assert result.verification.ruff_ok is False
        # But the sample is still accepted because pytest passes
        assert result.is_accepted is True

    def test_empty_hidden_tests_treated_as_pass(self):
        """When hidden_tests is blank, hidden_ok defaults to True."""
        sample = _make_sample(hidden_tests="")
        result = verify_sample(sample, run_ruff=False, pytest_timeout_s=15.0)
        assert result.hidden_ok is True
        assert result.is_accepted is True

    def test_result_duration_is_positive(self):
        sample = _make_sample()
        result = verify_sample(sample, run_ruff=False, pytest_timeout_s=15.0)
        assert result.duration_s > 0.0

    def test_mypy_ok_is_none_when_not_requested(self):
        sample = _make_sample()
        result = verify_sample(sample, run_ruff=False, run_mypy=False, pytest_timeout_s=15.0)
        assert result.mypy_ok is None

    def test_mypy_ok_is_bool_when_requested(self):
        sample = _make_sample()
        result = verify_sample(sample, run_ruff=False, run_mypy=True, pytest_timeout_s=15.0)
        assert isinstance(result.mypy_ok, bool)


# ---------------------------------------------------------------------------
# verify_broken_is_broken
# ---------------------------------------------------------------------------


class TestVerifyBrokenIsBroken:
    def _make_repair_sample(self, broken_code: str) -> Sample:
        """Create a static_repair sample with the given broken_code."""
        return _make_sample(
            task_type="static_repair",
            broken_code=broken_code,
        )

    def test_wrong_implementation_returns_true(self):
        """Code that subtracts instead of adds genuinely fails the tests."""
        broken = "def add(a, b):\n    return a - b\n"
        sample = self._make_repair_sample(broken)
        assert verify_broken_is_broken(sample, pytest_timeout_s=15.0) is True

    def test_correct_implementation_returns_false(self):
        """Code that actually passes all tests is NOT a valid broken sample."""
        correct = "def add(a, b):\n    return a + b\n"
        sample = self._make_repair_sample(correct)
        assert verify_broken_is_broken(sample, pytest_timeout_s=15.0) is False

    def test_syntax_error_broken_code_returns_true(self):
        """Syntactically invalid code cannot even be imported → pytest error → True."""
        invalid = "def add(a b):\n    return a + b\n"
        sample = self._make_repair_sample(invalid)
        assert verify_broken_is_broken(sample, pytest_timeout_s=15.0) is True

    def test_passes_public_fails_hidden_is_broken(self):
        """Broken code that passes public but fails hidden still returns True."""
        # A function correct for positive inputs, wrong for negative
        broken = (
            "def add(a, b):\n"
            "    if a < 0 or b < 0:\n"
            "        return 0\n"
            "    return a + b\n"
        )
        sample = self._make_repair_sample(broken)
        # Public test: add(1, 2) == 3 → passes
        # Hidden test: add(-1, -2) == -3 → fails
        assert verify_broken_is_broken(sample, pytest_timeout_s=15.0) is True

    def test_blank_broken_code_raises(self):
        """A generation sample (broken_code=None) must raise, not false-positive."""
        gen_sample = _make_sample(task_type="code_generation", broken_code=None)
        with pytest.raises(ValueError, match="non-empty broken_code"):
            verify_broken_is_broken(gen_sample, pytest_timeout_s=15.0)

    def test_whitespace_only_broken_code_raises(self):
        """Whitespace-only broken_code is treated as blank and raises.

        Uses a code_generation sample because the schema itself rejects blank
        broken_code for repair task types at construction time — here we are
        exercising verify_broken_is_broken's own guard.
        """
        sample = _make_sample(task_type="code_generation", broken_code="   \n  \n")
        with pytest.raises(ValueError, match="non-empty broken_code"):
            verify_broken_is_broken(sample, pytest_timeout_s=15.0)
