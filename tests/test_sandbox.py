"""Tests for src/sandbox.py.

All tests use REAL subprocess execution — no mocks.

Test classes
------------
TestRunPythonCode  — exercises run_python_code()
TestRunPytest      — exercises run_pytest()
TestCheckCodeSafety — exercises check_code_safety()
"""

from __future__ import annotations

import os
import time

import pytest

from src.sandbox import (
    ExecResult,
    PytestResult,
    check_code_safety,
    run_python_code,
    run_pytest,
)


# ===========================================================================
# run_python_code
# ===========================================================================


class TestRunPythonCode:
    """Tests for run_python_code()."""

    def test_simple_print_stdout(self):
        result = run_python_code("print('hello sandbox')")
        assert result.returncode == 0
        assert "hello sandbox" in result.stdout
        assert result.timed_out is False
        assert result.stderr == ""

    def test_nonzero_exit_code_captured(self):
        result = run_python_code("import sys; sys.exit(42)")
        assert result.returncode == 42
        assert result.timed_out is False

    def test_timeout_sets_flag_and_does_not_hang(self):
        """A program sleeping 10 s must be killed within ~3 s of the 1.5 s limit."""
        t0 = time.perf_counter()
        result = run_python_code("import time; time.sleep(10)", timeout_s=1.5)
        elapsed = time.perf_counter() - t0

        assert result.timed_out is True
        assert result.returncode is None
        # Must not take anywhere near 10 s.
        assert elapsed < 6.0, f"Took {elapsed:.1f}s — process not killed promptly"

    def test_stdout_truncated_to_max_output_chars(self):
        """Output exceeding max_output_chars is cut and a marker is appended."""
        code = "print('x' * 5000)"
        result = run_python_code(code, max_output_chars=100)
        # Length may exceed 100 by the length of the truncation marker, but
        # must not be dramatically larger.
        assert result.stdout.endswith("...[truncated]")
        # The non-marker prefix is at most max_output_chars chars.
        assert len(result.stdout) <= 100 + len("...[truncated]")

    def test_tempdir_cleaned_up_after_success(self):
        result = run_python_code("pass")
        assert not os.path.exists(result.tempdir), (
            f"Temp directory {result.tempdir!r} was not removed after execution"
        )

    def test_tempdir_cleaned_up_after_timeout(self):
        result = run_python_code("import time; time.sleep(10)", timeout_s=1.5)
        assert result.timed_out is True
        assert not os.path.exists(result.tempdir), (
            f"Temp directory {result.tempdir!r} was not removed after timeout"
        )

    def test_extra_files_accessible_from_code(self):
        code = "import helper; print(helper.VALUE)"
        extra = {"helper.py": "VALUE = 'extra_file_value'\n"}
        result = run_python_code(code, extra_files=extra)
        assert result.returncode == 0
        assert "extra_file_value" in result.stdout

    def test_stderr_captured_separately(self):
        result = run_python_code("import sys; sys.stderr.write('err_marker')")
        assert result.returncode == 0
        assert "err_marker" in result.stderr
        # Must NOT bleed into stdout.
        assert "err_marker" not in result.stdout

    def test_duration_is_positive(self):
        result = run_python_code("pass")
        assert result.duration_s >= 0.0

    def test_syntax_error_in_code_has_nonzero_exit(self):
        result = run_python_code("def broken(:\n    pass")
        assert result.returncode != 0
        assert result.timed_out is False

    def test_result_type(self):
        result = run_python_code("pass")
        assert isinstance(result, ExecResult)


# ===========================================================================
# run_pytest
# ===========================================================================


_CORRECT_SOLUTION = """\
def add(a, b):
    return a + b
"""

_CORRECT_TESTS = """\
from solution import add

def test_add_positive():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, 1) == 0
"""

_WRONG_SOLUTION = """\
def add(a, b):
    return a - b   # intentionally wrong
"""

# The infinite-loop solution hangs on import; pytest will be killed by timeout.
_INFINITE_SOLUTION = """\
while True:
    pass
"""


class TestRunPytest:
    """Tests for run_pytest()."""

    def test_correct_solution_passes(self):
        result = run_pytest(_CORRECT_SOLUTION, _CORRECT_TESTS, timeout_s=15.0)
        assert result.passed is True
        # The fixture has exactly 2 tests — tighten so a parse regression is caught.
        assert result.num_passed == 2
        assert result.num_failed == 0
        assert result.num_errors == 0
        assert result.timed_out is False
        assert result.returncode == 0

    def test_wrong_solution_fails(self):
        result = run_pytest(_WRONG_SOLUTION, _CORRECT_TESTS, timeout_s=15.0)
        assert result.passed is False
        assert result.num_failed >= 1
        assert result.timed_out is False

    def test_solution_with_syntax_error_reports_failure(self):
        """A solution that fails to import is a pytest *error*, not 'failed'.

        Downstream consumers rely on num_failed >= 1 to detect any non-success,
        so collection/import errors must fold into num_failed (and surface in
        num_errors). This is the dominant failure mode for model-generated code.
        """
        broken = "def add(a, b)\n    return a + b\n"  # missing colon -> SyntaxError
        result = run_pytest(broken, _CORRECT_TESTS, timeout_s=15.0)
        assert result.passed is False
        assert result.num_failed >= 1
        assert result.num_errors >= 1
        assert result.timed_out is False

    def test_solution_missing_symbol_reports_failure(self):
        """Importing a missing symbol is a collection error -> num_failed >= 1."""
        # solution.py exists but does not define `add`.
        result = run_pytest("x = 1\n", _CORRECT_TESTS, timeout_s=15.0)
        assert result.passed is False
        assert result.num_failed >= 1
        assert result.num_errors >= 1

    def test_timeout_on_infinite_loop_solution(self):
        """Pytest must be killed within a few seconds; must not hang the suite."""
        t0 = time.perf_counter()
        result = run_pytest(_INFINITE_SOLUTION, _CORRECT_TESTS, timeout_s=2.0)
        elapsed = time.perf_counter() - t0

        assert result.timed_out is True
        assert result.passed is False
        assert elapsed < 8.0, f"Took {elapsed:.1f}s — process not killed promptly"

    def test_tempdir_cleaned_up(self):
        result = run_pytest(_CORRECT_SOLUTION, _CORRECT_TESTS)
        assert not os.path.exists(result.tempdir), (
            f"Temp directory {result.tempdir!r} was not removed"
        )

    def test_result_type(self):
        result = run_pytest(_CORRECT_SOLUTION, _CORRECT_TESTS)
        assert isinstance(result, PytestResult)

    def test_stdout_contains_pytest_output(self):
        result = run_pytest(_CORRECT_SOLUTION, _CORRECT_TESTS)
        # pytest -q always reports "passed" somewhere in stdout.
        assert "passed" in result.stdout

    # --- bare-assert MBPP-style normalization (P2 fix) ---

    def test_bare_assert_correct_solution_passes(self):
        """MBPP-style top-level asserts (no `from solution`, no `def test_*`)
        must be auto-normalized so a correct solution passes."""
        bare = "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n"
        result = run_pytest(_CORRECT_SOLUTION, bare, timeout_s=15.0)
        assert result.passed is True
        assert result.num_collected == 1
        assert result.num_passed == 1
        assert result.num_failed == 0

    def test_bare_assert_wrong_solution_fails(self):
        """MBPP-style bare asserts against a wrong solution must fail."""
        bare = "assert add(1, 2) == 3\n"
        result = run_pytest(_WRONG_SOLUTION, bare, timeout_s=15.0)
        assert result.passed is False
        assert result.num_failed >= 1

    def test_bare_assert_different_param_name_passes(self):
        """Bare asserts against a solution with different parameter names
        (but equivalent logic) must pass — this is the regression case from
        P2 Stage 2 evaluation where count_char used `str1` instead of `string`."""
        code = "def count_char(s, ch):\n    return s.count(ch)\n"
        bare = (
            "assert count_char('Python','o')==1\n"
            "assert count_char('little','t')==2\n"
        )
        result = run_pytest(code, bare, timeout_s=15.0)
        assert result.passed is True
        assert result.num_passed == 1

    def test_bare_assert_with_syntax_error_solution_fails(self):
        """A solution with a syntax error must fail bare-assert tests."""
        broken = "def add(a, b)\n    return a + b\n"  # missing colon
        bare = "assert add(1, 2) == 3\n"
        result = run_pytest(broken, bare, timeout_s=15.0)
        assert result.passed is False
        assert result.num_failed >= 1



# ===========================================================================
# check_code_safety
# ===========================================================================


class TestCheckCodeSafety:
    """Tests for check_code_safety()."""

    # --- patterns that MUST be flagged ---

    def test_import_socket_flagged(self):
        warnings = check_code_safety("import socket")
        assert len(warnings) > 0
        assert any("socket" in w for w in warnings)

    def test_import_subprocess_flagged(self):
        warnings = check_code_safety("import subprocess")
        assert any("subprocess" in w for w in warnings)

    def test_import_ctypes_flagged(self):
        warnings = check_code_safety("import ctypes")
        assert any("ctypes" in w for w in warnings)

    def test_import_winreg_flagged(self):
        warnings = check_code_safety("import winreg")
        assert any("winreg" in w for w in warnings)

    def test_os_system_call_flagged(self):
        code = "import os\nos.system('echo hello')"
        warnings = check_code_safety(code)
        assert any("os.system" in w for w in warnings)

    def test_from_os_import_system_flagged(self):
        warnings = check_code_safety("from os import system")
        assert any("system" in w for w in warnings)

    def test_from_shutil_import_rmtree_flagged(self):
        warnings = check_code_safety("from shutil import rmtree")
        assert any("rmtree" in w for w in warnings)

    def test_network_import_urllib_flagged(self):
        warnings = check_code_safety("import urllib.request")
        assert any("urllib" in w for w in warnings)

    def test_network_import_requests_flagged(self):
        warnings = check_code_safety("import requests")
        assert any("requests" in w for w in warnings)

    def test_network_import_http_flagged(self):
        warnings = check_code_safety("import http.client")
        assert any("http" in w for w in warnings)

    def test_exec_call_flagged(self):
        warnings = check_code_safety("exec('print(1)')")
        assert any("exec" in w for w in warnings)

    def test_eval_call_flagged(self):
        warnings = check_code_safety("eval('1 + 1')")
        assert any("eval" in w for w in warnings)

    def test_dunder_import_call_flagged(self):
        warnings = check_code_safety("__import__('os').system('echo hi')")
        assert any("__import__" in w for w in warnings)

    def test_os_popen_call_flagged(self):
        code = "import os\nos.popen('echo hi')"
        warnings = check_code_safety(code)
        assert any("os.popen" in w for w in warnings)

    def test_os_execve_call_flagged(self):
        code = "import os\nos.execve('/bin/sh', [], {})"
        warnings = check_code_safety(code)
        assert any("os.execve" in w for w in warnings)

    def test_from_os_import_popen_flagged(self):
        warnings = check_code_safety("from os import popen")
        assert any("popen" in w for w in warnings)

    def test_open_write_mode_with_traversal_path_flagged(self):
        warnings = check_code_safety("open('../outside.txt', 'w')")
        assert any("open" in w for w in warnings)

    def test_open_append_mode_with_absolute_unix_path_flagged(self):
        warnings = check_code_safety("open('/etc/passwd', 'a')")
        assert any("open" in w for w in warnings)

    # --- patterns that must NOT be flagged ---

    def test_clean_numeric_function_returns_empty(self):
        code = """\
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(result)
"""
        warnings = check_code_safety(code)
        assert warnings == []

    def test_open_read_mode_not_flagged(self):
        warnings = check_code_safety("open('data.txt', 'r')")
        assert not any("open" in w for w in warnings)

    def test_open_default_mode_not_flagged(self):
        # open() with no mode defaults to 'r' — safe
        warnings = check_code_safety("open('data.txt')")
        assert not any("open" in w for w in warnings)

    def test_math_imports_not_flagged(self):
        code = "import math\nresult = math.sqrt(16)"
        warnings = check_code_safety(code)
        assert warnings == []

    # --- edge cases ---

    def test_syntax_error_returns_warning_not_exception(self):
        warnings = check_code_safety("def broken(:\n    pass")
        # Should return a warning about the SyntaxError rather than raising.
        assert len(warnings) > 0
        assert any("SyntaxError" in w or "syntax" in w.lower() for w in warnings)

    def test_returns_list(self):
        assert isinstance(check_code_safety("x = 1"), list)

    def test_no_duplicate_warnings(self):
        # Importing socket twice should not double-report.
        code = "import socket\nimport socket"
        warnings = check_code_safety(code)
        assert len(warnings) == len(set(warnings))
