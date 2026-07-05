"""
Code-execution sandbox for the Qwen3-0.6B Code Recovery Lab (spec §7.3).

Real protections provided by this module
-----------------------------------------
timeout            subprocess.run(..., timeout=...) raises TimeoutExpired;
                   the subprocess is killed (TerminateProcess on Windows) and
                   the child cannot hang indefinitely.
temp-dir isolation Every execution uses a fresh tempfile.mkdtemp() directory.
                   cwd is set to that dir, so relative file operations stay
                   inside it and never touch the project tree.
output capping     stdout and stderr are truncated to max_output_chars chars
                   each, preventing memory exhaustion from runaway programs.
independent process Code always runs in a child subprocess, never in-process.
cleanup            The temp directory is unconditionally deleted in a finally
                   block via shutil.rmtree(..., ignore_errors=True).

Known limitations (best-effort only)
------------------------------------
grandchild procs   Only the DIRECT child process is killed on timeout
                   (TerminateProcess on Windows; no Job Object / process-tree
                   kill is used).  If untrusted code spawns its own
                   subprocess, that grandchild survives the kill.  A surviving
                   grandchild can hold a file handle open inside the temp
                   directory, in which case ``shutil.rmtree(ignore_errors=True)``
                   silently fails and the temp directory is leaked.  Children
                   are launched in a new process group on Windows
                   (CREATE_NEW_PROCESS_GROUP) for isolation, but this does NOT
                   kill descendants.
check_code_safety  A lightweight AST-based static pre-screen.  This is NOT a
                   real security boundary — determined adversarial code can
                   bypass it trivially.  No OS-level sandboxing (AppContainer,
                   Job Objects, seccomp, etc.) is used.  Callers decide
                   whether to act on the returned warnings.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass


_TRUNCATION_MARKER = "...[truncated]"


def _truncate(s: str, max_chars: int) -> str:
    """Truncate *s* to *max_chars*, appending a marker when cut."""
    if len(s) > max_chars:
        return s[:max_chars] + _TRUNCATION_MARKER
    return s


def _decode_output(v: str | None) -> str:
    """Normalise subprocess output to a string.

    Every subprocess in this module runs with ``text=True``, so output is
    already ``str``.  The only special case is ``None`` — no output was
    captured before a timeout/kill — which becomes the empty string.
    """
    return v if v is not None else ""


# On Windows, launch children in their own process group so that, in principle,
# group-level signalling is possible and the child is cleanly isolated from the
# parent's console.  On POSIX this flag does not exist; 0 means "no extra flags".
_CREATIONFLAGS = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    """Result of running a Python script inside the sandbox.

    Attributes
    ----------
    returncode:
        Process exit code, or ``None`` when the run timed out and the
        process was killed before it could return.
    stdout:
        Captured standard output (possibly truncated).
    stderr:
        Captured standard error (possibly truncated).
    timed_out:
        ``True`` when the timeout fired and the child was killed.
    duration_s:
        Wall-clock seconds from process launch to completion / kill.
    tempdir:
        Path to the (now-deleted) temporary directory used for this run.
        Useful for tests that want to assert cleanup happened.
    """

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    tempdir: str


@dataclass
class PytestResult:
    """Result of running pytest inside the sandbox.

    Attributes
    ----------
    passed:
        ``True`` iff pytest exited with code 0 (all tests collected and
        passed) **and** at least one test was collected.
    num_collected:
        Total number of test items collected by pytest.  When 0, no tests
        were found — the result is always a failure regardless of exit code.
    num_passed:
        Number of tests reported as passing (parsed from pytest output).
    num_failed:
        Number of non-passing tests: assertion failures PLUS collection/import/
        fixture errors.  Guaranteed ``>= 1`` whenever ``passed`` is ``False``,
        so downstream consumers can rely on it to detect any non-success.
    num_errors:
        Subset of ``num_failed`` that pytest classified as "error" (collection
        / import / fixture errors) rather than ordinary test "failed".
    timed_out:
        ``True`` when the timeout fired.
    returncode:
        Pytest's exit code, or ``None`` on timeout.
    stdout / stderr:
        Captured output (possibly truncated).
    duration_s:
        Wall-clock seconds.
    tempdir:
        Path to the (now-deleted) temporary directory.
    """

    passed: bool
    num_collected: int
    num_passed: int
    num_failed: int
    num_errors: int
    timed_out: bool
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    tempdir: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_python_code(
    code: str,
    *,
    timeout_s: float = 5.0,
    max_output_chars: int = 10_000,
    extra_files: dict[str, str] | None = None,
) -> ExecResult:
    """Execute *code* as ``main.py`` inside a fresh temporary directory.

    Parameters
    ----------
    code:
        Python source code to run.
    timeout_s:
        Maximum wall-clock seconds before the child process is killed.
    max_output_chars:
        Maximum characters kept from each of stdout and stderr.  Excess is
        replaced with a ``...[truncated]`` suffix.
    extra_files:
        Optional ``{filename: content}`` mapping written alongside
        ``main.py``.  Useful for helper modules or fixture data files that
        the code needs to ``import`` or ``open`` by relative name.

    Returns
    -------
    ExecResult
        ``timed_out=True`` and ``returncode=None`` when the timeout fires.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "main.py"), "w", encoding="utf-8") as fh:
            fh.write(code)
        if extra_files:
            for fname, content in extra_files.items():
                with open(os.path.join(tmpdir, fname), "w", encoding="utf-8") as fh:
                    fh.write(content)

        timed_out = False
        returncode: int | None = None
        stdout = ""
        stderr = ""

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "main.py"],
                cwd=tmpdir,
                timeout=timeout_s,
                capture_output=True,
                text=True,
                creationflags=_CREATIONFLAGS,
            )
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills the direct child and waits for it
            # (communicate) before re-raising, so the child is dead before the
            # finally-block rmtree runs.  Grandchildren are NOT killed — see the
            # module docstring's "Known limitations" section.  exc.stdout/stderr
            # may be None (no output captured before the kill).
            timed_out = True
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
        duration_s = time.perf_counter() - t0

        return ExecResult(
            returncode=returncode,
            stdout=_truncate(stdout, max_output_chars),
            stderr=_truncate(stderr, max_output_chars),
            timed_out=timed_out,
            duration_s=duration_s,
            tempdir=tmpdir,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _has_bare_asserts_before_import(test_code: str) -> bool:
    """Return True if any ``assert `` statement appears before the first
    ``from solution import`` line in *test_code*.

    This detects the mixed-format case where bare-assert public tests are
    concatenated before a pytest test-function block that starts with its
    own ``from solution import`` header.  Without intervention, the bare
    asserts run at module top-level BEFORE the import line and raise
    ``NameError`` during pytest collection because the function under test
    is undefined in the test module's namespace.
    """
    for line in test_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("from solution"):
            return False  # reached the import line; no bare asserts before it
        if stripped.startswith("assert "):
            return True
    return False


def _normalize_test_code(test_code: str) -> str:
    """Normalize bare-assert test code to a pytest-collectable format.

    MBPP-style test snippets are top-level ``assert`` statements without
    ``from solution import ...`` and without ``def test_*`` wrappers.  pytest
    fails to collect them as tests (raises NameError during collection
    because the function under test is undefined in the test module's
    namespace, or reports "no tests ran" if names happen to resolve).

    This function detects such bare-assert tests and rewrites them to::

        from solution import *

        def test_solution():
            <indented asserts>

    Test code that already contains ``from solution`` or ``def test_`` is
    returned unchanged — UNLESS it is a mixed-format file where bare asserts
    appear BEFORE the first ``from solution`` line (produced by
    ``generate_boundary_variants.py`` concatenating bare-assert public tests
    with pytest test functions).  In that case ``from solution import *`` is
    prepended at the top so the bare asserts resolve the function under test.
    Empty / comment-only test code is also returned unchanged (the caller
    handles "no tests" as a failure).
    """
    if not test_code or not test_code.strip():
        return test_code
    # Mixed-format: `from solution` present AND bare asserts precede it.
    # Prepend `from solution import *` so the bare asserts resolve.
    if "from solution" in test_code:
        if _has_bare_asserts_before_import(test_code):
            return "from solution import *\n\n" + test_code
        return test_code
    # Already in pytest format (def test_* but no from solution) - leave as-is
    if "def test" in test_code:
        return test_code
    # No asserts to wrap
    if "assert" not in test_code:
        return test_code
    # Bare asserts: wrap in a test function with import header
    indented = textwrap.indent(test_code.strip(), "    ")
    return f"from solution import *\n\ndef test_solution():\n{indented}\n"


def run_pytest(
    target_code: str,
    test_code: str,
    *,
    timeout_s: float = 10.0,
    max_output_chars: int = 20_000,
) -> PytestResult:
    """Run *test_code* against *target_code* using pytest in a temp dir.

    Contract
    --------
    *target_code* is written to ``solution.py``; *test_code* to
    ``test_solution.py``.  The test file is expected to import the
    implementation from ``solution``, e.g.::

        from solution import my_function

    Bare-assert MBPP-style tests (no ``from solution`` and no ``def test_*``)
    are auto-normalized via :func:`_normalize_test_code` so that pytest can
    collect them as a single ``test_solution`` function.

    Parameters
    ----------
    target_code:
        The solution / implementation to be tested.
    test_code:
        A pytest test file (or bare asserts).
    timeout_s:
        Maximum wall-clock seconds before the child process is killed.
    max_output_chars:
        Maximum characters kept from each of stdout and stderr.

    Returns
    -------
    PytestResult
    """
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as fh:
            fh.write(target_code)
        with open(os.path.join(tmpdir, "test_solution.py"), "w", encoding="utf-8") as fh:
            fh.write(_normalize_test_code(test_code))

        timed_out = False
        returncode: int | None = None
        stdout = ""
        stderr = ""

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    "test_solution.py", "-q", "--no-header",
                ],
                cwd=tmpdir,
                timeout=timeout_s,
                capture_output=True,
                text=True,
                creationflags=_CREATIONFLAGS,
            )
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
        duration_s = time.perf_counter() - t0

        # Parse passed/failed/error/collected counts from pytest's output.
        num_passed, num_failed_only, num_errors, num_collected = _parse_pytest_counts(stdout)
        # passed requires: no timeout, exit code 0, AND at least 1 test collected
        passed = (not timed_out) and (returncode == 0) and (num_collected > 0)
        num_failed = num_failed_only + num_errors

        # If pytest reported a non-zero exit but no parseable failure/error
        # count (e.g. crash, or a summary format we don't recognise), surface at
        # least one failure so callers never mistake it for success.
        if (not passed) and num_failed == 0:
            num_failed = 1

        return PytestResult(
            passed=passed,
            num_collected=num_collected,
            num_passed=num_passed,
            num_failed=num_failed,
            num_errors=num_errors,
            timed_out=timed_out,
            returncode=returncode,
            stdout=_truncate(stdout, max_output_chars),
            stderr=_truncate(stderr, max_output_chars),
            duration_s=duration_s,
            tempdir=tmpdir,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Static safety pre-screen
# ---------------------------------------------------------------------------

_DANGEROUS_TOP_LEVEL = frozenset({"subprocess", "ctypes", "winreg"})
_NETWORK_TOP_LEVEL = frozenset({"socket", "urllib", "requests", "http"})
# os attributes that run external programs.  os.exec*/os.spawn* are matched by
# prefix in the call-site check rather than being enumerated here.
_DANGEROUS_OS_ATTRS = frozenset({"system", "popen"})
_DANGEROUS_BUILTINS = frozenset({"exec", "eval", "__import__"})


def check_code_safety(code: str) -> list[str]:
    """Lightweight AST-based static pre-screen for obviously dangerous patterns.

    .. warning::
       This is NOT a real security boundary.  It is a defense-in-depth hint
       that helps catch common accidents and obvious dangerous patterns.
       Determined adversarial code can bypass it trivially.  Callers decide
       whether to act on the returned warnings — this function never blocks
       execution by itself.

    Detected patterns
    -----------------
    * ``import subprocess`` / ``ctypes`` / ``winreg``.
    * ``import socket`` / ``urllib*`` / ``requests`` / ``http*``  (network).
    * ``from os import system`` / ``popen``  and ``from shutil import rmtree``.
    * Calls to ``os.system(...)`` / ``os.popen(...)`` / ``os.exec*`` /
      ``os.spawn*``.
    * Calls to ``exec(...)`` / ``eval(...)`` / ``__import__(...)`` — the
      easiest ways to bypass static screening by building dangerous code or
      module names dynamically.
    * ``open(...)`` with ``'w'`` or ``'a'`` mode on an absolute path or a
      path that contains ``..`` (path traversal).

    NOT covered (non-exhaustive — do not treat absence of warnings as proof of
    safety): dynamically-constructed import/open arguments, ``importlib``,
    ``compile()``, ``getattr``-based attribute access, ``pickle``/``marshal``
    deserialisation, C-extension side effects, and any obfuscation.  This is a
    heuristic, not a sandbox.

    Parameters
    ----------
    code:
        Python source code to inspect.

    Returns
    -------
    list[str]
        Human-readable warning strings; ``[]`` when nothing suspicious is
        found.
    """
    warnings: list[str] = []
    seen: set[str] = set()

    def warn(msg: str) -> None:
        if msg not in seen:
            seen.add(msg)
            warnings.append(msg)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        warn(f"SyntaxError during static analysis: {exc}")
        return warnings

    for node in ast.walk(tree):
        # ------------------------------------------------------------------
        # Import / ImportFrom
        # ------------------------------------------------------------------
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _DANGEROUS_TOP_LEVEL:
                    warn(f"Potentially dangerous import: {alias.name!r}")
                if top in _NETWORK_TOP_LEVEL:
                    warn(f"Network-related import: {alias.name!r}")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top in _DANGEROUS_TOP_LEVEL:
                warn(f"Potentially dangerous import from: {module!r}")
            if top in _NETWORK_TOP_LEVEL:
                warn(f"Network-related import from: {module!r}")
            # Special-case: from os import system/popen  /  from shutil import rmtree
            if module == "os":
                for alias in node.names:
                    if alias.name in _DANGEROUS_OS_ATTRS:
                        warn(f"Dangerous import: 'from os import {alias.name}'")
            if module == "shutil":
                for alias in node.names:
                    if alias.name == "rmtree":
                        warn("Dangerous import: 'from shutil import rmtree'")

        # ------------------------------------------------------------------
        # Call sites
        # ------------------------------------------------------------------
        elif isinstance(node, ast.Call):
            func = node.func
            # os.system(...) / os.popen(...) / os.exec*/spawn*(...)
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
                and (
                    func.attr in _DANGEROUS_OS_ATTRS
                    or func.attr.startswith("exec")
                    or func.attr.startswith("spawn")
                )
            ):
                warn(f"Call to os.{func.attr}()")

            # exec(...) / eval(...) / __import__(...) — easy static-screen bypass.
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_BUILTINS:
                warn(f"Call to {func.id}()")

            # open(..., 'w'|'a', ...) with dangerous path
            if _is_dangerous_open(node):
                warn("open() with write/append mode on absolute or traversal path")

    return warnings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_dangerous_open(node: ast.Call) -> bool:
    """Return True if *node* looks like ``open()`` writing to a dangerous path.

    Only flags cases where both the mode and the path are string literals, so
    there are some false negatives (dynamic paths / modes).  That is
    intentional — the goal is to catch obvious accidents, not to be complete.
    """
    func = node.func
    is_open = (isinstance(func, ast.Name) and func.id == "open") or (
        isinstance(func, ast.Attribute) and func.attr == "open"
    )
    if not is_open:
        return False

    args = node.args
    keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

    # Resolve mode: positional arg[1] or keyword 'mode'.
    mode_node: ast.expr | None = None
    if len(args) >= 2:
        mode_node = args[1]
    elif "mode" in keywords:
        mode_node = keywords["mode"]
    else:
        return False  # default mode is 'r' — safe

    if not isinstance(mode_node, ast.Constant) or not isinstance(mode_node.value, str):
        return False  # dynamic mode — skip conservatively
    if "w" not in mode_node.value and "a" not in mode_node.value:
        return False

    # Resolve path: positional arg[0].
    if not args:
        return False
    path_node = args[0]
    if not isinstance(path_node, ast.Constant) or not isinstance(path_node.value, str):
        return False  # dynamic path — skip conservatively
    path_val: str = path_node.value

    # Absolute path (Unix or Windows) or traversal sequence.
    is_absolute = (
        path_val.startswith("/")
        or path_val.startswith("\\")
        or (len(path_val) >= 3 and path_val[1] == ":" and path_val[2] in r"\/")
    )
    has_traversal = ".." in path_val
    return is_absolute or has_traversal


def _parse_pytest_counts(output: str) -> tuple[int, int, int, int]:
    """Extract passed/failed/error/collected counts from pytest's output.

    Handles lines such as::

        1 passed in 0.01s
        2 failed, 1 passed in 0.05s
        3 failed in 0.02s
        1 error in 0.03s              # collection/import/fixture error
        1 failed, 1 error in 0.04s

    Also parses the "collected N items" line::

        collected 4 items

    The "error" token is distinct from "failed": pytest reports a solution that
    fails to import (e.g. a SyntaxError or missing symbol) as an *error*, not a
    test failure.  Both are returned so the caller can fold errors into the
    non-passing count.

    Searches from the bottom of the output so the final summary line wins.

    Returns
    -------
    tuple[int, int, int, int]
        ``(num_passed, num_failed, num_errors, num_collected)``; first three
        are 0 when nothing could be parsed.  ``num_collected`` is parsed from
        the "collected N items" line, or falls back to
        ``num_passed + num_failed + num_errors`` when that line is absent.
    """
    passed = 0
    failed = 0
    errors = 0
    for line in reversed(output.splitlines()):
        m_passed = re.search(r"(\d+)\s+passed", line)
        m_failed = re.search(r"(\d+)\s+failed", line)
        # Match "error" / "errors" but avoid the unrelated word "errors" inside
        # tracebacks by requiring the count-prefixed summary form.
        m_error = re.search(r"(\d+)\s+errors?\b", line)
        if m_passed or m_failed or m_error:
            passed = int(m_passed.group(1)) if m_passed else 0
            failed = int(m_failed.group(1)) if m_failed else 0
            errors = int(m_error.group(1)) if m_error else 0
            break

    # Parse "collected N items" line
    num_collected = 0
    for line in output.splitlines():
        m_collected = re.search(r"collected\s+(\d+)\s+items?", line)
        if m_collected:
            num_collected = int(m_collected.group(1))
            break
    # Fallback: infer from summary counts
    if num_collected == 0:
        num_collected = passed + failed + errors

    return passed, failed, errors, num_collected
