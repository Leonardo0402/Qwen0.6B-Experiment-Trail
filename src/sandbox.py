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

Advisory / best-effort only
----------------------------
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
import time
from dataclasses import dataclass


_TRUNCATION_MARKER = "...[truncated]"


def _truncate(s: str, max_chars: int) -> str:
    """Truncate *s* to *max_chars*, appending a marker when cut."""
    if len(s) > max_chars:
        return s[:max_chars] + _TRUNCATION_MARKER
    return s


def _decode_output(v: bytes | str | None) -> str:
    """Safely decode subprocess output that may be bytes, str, or None."""
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


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
        passed).
    num_passed:
        Number of tests reported as passing (parsed from pytest output).
    num_failed:
        Number of tests reported as failing (parsed from pytest output).
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
    num_passed: int
    num_failed: int
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
            )
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills the child on TimeoutExpired (Python ≥ 3.3).
            # On Windows a second communicate() populates exc.stdout/stderr.
            # Guard against bytes (partial-read path) and None (no output yet).
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

    Parameters
    ----------
    target_code:
        The solution / implementation to be tested.
    test_code:
        A pytest test file.
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
            fh.write(test_code)

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
            )
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
        duration_s = time.perf_counter() - t0

        # Parse passed/failed counts from pytest's summary line before truncation.
        num_passed, num_failed = _parse_pytest_counts(stdout)
        passed = (not timed_out) and (returncode == 0)

        # Fallback: if parsing failed but everything passed, report at least 1.
        if passed and num_passed == 0 and num_failed == 0:
            num_passed = 1

        return PytestResult(
            passed=passed,
            num_passed=num_passed,
            num_failed=num_failed,
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
    * ``from os import system`` (os.system shorthand).
    * Calls to ``os.system(...)``.
    * ``from shutil import rmtree``.
    * ``open(...)`` with ``'w'`` or ``'a'`` mode on an absolute path or a
      path that contains ``..`` (path traversal).

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
            # Special-case: from os import system  /  from shutil import rmtree
            if module == "os":
                for alias in node.names:
                    if alias.name == "system":
                        warn("Dangerous import: 'from os import system'")
            if module == "shutil":
                for alias in node.names:
                    if alias.name == "rmtree":
                        warn("Dangerous import: 'from shutil import rmtree'")

        # ------------------------------------------------------------------
        # Call sites
        # ------------------------------------------------------------------
        elif isinstance(node, ast.Call):
            # os.system(...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "system"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                warn("Call to os.system()")

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


def _parse_pytest_counts(output: str) -> tuple[int, int]:
    """Extract passed/failed counts from pytest's compact summary line.

    Handles lines such as::

        1 passed in 0.01s
        2 failed, 1 passed in 0.05s
        3 failed in 0.02s

    Searches from the bottom of the output so the final summary line wins.

    Returns
    -------
    tuple[int, int]
        ``(num_passed, num_failed)``, both 0 when nothing could be parsed.
    """
    passed = 0
    failed = 0
    for line in reversed(output.splitlines()):
        m_passed = re.search(r"(\d+)\s+passed", line)
        m_failed = re.search(r"(\d+)\s+failed", line)
        if m_passed or m_failed:
            passed = int(m_passed.group(1)) if m_passed else 0
            failed = int(m_failed.group(1)) if m_failed else 0
            break
    return passed, failed
