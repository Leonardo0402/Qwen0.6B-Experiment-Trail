"""Executable verifier for the Qwen3-0.6B Code Recovery Lab (spec §7.2).

All decisions are made via real tools only — compile(), pytest, ruff, mypy.
Language-model judgment is never used here.

Accept rule (is_accepted)
--------------------------
A sample is accepted iff::

    syntax_ok AND pytest_ok AND NOT timeout

ruff_ok and mypy_ok are recorded but do NOT gate acceptance:
- ruff is advisory (best-practice linting, suggested by spec §7.2).
- mypy is opt-in (caller passes ``run_mypy=True`` for explicitly typed tasks).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.sandbox import run_pytest
from src.schemas import Verification

if TYPE_CHECKING:
    from src.schemas import Sample


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SampleVerification:
    """Full verification result for a single sample.

    The ``verification`` field mirrors ``schemas.Verification`` for easy
    serialisation.  Extra fields give the per-check breakdown.

    Attributes
    ----------
    verification:
        Core flags (syntax_ok, pytest_ok, ruff_ok, timeout).
    public_ok:
        True when the public test suite passed.
    hidden_ok:
        True when the hidden test suite passed, or when there are no hidden
        tests (vacuously true).
    mypy_ok:
        True/False when mypy was run; None when it was not requested.
    duration_s:
        Wall-clock seconds across all sub-checks.
    messages:
        Human-readable diagnostics (compile errors, lint output, …).
    """

    verification: Verification
    public_ok: bool
    hidden_ok: bool
    mypy_ok: bool | None
    duration_s: float
    messages: list[str] = field(default_factory=list)

    @property
    def is_accepted(self) -> bool:
        """True iff all *required* checks pass.

        Required:  syntax_ok AND pytest_ok AND NOT timeout.
        Advisory (recorded, not gating): ruff_ok, mypy_ok.
        """
        v = self.verification
        return v.syntax_ok and v.pytest_ok and not v.timeout


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def compile_check(code: str) -> tuple[bool, str]:
    """Check Python syntax using the built-in compiler.

    Uses ``compile(code, "<sample>", "exec")`` — pure in-process, no
    subprocess needed.

    Parameters
    ----------
    code:
        Python source to check.

    Returns
    -------
    (ok, error_message)
        *ok* is ``True`` when the code compiles without errors.
        *error_message* is the empty string on success and the
        ``SyntaxError`` description on failure.
    """
    try:
        compile(code, "<sample>", "exec")
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


def ruff_check(code: str, *, timeout_s: float = 15.0) -> tuple[bool, str]:
    """Lint *code* with ruff (advisory; does NOT execute the code).

    Writes code to a temp file, then runs::

        [sys.executable, "-m", "ruff", "check", "--quiet", <file>]

    Parameters
    ----------
    code:
        Python source to lint.
    timeout_s:
        Maximum seconds before the ruff subprocess is killed.

    Returns
    -------
    (ok, output)
        *ok* is ``True`` when ruff exits with code 0 (no lint issues).
        *output* contains any diagnostic text (stdout + stderr combined).
    """
    tmpdir = tempfile.mkdtemp()
    try:
        src_file = os.path.join(tmpdir, "sample.py")
        with open(src_file, "w", encoding="utf-8") as fh:
            fh.write(code)

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--quiet", src_file],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = (proc.stdout + proc.stderr).strip()
            return proc.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "ruff timed out"
        except FileNotFoundError:
            return False, "ruff not found"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def mypy_check(code: str, *, timeout_s: float = 30.0) -> tuple[bool, str]:
    """Type-check *code* with mypy (opt-in; does NOT execute the code).

    Only meaningful for tasks that explicitly require type annotations.
    Callers decide when to invoke this by passing ``run_mypy=True`` to
    :func:`verify_sample`.

    Writes code to a temp file, then runs::

        [sys.executable, "-m", "mypy", <file>]

    Parameters
    ----------
    code:
        Python source to type-check.
    timeout_s:
        Maximum seconds before the mypy subprocess is killed.

    Returns
    -------
    (ok, output)
        *ok* is ``True`` when mypy exits with code 0 (no type errors).
    """
    tmpdir = tempfile.mkdtemp()
    try:
        src_file = os.path.join(tmpdir, "sample.py")
        with open(src_file, "w", encoding="utf-8") as fh:
            fh.write(code)

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "mypy", src_file],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = (proc.stdout + proc.stderr).strip()
            return proc.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "mypy timed out"
        except FileNotFoundError:
            return False, "mypy not found"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------


def verify_sample(
    sample: "Sample",
    *,
    run_ruff: bool = True,
    run_mypy: bool = False,
    pytest_timeout_s: float = 10.0,
) -> SampleVerification:
    """Verify a Sample using real tools (compile / pytest / ruff / mypy).

    Execution order
    ---------------
    a. ``compile_check(target_code)``  → syntax_ok
    b. ``run_pytest(public_tests)``    → public_ok    (skipped if syntax fails)
    c. ``run_pytest(hidden_tests)``    → hidden_ok    (skipped if syntax fails
                                                        or hidden_tests is blank)
    d. ``ruff_check(target_code)``     → ruff_ok      (if run_ruff=True)
    e. ``mypy_check(target_code)``     → mypy_ok      (if run_mypy=True)

    Public and hidden tests run as **separate** pytest invocations so that
    test-function name collisions between the two suites are impossible.

    Accept rule
    -----------
    :attr:`SampleVerification.is_accepted` is True iff::

        syntax_ok AND pytest_ok AND NOT timeout

    ruff_ok and mypy_ok are advisory: they are recorded in the returned
    object but do NOT gate acceptance.

    Parameters
    ----------
    sample:
        The :class:`~src.schemas.Sample` to verify.
    run_ruff:
        When True (default), run ruff and record ruff_ok.
    run_mypy:
        When True, run mypy and record mypy_ok.  Defaults to False because
        mypy is only meaningful for explicitly typed tasks.
    pytest_timeout_s:
        Per-invocation timeout for each pytest subprocess call.

    Returns
    -------
    SampleVerification
    """
    t0 = time.perf_counter()
    messages: list[str] = []

    # ------------------------------------------------------------------
    # a. Syntax check
    # ------------------------------------------------------------------
    syntax_ok, compile_msg = compile_check(sample.target_code)
    if compile_msg:
        messages.append(f"compile: {compile_msg}")

    # ------------------------------------------------------------------
    # b/c. Pytest — public, then hidden
    # ------------------------------------------------------------------
    public_ok = False
    hidden_ok = True  # vacuously True when no hidden tests
    timed_out = False

    if syntax_ok:
        # Public tests
        pub_result = run_pytest(
            sample.target_code,
            sample.public_tests,
            timeout_s=pytest_timeout_s,
        )
        public_ok = pub_result.passed
        timed_out = timed_out or pub_result.timed_out
        if not public_ok:
            messages.append(
                f"public tests: {pub_result.num_failed} failure(s); "
                f"stdout={pub_result.stdout[:400]}"
            )

        # Hidden tests — skip when the field is blank
        hidden_tests = (sample.hidden_tests or "").strip()
        if hidden_tests:
            hid_result = run_pytest(
                sample.target_code,
                hidden_tests,
                timeout_s=pytest_timeout_s,
            )
            hidden_ok = hid_result.passed
            timed_out = timed_out or hid_result.timed_out
            if not hidden_ok:
                messages.append(
                    f"hidden tests: {hid_result.num_failed} failure(s); "
                    f"stdout={hid_result.stdout[:400]}"
                )

    pytest_ok = public_ok and hidden_ok

    # ------------------------------------------------------------------
    # d. Ruff (advisory)
    # ------------------------------------------------------------------
    ruff_ok = True
    if run_ruff:
        ruff_ok, ruff_msg = ruff_check(sample.target_code)
        if not ruff_ok and ruff_msg:
            messages.append(f"ruff: {ruff_msg[:400]}")

    # ------------------------------------------------------------------
    # e. Mypy (opt-in, advisory)
    # ------------------------------------------------------------------
    mypy_result: bool | None = None
    if run_mypy:
        mypy_ok_val, mypy_msg = mypy_check(sample.target_code)
        mypy_result = mypy_ok_val
        if not mypy_ok_val and mypy_msg:
            messages.append(f"mypy: {mypy_msg[:400]}")

    duration_s = time.perf_counter() - t0

    verification = Verification(
        syntax_ok=syntax_ok,
        pytest_ok=pytest_ok,
        ruff_ok=ruff_ok,
        timeout=timed_out,
    )

    return SampleVerification(
        verification=verification,
        public_ok=public_ok,
        hidden_ok=hidden_ok,
        mypy_ok=mypy_result,
        duration_s=duration_s,
        messages=messages,
    )


def verify_broken_is_broken(
    sample: "Sample",
    *,
    pytest_timeout_s: float = 10.0,
) -> bool:
    """Confirm that ``sample.broken_code`` genuinely fails at least one test.

    Per spec §11.2 every broken variant must trigger ≥ 1 test failure.
    Runs ``broken_code`` against both ``public_tests`` and ``hidden_tests``
    (as separate invocations).

    Parameters
    ----------
    sample:
        Must have a non-empty ``broken_code``; typically a static_repair or
        execution_repair sample.
    pytest_timeout_s:
        Per-invocation timeout for each pytest subprocess.

    Returns
    -------
    True
        ``broken_code`` fails at least one test — valid broken sample.
    False
        ``broken_code`` unexpectedly passes all tests — invalid broken sample.
    """
    broken = sample.broken_code or ""

    # Public tests
    pub_result = run_pytest(
        broken,
        sample.public_tests,
        timeout_s=pytest_timeout_s,
    )
    if pub_result.timed_out or pub_result.num_failed >= 1:
        return True

    # Hidden tests (skip when blank)
    hidden = (sample.hidden_tests or "").strip()
    if hidden:
        hid_result = run_pytest(
            broken,
            hidden,
            timeout_s=pytest_timeout_s,
        )
        if hid_result.timed_out or hid_result.num_failed >= 1:
            return True

    return False
