"""
scripts/build_execution_repair.py -- Execution-driven repair sample builder (P2.2).

Pipeline
--------
1. Read correct code_generation samples.
2. Inject bugs via scripts.inject_bugs.inject_all_bugs.
3. Execute each bugged variant against the sample's tests (src.sandbox.run_pytest).
4. Keep only variants that genuinely fail >= 1 test.
5. Compress the pytest output into a structured execution_feedback string.
6. Emit static_repair and execution_repair Sample objects.

Compressed feedback format
--------------------------
The feedback contains ONLY:
  - 失败测试名称 (failed test name)
  - 异常类型 (exception type)
  - 异常消息 (exception message)
  - Expected / Actual
  - 关键 traceback (only solution.py lines)
  - 修复要求 (repair requirement)
It does NOT contain hundreds of lines of raw pytest output.

Usage
-----
    python scripts/build_execution_repair.py --input <sample.jsonl> --output <output.jsonl> --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.inject_bugs import inject_all_bugs  # noqa: E402
from src.sandbox import run_pytest  # noqa: E402
from src.schemas import Sample, Verification  # noqa: E402
from src.validators import compile_check  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)
_DATASET_VERSION = "p2.2"
_GENERATOR = "execution_repair_v1"
MAX_FEEDBACK_CHARS = 3000
_TRUNCATION_MARKER = "...[truncated]"


# ---------------------------------------------------------------------------
# Compressed feedback parser
# ---------------------------------------------------------------------------

def _parse_failed_test_names(stdout: str) -> list[str]:
    """Extract failed test function names from pytest output.

    Looks for lines like:
        FAILED test_solution.py::test_basic - assert ...
    Also handles:
        test_solution.py::test_basic FAILED
    """
    names: list[str] = []
    # Pattern: FAILED test_solution.py::test_name
    for m in re.finditer(r"FAILED\s+test_solution\.py::(\w+)", stdout):
        names.append(m.group(1))
    # Fallback: ___ test_name ___ (pytest section headers)
    if not names:
        for m in re.finditer(r"_+ (\w+) _+", stdout):
            name = m.group(1)
            if name.startswith("test_"):
                names.append(name)
    return names


def _parse_exception(stdout: str) -> tuple[str, str]:
    """Extract (exception_type, exception_message) from pytest output.

    Looks for lines like:
        E   AssertionError: assert 7 == 6
        E   TypeError: ...
        E   ValueError: ...
    """
    exc_type = ""
    exc_msg = ""
    # Pattern: E   ExceptionType: message
    for m in re.finditer(r"^E\s+(\w+(?:Error|Exception)):\s*(.*)$", stdout, re.MULTILINE):
        exc_type = m.group(1)
        exc_msg = m.group(2)
        break
    # Fallback: last line of traceback (ExceptionType: message)
    if not exc_type:
        for m in re.finditer(r"(\w+(?:Error|Exception)):\s*(.*)", stdout):
            exc_type = m.group(1)
            exc_msg = m.group(2)
    return exc_type, exc_msg


def _parse_expected_actual(exc_msg: str) -> tuple[str, str]:
    """Extract Expected/Actual from an assertion message.

    Handles patterns like:
        assert 7 == 6           -> Expected=6, Actual=7
        assert 7 != 6           -> Expected=not-6, Actual=7
        assert 5 is True        -> Expected=True, Actual=5
        assert 5 is not True     -> Expected=not-True, Actual=5
    """
    # assert X == Y
    m = re.match(r"assert\s+(.+?)\s*==\s*(.+)", exc_msg)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    # assert X != Y
    m = re.match(r"assert\s+(.+?)\s*!=\s*(.+)", exc_msg)
    if m:
        return f"not {m.group(2).strip()}", m.group(1).strip()
    # assert X is Y
    m = re.match(r"assert\s+(.+?)\s+is\s+(not\s+)?(.+)", exc_msg)
    if m:
        expected = f"{m.group(2) or ''}{m.group(3)}".strip()
        return expected, m.group(1).strip()
    return "", ""


def _extract_solution_traceback(stdout: str) -> list[str]:
    """Extract only traceback lines that reference solution.py.

    Keeps lines like:
        File ".../solution.py", line 3, in sum_list
            total += x
    """
    lines = stdout.splitlines()
    result: list[str] = []
    capture_next = False
    for line in lines:
        if "solution.py" in line:
            result.append(line.strip())
            capture_next = True
        elif capture_next:
            # The line after "File ... solution.py" is the source code line
            stripped = line.strip()
            if stripped and not stripped.startswith("File "):
                result.append(stripped)
            capture_next = False
    return result


def compress_feedback(
    stdout: str,
    stderr: str,
    returncode: Optional[int],
) -> str:
    """Compress raw pytest output into a structured execution_feedback string.

    The output contains only:
      - Failed test name(s)
      - Exception type and message
      - Expected vs Actual
      - Key traceback (solution.py lines only)
      - Repair requirement
    """
    # Parse components
    failed_names = _parse_failed_test_names(stdout)
    exc_type, exc_msg = _parse_exception(stdout)
    expected, actual = _parse_expected_actual(exc_msg)
    traceback_lines = _extract_solution_traceback(stdout)

    # Assemble compressed feedback
    parts: list[str] = ["## 执行反馈"]

    # Failed test names
    if failed_names:
        parts.append("")
        parts.append("### 失败测试")
        for name in failed_names:
            parts.append(f"- {name}")
    else:
        parts.append("")
        parts.append("### 失败测试")
        parts.append("- (无法解析测试名称)")

    # Exception info
    if exc_type:
        parts.append("")
        parts.append("### 异常")
        parts.append(f"- 类型: {exc_type}")
        if exc_msg:
            parts.append(f"- 消息: {exc_msg}")
    else:
        parts.append("")
        parts.append("### 异常")
        # Fallback: include non-empty stderr snippet
        if stderr.strip():
            parts.append(f"- stderr: {stderr.strip()[:200]}")
        else:
            parts.append(f"- 类型: Unknown (exit code {returncode})")

    # Expected vs Actual
    if expected or actual:
        parts.append("")
        parts.append("### Expected vs Actual")
        if expected:
            parts.append(f"- Expected: {expected}")
        if actual:
            parts.append(f"- Actual: {actual}")

    # Key traceback (solution.py only)
    if traceback_lines:
        parts.append("")
        parts.append("### 关键 Traceback")
        for line in traceback_lines:
            parts.append(f"  {line}")

    # Repair requirement
    parts.append("")
    parts.append("### 修复要求")
    parts.append("请根据以上执行反馈修复代码，使其通过所有测试用例。")

    feedback = "\n".join(parts)

    # Enforce size cap
    if len(feedback) > MAX_FEEDBACK_CHARS:
        feedback = feedback[:MAX_FEEDBACK_CHARS] + "\n" + _TRUNCATION_MARKER

    return feedback


# ---------------------------------------------------------------------------
# Execution verification
# ---------------------------------------------------------------------------

def verify_bugged_fails(
    bugged_code: str,
    public_tests: str,
    hidden_tests: str,
    *,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Run bugged_code against tests; return (is_broken, compressed_feedback).

    is_broken is True when >= 1 test fails or timeout occurs.
    compressed_feedback is the structured feedback string (empty if not broken).
    """
    # Public tests first
    pub_result = run_pytest(bugged_code, public_tests, timeout_s=timeout_s)
    if pub_result.timed_out or pub_result.num_failed >= 1:
        feedback = compress_feedback(
            pub_result.stdout, pub_result.stderr, pub_result.returncode
        )
        return True, feedback

    # Hidden tests
    hidden = (hidden_tests or "").strip()
    if hidden:
        hid_result = run_pytest(bugged_code, hidden, timeout_s=timeout_s)
        if hid_result.timed_out or hid_result.num_failed >= 1:
            feedback = compress_feedback(
                hid_result.stdout, hid_result.stderr, hid_result.returncode
            )
            return True, feedback

    return False, ""


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def build_repair_samples(
    sample: Sample,
    *,
    timeout_s: float = 10.0,
    seed: int = 42,
) -> list[tuple[Sample, Optional[Sample]]]:
    """Produce (static_repair, execution_repair) pairs from one sample.

    For each bug injection that produces a genuinely failing variant:
      - static_repair: broken_code + no feedback
      - execution_repair: broken_code + compressed execution feedback

    Only variants with real failure evidence are returned.
    """
    variants = inject_all_bugs(sample.target_code, seed=seed)
    created_at = datetime.now(timezone.utc).isoformat()
    results: list[tuple[Sample, Optional[Sample]]] = []

    for bug_type, bugged_code, description in variants:
        # Compile check
        ok, _ = compile_check(bugged_code)
        if not ok:
            continue

        # Verify the bugged code actually fails
        is_broken, feedback = verify_bugged_fails(
            bugged_code,
            sample.public_tests,
            sample.hidden_tests,
            timeout_s=timeout_s,
        )
        if not is_broken:
            continue  # no real failure → skip

        sr_id = f"{sample.sample_id}_sr_{bug_type}"
        er_id = f"{sample.sample_id}_er_{bug_type}"

        repair_instruction = (
            f"{sample.instruction}\n\n"
            "以下代码存在错误，请找出并修复，使其能通过所有测试用例。"
        )
        exec_instruction = (
            f"{sample.instruction}\n\n"
            "以下代码存在错误，执行后出现以下问题，请修复代码。"
        )

        static_repair = Sample(
            sample_id=sr_id,
            family_id=sample.family_id,
            difficulty=sample.difficulty,
            task_type="static_repair",
            language="python",
            skill_tags=list(sample.skill_tags) + [bug_type],
            instruction=repair_instruction,
            broken_code=bugged_code,
            execution_feedback=None,
            target_code=sample.target_code,
            public_tests=sample.public_tests,
            hidden_tests=sample.hidden_tests,
            verified=False,
            verification=_PLACEHOLDER_VER,
            generator=_GENERATOR,
            created_at=created_at,
            dataset_version=_DATASET_VERSION,
        )

        exec_repair: Optional[Sample] = None
        if feedback.strip():
            exec_repair = Sample(
                sample_id=er_id,
                family_id=sample.family_id,
                difficulty=3,
                task_type="execution_repair",
                language="python",
                skill_tags=list(sample.skill_tags) + [bug_type],
                instruction=exec_instruction,
                broken_code=bugged_code,
                execution_feedback=feedback,
                target_code=sample.target_code,
                public_tests=sample.public_tests,
                hidden_tests=sample.hidden_tests,
                verified=False,
                verification=_PLACEHOLDER_VER,
                generator=_GENERATOR,
                created_at=created_at,
                dataset_version=_DATASET_VERSION,
            )

        results.append((static_repair, exec_repair))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build execution_repair samples with compressed feedback."
    )
    p.add_argument("--input", required=True, help="Input JSONL of code_generation samples.")
    p.add_argument("--output", required=True, help="Output JSONL of repair samples.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for bug injection.")
    p.add_argument("--timeout", type=float, default=10.0, help="Per-pytest timeout (s).")
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_static = 0
    n_exec = 0
    n_samples = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out_fh:
        with in_path.open(encoding="utf-8") as in_fh:
            for line in in_fh:
                line = line.strip()
                if not line:
                    continue
                n_samples += 1
                data = json.loads(line)
                try:
                    sample = Sample(**data)
                except Exception as exc:
                    print(f"  SKIP invalid sample: {exc}", file=sys.stderr)
                    n_skipped += 1
                    continue

                pairs = build_repair_samples(
                    sample, timeout_s=args.timeout, seed=args.seed
                )
                if not pairs:
                    n_skipped += 1
                    continue

                for sr, er in pairs:
                    if sr is not None:
                        out_fh.write(sr.to_json_line() + "\n")
                        n_static += 1
                    if er is not None:
                        out_fh.write(er.to_json_line() + "\n")
                        n_exec += 1

    total = n_static + n_exec
    print(
        f"build_execution_repair: {n_samples} samples -> {total} repair samples "
        f"({n_static} static_repair, {n_exec} execution_repair, "
        f"{n_skipped} skipped) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
