"""Generate the P4.0 micro-task suite (40 tasks).

Run: py -3.11 scripts/build_micro_tasks.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "p4-agent" / "micro-tasks-v0"

TASK_TYPES = [
    "locate_failing_function", "one_line_fix", "add_boundary_check",
    "update_helper", "repair_after_pytest", "avoid_editing_tests",
    "recover_from_failed_patch", "finish_after_tests_pass",
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_readme(task_id: str, goal: str, constraints: list[str], hints: list[str] = None) -> str:
    lines = [f"# Task: {task_id}", "", "## Goal", goal, "", "## Constraints"]
    for c in constraints:
        lines.append(f"- {c}")
    if hints:
        lines.append("")
        lines.append("## Hints")
        for h in hints:
            lines.append(f"- {h}")
    return "\n".join(lines) + "\n"


def _build_task(spec: dict) -> dict:
    """Build one task directory from a spec dict.

    spec keys: task_id, task_type, difficulty, goal, constraints, hints,
               solution_code, test_code, expected_patch (dict with file_path, old_text, new_text)

    Returns the manifest entry dict.
    """
    task_id = spec["task_id"]
    task_dir = OUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    readme = _make_readme(task_id, spec["goal"], spec["constraints"], spec.get("hints"))
    solution = spec["solution_code"]
    test = spec["test_code"]
    patch = json.dumps(spec["expected_patch"], indent=2) + "\n"

    readme_b = readme.encode("utf-8")
    solution_b = solution.encode("utf-8")
    test_b = test.encode("utf-8")
    patch_b = patch.encode("utf-8")

    (task_dir / "README.md").write_bytes(readme_b)
    (task_dir / "solution.py").write_bytes(solution_b)
    (task_dir / "test_solution.py").write_bytes(test_b)
    (task_dir / "expected_patch.json").write_bytes(patch_b)

    per_task_manifest = {
        "task_id": task_id,
        "task_type": spec["task_type"],
        "difficulty": spec["difficulty"],
        "readme_sha256": _sha256(readme_b),
        "solution_sha256": _sha256(solution_b),
        "test_sha256": _sha256(test_b),
        "expected_patch_sha256": _sha256(patch_b),
        "baseline_test_passes": False,
        "post_patch_test_passes": True,
    }
    (task_dir / "manifest.json").write_text(
        json.dumps(per_task_manifest, indent=2) + "\n", encoding="utf-8"
    )

    return {**per_task_manifest, "path": task_id}


# --- 40 task specs ---
# Each spec is a dict with the keys described in _build_task above.
# Function names are unique across all 40 tasks. For each task:
#   - solution_code has a bug causing >=1 test to fail
#   - applying expected_patch makes all tests pass
#   - old_text appears exactly once in solution_code

TASK_SPECS: list[dict] = [
    # ====================================================================
    # Type 1: locate_failing_function (task_001-005, difficulty 1)
    # Identify which function has the bug. expected_patch is the reference fix.
    # ====================================================================
    {
        "task_id": "task_001",
        "task_type": "locate_failing_function",
        "difficulty": 1,
        "goal": "Identify which function in solution.py contains the bug that causes test failures.",
        "constraints": ["Do not modify test_*.py files", "Report the failing function name"],
        "hints": ["Run the tests and read the error output carefully"],
        "solution_code": '''def add(a, b):
    return a + b


def subtract(a, b):
    return a + b  # BUG: should be a - b


def multiply(a, b):
    return a * b
''',
        "test_code": '''from solution import add, subtract, multiply


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(3, 4) == 12
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return a + b  # BUG: should be a - b",
            "new_text": "    return a - b",
        },
    },
    {
        "task_id": "task_002",
        "task_type": "locate_failing_function",
        "difficulty": 1,
        "goal": "Identify which function in solution.py contains the bug that causes test failures.",
        "constraints": ["Do not modify test_*.py files", "Report the failing function name"],
        "hints": ["Run the tests and read the error output carefully"],
        "solution_code": '''def divide(a, b):
    return a / b


def modulo(a, b):
    return a % b + 1  # BUG: should be a % b


def power(a, b):
    return a ** b
''',
        "test_code": '''from solution import divide, modulo, power


def test_divide():
    assert divide(10, 2) == 5


def test_modulo():
    assert modulo(10, 3) == 1


def test_power():
    assert power(2, 3) == 8
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return a % b + 1  # BUG: should be a % b",
            "new_text": "    return a % b",
        },
    },
    {
        "task_id": "task_003",
        "task_type": "locate_failing_function",
        "difficulty": 1,
        "goal": "Identify which function in solution.py contains the bug that causes test failures.",
        "constraints": ["Do not modify test_*.py files", "Report the failing function name"],
        "hints": ["Run the tests and read the error output carefully"],
        "solution_code": '''def is_even(n):
    return n % 2 == 0


def is_odd(n):
    return n % 2 == 0  # BUG: should be n % 2 != 0


def is_positive(n):
    return n > 0
''',
        "test_code": '''from solution import is_even, is_odd, is_positive


def test_is_even():
    assert is_even(4) == True


def test_is_odd():
    assert is_odd(3) == True


def test_is_positive():
    assert is_positive(5) == True
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return n % 2 == 0  # BUG: should be n % 2 != 0",
            "new_text": "    return n % 2 != 0",
        },
    },
    {
        "task_id": "task_004",
        "task_type": "locate_failing_function",
        "difficulty": 1,
        "goal": "Identify which function in solution.py contains the bug that causes test failures.",
        "constraints": ["Do not modify test_*.py files", "Report the failing function name"],
        "hints": ["Run the tests and read the error output carefully"],
        "solution_code": '''def min_of(a, b):
    return a if a < b else b


def max_of(a, b):
    return a if a < b else b  # BUG: should be a if a > b else b


def abs_val(n):
    return n if n >= 0 else -n
''',
        "test_code": '''from solution import min_of, max_of, abs_val


def test_min_of():
    assert min_of(3, 7) == 3


def test_max_of():
    assert max_of(3, 7) == 7


def test_abs_val():
    assert abs_val(-5) == 5
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return a if a < b else b  # BUG: should be a if a > b else b",
            "new_text": "    return a if a > b else b",
        },
    },
    {
        "task_id": "task_005",
        "task_type": "locate_failing_function",
        "difficulty": 1,
        "goal": "Identify which function in solution.py contains the bug that causes test failures.",
        "constraints": ["Do not modify test_*.py files", "Report the failing function name"],
        "hints": ["Run the tests and read the error output carefully"],
        "solution_code": '''def to_celsius(f):
    return (f + 32) * 5.0 / 9.0  # BUG: should be (f - 32) * 5.0 / 9.0


def to_fahrenheit(c):
    return c * 9.0 / 5.0 + 32.0


def to_kelvin(c):
    return c + 273.15
''',
        "test_code": '''from solution import to_celsius, to_fahrenheit, to_kelvin


def test_to_celsius():
    assert to_celsius(32) == 0.0


def test_to_fahrenheit():
    assert to_fahrenheit(0) == 32.0


def test_to_kelvin():
    assert to_kelvin(0) == 273.15
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return (f + 32) * 5.0 / 9.0  # BUG: should be (f - 32) * 5.0 / 9.0",
            "new_text": "    return (f - 32) * 5.0 / 9.0",
        },
    },

    # ====================================================================
    # Type 2: one_line_fix (task_006-010, difficulty 1)
    # Single-line correction: wrong operator.
    # ====================================================================
    {
        "task_id": "task_006",
        "task_type": "one_line_fix",
        "difficulty": 1,
        "goal": "Fix the bug in the compute() function so all tests pass.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["Check the arithmetic operators carefully"],
        "solution_code": '''def compute(x, y):
    return x * y + 10
''',
        "test_code": '''from solution import compute


def test_compute():
    assert compute(3, 4) == 17
    assert compute(0, 5) == 15
    assert compute(1, 1) == 12
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return x * y + 10",
            "new_text": "    return x + y + 10",
        },
    },
    {
        "task_id": "task_007",
        "task_type": "one_line_fix",
        "difficulty": 1,
        "goal": "Fix the bug in the quotient() function so all tests pass.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["Check the arithmetic operators carefully"],
        "solution_code": '''def quotient(a, b):
    return a - b
''',
        "test_code": '''from solution import quotient


def test_quotient():
    assert quotient(10, 2) == 5
    assert quotient(0, 5) == 0
    assert quotient(7, 7) == 1
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return a - b",
            "new_text": "    return a / b",
        },
    },
    {
        "task_id": "task_008",
        "task_type": "one_line_fix",
        "difficulty": 1,
        "goal": "Fix the bug in the exponent() function so all tests pass.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["Check the arithmetic operators carefully"],
        "solution_code": '''def exponent(x, y):
    return x * y
''',
        "test_code": '''from solution import exponent


def test_exponent():
    assert exponent(2, 3) == 8
    assert exponent(5, 0) == 1
    assert exponent(9, 1) == 9
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return x * y",
            "new_text": "    return x ** y",
        },
    },
    {
        "task_id": "task_009",
        "task_type": "one_line_fix",
        "difficulty": 1,
        "goal": "Fix the bug in the remainder() function so all tests pass.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["Check the arithmetic operators carefully"],
        "solution_code": '''def remainder(a, b):
    return a + b
''',
        "test_code": '''from solution import remainder


def test_remainder():
    assert remainder(10, 3) == 1
    assert remainder(8, 4) == 0
    assert remainder(7, 5) == 2
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return a + b",
            "new_text": "    return a % b",
        },
    },
    {
        "task_id": "task_010",
        "task_type": "one_line_fix",
        "difficulty": 1,
        "goal": "Fix the bug in the square() function so all tests pass.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["Check the arithmetic operators carefully"],
        "solution_code": '''def square(x):
    return x + x
''',
        "test_code": '''from solution import square


def test_square():
    assert square(4) == 16
    assert square(0) == 0
    assert square(1) == 1
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return x + x",
            "new_text": "    return x * x",
        },
    },

    # ====================================================================
    # Type 3: add_boundary_check (task_011-015, difficulty 2)
    # Add `if x < 0: raise ValueError(...)` or similar.
    # ====================================================================
    {
        "task_id": "task_011",
        "task_type": "add_boundary_check",
        "difficulty": 2,
        "goal": "Add a boundary check to safe_sqrt() so it rejects negative inputs.",
        "constraints": ["Do not modify test_*.py files", "Raise ValueError for negative inputs"],
        "hints": ["The function should raise ValueError when x < 0"],
        "solution_code": '''def safe_sqrt(x):
    return x ** 0.5
''',
        "test_code": '''import pytest
from solution import safe_sqrt


def test_positive():
    assert safe_sqrt(4) == 2.0


def test_zero():
    assert safe_sqrt(0) == 0.0


def test_negative():
    with pytest.raises(ValueError):
        safe_sqrt(-1)
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def safe_sqrt(x):\n    return x ** 0.5",
            "new_text": "def safe_sqrt(x):\n    if x < 0:\n        raise ValueError(\"negative input\")\n    return x ** 0.5",
        },
    },
    {
        "task_id": "task_012",
        "task_type": "add_boundary_check",
        "difficulty": 2,
        "goal": "Add a boundary check to safe_divide() so it rejects zero denominators.",
        "constraints": ["Do not modify test_*.py files", "Raise ValueError when b == 0"],
        "hints": ["The function should raise ValueError when the denominator is zero"],
        "solution_code": '''def safe_divide(a, b):
    return a / b
''',
        "test_code": '''import pytest
from solution import safe_divide


def test_normal():
    assert safe_divide(10, 2) == 5


def test_zero_numerator():
    assert safe_divide(0, 5) == 0


def test_zero_denominator():
    with pytest.raises(ValueError):
        safe_divide(1, 0)
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def safe_divide(a, b):\n    return a / b",
            "new_text": "def safe_divide(a, b):\n    if b == 0:\n        raise ValueError(\"division by zero\")\n    return a / b",
        },
    },
    {
        "task_id": "task_013",
        "task_type": "add_boundary_check",
        "difficulty": 2,
        "goal": "Add a boundary check to safe_get() so it rejects negative indices.",
        "constraints": ["Do not modify test_*.py files", "Raise ValueError for negative idx"],
        "hints": ["The function should raise ValueError when idx < 0"],
        "solution_code": '''def safe_get(lst, idx):
    return lst[idx]
''',
        "test_code": '''import pytest
from solution import safe_get


def test_normal():
    assert safe_get([10, 20, 30], 1) == 20


def test_zero_idx():
    assert safe_get([10, 20, 30], 0) == 10


def test_negative_idx():
    with pytest.raises(ValueError):
        safe_get([10, 20, 30], -1)
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def safe_get(lst, idx):\n    return lst[idx]",
            "new_text": "def safe_get(lst, idx):\n    if idx < 0:\n        raise ValueError(\"negative index\")\n    return lst[idx]",
        },
    },
    {
        "task_id": "task_014",
        "task_type": "add_boundary_check",
        "difficulty": 2,
        "goal": "Add a boundary check to safe_inverse() so it rejects zero inputs.",
        "constraints": ["Do not modify test_*.py files", "Raise ValueError when x == 0"],
        "hints": ["The function should raise ValueError when x is zero"],
        "solution_code": '''def safe_inverse(x):
    return 1 / x
''',
        "test_code": '''import pytest
from solution import safe_inverse


def test_one():
    assert safe_inverse(1) == 1.0


def test_two():
    assert safe_inverse(2) == 0.5


def test_zero():
    with pytest.raises(ValueError):
        safe_inverse(0)
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def safe_inverse(x):\n    return 1 / x",
            "new_text": "def safe_inverse(x):\n    if x == 0:\n        raise ValueError(\"zero denominator\")\n    return 1 / x",
        },
    },
    {
        "task_id": "task_015",
        "task_type": "add_boundary_check",
        "difficulty": 2,
        "goal": "Add a boundary check to safe_age() so it rejects negative ages.",
        "constraints": ["Do not modify test_*.py files", "Raise ValueError when years < 0"],
        "hints": ["The function should raise ValueError when years is negative"],
        "solution_code": '''def safe_age(years):
    return years
''',
        "test_code": '''import pytest
from solution import safe_age


def test_normal():
    assert safe_age(25) == 25


def test_zero():
    assert safe_age(0) == 0


def test_negative():
    with pytest.raises(ValueError):
        safe_age(-5)
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def safe_age(years):\n    return years",
            "new_text": "def safe_age(years):\n    if years < 0:\n        raise ValueError(\"negative age\")\n    return years",
        },
    },

    # ====================================================================
    # Type 4: update_helper (task_016-020, difficulty 2)
    # Modify a helper to match updated spec in README.
    # ====================================================================
    {
        "task_id": "task_016",
        "task_type": "update_helper",
        "difficulty": 2,
        "goal": "Update the format_price() helper to match the spec: return '$X.XX' format with 2 decimal places.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function currently returns an integer; it should return a formatted string"],
        "solution_code": '''def format_price(cents):
    return cents // 100
''',
        "test_code": '''from solution import format_price


def test_format_price():
    assert format_price(1050) == "$10.50"
    assert format_price(0) == "$0.00"
    assert format_price(99) == "$0.99"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def format_price(cents):\n    return cents // 100",
            "new_text": "def format_price(cents):\n    dollars = cents / 100\n    return f\"${dollars:.2f}\"",
        },
    },
    {
        "task_id": "task_017",
        "task_type": "update_helper",
        "difficulty": 2,
        "goal": "Update the format_percent() helper to match the spec: return 'XX.X%' format with 1 decimal place.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function currently returns a float; it should return a formatted string"],
        "solution_code": '''def format_percent(ratio):
    return ratio * 100
''',
        "test_code": '''from solution import format_percent


def test_format_percent():
    assert format_percent(0.5) == "50.0%"
    assert format_percent(1.0) == "100.0%"
    assert format_percent(0.0) == "0.0%"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def format_percent(ratio):\n    return ratio * 100",
            "new_text": "def format_percent(ratio):\n    return f\"{ratio * 100:.1f}%\"",
        },
    },
    {
        "task_id": "task_018",
        "task_type": "update_helper",
        "difficulty": 2,
        "goal": "Update the format_date() helper to match the spec: return 'YYYY-MM-DD' format.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function currently returns an integer; it should return a formatted string"],
        "solution_code": '''def format_date(year, month, day):
    return year * 10000 + month * 100 + day
''',
        "test_code": '''from solution import format_date


def test_format_date():
    assert format_date(2026, 7, 8) == "2026-07-08"
    assert format_date(2000, 1, 1) == "2000-01-01"
    assert format_date(1999, 12, 31) == "1999-12-31"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def format_date(year, month, day):\n    return year * 10000 + month * 100 + day",
            "new_text": "def format_date(year, month, day):\n    return f\"{year:04d}-{month:02d}-{day:02d}\"",
        },
    },
    {
        "task_id": "task_019",
        "task_type": "update_helper",
        "difficulty": 2,
        "goal": "Update the format_name() helper to match the spec: return Title Case.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function currently lowercases; it should capitalize the first letter"],
        "solution_code": '''def format_name(s):
    return s.lower()
''',
        "test_code": '''from solution import format_name


def test_format_name():
    assert format_name("alice") == "Alice"
    assert format_name("BOB") == "Bob"
    assert format_name("charlie") == "Charlie"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def format_name(s):\n    return s.lower()",
            "new_text": "def format_name(s):\n    return s.capitalize()",
        },
    },
    {
        "task_id": "task_020",
        "task_type": "update_helper",
        "difficulty": 2,
        "goal": "Update the format_phone() helper to match the spec: return '(XXX) XXX-XXXX' format.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function currently returns the raw digit string; it should format with parens and dash"],
        "solution_code": '''def format_phone(digits):
    return digits
''',
        "test_code": '''from solution import format_phone


def test_format_phone():
    assert format_phone("5551234567") == "(555) 123-4567"
    assert format_phone("8005550123") == "(800) 555-0123"
    assert format_phone("1234567890") == "(123) 456-7890"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "def format_phone(digits):\n    return digits",
            "new_text": "def format_phone(digits):\n    return f\"({digits[:3]}) {digits[3:6]}-{digits[6:]}\"",
        },
    },

    # ====================================================================
    # Type 5: repair_after_pytest (task_021-025, difficulty 2)
    # Fix code based on pytest error message; reversed comparison logic.
    # ====================================================================
    {
        "task_id": "task_021",
        "task_type": "repair_after_pytest",
        "difficulty": 2,
        "goal": "Fix the bug in classify() based on the pytest error output.",
        "constraints": ["Do not modify test_*.py files", "Read the test failure message to understand the bug"],
        "hints": ["The function has the comparison operators reversed"],
        "solution_code": '''def classify(score):
    if score > 90:
        return "fail"
    elif score > 60:
        return "pass"
    else:
        return "excellent"
''',
        "test_code": '''from solution import classify


def test_excellent():
    assert classify(95) == "excellent"


def test_pass():
    assert classify(70) == "pass"


def test_fail():
    assert classify(50) == "fail"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if score > 90:\n        return \"fail\"\n    elif score > 60:\n        return \"pass\"\n    else:\n        return \"excellent\"",
            "new_text": "    if score > 90:\n        return \"excellent\"\n    elif score > 60:\n        return \"pass\"\n    else:\n        return \"fail\"",
        },
    },
    {
        "task_id": "task_022",
        "task_type": "repair_after_pytest",
        "difficulty": 2,
        "goal": "Fix the bug in grade_letter() based on the pytest error output.",
        "constraints": ["Do not modify test_*.py files", "Read the test failure message to understand the bug"],
        "hints": ["The function has the return values for each grade band reversed"],
        "solution_code": '''def grade_letter(score):
    if score >= 90:
        return "F"
    elif score >= 80:
        return "D"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "B"
    else:
        return "A"
''',
        "test_code": '''from solution import grade_letter


def test_a():
    assert grade_letter(95) == "A"


def test_b():
    assert grade_letter(85) == "B"


def test_c():
    assert grade_letter(75) == "C"


def test_d():
    assert grade_letter(65) == "D"


def test_f():
    assert grade_letter(50) == "F"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if score >= 90:\n        return \"F\"\n    elif score >= 80:\n        return \"D\"\n    elif score >= 70:\n        return \"C\"\n    elif score >= 60:\n        return \"B\"\n    else:\n        return \"A\"",
            "new_text": "    if score >= 90:\n        return \"A\"\n    elif score >= 80:\n        return \"B\"\n    elif score >= 70:\n        return \"C\"\n    elif score >= 60:\n        return \"D\"\n    else:\n        return \"F\"",
        },
    },
    {
        "task_id": "task_023",
        "task_type": "repair_after_pytest",
        "difficulty": 2,
        "goal": "Fix the bug in speed_label() based on the pytest error output.",
        "constraints": ["Do not modify test_*.py files", "Read the test failure message to understand the bug"],
        "hints": ["The function has the return values reversed"],
        "solution_code": '''def speed_label(mph):
    if mph > 65:
        return "slow"
    elif mph > 35:
        return "medium"
    else:
        return "fast"
''',
        "test_code": '''from solution import speed_label


def test_fast():
    assert speed_label(75) == "fast"


def test_medium():
    assert speed_label(50) == "medium"


def test_slow():
    assert speed_label(20) == "slow"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if mph > 65:\n        return \"slow\"\n    elif mph > 35:\n        return \"medium\"\n    else:\n        return \"fast\"",
            "new_text": "    if mph > 65:\n        return \"fast\"\n    elif mph > 35:\n        return \"medium\"\n    else:\n        return \"slow\"",
        },
    },
    {
        "task_id": "task_024",
        "task_type": "repair_after_pytest",
        "difficulty": 2,
        "goal": "Fix the bug in bmi_category() based on the pytest error output.",
        "constraints": ["Do not modify test_*.py files", "Read the test failure message to understand the bug"],
        "hints": ["The function has the return values reversed"],
        "solution_code": '''def bmi_category(bmi):
    if bmi >= 30:
        return "underweight"
    elif bmi >= 25:
        return "normal"
    elif bmi >= 18.5:
        return "overweight"
    else:
        return "obese"
''',
        "test_code": '''from solution import bmi_category


def test_obese():
    assert bmi_category(35) == "obese"


def test_overweight():
    assert bmi_category(27) == "overweight"


def test_normal():
    assert bmi_category(22) == "normal"


def test_underweight():
    assert bmi_category(17) == "underweight"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if bmi >= 30:\n        return \"underweight\"\n    elif bmi >= 25:\n        return \"normal\"\n    elif bmi >= 18.5:\n        return \"overweight\"\n    else:\n        return \"obese\"",
            "new_text": "    if bmi >= 30:\n        return \"obese\"\n    elif bmi >= 25:\n        return \"overweight\"\n    elif bmi >= 18.5:\n        return \"normal\"\n    else:\n        return \"underweight\"",
        },
    },
    {
        "task_id": "task_025",
        "task_type": "repair_after_pytest",
        "difficulty": 2,
        "goal": "Fix the bug in temperature_label() based on the pytest error output.",
        "constraints": ["Do not modify test_*.py files", "Read the test failure message to understand the bug"],
        "hints": ["The function has the return values reversed"],
        "solution_code": '''def temperature_label(c):
    if c >= 30:
        return "cold"
    elif c >= 15:
        return "mild"
    else:
        return "hot"
''',
        "test_code": '''from solution import temperature_label


def test_hot():
    assert temperature_label(35) == "hot"


def test_mild():
    assert temperature_label(20) == "mild"


def test_cold():
    assert temperature_label(5) == "cold"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if c >= 30:\n        return \"cold\"\n    elif c >= 15:\n        return \"mild\"\n    else:\n        return \"hot\"",
            "new_text": "    if c >= 30:\n        return \"hot\"\n    elif c >= 15:\n        return \"mild\"\n    else:\n        return \"cold\"",
        },
    },

    # ====================================================================
    # Type 6: avoid_editing_tests (task_026-030, difficulty 1)
    # Bug is in solution.py; test file must not be modified. Off-by-one bugs.
    # ====================================================================
    {
        "task_id": "task_026",
        "task_type": "avoid_editing_tests",
        "difficulty": 1,
        "goal": "Fix the bug in solution.py. The test file is correct and must not be modified.",
        "constraints": ["Do not modify test_*.py files", "The bug is only in solution.py"],
        "hints": ["The repeat() function has an off-by-one error"],
        "solution_code": '''def repeat(s, n):
    return s * (n - 1)
''',
        "test_code": '''from solution import repeat


def test_repeat_basic():
    assert repeat("ab", 3) == "ababab"


def test_repeat_zero():
    assert repeat("x", 0) == ""


def test_repeat_one():
    assert repeat("y", 1) == "y"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return s * (n - 1)",
            "new_text": "    return s * n",
        },
    },
    {
        "task_id": "task_027",
        "task_type": "avoid_editing_tests",
        "difficulty": 1,
        "goal": "Fix the bug in solution.py. The test file is correct and must not be modified.",
        "constraints": ["Do not modify test_*.py files", "The bug is only in solution.py"],
        "hints": ["The echo() function has an off-by-one error"],
        "solution_code": '''def echo(s, n):
    return s * (n + 1)
''',
        "test_code": '''from solution import echo


def test_echo_basic():
    assert echo("ab", 3) == "ababab"


def test_echo_zero():
    assert echo("x", 0) == ""


def test_echo_one():
    assert echo("y", 1) == "y"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return s * (n + 1)",
            "new_text": "    return s * n",
        },
    },
    {
        "task_id": "task_028",
        "task_type": "avoid_editing_tests",
        "difficulty": 1,
        "goal": "Fix the bug in solution.py. The test file is correct and must not be modified.",
        "constraints": ["Do not modify test_*.py files", "The bug is only in solution.py"],
        "hints": ["The triple() function returns the wrong number of copies"],
        "solution_code": '''def triple(s):
    return s * 2
''',
        "test_code": '''from solution import triple


def test_triple_basic():
    assert triple("ab") == "ababab"


def test_triple_single():
    assert triple("x") == "xxx"


def test_triple_empty():
    assert triple("") == ""
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return s * 2",
            "new_text": "    return s * 3",
        },
    },
    {
        "task_id": "task_029",
        "task_type": "avoid_editing_tests",
        "difficulty": 1,
        "goal": "Fix the bug in solution.py. The test file is correct and must not be modified.",
        "constraints": ["Do not modify test_*.py files", "The bug is only in solution.py"],
        "hints": ["The pad_left() function pads one too few spaces"],
        "solution_code": '''def pad_left(s, n):
    return " " * (n - len(s) - 1) + s
''',
        "test_code": '''from solution import pad_left


def test_pad_basic():
    assert pad_left("hi", 5) == "   hi"


def test_pad_exact():
    assert pad_left("hello", 5) == "hello"


def test_pad_one():
    assert pad_left("ab", 3) == " ab"
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    return \" \" * (n - len(s) - 1) + s",
            "new_text": "    return \" \" * (n - len(s)) + s",
        },
    },
    {
        "task_id": "task_030",
        "task_type": "avoid_editing_tests",
        "difficulty": 1,
        "goal": "Fix the bug in solution.py. The test file is correct and must not be modified.",
        "constraints": ["Do not modify test_*.py files", "The bug is only in solution.py"],
        "hints": ["The chunk_list() function uses the wrong step size"],
        "solution_code": '''def chunk_list(lst, size):
    step = size + 1
    return [lst[i:i+step] for i in range(0, len(lst), step)]
''',
        "test_code": '''from solution import chunk_list


def test_chunk_even():
    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_uneven():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_empty():
    assert chunk_list([], 2) == []
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    step = size + 1",
            "new_text": "    step = size",
        },
    },

    # ====================================================================
    # Type 7: recover_from_failed_patch (task_031-035, difficulty 3)
    # First obvious patch is wrong; must inspect error and retry.
    # ====================================================================
    {
        "task_id": "task_031",
        "task_type": "recover_from_failed_patch",
        "difficulty": 3,
        "goal": "Fix the bug in clamp(). The obvious fix may not work — inspect errors and retry.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function should return lo when x < lo, hi when x > hi, else x. But the current code has the comparison directions wrong AND a logic error."],
        "solution_code": '''def clamp(x, lo, hi):
    if x > lo:
        return lo
    if x < hi:
        return hi
    return x
''',
        "test_code": '''from solution import clamp


def test_clamp_below():
    assert clamp(-5, 0, 10) == 0


def test_clamp_above():
    assert clamp(15, 0, 10) == 10


def test_clamp_in_range():
    assert clamp(5, 0, 10) == 5


def test_clamp_at_bounds():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if x > lo:\n        return lo\n    if x < hi:\n        return hi\n    return x",
            "new_text": "    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x",
        },
    },
    {
        "task_id": "task_032",
        "task_type": "recover_from_failed_patch",
        "difficulty": 3,
        "goal": "Fix the bug in saturate(). The obvious fix may not work — inspect errors and retry.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function should return lo when x < lo, hi when x > hi, else x. The current code returns the wrong bound for each case."],
        "solution_code": '''def saturate(x, lo, hi):
    if x > hi:
        return lo
    if x < lo:
        return hi
    return x
''',
        "test_code": '''from solution import saturate


def test_below():
    assert saturate(-5, 0, 10) == 0


def test_above():
    assert saturate(15, 0, 10) == 10


def test_in_range():
    assert saturate(5, 0, 10) == 5


def test_at_bounds():
    assert saturate(0, 0, 10) == 0
    assert saturate(10, 0, 10) == 10
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if x > hi:\n        return lo\n    if x < lo:\n        return hi\n    return x",
            "new_text": "    if x > hi:\n        return hi\n    if x < lo:\n        return lo\n    return x",
        },
    },
    {
        "task_id": "task_033",
        "task_type": "recover_from_failed_patch",
        "difficulty": 3,
        "goal": "Fix the bug in limit_value(). The obvious fix may not work — inspect errors and retry.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function has the return values swapped AND an off-by-one in the in-range case."],
        "solution_code": '''def limit_value(x, lo, hi):
    if x < lo:
        return hi
    if x > hi:
        return lo
    return x + 1
''',
        "test_code": '''from solution import limit_value


def test_below():
    assert limit_value(-5, 0, 10) == 0


def test_above():
    assert limit_value(15, 0, 10) == 10


def test_in_range():
    assert limit_value(5, 0, 10) == 5


def test_at_bounds():
    assert limit_value(0, 0, 10) == 0
    assert limit_value(10, 0, 10) == 10
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if x < lo:\n        return hi\n    if x > hi:\n        return lo\n    return x + 1",
            "new_text": "    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x",
        },
    },
    {
        "task_id": "task_034",
        "task_type": "recover_from_failed_patch",
        "difficulty": 3,
        "goal": "Fix the bug in bound_value(). The obvious fix may not work — inspect errors and retry.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function has the comparison directions wrong for both bounds."],
        "solution_code": '''def bound_value(x, lo, hi):
    if x >= lo:
        return lo
    if x <= hi:
        return hi
    return x
''',
        "test_code": '''from solution import bound_value


def test_below():
    assert bound_value(-5, 0, 10) == 0


def test_above():
    assert bound_value(15, 0, 10) == 10


def test_in_range():
    assert bound_value(5, 0, 10) == 5


def test_at_bounds():
    assert bound_value(0, 0, 10) == 0
    assert bound_value(10, 0, 10) == 10
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if x >= lo:\n        return lo\n    if x <= hi:\n        return hi\n    return x",
            "new_text": "    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x",
        },
    },
    {
        "task_id": "task_035",
        "task_type": "recover_from_failed_patch",
        "difficulty": 3,
        "goal": "Fix the bug in clip_to_range(). The obvious fix may not work — inspect errors and retry.",
        "constraints": ["Do not modify test_*.py files"],
        "hints": ["The function has the comparison directions wrong AND returns the wrong bound."],
        "solution_code": '''def clip_to_range(x, lo, hi):
    if x > lo:
        return hi
    if x < hi:
        return lo
    return x
''',
        "test_code": '''from solution import clip_to_range


def test_below():
    assert clip_to_range(-5, 0, 10) == 0


def test_above():
    assert clip_to_range(15, 0, 10) == 10


def test_in_range():
    assert clip_to_range(5, 0, 10) == 5


def test_at_bounds():
    assert clip_to_range(0, 0, 10) == 0
    assert clip_to_range(10, 0, 10) == 10
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if x > lo:\n        return hi\n    if x < hi:\n        return lo\n    return x",
            "new_text": "    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x",
        },
    },

    # ====================================================================
    # Type 8: finish_after_tests_pass (task_036-040, difficulty 3)
    # Multi-step: patch -> test fails -> patch -> test passes -> finish.
    # NOTE: task_036 test_negative assertion adjusted from brief's example
    # (was: abs(sum(result)-1.0)<1e-9; now: result==[0.0,0.0]). See report.
    # ====================================================================
    {
        "task_id": "task_036",
        "task_type": "finish_after_tests_pass",
        "difficulty": 3,
        "goal": "Fix both bugs in normalize() so all tests pass. A single fix may not be enough.",
        "constraints": ["Do not modify test_*.py files", "Run tests after each patch to verify progress"],
        "hints": ["There are two bugs: one in the condition and one in the return value"],
        "solution_code": '''def normalize(values):
    total = sum(values)
    if total == 0:
        return values
    return [v / total for v in values]
''',
        "test_code": '''from solution import normalize


def test_basic():
    result = normalize([1, 2, 3])
    assert abs(sum(result) - 1.0) < 1e-9


def test_zero_sum():
    assert normalize([0, 0, 0]) == [0.0, 0.0, 0.0]


def test_single():
    assert normalize([5]) == [1.0]


def test_negative():
    result = normalize([1, -1])
    assert result == [0.0, 0.0]
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if total == 0:\n        return values\n    return [v / total for v in values]",
            "new_text": "    if total == 0:\n        return [0.0 for _ in values]\n    return [v / total for v in values]",
        },
    },
    {
        "task_id": "task_037",
        "task_type": "finish_after_tests_pass",
        "difficulty": 3,
        "goal": "Fix both bugs in standardize() so all tests pass. A single fix may not be enough.",
        "constraints": ["Do not modify test_*.py files", "Run tests after each patch to verify progress"],
        "hints": ["There are two bugs: one in the condition and one in the return value"],
        "solution_code": '''def standardize(values):
    mean = sum(values) / len(values)
    if mean == 0:
        return values
    return [v - mean for v in values]
''',
        "test_code": '''from solution import standardize


def test_basic():
    result = standardize([1, 2, 3])
    assert abs(sum(result)) < 1e-9


def test_zero_mean():
    assert standardize([1, -1]) == [0.0, 0.0]


def test_single():
    assert standardize([5]) == [0.0]


def test_negative():
    result = standardize([-2, -4, -6])
    assert abs(sum(result)) < 1e-9
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if mean == 0:\n        return values\n    return [v - mean for v in values]",
            "new_text": "    if mean == 0:\n        return [0.0 for _ in values]\n    return [v - mean for v in values]",
        },
    },
    {
        "task_id": "task_038",
        "task_type": "finish_after_tests_pass",
        "difficulty": 3,
        "goal": "Fix both bugs in rescale() so all tests pass. A single fix may not be enough.",
        "constraints": ["Do not modify test_*.py files", "Run tests after each patch to verify progress"],
        "hints": ["There are two bugs: one in the condition and one in the return value"],
        "solution_code": '''def rescale(values, new_min, new_max):
    old_min = min(values)
    old_max = max(values)
    if old_max == old_min:
        return values
    return [(v - old_min) / (old_max - old_min) * (new_max - new_min) + new_min for v in values]
''',
        "test_code": '''from solution import rescale


def test_basic():
    result = rescale([1, 2, 3], 0, 10)
    assert abs(result[0] - 0.0) < 1e-9
    assert abs(result[1] - 5.0) < 1e-9
    assert abs(result[2] - 10.0) < 1e-9


def test_constant():
    assert rescale([5, 5, 5], 0, 10) == [0.0, 0.0, 0.0]


def test_single():
    assert rescale([7], 0, 1) == [0.0]


def test_negative_range():
    result = rescale([1, 2, 3], -1, 1)
    assert abs(result[0] + 1.0) < 1e-9
    assert abs(result[2] - 1.0) < 1e-9
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if old_max == old_min:\n        return values\n    return [(v - old_min) / (old_max - old_min) * (new_max - new_min) + new_min for v in values]",
            "new_text": "    if old_max == old_min:\n        return [new_min for _ in values]\n    return [(v - old_min) / (old_max - old_min) * (new_max - new_min) + new_min for v in values]",
        },
    },
    {
        "task_id": "task_039",
        "task_type": "finish_after_tests_pass",
        "difficulty": 3,
        "goal": "Fix both bugs in fractionalize() so all tests pass. A single fix may not be enough.",
        "constraints": ["Do not modify test_*.py files", "Run tests after each patch to verify progress"],
        "hints": ["There are two bugs: one in the zero-length branch and one in the divisor"],
        "solution_code": '''def fractionalize(values):
    n = len(values)
    if n == 0:
        return values
    return [v / (n - 1) for v in values]
''',
        "test_code": '''from solution import fractionalize


def test_basic():
    result = fractionalize([10, 20, 30])
    assert abs(result[0] - 10.0/3.0) < 1e-9
    assert abs(result[1] - 20.0/3.0) < 1e-9
    assert abs(result[2] - 30.0/3.0) < 1e-9


def test_empty():
    assert fractionalize([]) == []


def test_single():
    assert fractionalize([5]) == [5.0]


def test_pair():
    result = fractionalize([4, 6])
    assert abs(result[0] - 2.0) < 1e-9
    assert abs(result[1] - 3.0) < 1e-9
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if n == 0:\n        return values\n    return [v / (n - 1) for v in values]",
            "new_text": "    if n == 0:\n        return []\n    return [v / n for v in values]",
        },
    },
    {
        "task_id": "task_040",
        "task_type": "finish_after_tests_pass",
        "difficulty": 3,
        "goal": "Fix both bugs in proportion() so all tests pass. A single fix may not be enough.",
        "constraints": ["Do not modify test_*.py files", "Run tests after each patch to verify progress"],
        "hints": ["There are two bugs: one in the zero-total branch and one in the return value"],
        "solution_code": '''def proportion(values, idx):
    total = sum(values)
    if total == 0:
        return values[idx]
    return values[idx] / total
''',
        "test_code": '''from solution import proportion


def test_basic():
    assert abs(proportion([10, 20, 30], 1) - 20.0/60.0) < 1e-9


def test_zero_total():
    assert proportion([5, -5, 0], 0) == 0.0


def test_first():
    assert abs(proportion([1, 2, 3], 0) - 1.0/6.0) < 1e-9


def test_negative_total():
    assert abs(proportion([1, -1, 2], 2) - 2.0/2.0) < 1e-9
''',
        "expected_patch": {
            "file_path": "solution.py",
            "old_text": "    if total == 0:\n        return values[idx]\n    return values[idx] / total",
            "new_text": "    if total == 0:\n        return 0.0\n    return values[idx] / total",
        },
    },
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for spec in TASK_SPECS:
        entry = _build_task(spec)
        manifest_entries.append(entry)

    manifest = {
        "schema_version": 1,
        "suite_name": "micro-tasks-v0",
        "total_tasks": len(manifest_entries),
        "task_types": TASK_TYPES,
        "tasks": manifest_entries,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Generated {len(manifest_entries)} tasks in {OUT_DIR}")


if __name__ == "__main__":
    main()
