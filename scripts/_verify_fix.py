"""Verify the run_pytest fix handles bare-assert MBPP tests."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sandbox import run_pytest, _normalize_test_code

# Test cases
print("=== Test _normalize_test_code ===")
cases = [
    ("bare asserts", "assert max_of_nth([(5,6,7),(1,3,5),(8,9,19)], 2) == 19\n"),
    ("with from solution", "from solution import max_of_nth\nassert max_of_nth([], 2) == 0\n"),
    ("with def test", "def test_x():\n    assert foo() == 1\n"),
    ("empty", ""),
    ("comment only", "# no tests here\n"),
]
for label, code in cases:
    print(f"\n[{label}]")
    print(f"INPUT:\n{code}")
    print(f"OUTPUT:\n{_normalize_test_code(code)}")

print("\n=== Test run_pytest with bare asserts ===")
# count_char - correct code, bare asserts
result = run_pytest(
    "def count_char(str1, char):\n    count = str1.count(char)\n    return (count)\n",
    "assert count_char('Python','o')==1\nassert count_char('little','t')==2\n",
    timeout_s=10.0,
)
print(f"count_char (correct): passed={result.passed}, collected={result.num_collected}, "
      f"passed={result.num_passed}, failed={result.num_failed}, returncode={result.returncode}")

# count_char - wrong code, bare asserts (should fail)
result = run_pytest(
    "def count_char(str1, char):\n    return 0\n",
    "assert count_char('Python','o')==1\n",
    timeout_s=10.0,
)
print(f"count_char (wrong): passed={result.passed}, collected={result.num_collected}, "
      f"passed={result.num_passed}, failed={result.num_failed}, returncode={result.returncode}")

# remove_spaces - correct code with different param name
result = run_pytest(
    "def remove_spaces(text):\n    text = text.replace(' ', \"\")\n    return text\n",
    "assert remove_spaces('a b c') == 'abc'\nassert remove_spaces('1 2 3') == '123'\n",
    timeout_s=10.0,
)
print(f"remove_spaces (correct, diff param): passed={result.passed}, collected={result.num_collected}, "
      f"passed={result.num_passed}, failed={result.num_failed}")

# No tests case - should not pass (no tests collected)
result = run_pytest("x = 1\n", "# no tests here\n", timeout_s=5.0)
print(f"no-tests case: passed={result.passed}, collected={result.num_collected}")
