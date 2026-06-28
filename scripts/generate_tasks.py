"""
scripts/generate_tasks.py -- Procedural task bank for the Qwen3-0.6B Code Recovery Lab.

Generates Sample objects (task_type=code_generation) from 12 hand-authored task
families spanning difficulty levels 0-2:
  - L0 (3 families): simple two-liner functions good for syntax-level mutations.
  - L1 (5 families): single-function implementations with loops / strings.
  - L2 (4 families): boundary conditions and basic data structures.

LEVEL DESIGN (why only 0-2 here, not 0-3)
-----------------------------------------
This procedural task bank intentionally contains ONLY Level 0-2
code_generation families.  Level 3 in the curriculum is "execution-feedback
repair" -- a task that, by definition, has no standalone code_generation
reference (it is a broken-code-plus-real-feedback fix task).  Level-3 samples
are therefore NOT authored here; they are produced downstream by
``scripts/mutate_code.py`` as ``execution_repair`` samples (difficulty=3),
which carry broken_code + captured pytest output.  Together the two scripts
cover the full Level 0-3 curriculum: 0-2 from this bank, 3 from the mutator.

No LLM/network calls.  Pure procedural generation.

Usage
-----
    python scripts/generate_tasks.py [--count N] [--seed S] [--out PATH] [--levels 0 1 2]

Exit codes
----------
    0   success
    1   error
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, Verification  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)
_DATASET_VERSION = "v1.0"
_GENERATOR = "procedural_v1"


# ---------------------------------------------------------------------------
# TaskFamily dataclass
# ---------------------------------------------------------------------------


@dataclass
class TaskFamily:
    """Describes a single hand-authored task family."""

    family_id: str
    difficulty: int          # 0 = L0, 1 = L1, 2 = L2
    skill_tags: list[str]
    instruction: str         # Chinese task description
    target_code: str         # Reference (correct) implementation
    public_tests: str        # pytest file (imports from solution)
    hidden_tests: str        # Additional edge-case tests
    description: str = field(default="")  # English short description


# ---------------------------------------------------------------------------
# Task family definitions
# ---------------------------------------------------------------------------

# ---- L0: simple two-liner functions ----------------------------------------

FAMILY_L0_ADD_TWO = TaskFamily(
    family_id="fam_l0_add_two",
    difficulty=0,
    description="Sum of two numbers",
    skill_tags=["arithmetic", "basics"],
    instruction="实现函数 `add_two(a, b)`，返回两个数的和。",
    target_code="""\
def add_two(a, b):
    return a + b
""",
    public_tests="""\
from solution import add_two


def test_positive():
    assert add_two(1, 2) == 3


def test_zero():
    assert add_two(0, 5) == 5


def test_negative():
    assert add_two(-3, -2) == -5
""",
    hidden_tests="""\
from solution import add_two


def test_both_zero():
    assert add_two(0, 0) == 0


def test_large():
    assert add_two(10 ** 6, 10 ** 6) == 2 * 10 ** 6
""",
)

FAMILY_L0_IS_POSITIVE = TaskFamily(
    family_id="fam_l0_is_positive",
    difficulty=0,
    description="Check if number is strictly positive",
    skill_tags=["comparison", "basics"],
    instruction="实现函数 `is_positive(n)`，若 n 严格大于 0 则返回 True，否则返回 False。",
    target_code="""\
def is_positive(n):
    return n > 0
""",
    public_tests="""\
from solution import is_positive


def test_positive():
    assert is_positive(5) is True


def test_zero():
    assert is_positive(0) is False


def test_negative():
    assert is_positive(-3) is False
""",
    hidden_tests="""\
from solution import is_positive


def test_small_positive():
    assert is_positive(0.001) is True


def test_large_negative():
    assert is_positive(-1000) is False
""",
)

FAMILY_L0_CLAMP = TaskFamily(
    family_id="fam_l0_clamp",
    difficulty=0,
    description="Clamp a value to [lo, hi]",
    skill_tags=["comparison", "basics", "boundary"],
    instruction=(
        "实现函数 `clamp(x, lo, hi)`，将 x 限制在 [lo, hi] 范围内：\n"
        "若 x < lo 返回 lo；若 x > hi 返回 hi；否则返回 x。"
    ),
    target_code="""\
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
""",
    public_tests="""\
from solution import clamp


def test_below():
    assert clamp(-5, 0, 10) == 0


def test_above():
    assert clamp(15, 0, 10) == 10


def test_within():
    assert clamp(5, 0, 10) == 5


def test_at_lo():
    assert clamp(0, 0, 10) == 0


def test_at_hi():
    assert clamp(10, 0, 10) == 10
""",
    hidden_tests="""\
from solution import clamp


def test_equal_bounds():
    assert clamp(5, 3, 3) == 3


def test_negative_range():
    assert clamp(-2, -5, -1) == -2
""",
)

# ---- L1: single-function implementations -----------------------------------

FAMILY_L1_SUM_LIST = TaskFamily(
    family_id="fam_l1_sum_list",
    difficulty=1,
    description="Sum all elements in a list",
    skill_tags=["loops", "accumulation", "list"],
    instruction="实现函数 `sum_list(lst)`，返回列表中所有数字的总和；列表为空时返回 0。",
    target_code="""\
def sum_list(lst):
    total = 0
    for x in lst:
        total += x
    return total
""",
    public_tests="""\
from solution import sum_list


def test_basic():
    assert sum_list([1, 2, 3]) == 6


def test_empty():
    assert sum_list([]) == 0


def test_single():
    assert sum_list([42]) == 42


def test_negative():
    assert sum_list([-1, -2, -3]) == -6
""",
    hidden_tests="""\
from solution import sum_list


def test_mixed():
    assert sum_list([-1, 0, 1, 2]) == 2


def test_gauss():
    assert sum_list(list(range(101))) == 5050
""",
)

FAMILY_L1_REVERSE_STRING = TaskFamily(
    family_id="fam_l1_reverse_string",
    difficulty=1,
    description="Reverse a string",
    skill_tags=["string", "slicing"],
    instruction="实现函数 `reverse_string(s)`，返回字符串 s 的逆序。",
    target_code="""\
def reverse_string(s):
    return s[::-1]
""",
    public_tests="""\
from solution import reverse_string


def test_basic():
    assert reverse_string("hello") == "olleh"


def test_empty():
    assert reverse_string("") == ""


def test_single():
    assert reverse_string("a") == "a"


def test_numbers():
    assert reverse_string("12345") == "54321"
""",
    hidden_tests="""\
from solution import reverse_string


def test_palindrome():
    assert reverse_string("racecar") == "racecar"


def test_spaces():
    assert reverse_string("ab cd") == "dc ba"
""",
)

FAMILY_L1_COUNT_VOWELS = TaskFamily(
    family_id="fam_l1_count_vowels",
    difficulty=1,
    description="Count vowels in a string (case-insensitive)",
    skill_tags=["string", "loops", "counting"],
    instruction=(
        "实现函数 `count_vowels(s)`，统计字符串 s 中元音字母"
        "（a、e、i、o、u，大小写均计）的数量。"
    ),
    target_code="""\
def count_vowels(s):
    count = 0
    for ch in s.lower():
        if ch in "aeiou":
            count += 1
    return count
""",
    public_tests="""\
from solution import count_vowels


def test_basic():
    assert count_vowels("hello") == 2


def test_empty():
    assert count_vowels("") == 0


def test_no_vowels():
    assert count_vowels("rhythm") == 0


def test_all_vowels():
    assert count_vowels("aeiou") == 5
""",
    hidden_tests="""\
from solution import count_vowels


def test_uppercase():
    assert count_vowels("HELLO") == 2


def test_mixed():
    assert count_vowels("Hello World") == 3
""",
)

FAMILY_L1_IS_PALINDROME = TaskFamily(
    family_id="fam_l1_is_palindrome",
    difficulty=1,
    description="Check if a string is a palindrome (case-insensitive)",
    skill_tags=["string", "comparison", "slicing"],
    instruction="实现函数 `is_palindrome(s)`，判断字符串 s（忽略大小写）是否为回文串。",
    target_code="""\
def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]
""",
    public_tests="""\
from solution import is_palindrome


def test_palindrome():
    assert is_palindrome("racecar") is True


def test_not_palindrome():
    assert is_palindrome("hello") is False


def test_empty():
    assert is_palindrome("") is True


def test_single():
    assert is_palindrome("a") is True
""",
    hidden_tests="""\
from solution import is_palindrome


def test_uppercase_palindrome():
    assert is_palindrome("Racecar") is True


def test_two_chars_palindrome():
    assert is_palindrome("aa") is True


def test_two_chars_not():
    assert is_palindrome("ab") is False
""",
)

FAMILY_L1_FACTORIAL = TaskFamily(
    family_id="fam_l1_factorial",
    difficulty=1,
    description="Iterative factorial with negative-input guard",
    skill_tags=["loops", "math", "validation"],
    instruction=(
        "实现函数 `factorial(n)`，计算 n 的阶乘（迭代实现）。"
        "n 为负数时抛出 ValueError。"
    ),
    target_code="""\
def factorial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
""",
    public_tests="""\
from solution import factorial
import pytest


def test_zero():
    assert factorial(0) == 1


def test_one():
    assert factorial(1) == 1


def test_five():
    assert factorial(5) == 120


def test_negative():
    with pytest.raises(ValueError):
        factorial(-1)
""",
    hidden_tests="""\
from solution import factorial
import pytest


def test_ten():
    assert factorial(10) == 3628800


def test_two():
    assert factorial(2) == 2


def test_negative_guard():
    # Exercises the negative-input guard so the drop_guard mutation fails here.
    with pytest.raises(ValueError):
        factorial(-5)
""",
)

# ---- L2: boundary conditions and data structures ---------------------------

FAMILY_L2_SECOND_LARGEST = TaskFamily(
    family_id="fam_l2_second_largest",
    difficulty=2,
    description="Second largest distinct value in a list",
    skill_tags=["sorting", "set", "boundary"],
    instruction=(
        "实现函数 `second_largest(lst)`，返回列表中第二大的不重复值。"
        "若不足两个不同的值则抛出 ValueError。"
    ),
    target_code="""\
def second_largest(lst):
    unique = sorted(set(lst), reverse=True)
    if len(unique) < 2:
        raise ValueError("Need at least 2 distinct values")
    return unique[1]
""",
    public_tests="""\
from solution import second_largest
import pytest


def test_basic():
    assert second_largest([3, 1, 4, 1, 5, 9, 2, 6]) == 6


def test_simple():
    assert second_largest([1, 2]) == 1


def test_negatives():
    assert second_largest([-1, -3, -2]) == -2


def test_raises():
    with pytest.raises(ValueError):
        second_largest([5, 5, 5])
""",
    hidden_tests="""\
from solution import second_largest
import pytest


def test_sorted_input():
    assert second_largest([1, 2, 3, 4, 5]) == 4


def test_duplicates_of_max():
    assert second_largest([5, 5, 3, 1]) == 3


def test_too_few_distinct_guard():
    # Exercises the <2-distinct guard so the drop_guard mutation fails here.
    with pytest.raises(ValueError):
        second_largest([7])
""",
)

FAMILY_L2_BALANCED_PARENS = TaskFamily(
    family_id="fam_l2_balanced_parens",
    difficulty=2,
    description="Balanced parentheses/brackets checker using a stack",
    skill_tags=["stack", "string", "data_structures"],
    instruction=(
        "实现函数 `is_balanced(s)`，判断字符串 s 中的括号"
        "（`()`、`[]`、`{}`）是否完全匹配且正确嵌套。"
    ),
    target_code="""\
def is_balanced(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != mapping[ch]:
                return False
            stack.pop()
    return len(stack) == 0
""",
    public_tests="""\
from solution import is_balanced


def test_balanced():
    assert is_balanced("()[]{}") is True


def test_nested():
    assert is_balanced("([{}])") is True


def test_unbalanced():
    assert is_balanced("(]") is False


def test_empty():
    assert is_balanced("") is True


def test_unclosed():
    assert is_balanced("(") is False
""",
    hidden_tests="""\
from solution import is_balanced


def test_complex_balanced():
    assert is_balanced("{[()]}") is True


def test_wrong_order():
    assert is_balanced(")(") is False


def test_nested_mismatch():
    assert is_balanced("([)]") is False
""",
)

FAMILY_L2_MERGE_SORTED = TaskFamily(
    family_id="fam_l2_merge_sorted",
    difficulty=2,
    description="Merge two sorted lists",
    skill_tags=["sorting", "two_pointers", "list"],
    instruction="实现函数 `merge_sorted(a, b)`，将两个已排好序的列表合并为一个有序列表。",
    target_code="""\
def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
""",
    public_tests="""\
from solution import merge_sorted


def test_basic():
    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]


def test_empty_a():
    assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]


def test_empty_b():
    assert merge_sorted([1, 2, 3], []) == [1, 2, 3]


def test_duplicates():
    assert merge_sorted([1, 2], [1, 3]) == [1, 1, 2, 3]
""",
    hidden_tests="""\
from solution import merge_sorted


def test_all_same():
    assert merge_sorted([2, 2], [2, 2]) == [2, 2, 2, 2]


def test_one_longer():
    assert merge_sorted([1, 5, 9], [2, 3]) == [1, 2, 3, 5, 9]
""",
)

FAMILY_L2_FIND_DUPLICATES = TaskFamily(
    family_id="fam_l2_find_duplicates",
    difficulty=2,
    description="Find all duplicate elements in a list, sorted",
    skill_tags=["set", "list", "deduplication"],
    instruction=(
        "实现函数 `find_duplicates(lst)`，返回列表中所有重复出现的元素"
        "（排序后的列表，每个重复元素只出现一次）。"
    ),
    target_code="""\
def find_duplicates(lst):
    seen = set()
    dups = set()
    for x in lst:
        if x in seen:
            dups.add(x)
        seen.add(x)
    return sorted(dups)
""",
    public_tests="""\
from solution import find_duplicates


def test_basic():
    assert find_duplicates([1, 2, 3, 2, 1]) == [1, 2]


def test_no_dups():
    assert find_duplicates([1, 2, 3]) == []


def test_empty():
    assert find_duplicates([]) == []


def test_all_same():
    assert find_duplicates([5, 5, 5]) == [5]
""",
    hidden_tests="""\
from solution import find_duplicates


def test_many_dups():
    assert find_duplicates([3, 1, 2, 1, 3, 2]) == [1, 2, 3]


def test_strings():
    assert find_duplicates(['a', 'b', 'a', 'c']) == ['a']
""",
)

# ---- Additional L0 families ------------------------------------------------

FAMILY_L0_ABS_VALUE = TaskFamily(
    family_id="fam_l0_abs_value",
    difficulty=0,
    description="Absolute value of a number",
    skill_tags=["arithmetic", "basics", "comparison"],
    instruction="实现函数 `abs_value(n)`，返回 n 的绝对值。",
    target_code="""\
def abs_value(n):
    if n < 0:
        return -n
    return n
""",
    public_tests="""\
from solution import abs_value


def test_positive():
    assert abs_value(5) == 5


def test_negative():
    assert abs_value(-3) == 3


def test_zero():
    assert abs_value(0) == 0


def test_float():
    assert abs_value(-2.5) == 2.5
""",
    hidden_tests="""\
from solution import abs_value


def test_large_negative():
    assert abs_value(-1000000) == 1000000


def test_small_positive():
    assert abs_value(0.001) == 0.001
""",
)

FAMILY_L0_MAX_OF_TWO = TaskFamily(
    family_id="fam_l0_max_of_two",
    difficulty=0,
    description="Max of two numbers",
    skill_tags=["comparison", "basics"],
    instruction="实现函数 `max_of_two(a, b)`，返回两个数中较大的那个。",
    target_code="""\
def max_of_two(a, b):
    if a >= b:
        return a
    return b
""",
    public_tests="""\
from solution import max_of_two


def test_first_larger():
    assert max_of_two(5, 3) == 5


def test_second_larger():
    assert max_of_two(2, 7) == 7


def test_equal():
    assert max_of_two(4, 4) == 4


def test_negative():
    assert max_of_two(-1, -5) == -1
""",
    hidden_tests="""\
from solution import max_of_two


def test_zero_and_negative():
    assert max_of_two(0, -1) == 0


def test_floats():
    assert max_of_two(3.14, 2.72) == 3.14
""",
)

FAMILY_L0_MIN_OF_TWO = TaskFamily(
    family_id="fam_l0_min_of_two",
    difficulty=0,
    description="Min of two numbers",
    skill_tags=["comparison", "basics"],
    instruction="实现函数 `min_of_two(a, b)`，返回两个数中较小的那个。",
    target_code="""\
def min_of_two(a, b):
    if a <= b:
        return a
    return b
""",
    public_tests="""\
from solution import min_of_two


def test_first_smaller():
    assert min_of_two(3, 5) == 3


def test_second_smaller():
    assert min_of_two(7, 2) == 2


def test_equal():
    assert min_of_two(4, 4) == 4


def test_negative():
    assert min_of_two(-1, -5) == -5
""",
    hidden_tests="""\
from solution import min_of_two


def test_zero_and_positive():
    assert min_of_two(0, 1) == 0


def test_floats():
    assert min_of_two(3.14, 2.72) == 2.72
""",
)

FAMILY_L0_IS_EVEN = TaskFamily(
    family_id="fam_l0_is_even",
    difficulty=0,
    description="Check if a number is even",
    skill_tags=["arithmetic", "basics", "comparison"],
    instruction="实现函数 `is_even(n)`，若 n 是偶数则返回 True，否则返回 False。",
    target_code="""\
def is_even(n):
    return n % 2 == 0
""",
    public_tests="""\
from solution import is_even


def test_even():
    assert is_even(4) is True


def test_odd():
    assert is_even(3) is False


def test_zero():
    assert is_even(0) is True


def test_negative_even():
    assert is_even(-6) is True
""",
    hidden_tests="""\
from solution import is_even


def test_negative_odd():
    assert is_even(-3) is False


def test_large_even():
    assert is_even(1000000) is True
""",
)

FAMILY_L0_MULTIPLY = TaskFamily(
    family_id="fam_l0_multiply",
    difficulty=0,
    description="Multiply two numbers",
    skill_tags=["arithmetic", "basics"],
    instruction="实现函数 `multiply(a, b)`，返回两个数的乘积。",
    target_code="""\
def multiply(a, b):
    return a * b
""",
    public_tests="""\
from solution import multiply


def test_positive():
    assert multiply(3, 4) == 12


def test_zero():
    assert multiply(5, 0) == 0


def test_negative():
    assert multiply(-2, 3) == -6


def test_both_negative():
    assert multiply(-2, -3) == 6
""",
    hidden_tests="""\
from solution import multiply


def test_one():
    assert multiply(7, 1) == 7


def test_large():
    assert multiply(1000, 1000) == 1000000
""",
)

# ---- Additional L1 families ------------------------------------------------

FAMILY_L1_MAX_LIST = TaskFamily(
    family_id="fam_l1_max_list",
    difficulty=1,
    description="Find max element in a list",
    skill_tags=["loops", "list", "comparison"],
    instruction="实现函数 `max_list(lst)`，返回列表中的最大元素；列表为空时抛出 ValueError。",
    target_code="""\
def max_list(lst):
    if not lst:
        raise ValueError("Empty list")
    result = lst[0]
    for x in lst[1:]:
        if x > result:
            result = x
    return result
""",
    public_tests="""\
from solution import max_list
import pytest


def test_basic():
    assert max_list([3, 1, 4, 1, 5]) == 5


def test_single():
    assert max_list([42]) == 42


def test_negative():
    assert max_list([-5, -2, -9]) == -2


def test_empty():
    with pytest.raises(ValueError):
        max_list([])
""",
    hidden_tests="""\
from solution import max_list


def test_all_same():
    assert max_list([7, 7, 7]) == 7


def test_sorted():
    assert max_list([1, 2, 3, 4, 5]) == 5


def test_reverse_sorted():
    assert max_list([5, 4, 3, 2, 1]) == 5
""",
)

FAMILY_L1_MIN_LIST = TaskFamily(
    family_id="fam_l1_min_list",
    difficulty=1,
    description="Find min element in a list",
    skill_tags=["loops", "list", "comparison"],
    instruction="实现函数 `min_list(lst)`，返回列表中的最小元素；列表为空时抛出 ValueError。",
    target_code="""\
def min_list(lst):
    if not lst:
        raise ValueError("Empty list")
    result = lst[0]
    for x in lst[1:]:
        if x < result:
            result = x
    return result
""",
    public_tests="""\
from solution import min_list
import pytest


def test_basic():
    assert min_list([3, 1, 4, 1, 5]) == 1


def test_single():
    assert min_list([42]) == 42


def test_negative():
    assert min_list([-5, -2, -9]) == -9


def test_empty():
    with pytest.raises(ValueError):
        min_list([])
""",
    hidden_tests="""\
from solution import min_list


def test_all_same():
    assert min_list([7, 7, 7]) == 7


def test_sorted():
    assert min_list([1, 2, 3, 4, 5]) == 1


def test_reverse_sorted():
    assert min_list([5, 4, 3, 2, 1]) == 1
""",
)

FAMILY_L1_COUNT_CHAR = TaskFamily(
    family_id="fam_l1_count_char",
    difficulty=1,
    description="Count occurrences of a character in a string",
    skill_tags=["string", "loops", "counting"],
    instruction="实现函数 `count_char(s, c)`，统计字符 c 在字符串 s 中出现的次数。",
    target_code="""\
def count_char(s, c):
    count = 0
    for ch in s:
        if ch == c:
            count += 1
    return count
""",
    public_tests="""\
from solution import count_char


def test_basic():
    assert count_char("hello", "l") == 2


def test_not_found():
    assert count_char("hello", "z") == 0


def test_empty():
    assert count_char("", "a") == 0


def test_all_match():
    assert count_char("aaa", "a") == 3
""",
    hidden_tests="""\
from solution import count_char


def test_case_sensitive():
    assert count_char("Hello", "h") == 0


def test_space():
    assert count_char("a b c", " ") == 2
""",
)

FAMILY_L1_REMOVE_NEGATIVES = TaskFamily(
    family_id="fam_l1_remove_negatives",
    difficulty=1,
    description="Remove negative numbers from a list",
    skill_tags=["loops", "list", "filtering"],
    instruction="实现函数 `remove_negatives(lst)`，返回一个新列表，移除原列表中所有负数元素。",
    target_code="""\
def remove_negatives(lst):
    return [x for x in lst if x >= 0]
""",
    public_tests="""\
from solution import remove_negatives


def test_basic():
    assert remove_negatives([1, -2, 3, -4, 5]) == [1, 3, 5]


def test_no_negatives():
    assert remove_negatives([1, 2, 3]) == [1, 2, 3]


def test_all_negatives():
    assert remove_negatives([-1, -2, -3]) == []


def test_empty():
    assert remove_negatives([]) == []
""",
    hidden_tests="""\
from solution import remove_negatives


def test_with_zero():
    assert remove_negatives([-1, 0, 1]) == [0, 1]


def test_duplicates():
    assert remove_negatives([3, -1, 3, -1]) == [3, 3]
""",
)

FAMILY_L1_FLATTEN_ONCE = TaskFamily(
    family_id="fam_l1_flatten_once",
    difficulty=1,
    description="Flatten a list of lists one level",
    skill_tags=["loops", "list", "nested"],
    instruction="实现函数 `flatten_once(lst)`，将二维列表（列表的列表）展平为一维列表。",
    target_code="""\
def flatten_once(lst):
    result = []
    for sub in lst:
        for item in sub:
            result.append(item)
    return result
""",
    public_tests="""\
from solution import flatten_once


def test_basic():
    assert flatten_once([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]


def test_empty_outer():
    assert flatten_once([]) == []


def test_empty_inner():
    assert flatten_once([[], [1], []]) == [1]


def test_single():
    assert flatten_once([[1, 2, 3]]) == [1, 2, 3]
""",
    hidden_tests="""\
from solution import flatten_once


def test_strings():
    assert flatten_once([["a", "b"], ["c"]]) == ["a", "b", "c"]


def test_mixed_lengths():
    assert flatten_once([[1], [2, 3, 4], [5, 6]]) == [1, 2, 3, 4, 5, 6]
""",
)

FAMILY_L1_ZIP_SUM = TaskFamily(
    family_id="fam_l1_zip_sum",
    difficulty=1,
    description="Sum corresponding elements of two lists",
    skill_tags=["loops", "list", "accumulation"],
    instruction="实现函数 `zip_sum(a, b)`，返回一个新列表，其中每个元素是两个列表对应位置元素之和。两个列表长度保证相同。",
    target_code="""\
def zip_sum(a, b):
    return [x + y for x, y in zip(a, b)]
""",
    public_tests="""\
from solution import zip_sum


def test_basic():
    assert zip_sum([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def test_empty():
    assert zip_sum([], []) == []


def test_negatives():
    assert zip_sum([-1, 0, 1], [1, 0, -1]) == [0, 0, 0]


def test_single():
    assert zip_sum([10], [20]) == [30]
""",
    hidden_tests="""\
from solution import zip_sum


def test_large_values():
    assert zip_sum([100, 200], [300, 400]) == [400, 600]


def test_zeros():
    assert zip_sum([0, 0, 0], [0, 0, 0]) == [0, 0, 0]
""",
)

FAMILY_L1_RUNNING_SUM = TaskFamily(
    family_id="fam_l1_running_sum",
    difficulty=1,
    description="Compute running/prefix sum",
    skill_tags=["loops", "list", "accumulation"],
    instruction="实现函数 `running_sum(lst)`，返回列表的前缀和（累加和）列表。例如输入 [1, 2, 3] 返回 [1, 3, 6]。",
    target_code="""\
def running_sum(lst):
    result = []
    total = 0
    for x in lst:
        total += x
        result.append(total)
    return result
""",
    public_tests="""\
from solution import running_sum


def test_basic():
    assert running_sum([1, 2, 3]) == [1, 3, 6]


def test_empty():
    assert running_sum([]) == []


def test_single():
    assert running_sum([5]) == [5]


def test_negatives():
    assert running_sum([-1, 2, -3]) == [-1, 1, -2]
""",
    hidden_tests="""\
from solution import running_sum


def test_all_zeros():
    assert running_sum([0, 0, 0]) == [0, 0, 0]


def test_ascending():
    assert running_sum([1, 1, 1, 1]) == [1, 2, 3, 4]
""",
)

FAMILY_L1_FIRST_NON_REPEATING = TaskFamily(
    family_id="fam_l1_first_non_repeating",
    difficulty=1,
    description="First non-repeating character in a string",
    skill_tags=["string", "loops", "counting", "dict"],
    instruction=(
        "实现函数 `first_non_repeating(s)`，返回字符串 s 中第一个只出现一次的字符。"
        "若所有字符都重复则返回 None。"
    ),
    target_code="""\
def first_non_repeating(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in s:
        if counts[ch] == 1:
            return ch
    return None
""",
    public_tests="""\
from solution import first_non_repeating


def test_basic():
    assert first_non_repeating("aabbc") == "c"


def test_first_char():
    assert first_non_repeating("abcabc") is None


def test_empty():
    assert first_non_repeating("") is None


def test_single():
    assert first_non_repeating("x") == "x"


def test_all_same():
    assert first_non_repeating("aaaa") is None
""",
    hidden_tests="""\
from solution import first_non_repeating


def test_middle():
    assert first_non_repeating("aabbcd") == "c"


def test_space():
    assert first_non_repeating("a a b") == "b"
""",
)

# ---- Additional L2 families ------------------------------------------------

FAMILY_L2_BINARY_SEARCH = TaskFamily(
    family_id="fam_l2_binary_search",
    difficulty=2,
    description="Binary search in sorted array, return index or -1",
    skill_tags=["binary_search", "loops", "list"],
    instruction=(
        "实现函数 `binary_search(lst, target)`，在已排序列表 lst 中查找 target。"
        "找到则返回其索引，否则返回 -1。"
    ),
    target_code="""\
def binary_search(lst, target):
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
    public_tests="""\
from solution import binary_search


def test_found():
    assert binary_search([1, 3, 5, 7, 9], 5) == 2


def test_not_found():
    assert binary_search([1, 3, 5, 7, 9], 4) == -1


def test_first():
    assert binary_search([1, 3, 5, 7, 9], 1) == 0


def test_last():
    assert binary_search([1, 3, 5, 7, 9], 9) == 4


def test_empty():
    assert binary_search([], 1) == -1
""",
    hidden_tests="""\
from solution import binary_search


def test_single_found():
    assert binary_search([5], 5) == 0


def test_single_not_found():
    assert binary_search([5], 3) == -1


def test_two_elements():
    assert binary_search([1, 3], 3) == 1
""",
)

FAMILY_L2_TWO_SUM = TaskFamily(
    family_id="fam_l2_two_sum",
    difficulty=2,
    description="Find two indices that sum to target",
    skill_tags=["dict", "list", "two_sum"],
    instruction=(
        "实现函数 `two_sum(lst, target)`，返回列表中两个元素的下标，"
        "使得它们的和等于 target。保证有且仅有一组解，返回 [i, j]（i < j）。"
    ),
    target_code="""\
def two_sum(lst, target):
    seen = {}
    for i, x in enumerate(lst):
        complement = target - x
        if complement in seen:
            return [seen[complement], i]
        seen[x] = i
    return []
""",
    public_tests="""\
from solution import two_sum


def test_basic():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]


def test_middle():
    assert two_sum([3, 2, 4], 6) == [1, 2]


def test_same_values():
    assert two_sum([3, 3], 6) == [0, 1]


def test_negative():
    assert two_sum([-1, 2, 3], 2) == [0, 1]
""",
    hidden_tests="""\
from solution import two_sum


def test_later_pair():
    assert two_sum([1, 5, 3, 8, 2], 11) == [2, 3]


def test_large_target():
    assert two_sum([100, 200, 300], 500) == [1, 2]
""",
)

FAMILY_L2_GROUP_BY_PARITY = TaskFamily(
    family_id="fam_l2_group_by_parity",
    difficulty=2,
    description="Group numbers into even/odd lists",
    skill_tags=["loops", "list", "filtering"],
    instruction=(
        "实现函数 `group_by_parity(lst)`，将列表中的数字按奇偶分组。"
        "返回 [even_list, odd_list]，各自保持原始顺序。"
    ),
    target_code="""\
def group_by_parity(lst):
    evens = []
    odds = []
    for x in lst:
        if x % 2 == 0:
            evens.append(x)
        else:
            odds.append(x)
    return [evens, odds]
""",
    public_tests="""\
from solution import group_by_parity


def test_basic():
    assert group_by_parity([1, 2, 3, 4, 5]) == [[2, 4], [1, 3, 5]]


def test_empty():
    assert group_by_parity([]) == [[], []]


def test_all_even():
    assert group_by_parity([2, 4, 6]) == [[2, 4, 6], []]


def test_all_odd():
    assert group_by_parity([1, 3, 5]) == [[], [1, 3, 5]]
""",
    hidden_tests="""\
from solution import group_by_parity


def test_with_zero():
    assert group_by_parity([0, 1, 2]) == [[0, 2], [1]]


def test_negatives():
    assert group_by_parity([-3, -2, -1, 0]) == [[-2, 0], [-3, -1]]
""",
)

FAMILY_L2_REMOVE_DUPLICATES_ORDERED = TaskFamily(
    family_id="fam_l2_remove_duplicates_ordered",
    difficulty=2,
    description="Remove duplicates preserving first-occurrence order",
    skill_tags=["list", "set", "deduplication"],
    instruction=(
        "实现函数 `remove_duplicates_ordered(lst)`，移除列表中的重复元素，"
        "保留每个元素第一次出现的顺序。"
    ),
    target_code="""\
def remove_duplicates_ordered(lst):
    seen = set()
    result = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
""",
    public_tests="""\
from solution import remove_duplicates_ordered


def test_basic():
    assert remove_duplicates_ordered([1, 2, 3, 2, 1]) == [1, 2, 3]


def test_no_dups():
    assert remove_duplicates_ordered([1, 2, 3]) == [1, 2, 3]


def test_empty():
    assert remove_duplicates_ordered([]) == []


def test_all_same():
    assert remove_duplicates_ordered([5, 5, 5]) == [5]
""",
    hidden_tests="""\
from solution import remove_duplicates_ordered


def test_strings():
    assert remove_duplicates_ordered(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_preserves_order():
    assert remove_duplicates_ordered([3, 1, 2, 1, 3]) == [3, 1, 2]
""",
)

FAMILY_L2_ROTATE_LIST = TaskFamily(
    family_id="fam_l2_rotate_list",
    difficulty=2,
    description="Rotate list by k positions",
    skill_tags=["list", "slicing", "math"],
    instruction=(
        "实现函数 `rotate_list(lst, k)`，将列表向右旋转 k 个位置。"
        "例如 [1, 2, 3, 4, 5] 旋转 2 位得到 [4, 5, 1, 2, 3]。k 可能大于列表长度。"
    ),
    target_code="""\
def rotate_list(lst, k):
    if not lst:
        return []
    k = k % len(lst)
    return lst[-k:] + lst[:-k] if k else list(lst)
""",
    public_tests="""\
from solution import rotate_list


def test_basic():
    assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]


def test_zero():
    assert rotate_list([1, 2, 3], 0) == [1, 2, 3]


def test_full():
    assert rotate_list([1, 2, 3], 3) == [1, 2, 3]


def test_empty():
    assert rotate_list([], 5) == []


def test_one():
    assert rotate_list([1, 2, 3], 1) == [3, 1, 2]
""",
    hidden_tests="""\
from solution import rotate_list


def test_large_k():
    assert rotate_list([1, 2, 3], 7) == [3, 1, 2]


def test_single_element():
    assert rotate_list([1], 100) == [1]
""",
)

# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------

TASK_FAMILIES: list[TaskFamily] = [
    FAMILY_L0_ADD_TWO,
    FAMILY_L0_IS_POSITIVE,
    FAMILY_L0_CLAMP,
    FAMILY_L0_ABS_VALUE,
    FAMILY_L0_MAX_OF_TWO,
    FAMILY_L0_MIN_OF_TWO,
    FAMILY_L0_IS_EVEN,
    FAMILY_L0_MULTIPLY,
    FAMILY_L1_SUM_LIST,
    FAMILY_L1_REVERSE_STRING,
    FAMILY_L1_COUNT_VOWELS,
    FAMILY_L1_IS_PALINDROME,
    FAMILY_L1_FACTORIAL,
    FAMILY_L1_MAX_LIST,
    FAMILY_L1_MIN_LIST,
    FAMILY_L1_COUNT_CHAR,
    FAMILY_L1_REMOVE_NEGATIVES,
    FAMILY_L1_FLATTEN_ONCE,
    FAMILY_L1_ZIP_SUM,
    FAMILY_L1_RUNNING_SUM,
    FAMILY_L1_FIRST_NON_REPEATING,
    FAMILY_L2_SECOND_LARGEST,
    FAMILY_L2_BALANCED_PARENS,
    FAMILY_L2_MERGE_SORTED,
    FAMILY_L2_FIND_DUPLICATES,
    FAMILY_L2_BINARY_SEARCH,
    FAMILY_L2_TWO_SUM,
    FAMILY_L2_GROUP_BY_PARITY,
    FAMILY_L2_REMOVE_DUPLICATES_ORDERED,
    FAMILY_L2_ROTATE_LIST,
]


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def family_to_sample(
    family: TaskFamily,
    *,
    created_at: Optional[str] = None,
) -> Sample:
    """Convert a TaskFamily into an unverified code_generation Sample."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    return Sample(
        sample_id=f"gen_{family.family_id}_v1",
        family_id=family.family_id,
        difficulty=family.difficulty,
        task_type="code_generation",
        language="python",
        skill_tags=list(family.skill_tags),
        instruction=family.instruction,
        broken_code=None,
        execution_feedback=None,
        target_code=family.target_code,
        public_tests=family.public_tests,
        hidden_tests=family.hidden_tests,
        verified=False,
        verification=_PLACEHOLDER_VER,
        generator=_GENERATOR,
        created_at=created_at,
        dataset_version=_DATASET_VERSION,
    )


def generate_samples(
    families: list[TaskFamily],
    *,
    seed: int = 42,
    levels: Optional[list[int]] = None,
    count: Optional[int] = None,
) -> list[Sample]:
    """Generate Sample objects from the given task families.

    Parameters
    ----------
    families:
        Pool of TaskFamily objects to draw from.
    seed:
        RNG seed for determinism when ``count`` limits output.
    levels:
        If provided, restrict to families whose ``difficulty`` is in this list.
    count:
        Maximum number of samples to return.  ``None`` returns all matching.

    Returns
    -------
    list[Sample]
        Unverified code_generation Samples, one per selected family.
    """
    rng = random.Random(seed)
    created_at = datetime.now(timezone.utc).isoformat()

    selected = families
    if levels is not None:
        level_set = set(levels)
        selected = [f for f in families if f.difficulty in level_set]

    samples = [family_to_sample(f, created_at=created_at) for f in selected]

    if count is not None and count < len(samples):
        rng.shuffle(samples)
        samples = samples[:count]

    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate procedural code-generation task samples.",
    )
    p.add_argument("--count", type=int, default=None,
                   help="Max samples to emit (default: all families).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (used when --count limits output).")
    p.add_argument("--out", default="data/generated/tasks.jsonl",
                   help="Output JSONL path.")
    p.add_argument("--levels", type=int, nargs="+", default=None,
                   help="Difficulty levels to include, e.g. --levels 0 1 2.")
    return p


def main() -> int:
    """CLI entry point.  Returns exit code."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(
        TASK_FAMILIES,
        seed=args.seed,
        levels=args.levels,
        count=args.count,
    )

    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")

    print(f"generate_tasks: wrote {len(samples)} samples -> {out_path}")
    level_counts: dict[int, int] = {}
    for s in samples:
        level_counts[s.difficulty] = level_counts.get(s.difficulty, 0) + 1
    for lvl in sorted(level_counts):
        print(f"  L{lvl}: {level_counts[lvl]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
