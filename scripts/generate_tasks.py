"""
scripts/generate_tasks.py -- Procedural task bank for the Qwen3-0.6B Code Recovery Lab.

Generates Sample objects (task_type=code_generation) from 12 hand-authored task
families spanning difficulty levels 0-2:
  - L0 (3 families): simple two-liner functions good for syntax-level mutations.
  - L1 (5 families): single-function implementations with loops / strings.
  - L2 (4 families): boundary conditions and basic data structures.

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


def test_ten():
    assert factorial(10) == 3628800


def test_two():
    assert factorial(2) == 2
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


def test_sorted_input():
    assert second_largest([1, 2, 3, 4, 5]) == 4


def test_duplicates_of_max():
    assert second_largest([5, 5, 3, 1]) == 3
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

# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------

TASK_FAMILIES: list[TaskFamily] = [
    FAMILY_L0_ADD_TWO,
    FAMILY_L0_IS_POSITIVE,
    FAMILY_L0_CLAMP,
    FAMILY_L1_SUM_LIST,
    FAMILY_L1_REVERSE_STRING,
    FAMILY_L1_COUNT_VOWELS,
    FAMILY_L1_IS_PALINDROME,
    FAMILY_L1_FACTORIAL,
    FAMILY_L2_SECOND_LARGEST,
    FAMILY_L2_BALANCED_PARENS,
    FAMILY_L2_MERGE_SORTED,
    FAMILY_L2_FIND_DUPLICATES,
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
