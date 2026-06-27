"""
tests/test_mutate_code.py -- Tests for scripts/mutate_code.py.

Coverage
--------
- MUTATORS registry: at least 5 operators registered.
- flip_comparison: changes a simple comparison expression.
- wrong_arithmetic: changes + to - in AugAssign and BinOp.
- off_by_one: changes range stop arg.
- wrong_sort_dir: adds/flips reverse= in sorted().
- flip_slice_step: changes [::-1] to [::1].
- remove_first_extend: removes a .extend() call.
- Mutation filter: mutated variants that pass all tests are discarded.
- Key invariant: every returned repair sample's broken_code genuinely fails.
- execution_repair samples carry non-empty execution_feedback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.generate_tasks import (  # noqa: E402
    FAMILY_L0_ADD_TWO,
    FAMILY_L0_CLAMP,
    FAMILY_L0_IS_POSITIVE,
    FAMILY_L1_COUNT_VOWELS,
    FAMILY_L1_FACTORIAL,
    FAMILY_L1_IS_PALINDROME,
    FAMILY_L1_REVERSE_STRING,
    FAMILY_L1_SUM_LIST,
    FAMILY_L2_MERGE_SORTED,
    FAMILY_L2_SECOND_LARGEST,
    family_to_sample,
)
from scripts.mutate_code import (  # noqa: E402
    MUTATORS,
    apply_all_mutators,
    generate_repair_samples,
    mutate_and_get_feedback,
    _op_drop_guard,
    _op_flip_comparison,
    _op_wrong_arithmetic,
    _op_off_by_one_minus1,
    _op_off_by_one_plus1,
    _op_typo_identifier,
    _op_wrong_sort_dir,
    _op_flip_slice_step,
    _op_remove_first_extend,
)
from src.validators import verify_broken_is_broken, verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_mutators_registry_count():
    """At least 5 mutation operators are registered."""
    assert len(MUTATORS) >= 5


def test_all_expected_operators_present():
    """The 11 designed operators are in the registry (incl. typo + drop_guard)."""
    expected = {
        "typo_identifier",
        "flip_comparison",
        "wrong_arithmetic",
        "off_by_one_minus1",
        "off_by_one_plus1",
        "wrong_sort_dir",
        "wrong_index_plus1",
        "flip_bool_return",
        "drop_guard",
        "remove_first_extend",
        "flip_slice_step",
    }
    assert expected.issubset(set(MUTATORS.keys()))


def test_spec_categories_covered():
    """All spec §11.2 operator categories are represented in the registry."""
    keys = set(MUTATORS.keys())
    # typo
    assert "typo_identifier" in keys
    # off-by-one
    assert "off_by_one_minus1" in keys or "off_by_one_plus1" in keys
    # wrong comparison operator
    assert "flip_comparison" in keys
    # drop empty-input / boundary guard
    assert "drop_guard" in keys
    # wrong sort direction
    assert "wrong_sort_dir" in keys
    # wrong return structure
    assert "flip_bool_return" in keys
    # wrong index
    assert "wrong_index_plus1" in keys


# ---------------------------------------------------------------------------
# Individual operator unit tests
# ---------------------------------------------------------------------------


# --- typo_identifier ---------------------------------------------------------


def test_typo_identifier_renames_first_load_name():
    """typo_identifier renames the first Load-context identifier to a misspelling."""
    code = "def f(a, b):\n    return a + b\n"
    mutated = _op_typo_identifier(code)
    assert mutated is not None
    assert "a_typo" in mutated
    # function signature parameters are untouched (def still imports cleanly)
    assert mutated.startswith("def f(a, b):")


def test_typo_identifier_still_compiles():
    """The typo variant is syntactically valid (compiles)."""
    from src.validators import compile_check
    code = "def f(a, b):\n    return a + b\n"
    mutated = _op_typo_identifier(code)
    assert mutated is not None
    ok, _ = compile_check(mutated)
    assert ok, "typo variant must compile (failure is at runtime, not parse time)"


def test_typo_identifier_none_when_no_load_name():
    """typo_identifier returns None for a body with no Load-context names."""
    code = "def f():\n    return 1\n"
    mutated = _op_typo_identifier(code)
    assert mutated is None


def test_typo_identifier_genuinely_broken():
    """A typo variant fails the tests (NameError at runtime) for add_two."""
    sample = family_to_sample(FAMILY_L0_ADD_TWO)
    mutated = _op_typo_identifier(sample.target_code)
    assert mutated is not None
    sample.broken_code = mutated
    assert verify_broken_is_broken(sample) is True


# --- drop_guard --------------------------------------------------------------


def test_drop_guard_removes_factorial_guard():
    """drop_guard removes the `if n < 0: raise ...` guard from factorial."""
    sample = family_to_sample(FAMILY_L1_FACTORIAL)
    mutated = _op_drop_guard(sample.target_code)
    assert mutated is not None
    assert "raise ValueError" not in mutated
    assert "if n < 0" not in mutated


def test_drop_guard_removes_clamp_guard():
    """drop_guard removes the first `if x < lo: return lo` guard from clamp."""
    sample = family_to_sample(FAMILY_L0_CLAMP)
    mutated = _op_drop_guard(sample.target_code)
    assert mutated is not None
    assert mutated.strip() != sample.target_code.strip()


def test_drop_guard_none_when_no_guard():
    """drop_guard returns None for a function with no leading guard."""
    code = "def f(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total\n"
    mutated = _op_drop_guard(code)
    assert mutated is None


def test_drop_guard_still_compiles():
    """The drop_guard variant is syntactically valid."""
    from src.validators import compile_check
    sample = family_to_sample(FAMILY_L2_SECOND_LARGEST)
    mutated = _op_drop_guard(sample.target_code)
    assert mutated is not None
    ok, _ = compile_check(mutated)
    assert ok


def test_drop_guard_genuinely_broken_factorial():
    """Removing factorial's guard makes the negative-input test fail."""
    sample = family_to_sample(FAMILY_L1_FACTORIAL)
    mutated = _op_drop_guard(sample.target_code)
    assert mutated is not None
    sample.broken_code = mutated
    assert verify_broken_is_broken(sample) is True


def test_drop_guard_genuinely_broken_second_largest():
    """Removing second_largest's guard makes the <2-distinct test fail."""
    sample = family_to_sample(FAMILY_L2_SECOND_LARGEST)
    mutated = _op_drop_guard(sample.target_code)
    assert mutated is not None
    sample.broken_code = mutated
    assert verify_broken_is_broken(sample) is True


def test_flip_comparison_gt_becomes_lt():
    """flip_comparison changes > to < in a simple function."""
    code = "def f(n):\n    return n > 0\n"
    mutated = _op_flip_comparison(code)
    assert mutated is not None
    assert "<" in mutated
    assert ">" not in mutated or ">=" not in mutated


def test_flip_comparison_eq_becomes_neq():
    """flip_comparison changes == to != ."""
    code = "def f(a, b):\n    return a == b\n"
    mutated = _op_flip_comparison(code)
    assert mutated is not None
    assert "!=" in mutated


def test_flip_comparison_in_becomes_not_in():
    """flip_comparison changes 'in' to 'not in'."""
    code = "def f(x, s):\n    return x in s\n"
    mutated = _op_flip_comparison(code)
    assert mutated is not None
    assert "not in" in mutated


def test_flip_comparison_none_when_no_comparison():
    """flip_comparison returns None for code with no comparison operators."""
    code = "def f(a, b):\n    return a + b\n"
    mutated = _op_flip_comparison(code)
    assert mutated is None


def test_wrong_arithmetic_plus_to_minus_in_augassign():
    """wrong_arithmetic changes += to -= in sum_list."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    mutated = _op_wrong_arithmetic(sample.target_code)
    assert mutated is not None
    assert "-=" in mutated


def test_wrong_arithmetic_changes_factorial():
    """wrong_arithmetic changes some arithmetic in factorial (DFS hits range bound first)."""
    sample = family_to_sample(FAMILY_L1_FACTORIAL)
    mutated = _op_wrong_arithmetic(sample.target_code)
    assert mutated is not None
    # DFS visits n+1 inside range() before the *= AugAssign, so the range
    # bound changes (n+1 → n-1).  Either way the code compiles and breaks tests.
    assert mutated.strip() != sample.target_code.strip()
    # The mutated code should still compile
    from src.validators import compile_check
    ok, _ = compile_check(mutated)
    assert ok, "Mutated factorial must still compile"


def test_off_by_one_minus1_changes_range():
    """off_by_one_minus1 adjusts the range stop argument downward."""
    code = "def f(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n"
    mutated = _op_off_by_one_minus1(code)
    assert mutated is not None
    # range stop should be reduced
    assert mutated != code


def test_off_by_one_plus1_changes_range():
    """off_by_one_plus1 adjusts the range stop argument upward."""
    code = "def f(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n"
    mutated = _op_off_by_one_plus1(code)
    assert mutated is not None
    assert mutated != code


def test_wrong_sort_dir_adds_reverse_true():
    """wrong_sort_dir adds reverse=True to sorted() without it."""
    code = "def f(lst):\n    return sorted(lst)\n"
    mutated = _op_wrong_sort_dir(code)
    assert mutated is not None
    assert "reverse=True" in mutated


def test_wrong_sort_dir_flips_existing_reverse():
    """wrong_sort_dir flips reverse=True to reverse=False."""
    code = "def f(lst):\n    return sorted(lst, reverse=True)\n"
    mutated = _op_wrong_sort_dir(code)
    assert mutated is not None
    assert "reverse=False" in mutated


def test_flip_slice_step_changes_step():
    """flip_slice_step turns [::-1] into [::1]."""
    code = "def f(s):\n    return s[::-1]\n"
    mutated = _op_flip_slice_step(code)
    assert mutated is not None
    assert "::1" in mutated


def test_remove_first_extend_removes_it():
    """remove_first_extend drops the first .extend() call."""
    code = (
        "def f(a, b):\n"
        "    result = []\n"
        "    result.extend(a)\n"
        "    result.extend(b)\n"
        "    return result\n"
    )
    mutated = _op_remove_first_extend(code)
    assert mutated is not None
    # One extend should remain
    assert mutated.count(".extend(") == 1


# ---------------------------------------------------------------------------
# apply_all_mutators
# ---------------------------------------------------------------------------


def test_apply_all_mutators_returns_dict():
    """apply_all_mutators returns a dict of applicable operators."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    candidates = apply_all_mutators(sample.target_code)
    assert isinstance(candidates, dict)
    assert len(candidates) >= 1  # at least wrong_arithmetic applies


def test_apply_all_mutators_none_for_empty():
    """No operator fires on `def noop(): pass` (no mutable targets at all)."""
    code = "def noop(): pass\n"
    candidates = apply_all_mutators(code)
    # An empty-body function has no Load names, comparisons, arithmetic, ranges,
    # sorts, subscripts, bool returns, guards, extends or slices to mutate, so
    # the returned dict must be empty.
    assert candidates == {}


# ---------------------------------------------------------------------------
# mutate_and_get_feedback
# ---------------------------------------------------------------------------


def test_mutate_and_get_feedback_wrong_sum():
    """Mutated sum_list actually fails and produces feedback."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    broken_code = _op_wrong_arithmetic(sample.target_code)
    assert broken_code is not None
    is_broken, feedback = mutate_and_get_feedback(broken_code, sample)
    assert is_broken, "sum_list with -= should fail tests"
    assert feedback.strip(), "Feedback should be non-empty"


def test_mutate_and_get_feedback_passing_code():
    """Correct code is not classified as broken."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    is_broken, feedback = mutate_and_get_feedback(sample.target_code, sample)
    assert not is_broken
    assert feedback == ""


# ---------------------------------------------------------------------------
# _format_feedback truncation (M2) and per_sample_seed (M3)
# ---------------------------------------------------------------------------


def test_format_feedback_truncates_long_output():
    """_format_feedback caps assembled text at MAX_FEEDBACK_CHARS with a marker."""
    from scripts.mutate_code import _format_feedback, MAX_FEEDBACK_CHARS

    huge = "x" * (MAX_FEEDBACK_CHARS * 3)
    out = _format_feedback(huge, "", 1)
    assert len(out) <= MAX_FEEDBACK_CHARS + len("...[truncated]")
    assert out.endswith("...[truncated]")


def test_format_feedback_short_output_not_truncated():
    """Short feedback is returned verbatim, no marker."""
    from scripts.mutate_code import _format_feedback

    out = _format_feedback("boom", "", 1)
    assert out == "boom"


def test_per_sample_seed_is_stable_and_varies():
    """per_sample_seed is deterministic per id and differs across ids."""
    from scripts.mutate_code import per_sample_seed

    s1 = per_sample_seed(42, "gen_fam_a_v1")
    s1_again = per_sample_seed(42, "gen_fam_a_v1")
    s2 = per_sample_seed(42, "gen_fam_b_v1")
    assert s1 == s1_again          # stable for the same id
    assert s1 != s2                # differs across ids (diversifies shuffles)


def test_per_sample_seed_not_builtin_hash():
    """per_sample_seed uses a stable hash (reproducible across processes)."""
    import zlib
    from scripts.mutate_code import per_sample_seed

    sid = "gen_fam_l1_sum_list_v1"
    expected = 42 ^ (zlib.crc32(sid.encode("utf-8")) & 0xFFFF)
    assert per_sample_seed(42, sid) == expected


# ---------------------------------------------------------------------------
# generate_repair_samples: full integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [
    FAMILY_L0_ADD_TWO,
    FAMILY_L0_IS_POSITIVE,
    FAMILY_L1_SUM_LIST,
    FAMILY_L1_COUNT_VOWELS,
])
def test_generate_repair_samples_produces_broken_variants(family):
    """generate_repair_samples returns at least 1 pair for common families."""
    sample = family_to_sample(family)
    # Verify reference first
    ref_result = verify_sample(sample, run_ruff=False)
    assert ref_result.is_accepted, f"Reference for {family.family_id} must be valid"

    pairs = generate_repair_samples(sample, max_per_sample=5)
    assert len(pairs) >= 1, (
        f"Expected at least 1 repair pair for {family.family_id}, got 0"
    )


def test_all_static_repairs_are_genuinely_broken():
    """Every static_repair sample's broken_code must fail >= 1 test."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    pairs = generate_repair_samples(sample, max_per_sample=5)
    for sr, er in pairs:
        if sr is not None:
            is_broken = verify_broken_is_broken(sr)
            assert is_broken, (
                f"static_repair {sr.sample_id} should fail tests but passed all"
            )


def test_execution_repair_samples_have_feedback():
    """All execution_repair samples carry non-empty execution_feedback."""
    sample = family_to_sample(FAMILY_L1_FACTORIAL)
    pairs = generate_repair_samples(sample, max_per_sample=5)
    exec_repairs = [er for _, er in pairs if er is not None]
    if exec_repairs:
        for er in exec_repairs:
            assert er.execution_feedback and er.execution_feedback.strip(), (
                f"execution_repair {er.sample_id} has empty feedback"
            )


def test_static_repair_difficulty_matches_original():
    """static_repair samples inherit the difficulty of the original."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    assert sample.difficulty == 1
    pairs = generate_repair_samples(sample, max_per_sample=3)
    for sr, _ in pairs:
        if sr is not None:
            assert sr.difficulty == 1  # inherits L1


def test_execution_repair_difficulty_is_3():
    """execution_repair samples always have difficulty=3."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    pairs = generate_repair_samples(sample, max_per_sample=3)
    for _, er in pairs:
        if er is not None:
            assert er.difficulty == 3


def test_max_per_sample_respected():
    """generate_repair_samples returns at most max_per_sample pairs."""
    sample = family_to_sample(FAMILY_L2_SECOND_LARGEST)
    pairs = generate_repair_samples(sample, max_per_sample=2)
    assert len(pairs) <= 2


def test_repair_sample_ids_unique():
    """Static and execution repair sample IDs are unique."""
    sample = family_to_sample(FAMILY_L2_MERGE_SORTED)
    pairs = generate_repair_samples(sample, max_per_sample=5)
    ids = []
    for sr, er in pairs:
        if sr is not None:
            ids.append(sr.sample_id)
        if er is not None:
            ids.append(er.sample_id)
    assert len(ids) == len(set(ids))


def test_passing_mutations_excluded():
    """Mutations that pass all tests never appear in the output."""
    # is_positive's reference passes all tests.
    # Verify that generate_repair_samples returns nothing OR only genuinely broken ones.
    sample = family_to_sample(FAMILY_L0_IS_POSITIVE)
    pairs = generate_repair_samples(sample, max_per_sample=10)
    for sr, er in pairs:
        if sr is not None:
            assert verify_broken_is_broken(sr), (
                "A mutation that passes all tests slipped through the filter"
            )
