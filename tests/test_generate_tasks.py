"""
tests/test_generate_tasks.py -- Tests for scripts/generate_tasks.py.

Coverage
--------
- TASK_FAMILIES registry has expected size and difficulty distribution.
- generate_samples: unique IDs, level filtering, count limiting, valid schemas.
- family_to_sample: produces a valid Sample with correct fields.
- Reference self-verification: a subset of families pass their own tests.
  (This is the key quality gate: broken references must be caught.)
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
    FAMILY_L2_BALANCED_PARENS,
    FAMILY_L2_FIND_DUPLICATES,
    FAMILY_L2_MERGE_SORTED,
    FAMILY_L2_SECOND_LARGEST,
    TASK_FAMILIES,
    TaskFamily,
    family_to_sample,
    generate_samples,
)
from src.schemas import Sample  # noqa: E402
from src.validators import verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_family_count_at_least_eight():
    """We define at least 8 task families (spec calls for 8-15)."""
    assert len(TASK_FAMILIES) >= 8


def test_all_twelve_families_present():
    """All designed families are registered (at least 12, expanded in P2)."""
    assert len(TASK_FAMILIES) >= 12


def test_family_ids_unique():
    """Every family_id is distinct."""
    ids = [f.family_id for f in TASK_FAMILIES]
    assert len(ids) == len(set(ids))


def test_difficulty_distribution():
    """We have L0, L1, and L2 families."""
    by_level: dict[int, int] = {}
    for f in TASK_FAMILIES:
        by_level[f.difficulty] = by_level.get(f.difficulty, 0) + 1
    assert by_level.get(0, 0) >= 3, "Need at least 3 L0 families"
    assert by_level.get(1, 0) >= 4, "Need at least 4 L1 families"
    assert by_level.get(2, 0) >= 3, "Need at least 3 L2 families"


# ---------------------------------------------------------------------------
# generate_samples tests
# ---------------------------------------------------------------------------


def test_generate_all_unique_ids():
    """Generated sample IDs are globally unique."""
    samples = generate_samples(TASK_FAMILIES)
    ids = [s.sample_id for s in samples]
    assert len(ids) == len(set(ids))


def test_generate_all_returns_one_per_family():
    """Default call returns exactly one sample per family."""
    samples = generate_samples(TASK_FAMILIES)
    assert len(samples) == len(TASK_FAMILIES)


def test_generate_filter_by_level_zero():
    """--levels 0 returns only L0 samples."""
    samples = generate_samples(TASK_FAMILIES, levels=[0])
    assert all(s.difficulty == 0 for s in samples)
    expected_l0 = sum(1 for f in TASK_FAMILIES if f.difficulty == 0)
    assert len(samples) == expected_l0


def test_generate_filter_by_levels_01():
    """--levels 0 1 returns L0 and L1 samples only."""
    samples = generate_samples(TASK_FAMILIES, levels=[0, 1])
    assert all(s.difficulty in (0, 1) for s in samples)
    expected_01 = sum(1 for f in TASK_FAMILIES if f.difficulty in (0, 1))
    assert len(samples) == expected_01


def test_generate_count_limit():
    """--count N returns exactly N samples."""
    samples = generate_samples(TASK_FAMILIES, count=5, seed=42)
    assert len(samples) == 5


def test_generate_count_deterministic():
    """Same seed, same count → same sample IDs."""
    s1 = [s.sample_id for s in generate_samples(TASK_FAMILIES, count=4, seed=7)]
    s2 = [s.sample_id for s in generate_samples(TASK_FAMILIES, count=4, seed=7)]
    assert s1 == s2


def test_generate_count_different_seed():
    """Different seeds may produce different orderings."""
    s1 = [s.sample_id for s in generate_samples(TASK_FAMILIES, count=4, seed=1)]
    s2 = [s.sample_id for s in generate_samples(TASK_FAMILIES, count=4, seed=99)]
    # Not guaranteed to differ, but with 12 families it's very likely
    # (just check we don't crash)
    assert len(s1) == 4 and len(s2) == 4


# ---------------------------------------------------------------------------
# family_to_sample schema tests
# ---------------------------------------------------------------------------


def test_family_to_sample_valid_schema():
    """family_to_sample produces a valid Sample object."""
    s = family_to_sample(FAMILY_L1_SUM_LIST)
    assert isinstance(s, Sample)
    assert s.task_type == "code_generation"
    assert s.language == "python"
    assert s.verified is False
    assert s.difficulty == 1


def test_family_to_sample_not_verified():
    """Generated samples are explicitly unverified (verified=False)."""
    for fam in TASK_FAMILIES:
        s = family_to_sample(fam)
        assert s.verified is False


def test_family_to_sample_fields_non_empty():
    """Instruction, target_code, public_tests, hidden_tests are all non-empty."""
    for fam in TASK_FAMILIES:
        s = family_to_sample(fam)
        assert s.instruction.strip()
        assert s.target_code.strip()
        assert s.public_tests.strip()
        assert s.hidden_tests.strip()


# ---------------------------------------------------------------------------
# Reference self-verification (key quality gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [
    FAMILY_L0_ADD_TWO,
    FAMILY_L0_IS_POSITIVE,
    FAMILY_L0_CLAMP,
    FAMILY_L1_SUM_LIST,
    FAMILY_L1_REVERSE_STRING,
])
def test_reference_self_verifies(family: TaskFamily):
    """Each reference implementation passes its own public + hidden tests."""
    sample = family_to_sample(family)
    result = verify_sample(sample, run_ruff=False)
    assert result.is_accepted, (
        f"Family {family.family_id} failed verification: {result.messages}"
    )
    assert result.public_ok
    assert result.hidden_ok


@pytest.mark.parametrize("family", [
    FAMILY_L1_COUNT_VOWELS,
    FAMILY_L1_IS_PALINDROME,
    FAMILY_L1_FACTORIAL,
    FAMILY_L2_SECOND_LARGEST,
    FAMILY_L2_BALANCED_PARENS,
    FAMILY_L2_MERGE_SORTED,
    FAMILY_L2_FIND_DUPLICATES,
])
def test_remaining_references_self_verify(family: TaskFamily):
    """Remaining families' references also pass their tests."""
    sample = family_to_sample(family)
    result = verify_sample(sample, run_ruff=False)
    assert result.is_accepted, (
        f"Family {family.family_id} failed verification: {result.messages}"
    )
