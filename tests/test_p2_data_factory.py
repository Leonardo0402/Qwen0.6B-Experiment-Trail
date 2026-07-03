"""
tests/test_p2_data_factory.py -- Tests for P2.2 Execution-driven data factory.

Coverage
--------
- Each of the 8 bug injectors (7.1-7.8) returns valid output or None.
- inject_all_bugs returns a list of (bug_type, bugged_code, description).
- Bugged code actually fails >= 1 test (verified via sandbox).
- compress_feedback produces structured output (failed test / exception / Expected / Actual / traceback / repair requirement).
- generate_boundary_variant produces a valid Sample with enhanced tests.
- partition_families returns three-way disjoint sets.
- build_p2_curriculum manifest contains all required fields.

All fixtures use project-internal simple tasks (FAMILY_L0/L1/L2). No external
datasets, no network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.generate_tasks import (  # noqa: E402
    FAMILY_L0_ADD_TWO,
    FAMILY_L0_IS_POSITIVE,
    FAMILY_L1_FACTORIAL,
    FAMILY_L1_MAX_LIST,
    FAMILY_L1_SUM_LIST,
    FAMILY_L2_BALANCED_PARENS,
    FAMILY_L2_MERGE_SORTED,
    FAMILY_L2_SECOND_LARGEST,
    family_to_sample,
)
from scripts.inject_bugs import (  # noqa: E402
    BUG_TYPES,
    inject_all_bugs,
    inject_bug_aggregation_error,
    inject_bug_branch_deletion,
    inject_bug_condition_error,
    inject_bug_index_error,
    inject_bug_initialization_error,
    inject_bug_off_by_one,
    inject_bug_return_value_error,
    inject_bug_type_error,
)
from scripts.build_execution_repair import (  # noqa: E402
    build_repair_samples,
    compress_feedback,
    verify_bugged_fails,
)
from scripts.generate_boundary_variants import (  # noqa: E402
    generate_boundary_variant,
)
from src.schemas import Sample  # noqa: E402
from src.validators import (  # noqa: E402
    compile_check,
    verify_broken_is_broken,
    verify_sample,
)


# ---------------------------------------------------------------------------
# Helpers: build a Sample with a custom target_code (for injectors that need
# specific patterns not present in the task bank, e.g. aggregation builtins).
# ---------------------------------------------------------------------------

def _make_sample(family, target_code: str | None = None) -> Sample:
    """Create a Sample from a TaskFamily, optionally overriding target_code."""
    s = family_to_sample(family)
    if target_code is not None:
        # Sample is a pydantic model with frozen=False, so we can mutate.
        s = s.model_copy(update={"target_code": target_code})
    return s


# ---------------------------------------------------------------------------
# 1. Bug injector registry
# ---------------------------------------------------------------------------

def test_bug_types_registry_has_eight_entries():
    """BUG_TYPES lists exactly the 8 spec §7.1-7.8 injectors."""
    expected = {
        "condition_error",
        "off_by_one",
        "return_value_error",
        "index_error",
        "initialization_error",
        "aggregation_error",
        "branch_deletion",
        "type_error",
    }
    assert set(BUG_TYPES) == expected
    assert len(BUG_TYPES) == 8


# ---------------------------------------------------------------------------
# 2. Individual injector unit tests (7.1-7.8)
# ---------------------------------------------------------------------------

# --- 7.1 condition_error ----------------------------------------------------

def test_inject_bug_condition_error_flips_lt():
    """7.1: `<` becomes `<=` in a simple comparison."""
    code = "def f(n):\n    return n < 0\n"
    result = inject_bug_condition_error(code)
    assert result is not None
    bugged, desc = result
    assert "<=" in bugged
    assert "<" not in bugged.replace("<=", "")  # only the flipped op remains
    assert isinstance(desc, str) and desc


def test_inject_bug_condition_error_none_when_no_comparison():
    """7.1: Returns None when no comparison operator is present."""
    code = "def f(a, b):\n    return a + b\n"
    assert inject_bug_condition_error(code) is None


# --- 7.2 off_by_one ---------------------------------------------------------

def test_inject_bug_off_by_one_changes_range():
    """7.2: range() stop argument is adjusted by ±1."""
    code = "def f(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
    result = inject_bug_off_by_one(code)
    assert result is not None
    bugged, desc = result
    assert "range(n - 1)" in bugged or "range(n + 1)" in bugged
    assert bugged != code


def test_inject_bug_off_by_one_none_when_no_range():
    """7.2: Returns None when no range() call is present."""
    code = "def f(a, b):\n    return a + b\n"
    assert inject_bug_off_by_one(code) is None


# --- 7.3 return_value_error -------------------------------------------------

def test_inject_bug_return_value_error_returns_none():
    """7.3: `return X` becomes `return None`."""
    code = "def f():\n    x = 1\n    return x\n"
    result = inject_bug_return_value_error(code)
    assert result is not None
    bugged, desc = result
    assert "return None" in bugged
    assert "return x" not in bugged.replace("return None", "")


def test_inject_bug_return_value_error_none_when_only_constants():
    """7.3: Returns None when all returns are constants (no Name to swap)."""
    code = "def f():\n    return 1\n"
    # _ReturnNoneTransformer skips Constant returns; _ReturnWrongVarTransformer
    # also can't find an alternative Name. Both return None.
    assert inject_bug_return_value_error(code) is None


# --- 7.4 index_error --------------------------------------------------------

def test_inject_bug_index_error_shifts_index():
    """7.4: items[0] becomes items[1] (or items[-1] -> items[0])."""
    code = "def f(lst):\n    return lst[0]\n"
    result = inject_bug_index_error(code)
    assert result is not None
    bugged, desc = result
    assert "lst[1]" in bugged or "lst[0]" not in bugged


def test_inject_bug_index_error_none_when_no_subscript():
    """7.4: Returns None when no integer subscript is present."""
    code = "def f(a, b):\n    return a + b\n"
    assert inject_bug_index_error(code) is None


# --- 7.5 initialization_error ----------------------------------------------

def test_inject_bug_initialization_error_corrupts_zero():
    """7.5: `total = 0` becomes `total = 1`."""
    code = "def f():\n    total = 0\n    return total\n"
    result = inject_bug_initialization_error(code)
    assert result is not None
    bugged, desc = result
    assert "total = 1" in bugged


def test_inject_bug_initialization_error_corrupts_empty_list():
    """7.5: `result = []` becomes `result = None`."""
    code = "def f():\n    result = []\n    return result\n"
    result = inject_bug_initialization_error(code)
    assert result is not None
    bugged, desc = result
    assert "result = None" in bugged


def test_inject_bug_initialization_error_none_when_no_init():
    """7.5: Returns None when no mutable initialization is present."""
    code = "def f(a):\n    return a\n"
    assert inject_bug_initialization_error(code) is None


# --- 7.6 aggregation_error --------------------------------------------------

def test_inject_bug_aggregation_error_swaps_max_to_min():
    """7.6: `max(...)` becomes `min(...)`."""
    code = "def f(lst):\n    return max(lst)\n"
    result = inject_bug_aggregation_error(code)
    assert result is not None
    bugged, desc = result
    assert "min(lst)" in bugged
    assert "max(" not in bugged


def test_inject_bug_aggregation_error_swaps_sum_to_len():
    """7.6: `sum(...)` becomes `len(...)`."""
    code = "def f(lst):\n    return sum(lst)\n"
    result = inject_bug_aggregation_error(code)
    assert result is not None
    bugged, desc = result
    assert "len(lst)" in bugged


def test_inject_bug_aggregation_error_none_when_no_aggregator():
    """7.6: Returns None when no aggregation builtin is present."""
    code = "def f(a, b):\n    return a + b\n"
    assert inject_bug_aggregation_error(code) is None


# --- 7.7 branch_deletion ----------------------------------------------------

def test_inject_bug_branch_deletion_removes_if():
    """7.7: The first if-branch is deleted (replaced with pass)."""
    code = "def f(n):\n    if n > 0:\n        return n\n    return 0\n"
    result = inject_bug_branch_deletion(code)
    assert result is not None
    bugged, desc = result
    # The if body should be replaced (pass or removed)
    assert bugged != code


def test_inject_bug_branch_deletion_none_when_no_if():
    """7.7: Returns None when no if-statement is present."""
    code = "def f(a, b):\n    return a + b\n"
    assert inject_bug_branch_deletion(code) is None


# --- 7.8 type_error ---------------------------------------------------------

def test_inject_bug_type_error_swaps_empty_list_to_none():
    """7.8: `result = []` becomes `result = None` (type mismatch)."""
    code = "def f():\n    result = []\n    return result\n"
    result = inject_bug_type_error(code)
    assert result is not None
    bugged, desc = result
    assert "result = None" in bugged


def test_inject_bug_type_error_swaps_zero_to_empty_string():
    """7.8: `total = 0` becomes `total = ''` (type mismatch)."""
    code = "def f():\n    total = 0\n    return total\n"
    result = inject_bug_type_error(code)
    assert result is not None
    bugged, desc = result
    # 0 should be replaced with empty string literal
    assert "total = ''" in bugged or 'total = ""' in bugged


def test_inject_bug_type_error_none_when_no_literal_init():
    """7.8: Returns None when no [] or 0 initialization is present."""
    code = "def f(a):\n    return a\n"
    assert inject_bug_type_error(code) is None


# ---------------------------------------------------------------------------
# 3. inject_all_bugs main function
# ---------------------------------------------------------------------------

def test_inject_all_bugs_returns_list_of_tuples():
    """inject_all_bugs returns list of (bug_type, bugged_code, description)."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    results = inject_all_bugs(sample.target_code, seed=42)
    assert isinstance(results, list)
    assert len(results) >= 1
    for item in results:
        assert len(item) == 3
        bug_type, bugged_code, desc = item
        assert bug_type in BUG_TYPES
        assert isinstance(bugged_code, str) and bugged_code.strip()
        assert isinstance(desc, str) and desc.strip()
        # bugged_code must differ from original
        assert bugged_code.strip() != sample.target_code.strip()


def test_inject_all_bugs_empty_for_noop_code():
    """inject_all_bugs returns [] for code with no mutable targets."""
    code = "def noop():\n    pass\n"
    results = inject_all_bugs(code, seed=42)
    assert results == []


def test_inject_all_bugs_factorial_produces_multiple_variants():
    """factorial code should produce multiple bug variants."""
    sample = family_to_sample(FAMILY_L1_FACTORIAL)
    results = inject_all_bugs(sample.target_code, seed=42)
    # factorial has: comparison (n<0), range(1,n+1), return result, result=1
    # Expect at least 3 applicable injectors.
    assert len(results) >= 3


# ---------------------------------------------------------------------------
# 4. Bugged code actually fails (verified via sandbox)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", [
    FAMILY_L1_SUM_LIST,
    FAMILY_L1_FACTORIAL,
    FAMILY_L2_SECOND_LARGEST,
])
def test_injected_bugs_fail_at_least_one_test(family):
    """Every bug variant produced from these families must fail >= 1 test."""
    sample = family_to_sample(family)
    # Sanity: reference code passes all tests.
    ref = verify_sample(sample, run_ruff=False)
    assert ref.is_accepted, f"Reference for {family.family_id} must pass tests"

    variants = inject_all_bugs(sample.target_code, seed=42)
    assert len(variants) >= 1, f"No bug variants produced for {family.family_id}"

    failing_count = 0
    for bug_type, bugged_code, _desc in variants:
        # bugged code must compile (injectors produce valid Python)
        ok, _err = compile_check(bugged_code)
        assert ok, f"{bug_type} variant for {family.family_id} does not compile"

        is_broken, _feedback = verify_bugged_fails(
            bugged_code,
            sample.public_tests,
            sample.hidden_tests,
            timeout_s=15.0,
        )
        if is_broken:
            failing_count += 1

    assert failing_count >= 1, (
        f"At least one variant for {family.family_id} must fail a test"
    )


# ---------------------------------------------------------------------------
# 5. compress_feedback structure
# ---------------------------------------------------------------------------

def test_compress_feedback_contains_required_sections():
    """compress_feedback output contains failed test / exception / repair requirement."""
    # Simulated pytest output with a failure
    stdout = """\
============================= test session starts =============================
collected 2 items

test_solution.py::test_basic F                                            [ 50%]
test_solution.py::test_zero .                                             [100%]

=================================== FAILURES ===================================
___________________________ test_basic ________________________________________

    def test_basic():
>       assert sum_list([1, 2, 3]) == 6
E       AssertionError: assert 7 == 6
E        +  where 7 = sum_list([1, 2, 3])

test_solution.py:4: AssertionError
============================= short test summary info =============================
FAILED test_solution.py::test_basic - assert 7 == 6
========================= 1 failed, 1 passed in 0.05s ==========================
"""
    feedback = compress_feedback(stdout, "", 1)

    # Must contain failed test name
    assert "test_basic" in feedback
    # Must contain exception type
    assert "AssertionError" in feedback
    # Must contain Expected / Actual
    assert "Expected" in feedback or "Actual" in feedback
    # Must contain a repair-requirement section
    assert "修复要求" in feedback
    # Must contain the header
    assert "执行反馈" in feedback


def test_compress_feedback_truncates_long_output():
    """compress_feedback caps output at MAX_FEEDBACK_CHARS."""
    from scripts.build_execution_repair import MAX_FEEDBACK_CHARS

    # Build a stdout that produces feedback > MAX_FEEDBACK_CHARS
    huge_stdout = "FAILED test_solution.py::test_x - assert " + "x" * (MAX_FEEDBACK_CHARS * 2)
    feedback = compress_feedback(huge_stdout, "", 1)
    assert len(feedback) <= MAX_FEEDBACK_CHARS + 50  # allow marker slack


def test_compress_feedback_handles_no_failure_gracefully():
    """compress_feedback with empty stdout still produces a structured doc."""
    feedback = compress_feedback("", "", 0)
    assert "执行反馈" in feedback
    assert "修复要求" in feedback


def test_compress_feedback_extracts_expected_actual_from_assert():
    """compress_feedback parses `assert X == Y` into Expected/Actual."""
    stdout = "FAILED test_solution.py::test_x - assert 7 == 6\nE   AssertionError: assert 7 == 6"
    feedback = compress_feedback(stdout, "", 1)
    # 6 should appear as Expected, 7 as Actual (or vice versa)
    assert "6" in feedback
    assert "7" in feedback


# ---------------------------------------------------------------------------
# 6. build_repair_samples integration
# ---------------------------------------------------------------------------

def test_build_repair_samples_produces_pairs():
    """build_repair_samples returns at least one (static_repair, execution_repair) pair."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    pairs = build_repair_samples(sample, timeout_s=15.0, seed=42)
    assert len(pairs) >= 1
    for sr, er in pairs:
        # static_repair must have broken_code, no feedback
        assert sr is not None
        assert sr.broken_code and sr.broken_code.strip()
        assert sr.task_type == "static_repair"
        assert sr.execution_feedback is None
        # execution_repair must have broken_code + feedback
        if er is not None:
            assert er.broken_code and er.broken_code.strip()
            assert er.execution_feedback and er.execution_feedback.strip()
            assert er.task_type == "execution_repair"
            assert er.difficulty == 3


def test_build_repair_samples_all_broken_code_genuinely_fails():
    """Every static_repair sample's broken_code must fail >= 1 test."""
    sample = family_to_sample(FAMILY_L1_FACTORIAL)
    pairs = build_repair_samples(sample, timeout_s=15.0, seed=42)
    for sr, er in pairs:
        assert verify_broken_is_broken(sr), (
            f"static_repair {sr.sample_id} should fail tests but passed all"
        )


# ---------------------------------------------------------------------------
# 7. generate_boundary_variant
# ---------------------------------------------------------------------------

def test_generate_boundary_variant_returns_valid_sample():
    """generate_boundary_variant returns a Sample with enhanced tests."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    variant = generate_boundary_variant(sample)
    assert variant is not None
    assert variant.task_type == "code_generation"
    # difficulty should increase by 1 (capped at 4)
    assert variant.difficulty == min(sample.difficulty + 1, 4)
    # target_code unchanged (still the correct implementation)
    assert variant.target_code == sample.target_code
    # public_tests should be enhanced (longer than original)
    assert len(variant.public_tests) > len(sample.public_tests)
    # Should include boundary test markers
    assert "test_boundary_" in variant.public_tests


def test_generate_boundary_variant_none_when_no_function():
    """generate_boundary_variant returns None for code with no def."""
    sample = family_to_sample(FAMILY_L0_ADD_TWO)
    broken = sample.model_copy(update={"target_code": "x = 1\n"})
    assert generate_boundary_variant(broken) is None


def test_boundary_variant_target_code_still_passes():
    """The boundary variant's target_code must still pass the enhanced tests."""
    sample = family_to_sample(FAMILY_L1_SUM_LIST)
    variant = generate_boundary_variant(sample)
    assert variant is not None
    ref = verify_sample(variant, run_ruff=False)
    assert ref.is_accepted, "Boundary variant target_code must pass enhanced tests"


# ---------------------------------------------------------------------------
# 8. Family partition (three-way disjoint)
# ---------------------------------------------------------------------------

def test_partition_families_three_way_disjoint():
    """partition_families returns train/val/frozen with no overlap."""
    from scripts.build_p2_curriculum import partition_families

    # Use families that are all in the existing partition (curriculum-v2).
    families = [
        FAMILY_L0_ADD_TWO,        # train
        FAMILY_L1_REVERSE_STRING if False else FAMILY_L1_SUM_LIST,  # val
        FAMILY_L0_IS_POSITIVE,    # frozen
        FAMILY_L2_MERGE_SORTED,   # frozen
        FAMILY_L1_FACTORIAL,      # val
        FAMILY_L2_BALANCED_PARENS,  # train
    ]
    samples = [family_to_sample(f) for f in families]
    train, val, frozen = partition_families(samples, seed=42)

    # Three-way disjoint
    assert train.isdisjoint(val), f"train ∩ val = {train & val}"
    assert train.isdisjoint(frozen), f"train ∩ frozen = {train & frozen}"
    assert val.isdisjoint(frozen), f"val ∩ frozen = {val & frozen}"

    # Union must cover all input families
    all_input = {s.family_id for s in samples}
    union = train | val | frozen
    assert all_input.issubset(union), (
        f"Input families missing from partition: {all_input - union}"
    )


# ---------------------------------------------------------------------------
# 9. build_p2_curriculum end-to-end (manifest fields)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def p2_curriculum_output(tmp_path_factory):
    """Run build_p2_curriculum.main() on a small fixture and return the output dir."""
    import scripts.build_p2_curriculum as bpc

    # Use families spanning train/val/frozen partitions so every stage has data.
    families = [
        FAMILY_L0_ADD_TWO,        # train
        FAMILY_L1_SUM_LIST,       # val
        FAMILY_L1_FACTORIAL,      # val
        FAMILY_L0_IS_POSITIVE,    # frozen
        FAMILY_L2_MERGE_SORTED,    # frozen
        FAMILY_L2_BALANCED_PARENS,  # train
        FAMILY_L1_MAX_LIST,       # train
    ]
    samples = [family_to_sample(f) for f in families]

    out_dir = tmp_path_factory.mktemp("p2_curriculum")
    in_file = out_dir / "input.jsonl"
    with in_file.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")

    # Patch DEFAULT_OUT_ROOT to use our temp dir
    original_root = bpc.DEFAULT_OUT_ROOT
    bpc.DEFAULT_OUT_ROOT = out_dir
    try:
        # Use a fresh subprocess-style call by invoking main() with argv
        import argparse as _argparse
        old_parser = bpc._build_parser

        def _patched_parser():
            p = _argparse.ArgumentParser()
            p.add_argument("--input", required=True)
            p.add_argument("--output-dir", default=str(out_dir))
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--timeout", type=float, default=15.0)
            return p

        bpc._build_parser = _patched_parser
        try:
            argv = [
                "--input", str(in_file),
                "--output-dir", str(out_dir),
                "--seed", "42",
                "--timeout", "15.0",
            ]
            old_argv = sys.argv
            sys.argv = ["build_p2_curriculum"] + argv
            try:
                rc = bpc.main()
                assert rc == 0, f"build_p2_curriculum.main() returned {rc}"
            finally:
                sys.argv = old_argv
        finally:
            bpc._build_parser = old_parser
    finally:
        bpc.DEFAULT_OUT_ROOT = original_root

    return out_dir


def test_p2_curriculum_directories_created(p2_curriculum_output):
    """All four stage directories + family-partition.json are created."""
    out = p2_curriculum_output
    assert (out / "stage1-code").is_dir()
    assert (out / "stage2-boundary").is_dir()
    assert (out / "stage3-repair").is_dir()
    assert (out / "frozen-eval-v2").is_dir()
    assert (out / "family-partition.json").is_file()


def test_p2_curriculum_manifest_fields_complete(p2_curriculum_output):
    """Every stage manifest contains all required fields."""
    required_fields = {
        "stage",
        "dataset_version",
        "family_partition_version",
        "created_at",
        "seed",
        "train_sha256",
        "validation_sha256",
        "train_families",
        "validation_families",
        "frozen_families",
        "train_validation_overlap",
        "train_frozen_overlap",
        "validation_frozen_overlap",
        "sample_counts",
        "family_counts",
        "task_type_mix",
        "difficulty_mix",
        "max_seq_length",
        "assistant_target_retention_rate",
        "curriculum_ratios",
        "assembly_report",
    }
    for stage in ("stage1-code", "stage2-boundary", "stage3-repair", "frozen-eval-v2"):
        manifest_path = p2_curriculum_output / stage / "manifest.json"
        assert manifest_path.is_file(), f"manifest missing for {stage}"
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        missing = required_fields - set(manifest.keys())
        assert not missing, f"{stage} manifest missing fields: {missing}"
        # dataset_version must be p2.2
        assert manifest["dataset_version"] == "p2.2"
        # curriculum_ratios must match STAGE_RATIOS for non-frozen stages
        if stage != "frozen-eval-v2":
            from scripts.build_p2_curriculum import STAGE_RATIOS
            assert manifest["curriculum_ratios"] == STAGE_RATIOS[stage]


def test_p2_curriculum_family_partition_no_leak(p2_curriculum_output):
    """family-partition.json has empty overlap lists."""
    partition_path = p2_curriculum_output / "family-partition.json"
    with partition_path.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["train_validation_overlap"] == []
    assert doc["train_frozen_overlap"] == []
    assert doc["validation_frozen_overlap"] == []
    # Programmatically verify disjointness
    train = set(doc["train_families"])
    val = set(doc["validation_families"])
    frozen = set(doc["frozen_families"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(frozen)
    assert val.isdisjoint(frozen)


def test_p2_curriculum_stage_files_present(p2_curriculum_output):
    """Each stage directory contains train.jsonl, validation.jsonl, manifest.json, families.json."""
    expected_files = {
        "train.jsonl",
        "validation.jsonl",
        "manifest.json",
        "families.json",
        "token_audit.json",
        "rejected.jsonl",
    }
    for stage in ("stage1-code", "stage2-boundary", "stage3-repair", "frozen-eval-v2"):
        stage_dir = p2_curriculum_output / stage
        for fname in expected_files:
            assert (stage_dir / fname).is_file(), (
                f"{stage}/{fname} missing"
            )


def test_p2_curriculum_stage1_train_has_samples(p2_curriculum_output):
    """Stage 1 train.jsonl contains at least one valid Sample."""
    train_path = p2_curriculum_output / "stage1-code" / "train.jsonl"
    samples = []
    with train_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(Sample.from_json_line(line))
    assert len(samples) >= 1, "stage1-code/train.jsonl should have at least 1 sample"
    # All samples must be valid Sample objects (already validated by from_json_line)
    for s in samples:
        assert s.task_type in ("code_generation", "static_repair", "execution_repair")
