"""
P0 tests for evaluation trustworthiness.

Tests:
  1. ChatML test.jsonl is rejected
  2. Raw test_raw.jsonl can be loaded
  3. Missing sample_id is rejected
  4. Empty public_tests is rejected
  5. Malformed JSON is rejected
  6. Pydantic validation failure aborts
  7. Hello World canary fails
  8. pass canary fails
  9. return None canary fails
  10. Zero tests collected can't pass
  11. Different dataset SHA can't compare
  12. Different generation config can't compare
  13. Same conditions can compare normally
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, Verification
from src.sandbox import run_pytest, _parse_pytest_counts
from src.metrics import EvalOutcome, summarize, hidden_pass_rate


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_valid_sample(
    sample_id: str = "test_001",
    family_id: str = "fam_test",
    task_type: str = "code_generation",
    public_tests: str = "def test_basic():\n    assert True\n",
    hidden_tests: str = "def test_hidden():\n    assert True\n",
    **overrides,
) -> dict:
    """Create a valid Sample dict for testing."""
    data = {
        "sample_id": sample_id,
        "family_id": family_id,
        "difficulty": 0,
        "task_type": task_type,
        "language": "python",
        "skill_tags": ["test"],
        "instruction": "实现一个测试函数",
        "broken_code": None,
        "execution_feedback": None,
        "target_code": "def solution():\n    return 42\n",
        "public_tests": public_tests,
        "hidden_tests": hidden_tests,
        "verified": True,
        "verification": {
            "syntax_ok": True,
            "pytest_ok": True,
            "ruff_ok": True,
            "timeout": False,
        },
        "generator": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "dataset_version": "test_v1",
    }
    data.update(overrides)
    return data


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 1. ChatML test.jsonl is rejected
# ---------------------------------------------------------------------------

def test_chatml_format_rejected():
    """ChatML format (only messages key) must be rejected."""
    from scripts.evaluate_model import _load_and_validate_samples

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.jsonl"
        _write_jsonl(path, [
            {"messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "ast"},
            ]}
        ])
        with pytest.raises(ValueError, match="ChatML training format"):
            _load_and_validate_samples(path)


# ---------------------------------------------------------------------------
# 2. Raw test_raw.jsonl can be loaded
# ---------------------------------------------------------------------------

def test_raw_sample_format_loaded():
    """Raw Sample format must load and validate successfully."""
    from scripts.evaluate_model import _load_and_validate_samples

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_raw.jsonl"
        _write_jsonl(path, [_make_valid_sample()])
        samples = _load_and_validate_samples(path)
        assert len(samples) == 1
        assert isinstance(samples[0], Sample)
        assert samples[0].sample_id == "test_001"


# ---------------------------------------------------------------------------
# 3. Missing sample_id is rejected
# ---------------------------------------------------------------------------

def test_missing_sample_id_rejected():
    """Records without sample_id must fail Pydantic validation."""
    from scripts.evaluate_model import _load_and_validate_samples

    record = _make_valid_sample()
    del record["sample_id"]

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_raw.jsonl"
        _write_jsonl(path, [record])
        with pytest.raises(Exception):
            _load_and_validate_samples(path)


# ---------------------------------------------------------------------------
# 4. Empty public_tests is rejected
# ---------------------------------------------------------------------------

def test_empty_public_tests_rejected():
    """Empty public_tests must be rejected.

    Defense-in-depth: rejection may come from either
    (a) Pydantic Schema validation (public_tests field validator), or
    (b) the explicit EMPTY_PUBLIC_TESTS check in evaluate_model.py.
    Both achieve the same goal — empty public tests never proceed.
    """
    from scripts.evaluate_model import _load_and_validate_samples

    record = _make_valid_sample(public_tests="   ")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_raw.jsonl"
        _write_jsonl(path, [record])
        # Accept either rejection path — Schema validator or explicit check
        with pytest.raises(Exception) as exc_info:
            _load_and_validate_samples(path)
        msg = str(exc_info.value)
        assert (
            "EMPTY_PUBLIC_TESTS" in msg
            or "public_tests" in msg
            or "non-empty" in msg
            or "non-blank" in msg
        ), f"Empty public_tests should be rejected, got: {msg}"


# ---------------------------------------------------------------------------
# 5. Malformed JSON is rejected
# ---------------------------------------------------------------------------

def test_malformed_json_rejected():
    """Malformed JSON lines must be rejected."""
    from scripts.evaluate_model import _load_and_validate_samples

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_raw.jsonl"
        path.write_text("{bad json}\n", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            _load_and_validate_samples(path)


# ---------------------------------------------------------------------------
# 6. Pydantic validation failure aborts
# ---------------------------------------------------------------------------

def test_pydantic_validation_failure_aborts():
    """Invalid data (e.g. bad difficulty) must fail Pydantic validation."""
    from scripts.evaluate_model import _load_and_validate_samples

    record = _make_valid_sample(difficulty=99)  # difficulty must be 0..4

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_raw.jsonl"
        _write_jsonl(path, [record])
        with pytest.raises(Exception):
            _load_and_validate_samples(path)


# ---------------------------------------------------------------------------
# 7. Hello World canary fails
# ---------------------------------------------------------------------------

def test_canary_hello_world_fails():
    """print('Hello, World!') must fail against real tests."""
    from scripts.evaluate_model import run_canary, _CANARY_CODES

    sample = Sample.model_validate(_make_valid_sample(
        public_tests="from solution import *\n\ndef test_solution():\n    assert solution() == 42\n",
        target_code="def solution():\n    return 42\n",
    ))
    result = run_canary([sample])
    hello_case = next(c for c in result["cases"] if c["canary_name"] == "hello_world")
    assert not hello_case["public_passed"], "Hello World canary should FAIL"
    assert hello_case["num_collected"] > 0, "Should have collected tests"


# ---------------------------------------------------------------------------
# 8. pass canary fails
# ---------------------------------------------------------------------------

def test_canary_pass_fails():
    """pass statement must fail against real tests."""
    from scripts.evaluate_model import run_canary

    sample = Sample.model_validate(_make_valid_sample(
        public_tests="from solution import *\n\ndef test_solution():\n    assert solution() == 42\n",
        target_code="def solution():\n    return 42\n",
    ))
    result = run_canary([sample])
    pass_case = next(c for c in result["cases"] if c["canary_name"] == "pass_stmt")
    assert not pass_case["public_passed"], "pass canary should FAIL"


# ---------------------------------------------------------------------------
# 9. return None canary fails
# ---------------------------------------------------------------------------

def test_canary_return_none_fails():
    """return None solution must fail against real tests."""
    from scripts.evaluate_model import run_canary

    sample = Sample.model_validate(_make_valid_sample(
        public_tests="from solution import *\n\ndef test_solution():\n    assert solution() == 42\n",
        target_code="def solution():\n    return 42\n",
    ))
    result = run_canary([sample])
    none_case = next(c for c in result["cases"] if c["canary_name"] == "return_none")
    assert not none_case["public_passed"], "return None canary should FAIL"


# ---------------------------------------------------------------------------
# 10. Zero tests collected can't pass
# ---------------------------------------------------------------------------

def test_zero_collected_cant_pass():
    """When pytest collects 0 tests, passed must be False."""
    # Use an empty test file
    result = run_pytest(
        "x = 1\n",
        "# no tests here\n",
        timeout_s=5.0,
    )
    assert not result.passed, "Zero tests collected must not pass"
    # num_collected should be 0
    assert result.num_collected == 0


# ---------------------------------------------------------------------------
# 11. Different dataset SHA can't compare
# ---------------------------------------------------------------------------

def test_different_dataset_sha_cant_compare():
    """Different dataset_sha256 must produce incompatibility."""
    from scripts.compare_runs import check_compatibility

    baseline = {
        "dataset_sha256": "aaa111",
        "sample_count": 36,
        "generation_config": {"a": 1},
        "outcomes": [{"sample_id": "s1"}],
        "task_type_counts": {"code_generation": 1},
        "canary": {"passed": True},
    }
    candidate = {
        "dataset_sha256": "bbb222",
        "sample_count": 36,
        "generation_config": {"a": 1},
        "outcomes": [{"sample_id": "s1"}],
        "task_type_counts": {"code_generation": 1},
        "canary": {"passed": True},
    }
    issues = check_compatibility(baseline, candidate)
    assert any("dataset_sha256" in i for i in issues)


# ---------------------------------------------------------------------------
# 12. Different generation config can't compare
# ---------------------------------------------------------------------------

def test_different_generation_config_cant_compare():
    """Different generation_config must produce incompatibility."""
    from scripts.compare_runs import check_compatibility

    baseline = {
        "dataset_sha256": "same",
        "sample_count": 36,
        "generation_config": {"max_new_tokens": 256},
        "outcomes": [{"sample_id": "s1"}],
        "task_type_counts": {"code_generation": 1},
        "canary": {"passed": True},
    }
    candidate = {
        "dataset_sha256": "same",
        "sample_count": 36,
        "generation_config": {"max_new_tokens": 384},
        "outcomes": [{"sample_id": "s1"}],
        "task_type_counts": {"code_generation": 1},
        "canary": {"passed": True},
    }
    issues = check_compatibility(baseline, candidate)
    assert any("generation_config" in i for i in issues)


# ---------------------------------------------------------------------------
# 13. Same conditions can compare normally
# ---------------------------------------------------------------------------

def test_compatible_results_can_compare():
    """Identical conditions must produce no incompatibility issues."""
    from scripts.compare_runs import check_compatibility

    import hashlib

    frozen_eval_path = Path(_ROOT) / "data" / "frozen-eval" / "v1" / "test_raw.jsonl"
    if frozen_eval_path.exists():
        sha = hashlib.sha256()
        sha.update(frozen_eval_path.read_bytes())
        real_hash = sha.hexdigest()
    else:
        real_hash = "same_hash"

    shared = {
        "dataset_sha256": real_hash,
        "sample_count": 36,
        "generation_config": {"max_new_tokens": 384},
        "outcomes": [{"sample_id": f"s{i}"} for i in range(36)],
        "task_type_counts": {"code_generation": 6, "static_repair": 15, "execution_repair": 15},
        "canary": {"passed": True},
    }
    baseline = {**shared, "metrics": {"pass_at_1": 0.5}, "model_path": "base", "adapter": None}
    candidate = {**shared, "metrics": {"pass_at_1": 0.8}, "model_path": "base", "adapter": "adapter1"}
    issues = check_compatibility(baseline, candidate)
    assert len(issues) == 0, f"Expected no issues, got: {issues}"


# ---------------------------------------------------------------------------
# Bonus: num_collected parsing
# ---------------------------------------------------------------------------

def test_parse_pytest_counts_collected():
    """_parse_pytest_counts should extract collected count."""
    output = "collected 4 items\n\ntest_solution.py ....                                                  [100%]\n\n============================== 4 passed in 0.01s ===============================\n"
    passed, failed, errors, collected = _parse_pytest_counts(output)
    assert passed == 4
    assert failed == 0
    assert errors == 0
    assert collected == 4


def test_parse_pytest_counts_no_collection():
    """When no collection line, num_collected falls back to passed+failed+errors."""
    output = "1 failed in 0.01s"
    passed, failed, errors, collected = _parse_pytest_counts(output)
    assert passed == 0
    assert failed == 1
    assert collected == 1  # fallback


# ---------------------------------------------------------------------------
# Bonus: hidden_pass_rate doesn't count missing hidden tests
# ---------------------------------------------------------------------------

def test_hidden_pass_rate_excludes_missing():
    """hidden_pass_rate should only count samples with hidden tests."""
    outcomes = [
        EvalOutcome(
            task_type="code_generation",
            syntax_ok=True, public_passed=True, public_tests_collected=1,
            hidden_passed=True, hidden_tests_present=True, hidden_tests_collected=1,
            format_ok=True, timed_out=False, is_repair=False,
            repair_succeeded=None, broke_other_tests=None,
        ),
        EvalOutcome(
            task_type="code_generation",
            syntax_ok=True, public_passed=True, public_tests_collected=1,
            hidden_passed=False, hidden_tests_present=False, hidden_tests_collected=0,
            format_ok=True, timed_out=False, is_repair=False,
            repair_succeeded=None, broke_other_tests=None,
        ),
    ]
    # Only 1 sample has hidden tests, and it passed => rate = 1.0
    assert hidden_pass_rate(outcomes) == 1.0
