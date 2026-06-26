"""Tests for src/schemas.py — TDD first pass.

Covers:
- Valid construction for each of the 3 task_types.
- Every validation rule rejects bad input.
- Round-trip to_json_line / from_json_line.
- to_chatml structure and content rules.
"""

import json
import pytest
from pydantic import ValidationError

from src.schemas import Sample, Verification, to_chatml


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _verification(**kwargs) -> dict:
    defaults = {"syntax_ok": True, "pytest_ok": True, "ruff_ok": True, "timeout": False}
    defaults.update(kwargs)
    return defaults


def _base_sample(**kwargs) -> dict:
    """Return a minimal valid code_generation sample dict."""
    defaults = {
        "sample_id": "py_gen_l0_000001",
        "family_id": "gen_family_01",
        "difficulty": 1,
        "task_type": "code_generation",
        "language": "python",
        "skill_tags": ["basics"],
        "instruction": "写一个函数，返回两数之和。",
        "broken_code": None,
        "execution_feedback": None,
        "target_code": "def add(a, b):\n    return a + b",
        "public_tests": "def test_add():\n    assert add(1, 2) == 3",
        "hidden_tests": "def test_add_neg():\n    assert add(-1, -1) == -2",
        "verified": True,
        "verification": _verification(),
        "generator": "claude-code",
        "created_at": "2026-01-01T00:00:00Z",
        "dataset_version": "v1",
    }
    defaults.update(kwargs)
    return defaults


def _repair_sample(**kwargs) -> dict:
    """Return a minimal valid static_repair sample dict."""
    defaults = _base_sample(
        sample_id="py_repair_l2_000001",
        task_type="static_repair",
        broken_code="def add(a, b):\n    return a - b",
    )
    defaults.update(kwargs)
    return defaults


def _exec_repair_sample(**kwargs) -> dict:
    """Return a minimal valid execution_repair sample dict."""
    defaults = _repair_sample(
        sample_id="py_exec_l3_000001",
        task_type="execution_repair",
        execution_feedback="FAILED test_add: AssertionError: assert 0 == 3",
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Part A – construction
# ---------------------------------------------------------------------------

class TestValidConstruction:
    def test_code_generation(self):
        s = Sample(**_base_sample())
        assert s.task_type == "code_generation"
        assert s.broken_code is None
        assert s.execution_feedback is None
        assert isinstance(s.verification, Verification)

    def test_static_repair(self):
        s = Sample(**_repair_sample())
        assert s.task_type == "static_repair"
        assert s.broken_code is not None

    def test_execution_repair(self):
        s = Sample(**_exec_repair_sample())
        assert s.task_type == "execution_repair"
        assert s.execution_feedback is not None

    def test_difficulty_boundaries(self):
        for d in range(5):  # 0..4
            s = Sample(**_base_sample(difficulty=d))
            assert s.difficulty == d

    def test_verification_model_populated(self):
        s = Sample(**_base_sample())
        assert s.verification.syntax_ok is True
        assert s.verification.pytest_ok is True
        assert s.verification.ruff_ok is True
        assert s.verification.timeout is False

    def test_code_generation_without_optional_keys(self):
        # broken_code / execution_feedback omitted entirely should default to None
        data = _base_sample()
        del data["broken_code"]
        del data["execution_feedback"]
        s = Sample(**data)
        assert s.broken_code is None
        assert s.execution_feedback is None


# ---------------------------------------------------------------------------
# Part A – validation rejections
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_bad_task_type(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(task_type="unknown_type"))

    def test_difficulty_negative(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(difficulty=-1))

    def test_difficulty_too_high(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(difficulty=5))

    def test_non_python_language(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(language="javascript"))

    def test_execution_repair_missing_feedback(self):
        with pytest.raises(ValidationError):
            Sample(**_exec_repair_sample(execution_feedback=None))

    def test_execution_repair_empty_feedback(self):
        with pytest.raises(ValidationError):
            Sample(**_exec_repair_sample(execution_feedback=""))

    def test_execution_repair_missing_broken_code(self):
        with pytest.raises(ValidationError):
            Sample(**_exec_repair_sample(broken_code=None))

    def test_execution_repair_empty_broken_code(self):
        with pytest.raises(ValidationError):
            Sample(**_exec_repair_sample(broken_code=""))

    def test_static_repair_missing_broken_code(self):
        with pytest.raises(ValidationError):
            Sample(**_repair_sample(broken_code=None))

    def test_static_repair_empty_broken_code(self):
        with pytest.raises(ValidationError):
            Sample(**_repair_sample(broken_code=""))

    def test_empty_target_code(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(target_code=""))

    def test_empty_instruction(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(instruction=""))

    def test_empty_public_tests(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(public_tests=""))

    # --- whitespace-only rejections ---

    def test_whitespace_instruction(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(instruction="   \n\t "))

    def test_whitespace_target_code(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(target_code="   \n  "))

    def test_whitespace_public_tests(self):
        with pytest.raises(ValidationError):
            Sample(**_base_sample(public_tests="  \t\n"))

    def test_whitespace_broken_code_static_repair(self):
        with pytest.raises(ValidationError):
            Sample(**_repair_sample(broken_code="   \n "))

    def test_whitespace_broken_code_execution_repair(self):
        with pytest.raises(ValidationError):
            Sample(**_exec_repair_sample(broken_code="  \t "))

    def test_whitespace_execution_feedback(self):
        with pytest.raises(ValidationError):
            Sample(**_exec_repair_sample(execution_feedback="   \n  "))


# ---------------------------------------------------------------------------
# Part A – round-trip serialisation
# ---------------------------------------------------------------------------

class TestJsonLineSerialization:
    def test_roundtrip_code_generation(self):
        original = Sample(**_base_sample())
        line = original.to_json_line()
        assert "\n" not in line, "JSONL line must not contain embedded newlines"
        restored = Sample.from_json_line(line)
        assert restored == original

    def test_roundtrip_static_repair(self):
        original = Sample(**_repair_sample())
        restored = Sample.from_json_line(original.to_json_line())
        assert restored == original

    def test_roundtrip_execution_repair(self):
        original = Sample(**_exec_repair_sample())
        restored = Sample.from_json_line(original.to_json_line())
        assert restored == original

    def test_jsonline_is_valid_json(self):
        s = Sample(**_base_sample())
        line = s.to_json_line()
        parsed = json.loads(line)
        assert parsed["sample_id"] == s.sample_id

    def test_roundtrip_preserves_skill_tags(self):
        original = Sample(**_base_sample(skill_tags=["loops", "recursion"]))
        restored = Sample.from_json_line(original.to_json_line())
        assert restored.skill_tags == ["loops", "recursion"]


# ---------------------------------------------------------------------------
# Part B – ChatML conversion
# ---------------------------------------------------------------------------

class TestToChatml:
    def _assert_structure(self, chatml: dict):
        assert "messages" in chatml
        msgs = chatml["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_structure_code_generation(self):
        s = Sample(**_base_sample())
        result = to_chatml(s)
        self._assert_structure(result)

    def test_structure_static_repair(self):
        s = Sample(**_repair_sample())
        result = to_chatml(s)
        self._assert_structure(result)

    def test_structure_execution_repair(self):
        s = Sample(**_exec_repair_sample())
        result = to_chatml(s)
        self._assert_structure(result)

    def test_system_prompt_in_chinese(self):
        s = Sample(**_base_sample())
        system_content = to_chatml(s)["messages"][0]["content"]
        # Must contain at least some Chinese characters
        assert any("一" <= ch <= "鿿" for ch in system_content)

    def test_code_generation_user_content(self):
        s = Sample(**_base_sample())
        user_content = to_chatml(s)["messages"][1]["content"]
        assert s.instruction in user_content
        assert user_content.rstrip().endswith("请输出完整代码")

    def test_code_generation_no_broken_code_in_user(self):
        s = Sample(**_base_sample())
        user_content = to_chatml(s)["messages"][1]["content"]
        # For code_generation, broken_code is None, so no fenced block should appear
        # The only fenced block permitted would be in the assistant message
        assert "```" not in user_content

    def test_static_repair_user_has_broken_code_fence(self):
        s = Sample(**_repair_sample())
        user_content = to_chatml(s)["messages"][1]["content"]
        assert "```python" in user_content
        assert s.broken_code in user_content

    def test_static_repair_user_no_execution_feedback(self):
        s = Sample(**_repair_sample())
        user_content = to_chatml(s)["messages"][1]["content"]
        assert s.execution_feedback is None  # sanity
        # feedback text should not appear in user content
        assert "FAILED" not in user_content

    def test_execution_repair_user_has_broken_code_and_feedback(self):
        s = Sample(**_exec_repair_sample())
        user_content = to_chatml(s)["messages"][1]["content"]
        assert "```python" in user_content
        assert s.broken_code in user_content
        assert s.execution_feedback in user_content

    def test_assistant_content_fenced_python(self):
        for sample_dict, label in [
            (_base_sample(), "code_generation"),
            (_repair_sample(), "static_repair"),
            (_exec_repair_sample(), "execution_repair"),
        ]:
            s = Sample(**sample_dict)
            assistant_content = to_chatml(s)["messages"][2]["content"]
            assert assistant_content.startswith("```python"), f"Failed for {label}"
            assert assistant_content.strip().endswith("```"), f"Failed for {label}"
            assert s.target_code in assistant_content, f"Failed for {label}"
