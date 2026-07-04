"""Unified training-sample schema for the Qwen3-0.6B Code Recovery Lab.

Defines:
  - Verification: nested model for per-sample verification flags.
  - Sample: the project's canonical data contract for every training sample.
  - to_chatml(sample): converts a Sample to a ChatML dict (spec §8.2).
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    code_generation = "code_generation"
    static_repair = "static_repair"
    execution_repair = "execution_repair"


_ALLOWED_LANGUAGES = {"python"}

_SYSTEM_PROMPT = (
    "你是一个严谨的 Python 代码助手。"
    "根据任务（及真实执行反馈）输出正确代码。"
    "除非用户要求解释，否则只输出完整代码。"
)


# ---------------------------------------------------------------------------
# Nested model
# ---------------------------------------------------------------------------

class Verification(BaseModel):
    model_config = ConfigDict(frozen=True)

    syntax_ok: bool
    pytest_ok: bool
    ruff_ok: bool
    timeout: bool


# ---------------------------------------------------------------------------
# Main sample model
# ---------------------------------------------------------------------------

class Sample(BaseModel):
    model_config = ConfigDict(frozen=False, use_enum_values=True)

    sample_id: str
    family_id: str
    difficulty: int
    task_type: TaskType
    language: str
    skill_tags: list[str]
    instruction: str
    broken_code: Optional[str] = None
    execution_feedback: Optional[str] = None
    target_code: str
    public_tests: str
    # hidden_tests is a required str but intentionally NOT subjected to the
    # non-empty validator: the spec only mandates non-empty
    # instruction/target_code/public_tests, and some samples may legitimately
    # ship without hidden tests.
    hidden_tests: str
    verified: bool
    verification: Verification
    generator: str
    created_at: str
    dataset_version: str
    # P3 optional metadata fields (free-form strings, default None for
    # backward compatibility with pre-P3 JSONL files).
    variant_type: Optional[str] = None
    bug_type: Optional[str] = None
    source_split: Optional[str] = None

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("difficulty")
    @classmethod
    def difficulty_in_range(cls, v: int) -> int:
        if v < 0 or v > 4:
            raise ValueError(f"difficulty must be 0..4, got {v}")
        return v

    @field_validator("language")
    @classmethod
    def language_must_be_python(cls, v: str) -> str:
        # Intentionally strict: the data contract requires the canonical
        # lowercase token "python". Variants like "Python" or " python " are
        # rejected so upstream formatting bugs surface instead of being
        # silently normalised away.
        if v not in _ALLOWED_LANGUAGES:
            raise ValueError(f"language must be one of {_ALLOWED_LANGUAGES}, got {v!r}")
        return v

    @field_validator("instruction", "target_code", "public_tests")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must be a non-empty, non-blank string")
        return v

    # ------------------------------------------------------------------
    # Cross-field validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_repair_fields(self) -> "Sample":
        # With use_enum_values=True, task_type is stored as a plain str.
        tt = self.task_type

        if tt == TaskType.static_repair.value:
            if not (self.broken_code or "").strip():
                raise ValueError(
                    "static_repair task requires non-empty broken_code"
                )

        if tt == TaskType.execution_repair.value:
            if not (self.broken_code or "").strip():
                raise ValueError(
                    "execution_repair task requires non-empty broken_code"
                )
            if not (self.execution_feedback or "").strip():
                raise ValueError(
                    "execution_repair task requires non-empty execution_feedback"
                )

        return self

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json_line(self) -> str:
        """Serialise to a single compact JSON line (no embedded newlines)."""
        raw = self.model_dump(mode="json")
        return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "Sample":
        """Deserialise from a single JSON line produced by to_json_line."""
        data = json.loads(line)
        return cls(**data)


# ---------------------------------------------------------------------------
# ChatML conversion (spec §8.2)
# ---------------------------------------------------------------------------

def to_chatml(sample: Sample) -> dict:
    """Convert a Sample to a ChatML-formatted dict with system/user/assistant messages."""
    # With use_enum_values=True, task_type is stored as a plain str.
    tt = sample.task_type

    # --- user content ---
    parts: list[str] = [sample.instruction]

    if tt in (TaskType.static_repair.value, TaskType.execution_repair.value):
        parts.append(f"```python\n{sample.broken_code}\n```")

    if tt == TaskType.execution_repair.value:
        parts.append(sample.execution_feedback)  # type: ignore[arg-type]

    if tt == TaskType.code_generation.value:
        parts.append("请输出完整代码")
    else:
        parts.append("请输出修复后的完整代码")

    user_content = "\n\n".join(parts)

    # --- assistant content ---
    assistant_content = f"```python\n{sample.target_code}\n```"

    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
