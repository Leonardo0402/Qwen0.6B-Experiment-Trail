"""P4.0 Agent State — AgentMemory schema.

See docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §4.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentMemory(BaseModel):
    """Fixed 4-field agent memory (user-confirmed)."""
    notes: str = ""
    hypothesis: str = ""
    failed_attempts: list[str] = Field(default_factory=list)
    last_test_summary: str = ""
