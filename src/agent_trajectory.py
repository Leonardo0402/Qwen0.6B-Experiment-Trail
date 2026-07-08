"""P4.0 Phase D — Agent Trajectory Schema v0.

See docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §6.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.agent_actions import Action, ActionType, FinishAction
from src.agent_state import AgentMemory


# action_type -> expected result.kind
_ACTION_RESULT_KIND: dict[str, str] = {
    ActionType.list_files.value: "file_list",
    ActionType.read_file.value: "file_content",
    ActionType.search_text.value: "search",
    ActionType.inspect_task.value: "task",
    ActionType.propose_patch.value: "patch_proposal",
    ActionType.apply_patch.value: "patch",
    ActionType.rollback_patch.value: "patch",
    ActionType.run_tests.value: "test",
    ActionType.inspect_error.value: "error",
    ActionType.write_memory.value: "memory",
    ActionType.finish.value: "finish",
}

_MUTATING_ACTION_TYPES = frozenset({
    ActionType.apply_patch.value,
    ActionType.rollback_patch.value,
})


class TrajectoryStep(BaseModel):
    """One step in an agent trajectory.

    Invariants enforced here (step-level):
    - result.kind must be consistent with action.action_type

    Trajectory-level invariants (step_index, memory chain, terminal finish,
    success_label) are enforced in ``Trajectory``.
    """
    trajectory_id: str
    task_id: str
    workspace_id: str
    step_index: int
    goal: str
    state_summary: str = ""
    observation: dict
    action: Action
    result: dict
    memory_before: AgentMemory = Field(default_factory=AgentMemory)
    memory_after: AgentMemory = Field(default_factory=AgentMemory)
    success_label: bool = False
    source: Literal["human", "teacher_model", "scripted", "model_self_run"]
    verified: bool = False

    @model_validator(mode="after")
    def _validate_result_kind(self):
        expected = _ACTION_RESULT_KIND.get(self.action.action_type)
        if expected is not None:
            actual = self.result.get("kind")
            if actual != expected:
                raise ValueError(
                    f"result.kind mismatch: action_type="
                    f"{self.action.action_type} expects result.kind="
                    f"'{expected}', got '{actual}'"
                )
        return self


class Trajectory(BaseModel):
    """A complete agent trajectory (sequence of TrajectoryStep).

    Invariants enforced:
    - step_index monotonically increasing from 0
    - memory_before of step N == memory_after of step N-1
    - terminal step must have action.action_type == "finish"
    - success_label=True requires verified=True
    - success_label=True on terminal requires finish.tests_passed=True
    """
    trajectory_id: str
    task_id: str
    workspace_id: str
    goal: str
    steps: list[TrajectoryStep]
    final_success: bool = False
    final_verified: bool = False
    source: Literal["human", "teacher_model", "scripted", "model_self_run"]
    action_count: int = 0
    tool_distribution: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_and_compute(self):
        if not self.steps:
            raise ValueError("trajectory must have at least one step")

        # step_index monotonic from 0
        for i, step in enumerate(self.steps):
            if step.step_index != i:
                raise ValueError(
                    f"step_index not monotonic: expected {i}, "
                    f"got {step.step_index} at position {i}"
                )

        # memory chain: memory_before of step N == memory_after of step N-1
        for i in range(1, len(self.steps)):
            if self.steps[i].memory_before != self.steps[i - 1].memory_after:
                raise ValueError(
                    f"memory chain broken at step {i}: "
                    f"memory_before != previous memory_after"
                )

        # terminal step must be finish
        last = self.steps[-1]
        if last.action.action_type != ActionType.finish.value:
            raise ValueError(
                f"trajectory must end with finish action, "
                f"got {last.action.action_type}"
            )

        # success_label=True requires verified=True
        for i, step in enumerate(self.steps):
            if step.success_label and not step.verified:
                raise ValueError(
                    f"step {i}: success_label=True requires verified=True"
                )

        # terminal success_label=True requires finish.tests_passed=True
        if last.success_label:
            if not isinstance(last.action, FinishAction):
                raise ValueError(
                    "terminal success_label=True requires FinishAction"
                )
            if not last.action.arguments.tests_passed:
                raise ValueError(
                    "success_label=True requires terminal "
                    "finish.tests_passed=True"
                )

        # compute derived fields
        self.action_count = len(self.steps)
        dist: dict[str, int] = {}
        for step in self.steps:
            at = step.action.action_type
            dist[at] = dist.get(at, 0) + 1
        self.tool_distribution = dist

        # derive final_success / final_verified from last step
        self.final_success = last.success_label
        self.final_verified = last.verified

        return self


def is_mutating_action(action_type: str) -> bool:
    """Return True if the action type modifies workspace files.

    Mutating actions: apply_patch, rollback_patch.
    Per spec §10 State Transition Gate (lines 935-936).
    """
    return action_type in _MUTATING_ACTION_TYPES


def save_trajectory(traj: Trajectory, path: Path) -> None:
    """Write a Trajectory as JSONL (one TrajectoryStep per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for step in traj.steps:
            f.write(step.model_dump_json() + "\n")


def load_trajectory(path: Path) -> Trajectory:
    """Read a Trajectory from a JSONL file (one TrajectoryStep per line).

    Derives trajectory-level fields (trajectory_id, task_id, workspace_id,
    goal, source) from the first step.
    """
    path = Path(path)
    steps: list[TrajectoryStep] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            steps.append(TrajectoryStep.model_validate_json(stripped))

    if not steps:
        raise ValueError(f"empty trajectory file: {path}")

    first = steps[0]
    return Trajectory(
        trajectory_id=first.trajectory_id,
        task_id=first.task_id,
        workspace_id=first.workspace_id,
        goal=first.goal,
        steps=steps,
        source=first.source,
    )
