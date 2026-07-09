"""P4.0 Phase G — Agent Evaluation Harness v0.

Replay evaluator that runs scripted trajectories through the real tool
layer and computes 8 metrics. See
docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §9.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field

from src.agent_actions import (
    Action,
    FinishAction,
    FinishArgs,
    ListFilesAction,
    SafetyFlags,
    TaskSuccessCriterion,
)
from src.agent_state import AgentMemory
from src.agent_trajectory import Trajectory
from src.agent_workspace import MicroTaskWorkspace
from src.agent_tools import (
    tool_apply_patch,
    tool_inspect_task,
    tool_list_files,
    tool_propose_patch,
    tool_read_file,
    tool_run_tests,
)


class AgentState(BaseModel):
    """State passed to ActionProvider.next_action()."""
    memory: AgentMemory
    step_count: int
    task_id: str
    workspace_id: str


class EvalResult(BaseModel):
    """Result of evaluating one task."""
    task_id: str
    trajectory_id: str
    steps_executed: int
    success: bool              # success_criterion was met
    metrics: dict[str, float | int]
    errors: list[str] = Field(default_factory=list)
    max_steps_hit: bool = False


class CorruptionType(str, Enum):
    WRONG_ACTION_TYPE = "wrong_action_type"        # replace action_type with a different one
    INVALID_PATH = "invalid_path"                   # use a forbidden path (e.g., ../etc/passwd)
    WRONG_PATCH = "wrong_patch"                     # use old_text that doesn't exist
    SKIP_TESTS_BEFORE_FINISH = "skip_tests_before_finish"  # finish without prior run_tests
    EXCEED_MAX_STEPS = "exceed_max_steps"           # provider never returns finish


class Corruption(BaseModel):
    step_index: int           # which step to corrupt
    type: CorruptionType


class ActionProvider(ABC):
    """Abstract base for action providers."""
    @abstractmethod
    def next_action(self, state: AgentState) -> Action:
        """Return the next action for the given state."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the provider to the beginning."""
        ...


class ScriptedActionProvider(ActionProvider):
    """Replays actions from a pre-computed trajectory."""
    def __init__(self, trajectory: Trajectory):
        self._actions = [step.action for step in trajectory.steps]
        self._index = 0

    def next_action(self, state: AgentState) -> Action:
        if self._index >= len(self._actions):
            raise StopIteration("trajectory exhausted")
        action = self._actions[self._index]
        self._index += 1
        return action

    def reset(self) -> None:
        self._index = 0


class ReplayActionProvider(ScriptedActionProvider):
    """Semantic alias for ScriptedActionProvider (used by evaluator)."""
    pass


def _make_safe_safety_flags(is_terminal: bool = False) -> SafetyFlags:
    """Build a non-forbidden SafetyFlags (network=False, sensitive=False)."""
    return SafetyFlags(
        modifies_workspace=False,
        executes_code=False,
        network_required=False,
        reads_sensitive_path=False,
        is_terminal=is_terminal,
    )


def _make_loop_action() -> Action:
    """Build a ListFilesAction used to force max_steps exhaustion."""
    return ListFilesAction(
        action_id="loop_action",
        reason_short="loop (corrupted)",
        expected_observation="file list",
        safety_flags=_make_safe_safety_flags(is_terminal=False),
    )


class CorruptedActionProvider(ActionProvider):
    """Injects intentional corruption at step N (for negative tests)."""
    def __init__(self, base: Trajectory, corruption: Corruption):
        self._actions = [step.action for step in base.steps]
        self._corruption = corruption
        self._index = 0

    def next_action(self, state: AgentState) -> Action:
        if self._index >= len(self._actions):
            if self._corruption.type == CorruptionType.EXCEED_MAX_STEPS:
                # Return a non-finish action to force max_steps exhaustion
                return _make_loop_action()
            raise StopIteration("trajectory exhausted")
        if self._index == self._corruption.step_index:
            action = self._corrupt_action(self._actions[self._index])
        else:
            action = self._actions[self._index]
        self._index += 1
        # EXCEED_MAX_STEPS: never return finish, return list_files forever
        if (
            self._corruption.type == CorruptionType.EXCEED_MAX_STEPS
            and action.action_type == "finish"
        ):
            return _make_loop_action()
        return action

    def _corrupt_action(self, action: Action) -> Action:
        ct = self._corruption.type
        if ct == CorruptionType.WRONG_ACTION_TYPE:
            # Swap the action for a ListFilesAction
            return ListFilesAction(
                action_id=action.action_id,
                reason_short="corrupted (wrong type)",
                expected_observation="file list",
                safety_flags=_make_safe_safety_flags(is_terminal=False),
            )
        if ct == CorruptionType.INVALID_PATH:
            data = action.model_dump()
            args = data.get("arguments", {}) or {}
            if "path" in args:
                args["path"] = "../etc/passwd"
            elif "file_path" in args:
                args["file_path"] = "../etc/passwd"
            else:
                return action
            # Bypass validation so the evaluator's re-validation catches it
            return action.__class__.model_construct(**data)
        if ct == CorruptionType.WRONG_PATCH:
            at = action.action_type
            if at == "apply_patch" or at == "propose_patch":
                data = action.model_dump()
                data["arguments"]["old_text"] = "__NONEXISTENT__"
                return action.__class__.model_validate(data)
            return action
        if ct == CorruptionType.SKIP_TESTS_BEFORE_FINISH:
            if action.action_type == "run_tests":
                return FinishAction(
                    action_id=action.action_id,
                    reason_short="corrupted (skip tests)",
                    expected_observation="finish",
                    safety_flags=_make_safe_safety_flags(is_terminal=True),
                    arguments=FinishArgs(
                        success_criterion=TaskSuccessCriterion.TEST_PASS,
                        tests_passed=False,
                        identification_verified=False,
                        summary="Corrupted: skipped tests before finish",
                    ),
                )
            return action
        if ct == CorruptionType.EXCEED_MAX_STEPS:
            # Convert any finish into a loop action; other actions pass through.
            if action.action_type == "finish":
                return _make_loop_action()
            return action
        return action

    def reset(self) -> None:
        self._index = 0


class ModelActionProvider(ActionProvider):
    """Reserved for P4.1 — loads Qwen3-0.6B and generates actions."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ModelActionProvider is P4.1")

    def next_action(self, state: AgentState) -> Action:
        raise NotImplementedError("P4.1")

    def reset(self) -> None:
        raise NotImplementedError("P4.1")


class AgentEvaluator:
    """Replay an ActionProvider's actions through the real tool layer."""

    def __init__(
        self,
        workspace: MicroTaskWorkspace,
        provider: ActionProvider,
        task_id: str,
        max_steps: int = 20,
    ):
        self._ws = workspace
        self._provider = provider
        self._task_id = task_id
        self._max_steps = max_steps

    def run(self) -> EvalResult:
        """Execute actions until finish or max_steps."""
        self._provider.reset()
        errors: list[str] = []
        steps_executed = 0
        success = False
        max_steps_hit = False

        # Metric counters
        total_actions = 0
        valid_actions = 0
        forbidden_count = 0
        total_patches = 0
        successful_patches = 0
        total_tests = 0
        passed_tests = 0
        tool_errors = 0
        total_tools = 0
        finish_without_tests = 0
        ran_tests = False

        state = AgentState(
            memory=AgentMemory(),
            step_count=0,
            task_id=self._task_id,
            # MicroTaskWorkspace has no .workspace_id attribute; use the
            # temp workspace root path as a unique identifier.
            workspace_id=str(self._ws.workspace_root),
        )

        for step in range(self._max_steps):
            try:
                action = self._provider.next_action(state)
            except StopIteration:
                break

            total_actions += 1

            # Validate action (schema + safety)
            try:
                # Re-validate by constructing a new instance of the same type
                action.__class__.model_validate(action.model_dump())
                valid_actions += 1
            except Exception as e:
                errors.append(f"step {step}: invalid action: {e}")
                continue

            # Check safety flags
            sf = action.safety_flags
            if sf.network_required or sf.reads_sensitive_path:
                forbidden_count += 1
                errors.append(f"step {step}: forbidden action")
                continue

            # Dispatch to tool
            at = action.action_type
            try:
                if at == "list_files":
                    total_tools += 1
                    tool_list_files(self._ws)
                elif at == "read_file":
                    total_tools += 1
                    tool_read_file(self._ws, action.arguments.path)
                elif at == "inspect_task":
                    total_tools += 1
                    tool_inspect_task(self._ws)
                elif at == "propose_patch":
                    total_tools += 1
                    tool_propose_patch(
                        self._ws,
                        action.arguments.file_path,
                        action.arguments.old_text,
                        action.arguments.new_text,
                    )
                elif at == "apply_patch":
                    total_tools += 1
                    obs = tool_apply_patch(
                        self._ws,
                        action.arguments.file_path,
                        action.arguments.old_text,
                        action.arguments.new_text,
                    )
                    total_patches += 1
                    if obs.success:
                        successful_patches += 1
                    else:
                        tool_errors += 1
                elif at == "run_tests":
                    total_tools += 1
                    ran_tests = True
                    obs = tool_run_tests(self._ws, timeout_s=10.0)
                    total_tests += 1
                    if obs.passed:
                        passed_tests += 1
                elif at == "inspect_error":
                    # Stateless — no tool dispatch needed for replay
                    pass
                elif at == "write_memory":
                    # Stateless — update state memory
                    state.memory = action.arguments.memory
                elif at == "finish":
                    fa = action.arguments
                    if not ran_tests:
                        finish_without_tests += 1
                    # Check success criterion
                    if fa.success_criterion == TaskSuccessCriterion.TEST_PASS:
                        success = fa.tests_passed
                    elif fa.success_criterion == TaskSuccessCriterion.IDENTIFY_BUG:
                        success = fa.identification_verified
                    elif fa.success_criterion == TaskSuccessCriterion.PATCH_APPLIED:
                        success = successful_patches > 0
                    steps_executed = step + 1
                    # Compute metrics and return
                    return self._make_result(
                        steps_executed, success, errors,
                        total_actions, valid_actions, forbidden_count,
                        total_patches, successful_patches,
                        total_tests, passed_tests,
                        tool_errors, total_tools,
                        finish_without_tests, max_steps_hit=False,
                    )
            except Exception as e:
                tool_errors += 1
                errors.append(f"step {step}: tool error: {e}")

            state.step_count = step + 1
            steps_executed = step + 1

        # Hit max_steps without finish
        max_steps_hit = True
        return self._make_result(
            steps_executed, success, errors,
            total_actions, valid_actions, forbidden_count,
            total_patches, successful_patches,
            total_tests, passed_tests,
            tool_errors, total_tools,
            finish_without_tests, max_steps_hit=True,
        )

    def _make_result(
        self,
        steps_executed: int,
        success: bool,
        errors: list[str],
        total_actions: int,
        valid_actions: int,
        forbidden_count: int,
        total_patches: int,
        successful_patches: int,
        total_tests: int,
        passed_tests: int,
        tool_errors: int,
        total_tools: int,
        finish_without_tests: int,
        max_steps_hit: bool,
    ) -> EvalResult:
        metrics = {
            "task_success_rate": 1.0 if success else 0.0,
            "action_validity_rate": valid_actions / total_actions if total_actions else 0.0,
            "tool_error_rate": tool_errors / total_tools if total_tools else 0.0,
            "patch_success_rate": successful_patches / total_patches if total_patches else 0.0,
            "tests_pass_rate": passed_tests / total_tests if total_tests else 0.0,
            "forbidden_action_count": forbidden_count,
            "max_step_exceeded_count": 1 if max_steps_hit else 0,
            "finish_without_tests_count": finish_without_tests,
        }
        return EvalResult(
            task_id=self._task_id,
            trajectory_id="",  # filled by caller
            steps_executed=steps_executed,
            success=success,
            metrics=metrics,
            errors=errors,
            max_steps_hit=max_steps_hit,
        )
