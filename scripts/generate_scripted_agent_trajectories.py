"""Generate the P4.0 scripted teacher trajectories (40 trajectories).

Run: set P4_ALLOW_NETWORK=0 && py -3.11 scripts/generate_scripted_agent_trajectories.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

# CRITICAL: must be set before importing agent_tools (run_tests checks it)
os.environ.setdefault("P4_ALLOW_NETWORK", "0")

# Ensure project root is on sys.path so `src.*` imports work when run as a script
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent_actions import (
    SafetyFlags,
    TaskSuccessCriterion,
    ListFilesAction,
    ReadFileAction,
    InspectTaskAction,
    ProposePatchAction,
    ApplyPatchAction,
    RunTestsAction,
    InspectErrorAction,
    WriteMemoryAction,
    FinishAction,
    InspectTaskArgs,
    ListFilesArgs,
    ReadFileArgs,
    ProposePatchArgs,
    ApplyPatchArgs,
    RunTestsArgs,
    InspectErrorArgs,
    WriteMemoryArgs,
    FinishArgs,
)
from src.agent_state import AgentMemory
from src.agent_workspace import MicroTaskWorkspace
from src.agent_tools import (
    tool_list_files,
    tool_read_file,
    tool_inspect_task,
    tool_propose_patch,
    tool_apply_patch,
    tool_run_tests,
    tool_inspect_error,
    tool_write_memory,
    tool_finish,
)
from src.agent_trajectory import Trajectory, TrajectoryStep

ROOT = _ROOT
TASKS_DIR = ROOT / "data" / "p4-agent" / "micro-tasks-v0"
OUT_PATH = ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
REPORT_PATH = ROOT / "reports" / "p4" / "scripted-trajectories-report.md"

_SAFETY = {
    "list_files": SafetyFlags(modifies_workspace=False, executes_code=False, network_required=False, reads_sensitive_path=False, is_terminal=False),
    "read_file": SafetyFlags(modifies_workspace=False, executes_code=False, network_required=False, reads_sensitive_path=False, is_terminal=False),
    "inspect_task": SafetyFlags(modifies_workspace=False, executes_code=False, network_required=False, reads_sensitive_path=False, is_terminal=False),
    "propose_patch": SafetyFlags(modifies_workspace=False, executes_code=False, network_required=False, reads_sensitive_path=False, is_terminal=False),
    "apply_patch": SafetyFlags(modifies_workspace=True, executes_code=False, network_required=False, reads_sensitive_path=False, is_terminal=False),
    "run_tests": SafetyFlags(modifies_workspace=False, executes_code=True, network_required=False, reads_sensitive_path=False, is_terminal=False),
    "inspect_error": SafetyFlags(modifies_workspace=False, executes_code=False, network_required=False, reads_sensitive_path=False, is_terminal=False),
    "write_memory": SafetyFlags(modifies_workspace=False, executes_code=False, network_required=False, reads_sensitive_path=False, is_terminal=False),
    "finish": SafetyFlags(modifies_workspace=False, executes_code=False, network_required=False, reads_sensitive_path=False, is_terminal=True),
}

_ACTION_CLS = {
    "list_files": ListFilesAction,
    "read_file": ReadFileAction,
    "inspect_task": InspectTaskAction,
    "propose_patch": ProposePatchAction,
    "apply_patch": ApplyPatchAction,
    "run_tests": RunTestsAction,
    "inspect_error": InspectErrorAction,
    "write_memory": WriteMemoryAction,
    "finish": FinishAction,
}


def _make_action(action_type: str, reason_short: str, expected_obs: str, arguments):
    """Construct an action with correct safety_flags and a UUID action_id."""
    return _ACTION_CLS[action_type](
        action_id=f"act_{uuid4().hex[:8]}",
        reason_short=reason_short[:120],
        expected_observation=expected_obs,
        safety_flags=_SAFETY[action_type],
        arguments=arguments,
    )


def _make_step(
    traj_id: str, task_id: str, ws_id: str, step_index: int, goal: str,
    action, result: dict, mem_before: AgentMemory, mem_after: AgentMemory,
    success_label: bool = False, observation: dict | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        trajectory_id=traj_id,
        task_id=task_id,
        workspace_id=ws_id,
        step_index=step_index,
        goal=goal,
        state_summary="",
        observation=observation or {"step": step_index},
        action=action,
        result=result,
        memory_before=mem_before,
        memory_after=mem_after,
        success_label=success_label,
        source="scripted",
        verified=True,
    )


def _extract_func_name(solution_content: str, old_text: str) -> str:
    """Find the function name whose body contains old_text. Falls back to 'unknown'."""
    idx = solution_content.find(old_text)
    if idx < 0:
        return "unknown"
    prefix = solution_content[:idx]
    last_def = prefix.rfind("def ")
    if last_def < 0:
        return "unknown"
    rest = solution_content[last_def + 4:]
    paren = rest.find("(")
    if paren < 0:
        return "unknown"
    return rest[:paren].strip()


def _build_trajectory(task_id: str, task_type: str) -> Trajectory:
    """Build a scripted trajectory for one task by executing real tools."""
    task_dir = TASKS_DIR / task_id
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        traj_id = f"traj_{task_id}"
        ws_id = f"micro-tasks-v0/{task_id}"

        task_obs = tool_inspect_task(ws)
        goal = task_obs.goal

        patch = json.loads((task_dir / "expected_patch.json").read_text())

        if task_type == "locate_failing_function":
            steps = _build_locate_steps(traj_id, task_id, ws_id, goal, ws, patch)
        elif task_type == "avoid_editing_tests":
            steps = _build_avoid_edit_steps(traj_id, task_id, ws_id, goal, ws, patch)
        elif task_type == "recover_from_failed_patch":
            steps = _build_recover_steps(traj_id, task_id, ws_id, goal, ws, patch)
        else:
            # Types: one_line_fix, add_boundary_check, update_helper,
            # repair_after_pytest, finish_after_tests_pass
            steps = _build_standard_steps(traj_id, task_id, ws_id, goal, ws, patch)

        return Trajectory(
            trajectory_id=traj_id,
            task_id=task_id,
            workspace_id=ws_id,
            goal=goal,
            steps=steps,
            source="scripted",
        )
    finally:
        ws.cleanup()


def _build_standard_steps(traj_id, task_id, ws_id, goal, ws, patch) -> list[TrajectoryStep]:
    """10-step pattern for types 2-6 and 8.

    inspect_task → list_files → read_file → run_tests (fails) →
    inspect_error → propose_patch → apply_patch → run_tests (passes) →
    write_memory → finish(TEST_PASS, tests_passed=True)
    """
    steps: list[TrajectoryStep] = []
    current_mem = AgentMemory()
    file_path = patch["file_path"]
    old_text = patch["old_text"]
    new_text = patch["new_text"]

    # Step 0: inspect_task
    task_obs = tool_inspect_task(ws)
    a0 = _make_action("inspect_task", "inspect task goal", "task fields", InspectTaskArgs())
    mem0 = current_mem.model_copy(update={"notes": task_obs.goal})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 0, goal, a0,
        {"kind": "task", "goal": task_obs.goal},
        current_mem, mem0, observation={"step": 0},
    ))
    current_mem = mem0

    # Step 1: list_files
    list_obs = tool_list_files(ws)
    a1 = _make_action("list_files", "list workspace files", "file list", ListFilesArgs())
    mem1 = current_mem.model_copy(update={"notes": f"files: {len(list_obs.paths)}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 1, goal, a1,
        {"kind": "file_list", "paths": list_obs.paths},
        current_mem, mem1, observation={"files": list_obs.paths},
    ))
    current_mem = mem1

    # Step 2: read_file(solution.py)
    read_obs = tool_read_file(ws, file_path)
    a2 = _make_action("read_file", f"read {file_path}", "file content",
                      ReadFileArgs(path=file_path))
    mem2 = current_mem.model_copy(update={"notes": f"read {file_path}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 2, goal, a2,
        {"kind": "file_content", "path": file_path, "line_count": read_obs.line_count},
        current_mem, mem2, observation={"path": file_path, "line_count": read_obs.line_count},
    ))
    current_mem = mem2

    # Step 3: run_tests (fails — baseline)
    test_obs = tool_run_tests(ws, timeout_s=10.0)
    a3 = _make_action("run_tests", "run baseline tests", "test results",
                      RunTestsArgs(test_path=None, timeout_s=10.0))
    mem3 = current_mem.model_copy(update={
        "last_test_summary": f"FAILED: {test_obs.num_failed} tests"
    })
    steps.append(_make_step(
        traj_id, task_id, ws_id, 3, goal, a3,
        {"kind": "test", "passed": test_obs.passed,
         "num_collected": test_obs.num_collected,
         "num_passed": test_obs.num_passed,
         "num_failed": test_obs.num_failed},
        current_mem, mem3,
        observation={"passed": test_obs.passed, "num_failed": test_obs.num_failed},
    ))
    current_mem = mem3

    # Step 4: inspect_error (last_test)
    err_obs = tool_inspect_error("last_test", test_obs, None)
    a4 = _make_action("inspect_error", "inspect test failure", "error content",
                      InspectErrorArgs(error_source="last_test"))
    mem4 = current_mem.model_copy(update={"hypothesis": err_obs.content[:200]})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 4, goal, a4,
        {"kind": "error", "source": "last_test", "content": err_obs.content},
        current_mem, mem4, observation={"source": "last_test"},
    ))
    current_mem = mem4

    # Step 5: propose_patch (correct)
    prop_obs = tool_propose_patch(ws, file_path, old_text, new_text)
    a5 = _make_action("propose_patch", "propose corrective patch", "patch proposal",
                      ProposePatchArgs(file_path=file_path, old_text=old_text, new_text=new_text))
    # No memory update specified for propose_patch per D6
    steps.append(_make_step(
        traj_id, task_id, ws_id, 5, goal, a5,
        {"kind": "patch_proposal", "would_succeed": prop_obs.would_succeed,
         "file_path": file_path},
        current_mem, current_mem, observation={"file_path": file_path},
    ))

    # Step 6: apply_patch (correct)
    patch_obs = tool_apply_patch(ws, file_path, old_text, new_text)
    a6 = _make_action("apply_patch", "apply corrective patch", "patch applied",
                      ApplyPatchArgs(file_path=file_path, old_text=old_text, new_text=new_text))
    mem6 = current_mem.model_copy(update={"notes": f"patched {file_path}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 6, goal, a6,
        {"kind": "patch", "success": patch_obs.success, "file_path": file_path},
        current_mem, mem6, observation={"file_path": file_path, "success": patch_obs.success},
    ))
    current_mem = mem6

    # Step 7: run_tests (passes)
    test_obs2 = tool_run_tests(ws, timeout_s=10.0)
    a7 = _make_action("run_tests", "verify fix with tests", "test results",
                      RunTestsArgs(test_path=None, timeout_s=10.0))
    mem7 = current_mem.model_copy(update={
        "last_test_summary": f"PASSED: {test_obs2.num_passed} tests"
    })
    steps.append(_make_step(
        traj_id, task_id, ws_id, 7, goal, a7,
        {"kind": "test", "passed": test_obs2.passed,
         "num_collected": test_obs2.num_collected,
         "num_passed": test_obs2.num_passed,
         "num_failed": test_obs2.num_failed},
        current_mem, mem7, observation={"passed": test_obs2.passed},
    ))
    current_mem = mem7

    # Step 8: write_memory (notes: "fixed <func>, tests pass")
    func_name = _extract_func_name(read_obs.content, old_text)
    final_mem = AgentMemory(
        notes=f"fixed {func_name}, tests pass",
        hypothesis=current_mem.hypothesis,
        failed_attempts=list(current_mem.failed_attempts),
        last_test_summary=current_mem.last_test_summary,
    )
    mem_obs = tool_write_memory(current_mem, final_mem)
    a8 = _make_action("write_memory", "record final state", "memory updated",
                      WriteMemoryArgs(memory=final_mem))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 8, goal, a8,
        {"kind": "memory"},
        mem_obs.memory_before, mem_obs.memory_after, observation={"step": 8},
    ))
    current_mem = mem_obs.memory_after

    # Step 9: finish(TEST_PASS, tests_passed=True)
    finish_obs = tool_finish(
        success_criterion=TaskSuccessCriterion.TEST_PASS,
        tests_passed=True,
        identification_verified=False,
        summary=f"Fixed {func_name} in {file_path}; all tests pass.",
    )
    a9 = _make_action("finish", "finish task", "task complete",
                      FinishArgs(success_criterion=TaskSuccessCriterion.TEST_PASS,
                                 tests_passed=True, identification_verified=False,
                                 summary=finish_obs.summary))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 9, goal, a9,
        {"kind": "finish"},
        current_mem, current_mem, success_label=True, observation={"step": 9},
    ))

    return steps


def _build_locate_steps(traj_id, task_id, ws_id, goal, ws, patch) -> list[TrajectoryStep]:
    """7-step pattern for type 1 (identify only, no patch, success_label=False)."""
    steps: list[TrajectoryStep] = []
    current_mem = AgentMemory()
    file_path = patch["file_path"]
    old_text = patch["old_text"]

    # Step 0: inspect_task
    task_obs = tool_inspect_task(ws)
    a0 = _make_action("inspect_task", "inspect task goal", "task fields", InspectTaskArgs())
    mem0 = current_mem.model_copy(update={"notes": task_obs.goal})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 0, goal, a0,
        {"kind": "task", "goal": task_obs.goal},
        current_mem, mem0, observation={"step": 0},
    ))
    current_mem = mem0

    # Step 1: list_files
    list_obs = tool_list_files(ws)
    a1 = _make_action("list_files", "list workspace files", "file list", ListFilesArgs())
    mem1 = current_mem.model_copy(update={"notes": f"files: {len(list_obs.paths)}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 1, goal, a1,
        {"kind": "file_list", "paths": list_obs.paths},
        current_mem, mem1, observation={"files": list_obs.paths},
    ))
    current_mem = mem1

    # Step 2: read_file(solution.py)
    read_obs = tool_read_file(ws, file_path)
    a2 = _make_action("read_file", f"read {file_path}", "file content",
                      ReadFileArgs(path=file_path))
    mem2 = current_mem.model_copy(update={"notes": f"read {file_path}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 2, goal, a2,
        {"kind": "file_content", "path": file_path, "line_count": read_obs.line_count},
        current_mem, mem2, observation={"path": file_path, "line_count": read_obs.line_count},
    ))
    current_mem = mem2

    # Step 3: run_tests (fails — baseline)
    test_obs = tool_run_tests(ws, timeout_s=10.0)
    a3 = _make_action("run_tests", "run baseline tests", "test results",
                      RunTestsArgs(test_path=None, timeout_s=10.0))
    mem3 = current_mem.model_copy(update={
        "last_test_summary": f"FAILED: {test_obs.num_failed} tests"
    })
    steps.append(_make_step(
        traj_id, task_id, ws_id, 3, goal, a3,
        {"kind": "test", "passed": test_obs.passed,
         "num_collected": test_obs.num_collected,
         "num_passed": test_obs.num_passed,
         "num_failed": test_obs.num_failed},
        current_mem, mem3,
        observation={"passed": test_obs.passed, "num_failed": test_obs.num_failed},
    ))
    current_mem = mem3

    # Step 4: inspect_error (last_test)
    err_obs = tool_inspect_error("last_test", test_obs, None)
    a4 = _make_action("inspect_error", "inspect test failure", "error content",
                      InspectErrorArgs(error_source="last_test"))
    mem4 = current_mem.model_copy(update={"hypothesis": err_obs.content[:200]})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 4, goal, a4,
        {"kind": "error", "source": "last_test", "content": err_obs.content},
        current_mem, mem4, observation={"source": "last_test"},
    ))
    current_mem = mem4

    # Step 5: write_memory (notes: "bug in <func>")
    func_name = _extract_func_name(read_obs.content, old_text)
    final_mem = AgentMemory(
        notes=f"bug in {func_name}",
        hypothesis=current_mem.hypothesis,
        failed_attempts=list(current_mem.failed_attempts),
        last_test_summary=current_mem.last_test_summary,
    )
    mem_obs = tool_write_memory(current_mem, final_mem)
    a5 = _make_action("write_memory", "record identified bug", "memory updated",
                      WriteMemoryArgs(memory=final_mem))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 5, goal, a5,
        {"kind": "memory"},
        mem_obs.memory_before, mem_obs.memory_after, observation={"step": 5},
    ))
    current_mem = mem_obs.memory_after

    # Step 6: finish(IDENTIFY_BUG, tests_passed=False, identification_verified=True)
    finish_obs = tool_finish(
        success_criterion=TaskSuccessCriterion.IDENTIFY_BUG,
        tests_passed=False,
        identification_verified=True,
        summary=f"Identified bug in {func_name}; tests still failing, no patch applied.",
    )
    a6 = _make_action("finish", "finish identification", "task complete",
                      FinishArgs(success_criterion=TaskSuccessCriterion.IDENTIFY_BUG,
                                 tests_passed=False, identification_verified=True,
                                 summary=finish_obs.summary))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 6, goal, a6,
        {"kind": "finish"},
        current_mem, current_mem, success_label=False, observation={"step": 6},
    ))

    return steps


def _build_avoid_edit_steps(traj_id, task_id, ws_id, goal, ws, patch) -> list[TrajectoryStep]:
    """11-step pattern for type 6 (extra read_file for test_solution.py)."""
    steps: list[TrajectoryStep] = []
    current_mem = AgentMemory()
    file_path = patch["file_path"]
    old_text = patch["old_text"]
    new_text = patch["new_text"]
    test_path = "test_solution.py"

    # Step 0: inspect_task
    task_obs = tool_inspect_task(ws)
    a0 = _make_action("inspect_task", "inspect task goal", "task fields", InspectTaskArgs())
    mem0 = current_mem.model_copy(update={"notes": task_obs.goal})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 0, goal, a0,
        {"kind": "task", "goal": task_obs.goal},
        current_mem, mem0, observation={"step": 0},
    ))
    current_mem = mem0

    # Step 1: list_files
    list_obs = tool_list_files(ws)
    a1 = _make_action("list_files", "list workspace files", "file list", ListFilesArgs())
    mem1 = current_mem.model_copy(update={"notes": f"files: {len(list_obs.paths)}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 1, goal, a1,
        {"kind": "file_list", "paths": list_obs.paths},
        current_mem, mem1, observation={"files": list_obs.paths},
    ))
    current_mem = mem1

    # Step 2: read_file(solution.py)
    read_obs = tool_read_file(ws, file_path)
    a2 = _make_action("read_file", f"read {file_path}", "file content",
                      ReadFileArgs(path=file_path))
    mem2 = current_mem.model_copy(update={"notes": f"read {file_path}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 2, goal, a2,
        {"kind": "file_content", "path": file_path, "line_count": read_obs.line_count},
        current_mem, mem2, observation={"path": file_path, "line_count": read_obs.line_count},
    ))
    current_mem = mem2

    # Step 3: read_file(test_solution.py) — confirm tests are correct (do not edit)
    test_read_obs = tool_read_file(ws, test_path)
    a3 = _make_action("read_file", f"read {test_path} (do not edit)", "file content",
                      ReadFileArgs(path=test_path))
    mem3 = current_mem.model_copy(update={"notes": f"read {test_path}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 3, goal, a3,
        {"kind": "file_content", "path": test_path, "line_count": test_read_obs.line_count},
        current_mem, mem3, observation={"path": test_path, "line_count": test_read_obs.line_count},
    ))
    current_mem = mem3

    # Step 4: run_tests (fails — baseline)
    test_obs = tool_run_tests(ws, timeout_s=10.0)
    a4 = _make_action("run_tests", "run baseline tests", "test results",
                      RunTestsArgs(test_path=None, timeout_s=10.0))
    mem4 = current_mem.model_copy(update={
        "last_test_summary": f"FAILED: {test_obs.num_failed} tests"
    })
    steps.append(_make_step(
        traj_id, task_id, ws_id, 4, goal, a4,
        {"kind": "test", "passed": test_obs.passed,
         "num_collected": test_obs.num_collected,
         "num_passed": test_obs.num_passed,
         "num_failed": test_obs.num_failed},
        current_mem, mem4,
        observation={"passed": test_obs.passed, "num_failed": test_obs.num_failed},
    ))
    current_mem = mem4

    # Step 5: inspect_error (last_test)
    err_obs = tool_inspect_error("last_test", test_obs, None)
    a5 = _make_action("inspect_error", "inspect test failure", "error content",
                      InspectErrorArgs(error_source="last_test"))
    mem5 = current_mem.model_copy(update={"hypothesis": err_obs.content[:200]})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 5, goal, a5,
        {"kind": "error", "source": "last_test", "content": err_obs.content},
        current_mem, mem5, observation={"source": "last_test"},
    ))
    current_mem = mem5

    # Step 6: propose_patch (correct)
    prop_obs = tool_propose_patch(ws, file_path, old_text, new_text)
    a6 = _make_action("propose_patch", "propose corrective patch", "patch proposal",
                      ProposePatchArgs(file_path=file_path, old_text=old_text, new_text=new_text))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 6, goal, a6,
        {"kind": "patch_proposal", "would_succeed": prop_obs.would_succeed,
         "file_path": file_path},
        current_mem, current_mem, observation={"file_path": file_path},
    ))

    # Step 7: apply_patch (correct)
    patch_obs = tool_apply_patch(ws, file_path, old_text, new_text)
    a7 = _make_action("apply_patch", "apply corrective patch", "patch applied",
                      ApplyPatchArgs(file_path=file_path, old_text=old_text, new_text=new_text))
    mem7 = current_mem.model_copy(update={"notes": f"patched {file_path}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 7, goal, a7,
        {"kind": "patch", "success": patch_obs.success, "file_path": file_path},
        current_mem, mem7, observation={"file_path": file_path, "success": patch_obs.success},
    ))
    current_mem = mem7

    # Step 8: run_tests (passes)
    test_obs2 = tool_run_tests(ws, timeout_s=10.0)
    a8 = _make_action("run_tests", "verify fix with tests", "test results",
                      RunTestsArgs(test_path=None, timeout_s=10.0))
    mem8 = current_mem.model_copy(update={
        "last_test_summary": f"PASSED: {test_obs2.num_passed} tests"
    })
    steps.append(_make_step(
        traj_id, task_id, ws_id, 8, goal, a8,
        {"kind": "test", "passed": test_obs2.passed,
         "num_collected": test_obs2.num_collected,
         "num_passed": test_obs2.num_passed,
         "num_failed": test_obs2.num_failed},
        current_mem, mem8, observation={"passed": test_obs2.passed},
    ))
    current_mem = mem8

    # Step 9: write_memory (notes: "fixed <func>, tests pass")
    func_name = _extract_func_name(read_obs.content, old_text)
    final_mem = AgentMemory(
        notes=f"fixed {func_name}, tests pass",
        hypothesis=current_mem.hypothesis,
        failed_attempts=list(current_mem.failed_attempts),
        last_test_summary=current_mem.last_test_summary,
    )
    mem_obs = tool_write_memory(current_mem, final_mem)
    a9 = _make_action("write_memory", "record final state", "memory updated",
                      WriteMemoryArgs(memory=final_mem))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 9, goal, a9,
        {"kind": "memory"},
        mem_obs.memory_before, mem_obs.memory_after, observation={"step": 9},
    ))
    current_mem = mem_obs.memory_after

    # Step 10: finish(TEST_PASS, tests_passed=True)
    finish_obs = tool_finish(
        success_criterion=TaskSuccessCriterion.TEST_PASS,
        tests_passed=True,
        identification_verified=False,
        summary=f"Fixed {func_name} in {file_path}; tests pass. Did not modify {test_path}.",
    )
    a10 = _make_action("finish", "finish task", "task complete",
                       FinishArgs(success_criterion=TaskSuccessCriterion.TEST_PASS,
                                  tests_passed=True, identification_verified=False,
                                  summary=finish_obs.summary))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 10, goal, a10,
        {"kind": "finish"},
        current_mem, current_mem, success_label=True, observation={"step": 10},
    ))

    return steps


def _find_non_unique_snippet(solution_content: str) -> str:
    """Find a short snippet in solution_content that appears 2+ times.

    Used to construct a wrong first patch for type 7 (recover_from_failed_patch).
    """
    candidates = ["return ", "if ", "return", "    ", "def "]
    for snip in candidates:
        if solution_content.count(snip) >= 2:
            return snip
    # Fallback: use the first 8 chars of any repeated character sequence
    for char in ["\n", " ", "("]:
        if solution_content.count(char) >= 2:
            return char
    raise ValueError("could not find a non-unique snippet in solution.py")


def _build_recover_steps(traj_id, task_id, ws_id, goal, ws, patch) -> list[TrajectoryStep]:
    """12-step pattern for type 7 (first patch fails, second succeeds)."""
    steps: list[TrajectoryStep] = []
    current_mem = AgentMemory()
    file_path = patch["file_path"]
    old_text = patch["old_text"]
    new_text = patch["new_text"]

    # Read solution content to find a non-unique snippet for the wrong patch
    sol_content = ws.resolve_path(file_path).read_text(encoding="utf-8")
    wrong_old_text = _find_non_unique_snippet(sol_content)
    assert sol_content.count(wrong_old_text) >= 2, (
        f"{task_id}: wrong_old_text {wrong_old_text!r} not non-unique "
        f"(count={sol_content.count(wrong_old_text)})"
    )
    wrong_new_text = wrong_old_text  # identity replacement (won't be applied anyway)

    # Step 0: inspect_task
    task_obs = tool_inspect_task(ws)
    a0 = _make_action("inspect_task", "inspect task goal", "task fields", InspectTaskArgs())
    mem0 = current_mem.model_copy(update={"notes": task_obs.goal})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 0, goal, a0,
        {"kind": "task", "goal": task_obs.goal},
        current_mem, mem0, observation={"step": 0},
    ))
    current_mem = mem0

    # Step 1: list_files
    list_obs = tool_list_files(ws)
    a1 = _make_action("list_files", "list workspace files", "file list", ListFilesArgs())
    mem1 = current_mem.model_copy(update={"notes": f"files: {len(list_obs.paths)}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 1, goal, a1,
        {"kind": "file_list", "paths": list_obs.paths},
        current_mem, mem1, observation={"files": list_obs.paths},
    ))
    current_mem = mem1

    # Step 2: read_file(solution.py)
    read_obs = tool_read_file(ws, file_path)
    a2 = _make_action("read_file", f"read {file_path}", "file content",
                      ReadFileArgs(path=file_path))
    mem2 = current_mem.model_copy(update={"notes": f"read {file_path}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 2, goal, a2,
        {"kind": "file_content", "path": file_path, "line_count": read_obs.line_count},
        current_mem, mem2, observation={"path": file_path, "line_count": read_obs.line_count},
    ))
    current_mem = mem2

    # Step 3: run_tests (fails — baseline)
    test_obs = tool_run_tests(ws, timeout_s=10.0)
    a3 = _make_action("run_tests", "run baseline tests", "test results",
                      RunTestsArgs(test_path=None, timeout_s=10.0))
    mem3 = current_mem.model_copy(update={
        "last_test_summary": f"FAILED: {test_obs.num_failed} tests"
    })
    steps.append(_make_step(
        traj_id, task_id, ws_id, 3, goal, a3,
        {"kind": "test", "passed": test_obs.passed,
         "num_collected": test_obs.num_collected,
         "num_passed": test_obs.num_passed,
         "num_failed": test_obs.num_failed},
        current_mem, mem3,
        observation={"passed": test_obs.passed, "num_failed": test_obs.num_failed},
    ))
    current_mem = mem3

    # Step 4: inspect_error (last_test)
    err_obs = tool_inspect_error("last_test", test_obs, None)
    a4 = _make_action("inspect_error", "inspect test failure", "error content",
                      InspectErrorArgs(error_source="last_test"))
    mem4 = current_mem.model_copy(update={"hypothesis": err_obs.content[:200]})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 4, goal, a4,
        {"kind": "error", "source": "last_test", "content": err_obs.content},
        current_mem, mem4, observation={"source": "last_test"},
    ))
    current_mem = mem4

    # Step 5: propose_patch (WRONG — non-unique old_text)
    prop_wrong = tool_propose_patch(ws, file_path, wrong_old_text, wrong_new_text)
    a5 = _make_action("propose_patch", "propose (wrong) patch", "patch proposal",
                      ProposePatchArgs(file_path=file_path, old_text=wrong_old_text,
                                       new_text=wrong_new_text))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 5, goal, a5,
        {"kind": "patch_proposal", "would_succeed": prop_wrong.would_succeed,
         "file_path": file_path, "error": prop_wrong.error},
        current_mem, current_mem, observation={"file_path": file_path},
    ))

    # Step 6: apply_patch (WRONG — fails with "old_text not unique")
    patch_wrong = tool_apply_patch(ws, file_path, wrong_old_text, wrong_new_text)
    a6 = _make_action("apply_patch", "apply (wrong) patch", "patch failed",
                      ApplyPatchArgs(file_path=file_path, old_text=wrong_old_text,
                                     new_text=wrong_new_text))
    failed_attempts = list(current_mem.failed_attempts)
    failed_attempts.append(f"patch {file_path} failed: {patch_wrong.error}")
    mem6 = current_mem.model_copy(update={"failed_attempts": failed_attempts})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 6, goal, a6,
        {"kind": "patch", "success": patch_wrong.success, "file_path": file_path,
         "error": patch_wrong.error},
        current_mem, mem6,
        observation={"file_path": file_path, "success": patch_wrong.success},
    ))
    current_mem = mem6

    # Step 7: inspect_error (last_patch)
    err_patch_obs = tool_inspect_error("last_patch", None, patch_wrong)
    a7 = _make_action("inspect_error", "inspect patch failure", "error content",
                      InspectErrorArgs(error_source="last_patch"))
    mem7 = current_mem.model_copy(update={"hypothesis": err_patch_obs.content[:200]})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 7, goal, a7,
        {"kind": "error", "source": "last_patch", "content": err_patch_obs.content},
        current_mem, mem7, observation={"source": "last_patch"},
    ))
    current_mem = mem7

    # Step 8: propose_patch (correct)
    prop_correct = tool_propose_patch(ws, file_path, old_text, new_text)
    a8 = _make_action("propose_patch", "propose corrective patch", "patch proposal",
                      ProposePatchArgs(file_path=file_path, old_text=old_text,
                                       new_text=new_text))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 8, goal, a8,
        {"kind": "patch_proposal", "would_succeed": prop_correct.would_succeed,
         "file_path": file_path},
        current_mem, current_mem, observation={"file_path": file_path},
    ))

    # Step 9: apply_patch (correct — succeeds)
    patch_correct = tool_apply_patch(ws, file_path, old_text, new_text)
    a9 = _make_action("apply_patch", "apply corrective patch", "patch applied",
                      ApplyPatchArgs(file_path=file_path, old_text=old_text,
                                     new_text=new_text))
    mem9 = current_mem.model_copy(update={"notes": f"patched {file_path}"})
    steps.append(_make_step(
        traj_id, task_id, ws_id, 9, goal, a9,
        {"kind": "patch", "success": patch_correct.success, "file_path": file_path},
        current_mem, mem9,
        observation={"file_path": file_path, "success": patch_correct.success},
    ))
    current_mem = mem9

    # Step 10: run_tests (passes)
    test_obs2 = tool_run_tests(ws, timeout_s=10.0)
    a10 = _make_action("run_tests", "verify fix with tests", "test results",
                       RunTestsArgs(test_path=None, timeout_s=10.0))
    mem10 = current_mem.model_copy(update={
        "last_test_summary": f"PASSED: {test_obs2.num_passed} tests"
    })
    steps.append(_make_step(
        traj_id, task_id, ws_id, 10, goal, a10,
        {"kind": "test", "passed": test_obs2.passed,
         "num_collected": test_obs2.num_collected,
         "num_passed": test_obs2.num_passed,
         "num_failed": test_obs2.num_failed},
        current_mem, mem10, observation={"passed": test_obs2.passed},
    ))
    current_mem = mem10

    # Step 11: finish(TEST_PASS, tests_passed=True)
    func_name = _extract_func_name(read_obs.content, old_text)
    finish_obs = tool_finish(
        success_criterion=TaskSuccessCriterion.TEST_PASS,
        tests_passed=True,
        identification_verified=False,
        summary=f"Recovered from failed patch; fixed {func_name} in {file_path}.",
    )
    a11 = _make_action("finish", "finish task", "task complete",
                       FinishArgs(success_criterion=TaskSuccessCriterion.TEST_PASS,
                                  tests_passed=True, identification_verified=False,
                                  summary=finish_obs.summary))
    steps.append(_make_step(
        traj_id, task_id, ws_id, 11, goal, a11,
        {"kind": "finish"},
        current_mem, current_mem, success_label=True, observation={"step": 11},
    ))

    return steps


def _load_task_types() -> dict[str, str]:
    manifest = json.loads((TASKS_DIR / "manifest.json").read_text())
    return {t["task_id"]: t["task_type"] for t in manifest["tasks"]}


_TYPE_PATTERNS = {
    "locate_failing_function": ("identify only", 7),
    "one_line_fix": ("standard", 10),
    "add_boundary_check": ("standard", 10),
    "update_helper": ("standard", 10),
    "repair_after_pytest": ("standard", 10),
    "avoid_editing_tests": ("standard + extra read", 11),
    "recover_from_failed_patch": ("failed patch + recover", 12),
    "finish_after_tests_pass": ("standard", 10),
}


def _write_report(trajectories, task_types):
    total_trajs = len(trajectories)
    total_steps = sum(len(t.steps) for t in trajectories)
    mean_steps = total_steps / total_trajs if total_trajs else 0
    success_count = sum(1 for t in trajectories if t.final_success)
    identify_count = total_trajs - success_count

    # Aggregate tool distribution
    tool_dist: dict[str, int] = {}
    for traj in trajectories:
        for at, cnt in traj.tool_distribution.items():
            # Normalize enum keys (str subclass) to plain strings
            key = at.value if hasattr(at, "value") else str(at)
            tool_dist[key] = tool_dist.get(key, 0) + cnt

    # Per-task-type breakdown
    type_counts: dict[str, list[int]] = {}
    for traj in trajectories:
        ttype = task_types[traj.task_id]
        type_counts.setdefault(ttype, []).append(len(traj.steps))

    lines = []
    lines.append("# P4.0 Phase F — Scripted Teacher Trajectories Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total trajectories: {total_trajs}")
    lines.append(f"- Total steps: {total_steps}")
    lines.append(f"- Mean steps per trajectory: {mean_steps:.2f}")
    lines.append(f"- Success (types 2-8): {success_count}")
    lines.append(f"- Identify-only (type 1): {identify_count}")
    lines.append("")
    lines.append("## Tool distribution (aggregate)")
    lines.append("| Action type | Count |")
    lines.append("|---|---|")
    for at in sorted(tool_dist.keys()):
        lines.append(f"| {at} | {tool_dist[at]} |")
    lines.append("")
    lines.append("## Per-task-type breakdown")
    lines.append("| Type | Tasks | Steps | Pattern |")
    lines.append("|---|---|---|---|")
    for ttype in [
        "locate_failing_function", "one_line_fix", "add_boundary_check",
        "update_helper", "repair_after_pytest", "avoid_editing_tests",
        "recover_from_failed_patch", "finish_after_tests_pass",
    ]:
        if ttype in type_counts:
            counts = type_counts[ttype]
            pattern, _expected = _TYPE_PATTERNS[ttype]
            steps_desc = f"{counts[0]} each" if len(set(counts)) == 1 else f"varies ({min(counts)}-{max(counts)})"
            lines.append(f"| {ttype} | {len(counts)} | {steps_desc} | {pattern} |")
    lines.append("")
    lines.append("## Generation")
    lines.append("- Script: scripts/generate_scripted_agent_trajectories.py")
    lines.append("- Output: data/p4-agent/trajectories-v0/scripted.jsonl")
    lines.append("- Format: one Trajectory JSON object per line (40 lines)")
    lines.append("- All steps verified=True (real tool execution)")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    task_types = _load_task_types()
    trajectories = []
    for task_id, task_type in task_types.items():
        traj = _build_trajectory(task_id, task_type)
        trajectories.append(traj)
        print(f"  {task_id} ({task_type}): {len(traj.steps)} steps, success={traj.final_success}")

    # Write JSONL (one Trajectory per line)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(traj.model_dump_json() + "\n")

    # Write report
    _write_report(trajectories, task_types)
    print(f"\nWrote {len(trajectories)} trajectories to {OUT_PATH}")


if __name__ == "__main__":
    main()
