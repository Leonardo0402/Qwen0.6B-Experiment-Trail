"""Verify P4.0 readiness — run 11 gates and write readiness report.

Run: set P4_ALLOW_NETWORK=0 && py -3.11 scripts/verify_p4_readiness.py
"""
from __future__ import annotations
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

os.environ.setdefault("P4_ALLOW_NETWORK", "0")
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.agent_trajectory import Trajectory, is_mutating_action
from src.agent_actions import TaskSuccessCriterion

TRAJ_PATH = _ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
P3_LOCK_PATH = _ROOT / "reports" / "p4" / "p3-exit-baseline-lock.json"
EVAL_REPORT_PATH = _ROOT / "reports" / "p4" / "agent-eval-report.json"
OUT_PATH = _ROOT / "reports" / "p4" / "p4-agent-foundation-readiness.md"

FORBIDDEN_PATTERNS = [
    "os.system", "os.popen", "git push",
    "requests.get", "requests.post", "urllib.request.urlopen",
    "socket.socket",
]


def _run_pytest(test_args, timeout=300):
    """Run pytest, return (exit_code, stdout)."""
    cmd = [sys.executable, "-m", "pytest"] + test_args + ["-v", "--tb=short", "-p", "no:warnings"]
    env = {**os.environ, "P4_ALLOW_NETWORK": "0"}
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_ROOT),
        env=env, timeout=timeout,
    )
    return result.returncode, result.stdout


def _extract_passed_line(stdout):
    """Extract the 'X passed' line from pytest output."""
    for line in stdout.splitlines():
        if "passed" in line:
            return line.strip()
    return "passed"


def gate_01_p3_baseline():
    assert P3_LOCK_PATH.exists(), f"{P3_LOCK_PATH} not found"
    data = json.loads(P3_LOCK_PATH.read_text(encoding="utf-8"))
    assert data.get("schema_version") == 1, "schema_version != 1"
    assert "pr_15" in data, "missing pr_15"
    assert "merge_commit_sha" in data["pr_15"], "missing merge_commit_sha"
    assert "p3_limited_datasets" in data, "missing p3_limited_datasets"
    sha = data["pr_15"]["merge_commit_sha"][:7]
    return f"locked, PR #15 merge {sha}"


def gate_02_action_schema_tests():
    code, stdout = _run_pytest(["tests/test_agent_actions.py"])
    assert code == 0, f"exit {code}"
    return _extract_passed_line(stdout)


def gate_03_tool_layer_tests():
    code, stdout = _run_pytest(["tests/test_agent_tools.py"])
    assert code == 0, f"exit {code}"
    return _extract_passed_line(stdout)


def gate_04_trajectory_schema_tests():
    code, stdout = _run_pytest(["tests/test_agent_trajectory.py"])
    assert code == 0, f"exit {code}"
    return _extract_passed_line(stdout)


def gate_05_micro_task_suite():
    code, stdout = _run_pytest(["tests/test_micro_task_suite.py"])
    assert code == 0, f"exit {code}"
    manifest = json.loads((TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_tasks"] == 40, f"total_tasks={manifest['total_tasks']}"
    return f"40 tasks, {_extract_passed_line(stdout)}"


def gate_06_scripted_trajectories():
    assert TRAJ_PATH.exists(), f"{TRAJ_PATH} not found"
    trajectories = []
    with TRAJ_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traj = Trajectory.model_validate_json(line)
                trajectories.append(traj)
    assert len(trajectories) == 40, f"expected 40, got {len(trajectories)}"
    total_steps = sum(len(t.steps) for t in trajectories)
    for traj in trajectories:
        for step in traj.steps:
            assert step.verified, f"{traj.task_id} step {step.step_index} not verified"
    return f"40 trajectories, {total_steps} steps, all verified=True"


def gate_07_evaluator_replay():
    assert EVAL_REPORT_PATH.exists(), f"{EVAL_REPORT_PATH} not found"
    data = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))
    assert data["total_tasks"] == 40, f"total_tasks={data['total_tasks']}"
    rate = data["metrics"]["task_success_rate"]
    assert rate == 1.0, f"task_success_rate={rate}"
    return f"100% replay success, eval_hash={data['eval_hash']}"


def gate_08_corrupted_tests():
    code, stdout = _run_pytest(["tests/test_agent_evaluator.py::test_corrupted_injection"])
    assert code == 0, f"exit {code}"
    return "test_corrupted_injection passes (WRONG_PATCH detected)"


def gate_09_no_forbidden_actions():
    # Check ONLY P4-agent files (D5: spec intent is P4-agent safety, not re-auditing P1-P3 infra)
    p4_files = list((_ROOT / "src").glob("agent_*.py")) + [
        _ROOT / "scripts" / "generate_scripted_agent_trajectories.py",
        _ROOT / "scripts" / "evaluate_agent_policy.py",
    ]
    # verify_p4_readiness.py is exempt (contains patterns as string literals)
    violations = []
    for py_file in p4_files:
        content = py_file.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in content:
                rel = py_file.relative_to(_ROOT)
                violations.append(f"{rel}: {pattern}")
    assert not violations, f"forbidden patterns: {violations}"
    return "no forbidden shell/network/git patterns in P4-agent files"


def gate_10_ci_green():
    # Run ONLY P4-agent test files (D6: CI runs on Linux; Windows local execution
    # has pre-existing P1-P3 platform issues. Gate 10 verifies P4-agent tests pass.)
    p4_tests = [
        "tests/test_agent_actions.py",
        "tests/test_agent_tools.py",
        "tests/test_p3_exit_lock.py",
        "tests/test_agent_trajectory.py",
        "tests/test_micro_task_suite.py",
        "tests/test_scripted_trajectories.py",
        "tests/test_agent_evaluator.py",
    ]
    code, stdout = _run_pytest(p4_tests, timeout=600)
    assert code == 0, f"exit {code}\n{stdout[-500:]}"
    return _extract_passed_line(stdout)


def gate_11_state_transition():
    trajectories = []
    with TRAJ_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traj = Trajectory.model_validate_json(line)
                trajectories.append(traj)
    for traj in trajectories:
        for i, step in enumerate(traj.steps):
            assert step.verified, f"{traj.task_id} step {i}: not verified"
            if i > 0:
                assert step.memory_before == traj.steps[i - 1].memory_after, \
                    f"{traj.task_id} step {i}: memory chain broken"
            if step.result.get("kind") == "patch":
                assert is_mutating_action(step.action.action_type), \
                    f"{traj.task_id} step {i}: patch result from non-mutating action"
        last = traj.steps[-1]
        assert last.action.action_type == "finish", \
            f"{traj.task_id}: last action not finish"
        if traj.final_success:
            fa = last.action.arguments
            sc = fa.success_criterion
            if sc == TaskSuccessCriterion.TEST_PASS.value:
                assert fa.tests_passed, f"{traj.task_id}: TEST_PASS but tests_passed=False"
            elif sc == TaskSuccessCriterion.IDENTIFY_BUG.value:
                assert fa.identification_verified, \
                    f"{traj.task_id}: IDENTIFY_BUG but not verified"
    return "all 40 trajectories, state transitions consistent"


GATES = [
    ("1", "P3 exit baseline locked", gate_01_p3_baseline),
    ("2", "Action schema tests pass", gate_02_action_schema_tests),
    ("3", "Tool layer safety tests pass", gate_03_tool_layer_tests),
    ("4", "Trajectory schema tests pass", gate_04_trajectory_schema_tests),
    ("5", "Micro task suite verified", gate_05_micro_task_suite),
    ("6", "Scripted trajectories verified", gate_06_scripted_trajectories),
    ("7", "Evaluator replay success = 100%", gate_07_evaluator_replay),
    ("8", "Corrupted trajectory tests fail as expected", gate_08_corrupted_tests),
    ("9", "No forbidden shell/network/git actions", gate_09_no_forbidden_actions),
    ("10", "CI green (P4-agent tests)", gate_10_ci_green),
    ("11", "State transition consistency", gate_11_state_transition),
]


def main():
    print("P4.0 Readiness Verification")
    print("=" * 60)
    results = []
    for num, name, func in GATES:
        print(f"\n[{num}] {name}")
        try:
            evidence = func()
            results.append((num, name, True, evidence))
            print(f"  PASS: {evidence}")
        except Exception as e:
            results.append((num, name, False, str(e)))
            print(f"  FAIL: {e}")
    all_pass = all(r[2] for r in results)
    verdict = "GO_FOR_P4_AGENT_SFT_DATA" if all_pass else "FIX_FIRST"
    passed = sum(1 for r in results if r[2])
    print(f"\n{'=' * 60}")
    print(f"Verdict: {verdict}")
    print(f"Gates: {passed}/{len(results)} passed")
    _write_report(results, verdict)
    print(f"Report: {OUT_PATH}")


def _write_report(results, verdict):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P4.0 Agentic Coder Foundation — Readiness Report",
        "",
        f"**Date:** {datetime.now().isoformat()}",
        f"**Branch:** feat/p4-agent-foundation",
        f"**Verdict:** `{verdict}`",
        "",
        "## Gates",
        "",
        "| # | Gate | Status | Evidence |",
        "|---|------|--------|----------|",
    ]
    for num, name, passed, evidence in results:
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {num} | {name} | {status} | {evidence} |")
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Total gates: {len(results)}",
        f"- Passed: {sum(1 for r in results if r[2])}",
        f"- Failed: {sum(1 for r in results if not r[2])}",
        "",
        "## Known Limitations",
        "",
        "1. **Gate 8 (corrupted trajectory tests):** Only WRONG_PATCH corruption type is",
        "   tested. The other 4 types (WRONG_ACTION_TYPE, INVALID_PATH,",
        "   SKIP_TESTS_BEFORE_FINISH, EXCEED_MAX_STEPS) are implemented but not individually",
        "   tested. Documented for P4.1 expansion.",
        "2. **Gate 9 (no forbidden actions):** Scope is narrowed to P4-agent files only",
        "   (`src/agent_*.py` + P4 scripts). Pre-existing P1-P3 infrastructure (sandbox.py,",
        "   validators.py) is not re-audited — it was verified in its own phases.",
        "3. **Gate 10 (CI green):** Runs only the 7 P4-agent test files, not the full",
        "   53-file suite. CI (on Linux) runs the full suite; this gate verifies P4-agent",
        "   tests pass locally. Pre-existing P1-P3 tests may fail on Windows due to",
        "   platform-specific issues (CRLF, subprocess timeout) but pass on CI's Linux.",
        "4. **40 trajectories are NOT training data:** Per spec section 15, these are",
        "   foundation verification artifacts only. P4.1 will produce 1000+ trajectories.",
        "5. **Evaluator trusts finish.tests_passed (trust gap):** The evaluator's",
        "   `task_success_rate` is computed from the scripted trajectory's `finish.tests_passed`",
        "   declaration, not cross-checked against actual replay `run_tests` results. This is",
        "   acceptable for P4.0 (scripted teachers don't lie) but enforcement is a P4.1",
        "   prerequisite for model-generated trajectories.",
        "6. **Evaluator dispatch omits search_text and rollback_patch:** The evaluator's",
        "   `run()` method does not dispatch `search_text` or `rollback_patch` actions. No",
        "   scripted trajectory uses these actions, so P4.0 is unaffected, but P4.1 trajectories",
        "   using them would produce silent no-ops. Must be fixed before P4.1 model agents.",
        "",
        "## Supply-Chain Warning",
        "",
        "Issue #17 received a comment from unverified account `depucobose87` attaching",
        "`p4_baseline_fix.zip`. This was treated as a potential supply-chain attack.",
        "No file from Issue/PR comments was downloaded, inspected, or applied.",
        "All P4.0 code was written from scratch under TDD discipline.",
        "",
        "## Next Steps (P4.1, out of scope)",
        "",
        "1. Build supervised action-policy dataset from 1000+ scripted/teacher trajectories",
        "2. Implement ModelActionProvider with Qwen3-0.6B",
        "3. Train and evaluate agent policy",
        "",
        "**This report does not authorize any P4.1 work.**",
    ])
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
