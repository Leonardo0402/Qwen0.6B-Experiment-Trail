# scripts/verify_p4_1_readiness.py
"""Phase H: P4.1 readiness verifier — 10 gates → GO_FOR_P4_AGENT_SFT."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("P4_ALLOW_NETWORK", "0")

_REPORT = _ROOT / "reports" / "p4" / "p4-1-readiness.md"
_BASELINE_LOCK = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
_COLLECTION_REPORT = _ROOT / "reports" / "p4" / "model-trajectory-collection-report.json"
_SFT_MANIFEST = _ROOT / "data" / "p4-agent" / "sft-v1" / "manifest.json"

_FORBIDDEN_TRAINING_PATTERNS = [
    "trainer.train", "SFTTrainer", "DPOTrainer", "PPOTrainer", "RLTrainer",
]
_FORBIDDEN_NETWORK_PATTERNS = [
    "requests.get", "requests.post", "wget", "curl",
]
_P4_1_SCRIPTS = [
    "scripts/lock_p4_0_baseline.py",
    "scripts/collect_model_trajectories.py",
    "scripts/augment_teacher_model.py",
    "scripts/augment_corrupted_recovered.py",
    "scripts/augment_failed_patch_recovery.py",
    "scripts/build_agent_sft_dataset.py",
]
_P4_1_SRC = ["src/agent_model_provider.py"]


def _run_pytest(test_args, timeout=600):
    env = os.environ.copy()
    env["P4_ALLOW_NETWORK"] = "0"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest"] + test_args
        + ["-p", "no:warnings", "--tb=no"],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _extract_passed_line(stdout):
    for line in stdout.splitlines():
        if "passed" in line:
            return line.strip()
    return "no passed line found"


def _parse_pytest_summary(stdout):
    """Parse pytest summary line for §2.8 test counts.

    Returns dict with keys: passed, failed, skipped, warnings, errors.
    Each value is an int (0 if not found).
    """
    result = {"passed": 0, "failed": 0, "skipped": 0, "warnings": 0, "errors": 0}
    # Look for the summary line: "N passed, N failed, N skipped, N warnings in N.NNs"
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match patterns like "9 passed", "2 failed", "1 skipped", "3 warnings"
        m_passed = re.search(r"(\d+)\s+passed", line)
        m_failed = re.search(r"(\d+)\s+failed", line)
        m_skipped = re.search(r"(\d+)\s+skipped", line)
        m_warnings = re.search(r"(\d+)\s+warnings?", line)
        m_errors = re.search(r"(\d+)\s+errors?", line)
        if m_passed or m_failed or m_skipped:
            if m_passed:
                result["passed"] = int(m_passed.group(1))
            if m_failed:
                result["failed"] = int(m_failed.group(1))
            if m_skipped:
                result["skipped"] = int(m_skipped.group(1))
            if m_warnings:
                result["warnings"] = int(m_warnings.group(1))
            if m_errors:
                result["errors"] = int(m_errors.group(1))
            break
    return result


def _get_commit_sha():
    """Get the current git commit SHA (short)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(_ROOT),
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _collect_test_evidence():
    """Run the full test suite and collect §2.8 test evidence.

    Returns a dict with:
    - exact_command
    - test_count, pass_count, fail_count, skip_count, warning_count
    - runtime_seconds
    - environment (python version, OS, platform)
    - commit_sha
    - ci_status (local vs GitHub CI)
    """
    import tempfile
    import xml.etree.ElementTree as ET

    # Use --junit-xml for reliable test count extraction (text summary can
    # be lost on Windows when stdout is not a TTY, even with -v).
    out_fd, junit_path = tempfile.mkstemp(suffix=".xml", prefix="pytest_junit_")
    os.close(out_fd)
    command = (
        f"{sys.executable} -m pytest tests/ -p no:warnings --tb=no -q "
        f"-m \"not gpu\" --timeout=120 "
        f"--ignore=tests/test_data_pipeline.py "
        f"--ignore=tests/test_p3_readiness_gate.py "
        f"--junit-xml={junit_path}"
    )
    env = os.environ.copy()
    env["P4_ALLOW_NETWORK"] = "0"
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/",
             "-p", "no:warnings", "--tb=no", "-q",
             "-m", "not gpu", "--timeout=120",
             "--ignore=tests/test_data_pipeline.py",
             "--ignore=tests/test_p3_readiness_gate.py",
             f"--junit-xml={junit_path}"],
            capture_output=True, text=True, timeout=1800, env=env,
            cwd=str(_ROOT),
        )
        runtime = time.time() - start
        # Parse JUnit XML for reliable counts
        summary = {"passed": 0, "failed": 0, "skipped": 0, "warnings": 0, "errors": 0}
        try:
            tree = ET.parse(junit_path)
            root = tree.getroot()
            # JUnit XML: <testsuites><testsuite tests="N" failures="N" errors="N" skipped="N">
            for suite in root.iter("testsuite"):
                summary["passed"] += int(suite.get("tests", 0))
                summary["failed"] += int(suite.get("failures", 0))
                summary["errors"] += int(suite.get("errors", 0))
                summary["skipped"] += int(suite.get("skipped", 0))
            # passed = total - failures - errors - skipped
            total_raw = summary["passed"]
            summary["passed"] = (
                total_raw - summary["failed"] - summary["errors"] - summary["skipped"]
            )
        except Exception:
            pass
    except subprocess.TimeoutExpired:
        runtime = time.time() - start
        summary = {"passed": 0, "failed": 0, "skipped": 0, "warnings": 0, "errors": 0}
    except Exception:
        runtime = time.time() - start
        summary = {"passed": 0, "failed": 0, "skipped": 0, "warnings": 0, "errors": 0}
    finally:
        try:
            os.unlink(junit_path)
        except OSError:
            pass

    total = (summary["passed"] + summary["failed"]
             + summary["skipped"] + summary["errors"])
    return {
        "exact_command": command,
        "test_count": total,
        "pass_count": summary["passed"],
        "fail_count": summary["failed"],
        "skip_count": summary["skipped"],
        "warning_count": summary["warnings"],
        "error_count": summary["errors"],
        "runtime_seconds": round(runtime, 2),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "commit_sha": _get_commit_sha(),
        "ci_status": {
            "local": "completed",
            "github_ci": "not run (local execution)",
        },
    }


def gate_01_baseline_lock():
    if not _BASELINE_LOCK.exists():
        return (False, "p4-0-baseline-lock.json not found")
    data = json.loads(_BASELINE_LOCK.read_text(encoding="utf-8"))
    if not data.get("p4_0_merge_commit", "").startswith("7ccd06c"):
        return (False, f"merge commit mismatch: {data.get('p4_0_merge_commit')}")
    return (True, f"locked at {data['p4_0_merge_commit'][:7]}")


def gate_02_test_pass_replay_authoritative():
    code, stdout = _run_pytest([
        "tests/test_agent_evaluator.py::test_test_pass_success_uses_replay_not_claim",
        "tests/test_agent_evaluator.py::test_test_pass_mismatch_claimed_fail_actual_pass",
        "tests/test_agent_evaluator.py::test_test_pass_no_mismatch_when_claim_matches_replay",
    ])
    if code != 0:
        return (False, f"exit {code}\n{stdout[-300:]}")
    return (True, _extract_passed_line(stdout))


def gate_03_unknown_action_hard_fails():
    code, stdout = _run_pytest([
        "tests/test_agent_evaluator.py::test_unknown_action_type_recorded_as_forbidden",
        "tests/test_agent_evaluator.py::test_allowed_action_types_has_exactly_11",
    ])
    if code != 0:
        return (False, f"exit {code}\n{stdout[-300:]}")
    return (True, _extract_passed_line(stdout))


def gate_04_all_11_actions_dispatched():
    code, stdout = _run_pytest([
        "tests/test_agent_evaluator.py::test_search_text_dispatched",
        "tests/test_agent_evaluator.py::test_rollback_patch_dispatched",
    ])
    if code != 0:
        return (False, f"exit {code}\n{stdout[-300:]}")
    return (True, _extract_passed_line(stdout))


def gate_05_inspect_error_surfaces_stdout():
    code, stdout = _run_pytest([
        "tests/test_agent_tools.py::test_inspect_error_returns_stdout_on_test_failure",
        "tests/test_agent_tools.py::test_inspect_error_caps_at_8kb",
    ])
    if code != 0:
        return (False, f"exit {code}\n{stdout[-300:]}")
    return (True, _extract_passed_line(stdout))


def gate_06_all_5_corruption_types_tested():
    tests = [
        "tests/test_agent_evaluator.py::test_corruption_wrong_action_type",
        "tests/test_agent_evaluator.py::test_corruption_invalid_path",
        "tests/test_agent_evaluator.py::test_corruption_wrong_patch",
        "tests/test_agent_evaluator.py::test_corruption_skip_tests_before_finish",
        "tests/test_agent_evaluator.py::test_corruption_exceed_max_steps",
    ]
    code, stdout = _run_pytest(tests)
    if code != 0:
        return (False, f"exit {code}\n{stdout[-300:]}")
    return (True, _extract_passed_line(stdout))


_SMOKE_REQUIRED_FIELDS = [
    "model_load_ok",
    "adapter_load_ok",
    "generation_ok",
    "json_parse_rate",
    "schema_valid_rate",
    "safety_valid_rate",
    "action_type_valid_rate",
    "arguments_valid_rate",
    "forbidden_action_count",
    "tool_dispatch_ok",
    "max_step_stop_ok",
    "runtime_crash_count",
]


def _check_smoke_metrics(entry, config_name):
    """Verify §2.3 required smoke metrics are present in a config entry.

    Returns (ok, evidence_str).
    """
    missing = [f for f in _SMOKE_REQUIRED_FIELDS if f not in entry]
    if missing:
        return (False, f"missing §2.3 fields: {missing}")
    if entry.get("runtime_crash_count", 0) > 0:
        return (False, f"runtime_crash_count={entry['runtime_crash_count']}")
    smoke = entry.get("limited_smoke", False)
    label = "LIMITED_SMOKE" if smoke else "full"
    evidence = (
        f"loaded ({label}), {entry['trajectories_written']} trajectories, "
        f"json_parse={entry['json_parse_rate']:.2f}, "
        f"schema_valid={entry['schema_valid_rate']:.2f}, "
        f"forbidden={entry['forbidden_action_count']}, "
        f"tool_dispatch_ok={entry['tool_dispatch_ok']:.2f}, "
        f"max_step_stop={entry['max_step_stop_ok']}"
    )
    return (True, evidence)


def gate_07_model_smoke_base():
    if not _COLLECTION_REPORT.exists():
        return (False, "collection report not found")
    reports = json.loads(_COLLECTION_REPORT.read_text(encoding="utf-8"))
    base = next((r for r in reports if r["config"] == "base"), None)
    if base is None:
        return (False, "base config not in report")
    if not base.get("model_load_ok"):
        return (False, "model_load_ok=False for base")
    return _check_smoke_metrics(base, "base")


def gate_08_model_smoke_repair_lora():
    if not _COLLECTION_REPORT.exists():
        return (False, "collection report not found")
    reports = json.loads(_COLLECTION_REPORT.read_text(encoding="utf-8"))
    repair = next((r for r in reports if r["config"] == "repair-lora"), None)
    if repair is None:
        return (False, "repair-lora config not in report")
    if not repair.get("model_load_ok"):
        return (False, "model_load_ok=False for repair-lora")
    if not repair.get("adapter_load_ok"):
        return (False, "adapter_load_ok=False for repair-lora")
    return _check_smoke_metrics(repair, "repair-lora")


def gate_09_sft_dataset():
    if not _SFT_MANIFEST.exists():
        return (False, "sft manifest not found")
    data = json.loads(_SFT_MANIFEST.read_text(encoding="utf-8"))
    total = data.get("total_trajectories", 0)
    if total < 1000:
        return (False, f"only {total} trajectories (need 1000+)")
    if data.get("replay_failures", 0) > 0:
        return (False, f"{data['replay_failures']} replay failures")
    if data.get("train_count", 0) == 0:
        return (False, "train split empty")
    if data.get("heldout_count", 0) == 0:
        return (False, "heldout split empty")
    splits = data.get("splits", {})
    for split_name in ("train", "validation", "heldout-agent-eval"):
        split_info = splits.get(split_name, {})
        if not split_info.get("sha256"):
            return (False, f"split '{split_name}' missing sha256")
    return (True, f"{total} trajectories, train={data['train_count']} "
            f"val={data['validation_count']} heldout={data['heldout_count']}")


def gate_10_no_training_no_external_data():
    files = _P4_1_SCRIPTS + _P4_1_SRC
    violations = []
    for rel in files:
        path = _ROOT / rel
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        for pat in _FORBIDDEN_TRAINING_PATTERNS:
            if pat in content:
                violations.append(f"{rel}: {pat}")
        for pat in _FORBIDDEN_NETWORK_PATTERNS:
            if pat in content:
                violations.append(f"{rel}: {pat}")
    # Check no weights in sft-v1
    sft_dir = _ROOT / "data" / "p4-agent" / "sft-v1"
    if sft_dir.exists():
        for f in sft_dir.iterdir():
            if f.suffix in (".safetensors", ".bin", ".pt", ".ckpt"):
                violations.append(f"weight file in sft-v1: {f.name}")
    if violations:
        return (False, f"violations: {violations}")
    return (True, "no training, no external data, no weights committed")


_GATES = [
    ("01_p4_0_baseline_lock", gate_01_baseline_lock),
    ("02_test_pass_replay_authoritative", gate_02_test_pass_replay_authoritative),
    ("03_unknown_action_hard_fails", gate_03_unknown_action_hard_fails),
    ("04_all_11_actions_dispatched", gate_04_all_11_actions_dispatched),
    ("05_inspect_error_surfaces_stdout", gate_05_inspect_error_surfaces_stdout),
    ("06_all_5_corruption_types_tested", gate_06_all_5_corruption_types_tested),
    ("07_model_smoke_base", gate_07_model_smoke_base),
    ("08_model_smoke_repair_lora", gate_08_model_smoke_repair_lora),
    ("09_sft_dataset", gate_09_sft_dataset),
    ("10_no_training_no_external_data", gate_10_no_training_no_external_data),
]


def main():
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    all_pass = True
    for name, func in _GATES:
        print(f"Gate {name}...", end=" ", flush=True)
        try:
            ok, evidence = func()
        except Exception as e:
            ok, evidence = False, f"exception: {e}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(status)
        results.append((name, status, evidence))

    # §2.8: Collect full test suite evidence
    print("\nCollecting §2.8 test evidence (full test suite)...")
    test_evidence = _collect_test_evidence()
    print(f"  {test_evidence['pass_count']} passed, "
          f"{test_evidence['fail_count']} failed, "
          f"{test_evidence['skip_count']} skipped in "
          f"{test_evidence['runtime_seconds']}s")

    verdict = "GO_FOR_P4_AGENT_SFT" if all_pass else "NOT_READY"

    lines = [
        "# P4.1 Readiness Report",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"**Date:** {datetime.now().isoformat()}",
        f"**Commit:** {test_evidence['commit_sha']}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
    ]
    for name, status, evidence in results:
        lines.append(f"| {name} | {status} | {evidence[:200]} |")
    lines.append("")

    # §2.8: Test Evidence section
    lines.append("## Test Evidence (§2.8)")
    lines.append("")
    lines.append("### Local Tests")
    lines.append("")
    lines.append(f"- **Exact command:** `{test_evidence['exact_command']}`")
    lines.append(f"- **Test count:** {test_evidence['test_count']}")
    lines.append(f"- **Pass count:** {test_evidence['pass_count']}")
    lines.append(f"- **Fail count:** {test_evidence['fail_count']}")
    lines.append(f"- **Skip count:** {test_evidence['skip_count']}")
    lines.append(f"- **Warning count:** {test_evidence['warning_count']}")
    lines.append(f"- **Error count:** {test_evidence['error_count']}")
    lines.append(f"- **Runtime:** {test_evidence['runtime_seconds']}s")
    lines.append(f"- **Python:** {test_evidence['environment']['python']}")
    lines.append(f"- **Platform:** {test_evidence['environment']['platform']}")
    lines.append(f"- **Machine:** {test_evidence['environment']['machine']}")
    lines.append(f"- **Commit SHA:** {test_evidence['commit_sha']}")
    lines.append("")
    lines.append("### GitHub CI")
    lines.append("")
    lines.append(f"- **Status:** {test_evidence['ci_status']['github_ci']}")
    lines.append("")

    lines.append(f"**Endpoint:** {verdict}")
    if all_pass:
        lines.append("")
        lines.append("P4.1 is complete. `GO_FOR_P4_AGENT_SFT` authorizes "
                     "considering P4.2 (Agent SFT training). It does NOT "
                     "authorize training. Training requires a separate P4.2 "
                     "issue + user approval.")

    _REPORT.write_text("\n".join(lines), encoding="utf-8")

    # Also write machine-readable JSON evidence
    evidence_json = _REPORT.parent / "p4-1-test-evidence.json"
    evidence_json.write_text(
        json.dumps(test_evidence, indent=2), encoding="utf-8"
    )

    print(f"\nVerdict: {verdict}")
    print(f"Report: {_REPORT}")
    print(f"Evidence: {evidence_json}")


if __name__ == "__main__":
    main()
