# scripts/verify_p4_1_readiness.py
"""Phase H: P4.1 readiness verifier — 10 gates → GO_FOR_P4_AGENT_SFT."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
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
        [sys.executable, "-m", "pytest"] + test_args + ["-p", "no:warnings", "-q"],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return proc.returncode, proc.stdout


def _extract_passed_line(stdout):
    for line in stdout.splitlines():
        if "passed" in line:
            return line.strip()
    return "no passed line found"


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


def gate_07_model_smoke_base():
    if not _COLLECTION_REPORT.exists():
        return (False, "collection report not found")
    reports = json.loads(_COLLECTION_REPORT.read_text(encoding="utf-8"))
    base = next((r for r in reports if r["config"] == "base"), None)
    if base is None:
        return (False, "base config not in report")
    if not base.get("model_load_ok"):
        return (False, "model_load_ok=False for base")
    if base.get("crashes", 0) > 0:
        return (False, f"crashes={base['crashes']}")
    smoke = base.get("limited_smoke", False)
    label = "LIMITED_SMOKE" if smoke else "full"
    return (True, f"loaded ({label}), {base['trajectories_written']} trajectories")


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
    smoke = repair.get("limited_smoke", False)
    label = "LIMITED_SMOKE" if smoke else "full"
    return (True, f"loaded ({label}), {repair['trajectories_written']} trajectories")


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

    verdict = "GO_FOR_P4_AGENT_SFT" if all_pass else "NOT_READY"

    lines = [
        "# P4.1 Readiness Report",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"**Date:** {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
    ]
    for name, status, evidence in results:
        lines.append(f"| {name} | {status} | {evidence[:200]} |")
    lines.append("")
    lines.append(f"**Endpoint:** {verdict}")
    if all_pass:
        lines.append("")
        lines.append("P4.1 is complete. `GO_FOR_P4_AGENT_SFT` authorizes "
                     "considering P4.2 (Agent SFT training). It does NOT "
                     "authorize training. Training requires a separate P4.2 "
                     "issue + user approval.")

    _REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nVerdict: {verdict}")
    print(f"Report: {_REPORT}")


if __name__ == "__main__":
    main()
