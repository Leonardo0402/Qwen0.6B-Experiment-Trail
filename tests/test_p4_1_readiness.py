# tests/test_p4_1_readiness.py
"""Tests for the P4.1 readiness report."""
import os
from pathlib import Path

os.environ["P4_ALLOW_NETWORK"] = "0"

from scripts.verify_p4_1_readiness import _GATES  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_REPORT = _ROOT / "reports" / "p4" / "p4-1-readiness.md"

# Roadmap §2.7: actual_gate_ids == EXPECTED_GATE_IDS (not just len == 10).
# A required gate must not be replaceable by an irrelevant passing gate.
EXPECTED_GATE_IDS = {
    "01_p4_0_baseline_lock",
    "02_test_pass_replay_authoritative",
    "03_unknown_action_hard_fails",
    "04_all_11_actions_dispatched",
    "05_inspect_error_surfaces_stdout",
    "06_all_5_corruption_types_tested",
    "07_model_smoke_base",
    "08_model_smoke_repair_lora",
    "09_sft_dataset",
    "10_no_training_no_external_data",
}


def test_readiness_report_exists():
    assert _REPORT.exists(), "p4-1-readiness.md not found — run verify_p4_1_readiness.py"


def test_verdict_is_go_for_p4_agent_sft():
    content = _REPORT.read_text(encoding="utf-8")
    assert "GO_FOR_P4_AGENT_SFT" in content, \
        f"verdict not GO_FOR_P4_AGENT_SFT; content:\n{content[:500]}"


def test_all_10_gates_listed():
    content = _REPORT.read_text(encoding="utf-8")
    for i in range(1, 11):
        assert f"0{i}_" in content or f"{i:02d}_" in content, \
            f"gate {i} not found in report"


def test_augmentation_scripts_exist():
    """T10/T11/T12 augmentation scripts must exist."""
    for script in [
        "scripts/augment_teacher_model.py",
        "scripts/augment_corrupted_recovered.py",
        "scripts/augment_failed_patch_recovery.py",
    ]:
        assert (_ROOT / script).exists(), f"{script} not found"


def test_gate_ids_match_expected_exactly():
    """Roadmap §2.7: the verifier must register exactly the expected 10 gates.

    Asserts set equality (not just count) so a required gate cannot be
    silently replaced by an irrelevant passing gate. Belt and suspenders:
    also asserts the raw list length is 10 to catch duplicate registrations.
    """
    actual_gate_ids = {name for name, _ in _GATES}
    assert actual_gate_ids == EXPECTED_GATE_IDS, (
        f"gate ID set mismatch.\n"
        f"  missing:   {sorted(EXPECTED_GATE_IDS - actual_gate_ids)}\n"
        f"  unexpected: {sorted(actual_gate_ids - EXPECTED_GATE_IDS)}"
    )
    assert len(_GATES) == 10, (
        f"expected 10 gates in _GATES, got {len(_GATES)} "
        f"(possible duplicate registration)"
    )


# ---------------------------------------------------------------------------
# §2.8: Test evidence requirements
# ---------------------------------------------------------------------------

# Required fields in the test evidence JSON (roadmap §2.8)
_EVIDENCE_REQUIRED_FIELDS = [
    "exact_command",
    "test_count",
    "pass_count",
    "fail_count",
    "skip_count",
    "warning_count",
    "runtime_seconds",
    "environment",
    "commit_sha",
    "ci_status",
]


def test_parse_pytest_summary_extracts_counts():
    """§2.8: _parse_pytest_summary must extract pass/fail/skip/warning counts."""
    from scripts.verify_p4_1_readiness import _parse_pytest_summary
    # Typical pytest summary line
    stdout = "=================== 9 passed, 2 failed, 1 skipped, 3 warnings in 12.34s ===================\n"
    result = _parse_pytest_summary(stdout)
    assert result["passed"] == 9
    assert result["failed"] == 2
    assert result["skipped"] == 1
    assert result["warnings"] == 3


def test_parse_pytest_summary_all_passed():
    """§2.8: parse when all tests pass (no failures)."""
    from scripts.verify_p4_1_readiness import _parse_pytest_summary
    stdout = "9 passed in 0.51s\n"
    result = _parse_pytest_summary(stdout)
    assert result["passed"] == 9
    assert result["failed"] == 0
    assert result["skipped"] == 0


def test_parse_pytest_summary_empty():
    """§2.8: parse gracefully handles empty/no-summary output."""
    from scripts.verify_p4_1_readiness import _parse_pytest_summary
    result = _parse_pytest_summary("no output here\n")
    assert result["passed"] == 0
    assert result["failed"] == 0
    assert result["skipped"] == 0


def test_test_evidence_json_has_all_required_fields():
    """§2.8: the test evidence JSON must contain all required fields."""
    import json
    evidence_path = _ROOT / "reports" / "p4" / "p4-1-test-evidence.json"
    if not evidence_path.exists():
        import pytest
        pytest.skip("test evidence not generated yet — run verify_p4_1_readiness.py")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for field in _EVIDENCE_REQUIRED_FIELDS:
        assert field in evidence, f"missing §2.8 evidence field: {field}"
    # environment sub-fields
    env = evidence["environment"]
    assert "python" in env
    assert "platform" in env
    # ci_status sub-fields
    ci = evidence["ci_status"]
    assert "local" in ci
    assert "github_ci" in ci
