# tests/test_p4_1_readiness.py
"""Tests for the P4.1 readiness report."""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_REPORT = _ROOT / "reports" / "p4" / "p4-1-readiness.md"


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
