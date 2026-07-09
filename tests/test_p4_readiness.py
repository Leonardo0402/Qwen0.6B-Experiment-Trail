"""Phase H: P4 readiness verifier tests."""
from pathlib import Path

REPORT_PATH = Path(__file__).parent.parent / "reports" / "p4" / "p4-agent-foundation-readiness.md"


def test_readiness_report_exists():
    """The readiness report file exists."""
    assert REPORT_PATH.exists(), f"readiness report not found at {REPORT_PATH}"


def test_verdict_is_go():
    """The readiness report verdict is GO_FOR_P4_AGENT_SFT_DATA."""
    content = REPORT_PATH.read_text(encoding="utf-8")
    assert "GO_FOR_P4_AGENT_SFT_DATA" in content, \
        "verdict not GO_FOR_P4_AGENT_SFT_DATA"


def test_all_11_gates_listed():
    """The readiness report lists all 11 gates."""
    content = REPORT_PATH.read_text(encoding="utf-8")
    expected_gates = [
        "P3 exit baseline locked",
        "Action schema tests pass",
        "Tool layer safety tests pass",
        "Trajectory schema tests pass",
        "Micro task suite verified",
        "Scripted trajectories verified",
        "Evaluator replay success",
        "Corrupted trajectory tests",
        "No forbidden",
        "CI green",
        "State transition consistency",
    ]
    for gate in expected_gates:
        assert gate in content, f"gate '{gate}' not found in report"
