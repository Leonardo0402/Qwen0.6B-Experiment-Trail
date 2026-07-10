# tests/test_p4_0_baseline_lock.py
import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def test_baseline_lock_exists():
    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
    assert lock_path.exists(), "p4-0-baseline-lock.json not found"


def test_baseline_lock_has_required_fields():
    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {
        "p4_0_merge_commit", "micro_tasks_manifest_sha256",
        "scripted_trajectories_sha256", "agent_evaluator_sha256",
        "readiness_report_sha256", "p4_0_verdict", "p4_0_test_count",
    }
    assert required.issubset(data.keys()), f"missing: {required - data.keys()}"


def test_baseline_lock_shas_match_files():
    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
    data = json.loads(lock_path.read_text(encoding="utf-8"))

    def sha256(path):
        # CRLF→LF normalization for cross-platform consistency
        content = path.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(content).hexdigest()

    assert data["micro_tasks_manifest_sha256"] == sha256(
        _ROOT / "data" / "p4-agent" / "micro-tasks-v0" / "manifest.json"
    )
    assert data["scripted_trajectories_sha256"] == sha256(
        _ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
    )
    assert data["agent_evaluator_sha256"] == sha256(
        _ROOT / "src" / "agent_evaluator.py"
    )
    assert data["readiness_report_sha256"] == sha256(
        _ROOT / "reports" / "p4" / "p4-agent-foundation-readiness.md"
    )


def test_baseline_lock_p4_0_merge_commit_is_7ccd06c():
    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["p4_0_merge_commit"].startswith("7ccd06c"), \
        f"expected 7ccd06c..., got {data['p4_0_merge_commit']}"


def test_baseline_lock_verdict_is_go():
    lock_path = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["p4_0_verdict"] == "GO_FOR_P4_AGENT_SFT_DATA"
    assert data["p4_0_test_count"] == 81
