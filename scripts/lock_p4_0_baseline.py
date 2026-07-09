"""Phase A: lock P4.0 baseline SHAs into a JSON file.

Idempotent: re-running produces the same JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "reports" / "p4" / "p4-0-baseline-lock.json"

_P4_0_MERGE_COMMIT = "7ccd06c4d479b269f7708a6a430b9965af5f17e6"
_P4_0_VERDICT = "GO_FOR_P4_AGENT_SFT_DATA"
_P4_0_TEST_COUNT = 81


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lock = {
        "p4_0_merge_commit": _P4_0_MERGE_COMMIT,
        "micro_tasks_manifest_sha256": _sha256(
            _ROOT / "data" / "p4-agent" / "micro-tasks-v0" / "manifest.json"
        ),
        "scripted_trajectories_sha256": _sha256(
            _ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
        ),
        "agent_evaluator_sha256": _sha256(_ROOT / "src" / "agent_evaluator.py"),
        "readiness_report_sha256": _sha256(
            _ROOT / "reports" / "p4" / "p4-agent-foundation-readiness.md"
        ),
        "p4_0_verdict": _P4_0_VERDICT,
        "p4_0_test_count": _P4_0_TEST_COUNT,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
