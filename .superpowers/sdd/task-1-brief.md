## Task 1: Phase A — P4.0 Baseline Lock

**Files:**
- Create: `scripts/lock_p4_0_baseline.py`
- Create: `tests/test_p4_0_baseline_lock.py`
- Generated: `reports/p4/p4-0-baseline-lock.json`

**Interfaces:**
- Consumes: `data/p4-agent/micro-tasks-v0/manifest.json`, `data/p4-agent/trajectories-v0/scripted.jsonl`, `src/agent_evaluator.py`, `reports/p4/p4-agent-foundation-readiness.md`
- Produces: `reports/p4/p4-0-baseline-lock.json` with fields: `p4_0_merge_commit`, `micro_tasks_manifest_sha256`, `scripted_trajectories_sha256`, `agent_evaluator_sha256`, `readiness_report_sha256`, `p4_0_verdict`, `p4_0_test_count`

- [ ] **Step 1: Write the failing test**

```python
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
        return hashlib.sha256(path.read_bytes()).hexdigest()

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_p4_0_baseline_lock.py -v -p no:warnings`
Expected: FAIL — `p4-0-baseline-lock.json not found`

- [ ] **Step 3: Write the lock script**

```python
# scripts/lock_p4_0_baseline.py
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
```

- [ ] **Step 4: Run the script to generate the lock file**

Run: `py -3.11 scripts/lock_p4_0_baseline.py`
Expected: `wrote reports/p4/p4-0-baseline-lock.json`

- [ ] **Step 5: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_p4_0_baseline_lock.py -v -p no:warnings`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/lock_p4_0_baseline.py tests/test_p4_0_baseline_lock.py reports/p4/p4-0-baseline-lock.json
git commit -m "feat(p4-1): Phase A — P4.0 baseline lock"
```

---

