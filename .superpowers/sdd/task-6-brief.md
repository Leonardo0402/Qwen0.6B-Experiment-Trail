# Task 6: Baseline Lock + Smoke Run Script

## Project Context

This is Task 6 of 8 in the P4.1b Protocol Ablation plan (Issue #29). The plan compares 3 action protocols (JSON/Tag/DSL) × 2 configs × 40 micro-tasks = 240 runs on Qwen3-0.6B.

**Branch:** `feat/p4-1b-protocol-ablation`
**Base SHA:** `d92787c` (T5 commit)
**Previous tasks COMPLETE:** T1 (ProtocolBase ABC), T2 (JsonProtocol), T3 (TagProtocol), T4 (DslProtocol), T5 (ModelActionProvider protocol adapter)

T6 creates the orchestration script that runs all protocol×config combinations, records trajectories, and computes metrics. T7 will add the report generator; T8 adds the verdict logic.

## Interface Confirmations (verified by controller)

All interfaces the plan code depends on have been verified against the current codebase:

- `MicroTaskWorkspace.from_task(task_dir: Path)` — creates a temp copy via `tempfile.mkdtemp()` + `shutil.copytree()`. ✅
- `MicroTaskWorkspace.cleanup()` — removes temp dir with `ignore_errors=True`. ✅
- `AgentEvaluator(workspace, provider, task_id, max_steps=20)` — exact signature match. ✅
- `EvalResult` fields: `success`, `metrics`, `steps_executed`, `max_steps_hit`, `finish_claim_mismatch` — all present. ✅
- `AgentState` and `ActionProvider` are imported from `src.agent_evaluator`. ✅
- `ModelActionProvider.__init__(model_path, adapter_path, max_new_tokens, protocol)` — T5 added `protocol` param. ✅
- `ModelActionProvider._load_model()`, `.reset()`, `.diagnostics` — all exist. ✅
- `data/p4-agent/micro-tasks-v0/manifest.json` exists. ✅
- `scripts/` directory exists with `__init__.py`. ✅

## IMPORTANT PROACTIVE FIX (not in original plan)

The plan's test `test_run_combination_with_mock_protocol` has a bug: the mock's `actions_iter` is created once when `make_provider` is called, but `run_combination` creates ONE provider and reuses it for ALL tasks. The mock's `reset = MagicMock()` does nothing, so the iterator is NOT reset between tasks.

**Result without fix:**
- Task 1: `next_action` → "list_files" → "finish" (2 actions consumed)
- Task 2: `next_action` → `StopIteration` (crash, caught by try/except)
- `trajectories_written` = 1, not 2
- Test assertion `result["trajectories_written"] == 2` FAILS

**Fix:** Replace the mock's `reset = MagicMock()` with a real function that resets the iterator. Use this corrected version of `test_run_combination_with_mock_protocol` INSTEAD of the plan's version:

```python
def test_run_combination_with_mock_protocol():
    """Test that run_combination works with a mock protocol (no model loading)."""
    from scripts.run_protocol_ablation import run_combination, _TASKS_DIR
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    proto = MockProtocol(action_type="list_files")
    config = {"name": "mock-base", "model_path": "fake", "adapter_path": None}

    # Use only first 2 tasks for speed
    manifest = json.loads((_TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    task_ids = [t["task_id"] for t in manifest["tasks"][:2]]

    with patch("scripts.run_protocol_ablation.ModelActionProvider") as MockProvider:
        # Create a mock provider that uses our protocol
        def make_provider(*args, **kwargs):
            provider = MagicMock()
            provider._protocol = kwargs.get("protocol")
            provider._model = MagicMock()
            provider._load_model = MagicMock()  # succeed, don't raise
            provider.diagnostics = []

            # Iterator state — reset() recreates it so each task gets fresh actions
            actions_state = {"iter": iter(["list_files", "finish"])}

            def reset():
                actions_state["iter"] = iter(["list_files", "finish"])
                provider.diagnostics = []

            def next_action(state):
                at = next(actions_state["iter"])
                if at == "list_files":
                    from src.agent_actions import ListFilesAction, SafetyFlags
                    return ListFilesAction(
                        action_id="a1", reason_short="test",
                        expected_observation="files",
                        safety_flags=SafetyFlags(
                            modifies_workspace=False, executes_code=False,
                            network_required=False, reads_sensitive_path=False,
                            is_terminal=False,
                        ),
                    )
                else:
                    from src.agent_actions import FinishAction, FinishArgs, SafetyFlags, TaskSuccessCriterion
                    return FinishAction(
                        action_id="a2", reason_short="done",
                        expected_observation="finished",
                        safety_flags=SafetyFlags(
                            modifies_workspace=False, executes_code=False,
                            network_required=False, reads_sensitive_path=False,
                            is_terminal=True,
                        ),
                        arguments=FinishArgs(
                            success_criterion=TaskSuccessCriterion.TEST_PASS,
                            tests_passed=False, identification_verified=False,
                            summary="mock finish",
                        ),
                    )
            provider.next_action = next_action
            provider.reset = reset
            return provider
        MockProvider.side_effect = make_provider

        result = run_combination(proto, config, task_ids, max_steps=5)
        assert result["config"] == "mock-base"
        assert result["trajectories_written"] == 2
        assert "metrics" in result
```

## Plan Text (verbatim from plan, Task 6)

### Files
- Create: `scripts/run_protocol_ablation.py`
- Create: `tests/test_protocol_ablation.py`
- Output dir: `reports/p4/protocol-ablation/`

### Interfaces
- Consumes: `JsonProtocol`, `TagProtocol`, `DslProtocol` from Tasks 2-4; `ModelActionProvider` from Task 5; `AgentEvaluator`, `MicroTaskWorkspace` from existing code
- Produces: `run_protocol_ablation.py` with `baseline_lock()`, `run_combination()`, `main()`; writes `baseline-lock.json` and trajectory JSONL files

### Step 1: Write the failing tests

```python
# tests/test_protocol_ablation.py
"""Integration tests for protocol ablation script.

Tests use MockProtocol to avoid loading the actual model.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.protocols.base import ProtocolBase, ProtocolDiagnostics
from src.agent_model_provider import SentinelAction


class MockProtocol(ProtocolBase):
    """Protocol that returns a predefined action for testing."""
    def __init__(self, action_type="list_files"):
        self._action_type = action_type

    @property
    def name(self):
        return "mock"

    def build_system_prompt(self, task_context):
        return f"Mock prompt for {task_context}"

    def parse_output(self, raw):
        diag = ProtocolDiagnostics(
            raw_output=raw, format_parse_ok=True, schema_valid=True,
            safety_valid=True, action_type_valid=True, arguments_valid=True,
            repair_attempted=False, repair_success=False, latency_ms=1,
        )
        if self._action_type == "invalid":
            return SentinelAction(reason="mock invalid"), diag
        from src.agent_actions import ListFilesAction, SafetyFlags
        action = ListFilesAction(
            action_id="mock_a1", reason_short="mock",
            expected_observation="mock",
            safety_flags=SafetyFlags(
                modifies_workspace=False, executes_code=False,
                network_required=False, reads_sensitive_path=False,
                is_terminal=False,
            ),
        )
        return action, diag


def test_baseline_lock_records_all_fields():
    from scripts.run_protocol_ablation import baseline_lock
    lock = baseline_lock()
    assert "commit_sha" in lock
    assert "micro_task_manifest_sha256" in lock
    assert "model_path" in lock
    assert "adapter_path" in lock
    assert "generation_config" in lock
    assert lock["generation_config"]["temperature"] == 0.0
    assert lock["generation_config"]["do_sample"] is False
    assert "total_tasks" in lock
    assert lock["total_tasks"] == 40


def test_aggregate_metrics_computes_all_fields():
    from scripts.run_protocol_ablation import aggregate_metrics
    trajectories = [
        {
            "step_diagnostics": [
                {"format_parse_ok": True, "schema_valid": True,
                 "safety_valid": True, "action_type_valid": True,
                 "arguments_valid": True, "failure_class": None},
                {"format_parse_ok": True, "schema_valid": False,
                 "safety_valid": False, "action_type_valid": False,
                 "arguments_valid": False, "failure_class": "SCHEMA_VALIDATION_FAIL"},
            ],
            "metrics": {"forbidden_action_count": 0, "tool_error_rate": 0.0,
                        "max_step_exceeded_count": 1},
            "success": False,
            "finish_claim_mismatch": True,
            "steps_executed": 12,
        },
    ]
    metrics = aggregate_metrics(trajectories, crashes=0)
    # Spec §6.2 requires 12+ metrics
    assert "format_parse_rate" in metrics
    assert "schema_valid_rate" in metrics
    assert "safety_valid_rate" in metrics
    assert "action_type_valid_rate" in metrics
    assert "arguments_valid_rate" in metrics
    assert "forbidden_action_count" in metrics
    assert "unknown_action_count" in metrics
    assert "task_success_rate" in metrics
    assert "finish_without_tests_count" in metrics
    assert "finish_claim_mismatch_count" in metrics
    assert "max_steps_hit_rate" in metrics
    assert "runtime_crash_count" in metrics
    assert metrics["format_parse_rate"] == 1.0
    assert metrics["schema_valid_rate"] == 0.5
    assert metrics["unknown_action_count"] == 1  # one step had action_type_valid=False
    assert metrics["finish_claim_mismatch_count"] == 1


def test_classify_failures_returns_taxonomy():
    from scripts.run_protocol_ablation import classify_failures
    trajectories = [
        {
            "step_diagnostics": [
                {"failure_class": "FORMAT_PARSE_FAIL"},
                {"failure_class": "SCHEMA_VALIDATION_FAIL"},
                {"failure_class": "UNKNOWN_ACTION_TYPE"},
                {"failure_class": None},  # success
            ],
            "actions": [
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
            ],
        },
    ]
    taxonomy = classify_failures(trajectories)
    assert taxonomy["FORMAT_PARSE_FAIL"] == 1
    assert taxonomy["SCHEMA_VALIDATION_FAIL"] == 1
    assert taxonomy["UNKNOWN_ACTION_TYPE"] == 1
    assert taxonomy["REPEATED_ACTION_LOOP"] == 1  # detected from actions


# NOTE: Use the CORRECTED version of test_run_combination_with_mock_protocol
# from the PROACTIVE FIX section above, NOT the plan's original version.
```

### Step 2: Run test to verify it fails

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'`

### Step 3: Write minimal implementation

```python
# scripts/run_protocol_ablation.py
"""P4.1b Protocol Ablation: run 3 protocols x 2 configs on 40 micro-tasks.

Produces:
- reports/p4/protocol-ablation/baseline-lock.json
- reports/p4/protocol-ablation/trajectories/{protocol}-{config}.jsonl
- reports/p4/protocol-ablation/comparison-matrix.json
- reports/p4/protocol-ablation/failure-taxonomy.json
- reports/p4/protocol-ablation/comparison-report.md

Usage:
    py -3.11 scripts/run_protocol_ablation.py
    py -3.11 scripts/run_protocol_ablation.py --task-limit 5  # quick smoke
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("P4_ALLOW_NETWORK", "0")

from src.protocols import JsonProtocol, TagProtocol, DslProtocol
from src.agent_model_provider import ModelActionProvider, SentinelAction
from src.agent_evaluator import AgentEvaluator, ActionProvider, AgentState
from src.agent_workspace import MicroTaskWorkspace

_PROTOCOLS = [
    {"name": "json", "class": JsonProtocol},
    {"name": "tag", "class": TagProtocol},
    {"name": "dsl", "class": DslProtocol},
]

_CONFIGS = [
    {"name": "base", "model_path": "models/Qwen3-0.6B", "adapter_path": None},
    {"name": "repair-lora", "model_path": "models/Qwen3-0.6B",
     "adapter_path": "adapters/p3/repair-limited"},
]

_TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
_REPORT_DIR = _ROOT / "reports" / "p4" / "protocol-ablation"
MAX_STEPS = 12

_FAILURE_CLASSES = [
    "FORMAT_PARSE_FAIL", "SCHEMA_VALIDATION_FAIL", "UNKNOWN_ACTION_TYPE",
    "FORBIDDEN_ACTION", "INVALID_PATH", "EMPTY_OR_USELESS_ACTION",
    "MODEL_REFUSAL_OR_CHATTER", "REPEATED_ACTION_LOOP",
]


def _load_task_ids():
    manifest = json.loads((_TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    return [t["task_id"] for t in manifest["tasks"]]


def _git_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(_ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def baseline_lock() -> dict:
    """Record experiment starting state for reproducibility."""
    manifest_path = _TASKS_DIR / "manifest.json"
    return {
        "commit_sha": _git_sha(),
        "micro_task_manifest_path": str(manifest_path.relative_to(_ROOT)),
        "micro_task_manifest_sha256": _file_sha256(manifest_path),
        "model_path": "models/Qwen3-0.6B",
        "adapter_path": "adapters/p3/repair-limited",
        "generation_config": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 128,
            "dtype": "float16",
        },
        "max_steps": MAX_STEPS,
        "total_tasks": 40,
        "protocols": [p["name"] for p in _PROTOCOLS],
        "configs": [c["name"] for c in _CONFIGS],
        "total_combinations": len(_PROTOCOLS) * len(_CONFIGS),
        "total_runs": len(_PROTOCOLS) * len(_CONFIGS) * 40,
    }


class RecordingProvider(ActionProvider):
    """Wraps an ActionProvider and records each action for later replay."""

    def __init__(self, inner: ActionProvider):
        self._inner = inner
        self._recorded: list[dict] = []

    def next_action(self, state: AgentState):
        action = self._inner.next_action(state)
        if isinstance(action, SentinelAction):
            self._recorded.append({
                "__sentinel__": True,
                "is_invalid": True,
                "reason": action.reason,
            })
        else:
            self._recorded.append(action.model_dump())
        return action

    @property
    def recorded_actions(self) -> list[dict]:
        return list(self._recorded)

    @property
    def diagnostics(self):
        return self._inner.diagnostics if hasattr(self._inner, "diagnostics") else []

    def reset(self) -> None:
        self._recorded.clear()
        if hasattr(self._inner, "reset"):
            self._inner.reset()


def run_combination(protocol, config, task_ids, max_steps=MAX_STEPS):
    """Run one protocol x config combination on all task_ids."""
    trajectories = []
    crashes = 0
    model_load_ok = False

    inner_provider = ModelActionProvider(
        model_path=config["model_path"],
        adapter_path=config["adapter_path"],
        protocol=protocol,
    )

    # Try to load the model once
    try:
        inner_provider._load_model()
        model_load_ok = True
    except Exception as e:
        print(f"[{protocol.name}/{config['name']}] model load failed: {e}")
        return {
            "protocol": protocol.name,
            "config": config["name"],
            "total_tasks": len(task_ids),
            "trajectories_written": 0,
            "model_load_ok": False,
            "metrics": {},
            "trajectories": [],
        }

    for i, task_id in enumerate(task_ids):
        task_dir = _TASKS_DIR / task_id
        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            inner_provider.reset()
            provider = RecordingProvider(inner_provider)
            evaluator = AgentEvaluator(ws, provider, task_id, max_steps=max_steps)
            result = evaluator.run()
            trajectories.append({
                "trajectory_id": f"{protocol.name}_{config['name']}_{task_id}",
                "task_id": task_id,
                "protocol": protocol.name,
                "config": config["name"],
                "success": result.success,
                "finish_claim_mismatch": result.finish_claim_mismatch,
                "metrics": result.metrics,
                "steps_executed": result.steps_executed,
                "max_steps_hit": result.max_steps_hit,
                "actions": provider.recorded_actions,
                "step_diagnostics": [d.model_dump() for d in inner_provider.diagnostics],
            })
        except Exception:
            crashes += 1
            traceback.print_exc()
        finally:
            ws.cleanup()
        print(f"\r[{protocol.name}/{config['name']}] {i+1}/{len(task_ids)} {task_id}", end="", flush=True)
    print()

    metrics = aggregate_metrics(trajectories, crashes, model_load_ok)
    return {
        "protocol": protocol.name,
        "config": config["name"],
        "total_tasks": len(task_ids),
        "trajectories_written": len(trajectories),
        "model_load_ok": model_load_ok,
        "metrics": metrics,
        "trajectories": trajectories,
    }


def aggregate_metrics(trajectories: list[dict], crashes: int = 0,
                      model_load_ok: bool = True) -> dict:
    """Compute aggregated metrics from trajectory step_diagnostics.

    Returns 13 metrics per spec §6.2 (12+ required by acceptance criteria).
    """
    all_diags: list[dict] = []
    for traj in trajectories:
        all_diags.extend(traj.get("step_diagnostics", []))

    total_diags = len(all_diags)
    if total_diags > 0:
        format_parse_rate = sum(1 for d in all_diags if d.get("format_parse_ok")) / total_diags
        schema_valid_rate = sum(1 for d in all_diags if d.get("schema_valid")) / total_diags
        safety_valid_rate = sum(1 for d in all_diags if d.get("safety_valid")) / total_diags
        action_type_valid_rate = sum(1 for d in all_diags if d.get("action_type_valid")) / total_diags
        arguments_valid_rate = sum(1 for d in all_diags if d.get("arguments_valid")) / total_diags
    else:
        format_parse_rate = 0.0
        schema_valid_rate = 0.0
        safety_valid_rate = 0.0
        action_type_valid_rate = 0.0
        arguments_valid_rate = 0.0

    forbidden_action_count = sum(
        t.get("metrics", {}).get("forbidden_action_count", 0) for t in trajectories
    )
    unknown_action_count = sum(1 for d in all_diags if not d.get("action_type_valid"))
    total_tasks = len(trajectories)
    task_success_rate = sum(1 for t in trajectories if t.get("success")) / total_tasks if total_tasks else 0.0
    max_steps_hit_rate = sum(1 for t in trajectories if t.get("max_steps_hit")) / total_tasks if total_tasks else 0.0

    # Finish-related metrics (from evaluator)
    finish_without_tests_count = sum(
        1 for t in trajectories
        if t.get("success") and not t.get("metrics", {}).get("tests_passed", False)
    )
    finish_claim_mismatch_count = sum(
        1 for t in trajectories if t.get("finish_claim_mismatch")
    )

    return {
        "format_parse_rate": format_parse_rate,
        "schema_valid_rate": schema_valid_rate,
        "safety_valid_rate": safety_valid_rate,
        "action_type_valid_rate": action_type_valid_rate,
        "arguments_valid_rate": arguments_valid_rate,
        "forbidden_action_count": forbidden_action_count,
        "unknown_action_count": unknown_action_count,
        "task_success_rate": task_success_rate,
        "finish_without_tests_count": finish_without_tests_count,
        "finish_claim_mismatch_count": finish_claim_mismatch_count,
        "max_steps_hit_rate": max_steps_hit_rate,
        "runtime_crash_count": crashes,
        "model_load_ok": model_load_ok,
    }


def classify_failures(trajectories: list[dict]) -> dict:
    """Classify all failed steps into failure taxonomy.

    Also detects REPEATED_ACTION_LOOP at trajectory level (3+ consecutive
    identical actions in a single trajectory).
    """
    taxonomy = {fc: 0 for fc in _FAILURE_CLASSES}
    for traj in trajectories:
        for d in traj.get("step_diagnostics", []):
            fc = d.get("failure_class")
            if fc and fc in taxonomy:
                taxonomy[fc] += 1
        # Trajectory-level: detect repeated action loops
        actions = traj.get("actions", [])
        if _detect_repeated_loop(actions):
            taxonomy["REPEATED_ACTION_LOOP"] += 1
    return taxonomy


def _detect_repeated_loop(actions: list[dict]) -> bool:
    """Return True if 3+ consecutive identical actions found."""
    if len(actions) < 3:
        return False
    for i in range(len(actions) - 2):
        a1 = actions[i]
        a2 = actions[i + 1]
        a3 = actions[i + 2]
        if (a1.get("action_type") == a2.get("action_type") == a3.get("action_type")
                and a1.get("arguments") == a2.get("arguments") == a3.get("arguments")):
            return True
    return False


def main():
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    traj_dir = _REPORT_DIR / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Baseline lock
    print("=== Step 1: Baseline Lock ===")
    lock = baseline_lock()
    (_REPORT_DIR / "baseline-lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(f"Wrote {_REPORT_DIR / 'baseline-lock.json'}")

    # Step 2: Run all 6 combinations
    print("\n=== Step 2: Run Protocol x Config Combinations ===")
    all_task_ids = _load_task_ids()
    task_limit = os.environ.get("P4_1B_TASK_LIMIT")
    if task_limit:
        all_task_ids = all_task_ids[:int(task_limit)]
        print(f"P4_1B_TASK_LIMIT={task_limit}: running first {len(all_task_ids)} tasks")

    all_results = []
    for proto_spec in _PROTOCOLS:
        for config in _CONFIGS:
            print(f"\n--- Protocol: {proto_spec['name']} | Config: {config['name']} ---")
            proto = proto_spec["class"]()
            result = run_combination(proto, config, all_task_ids)
            # Write trajectories JSONL
            out_file = traj_dir / f"{proto_spec['name']}-{config['name']}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for traj in result["trajectories"]:
                    f.write(json.dumps(traj) + "\n")
            # Strip trajectories from summary
            summary = {k: v for k, v in result.items() if k != "trajectories"}
            all_results.append(summary)
            print(f"  schema_valid_rate: {summary['metrics'].get('schema_valid_rate', 0):.2%}")

    # Step 3: Comparison matrix
    print("\n=== Step 3: Comparison Matrix ===")
    (_REPORT_DIR / "comparison-matrix.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # Step 4: Failure taxonomy
    print("\n=== Step 4: Failure Taxonomy ===")
    # Reload trajectories for taxonomy
    all_trajectories = []
    for proto_spec in _PROTOCOLS:
        for config in _CONFIGS:
            traj_file = traj_dir / f"{proto_spec['name']}-{config['name']}.jsonl"
            if traj_file.exists():
                for line in traj_file.read_text(encoding="utf-8").strip().split("\n"):
                    if line:
                        all_trajectories.append(json.loads(line))
    taxonomy = classify_failures(all_trajectories)
    (_REPORT_DIR / "failure-taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2), encoding="utf-8"
    )

    print(f"\nDone. Reports in {_REPORT_DIR}")


if __name__ == "__main__":
    main()
```

### Step 4: Run tests to verify they pass

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
Expected: PASS (4 tests)

### Step 5: Commit

```bash
git add scripts/run_protocol_ablation.py tests/test_protocol_ablation.py
git commit -m "feat(p4-1b): add baseline lock and smoke run script (P4.1b T6)"
```

## Report Contract

Write your full report to: `.superpowers/sdd/task-6-report.md`

Return only:
- Status: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED
- Commit SHA(s)
- One-line test summary (e.g., "4/4 tests pass")
- Any concerns

The report file should contain:
1. Files created/modified
2. Test results (full pytest output)
3. Any deviations from the brief and why
4. Self-review notes
