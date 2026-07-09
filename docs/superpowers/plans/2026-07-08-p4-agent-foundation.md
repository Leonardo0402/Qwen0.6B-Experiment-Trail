# P4.0 Agentic Coder Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the P4.0 Agentic Coder Foundation — action schemas, safe tool layer, trajectory format, 40-task micro suite, scripted teacher trajectories, and a replay evaluator — that unblocks P4.1 agent SFT data collection.

**Architecture:** TDD-built, phase-gated foundation. Eleven action types (incl. `rollback_patch`), 6 safe tools, JSONL trajectory schema with state-transition invariants, 40 toy workspaces (8 types × 5), per-task scripted teacher functions, and an `ActionProvider`-abstracted evaluator that replays scripted trajectories through the real tool layer. No model load in P4.0.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, pathlib, tempfile, hashlib. No new runtime dependencies. Reuses `src/sandbox.py::run_pytest` for test isolation.

**Spec:** [docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md](file:///e:/agent/Qwen/qwen3-code-lab/docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md)

## Global Constraints

- **Branch:** `feat/p4-agent-foundation` (off `main` @ merge commit of PR #15)
- **Delivery:** single PR covering phases A–H, `Closes #17` in commit message
- **Python:** 3.11 (matches CI)
- **No new runtime deps:** Pydantic v2 and pytest already in `requirements.txt`
- **No model weights:** P4.0 does not load Qwen3-0.6B; `ModelActionProvider` raises `NotImplementedError("P4.1")`
- **No external datasets:** micro-tasks are toy workspaces authored from scratch; no HumanEval/LeetCode/SWE-bench
- **No silent fallback:** `run_tests` network isolation failure raises `ToolUnavailableError`, never falls back
- **Supply-chain rule:** no file from Issue/PR comments is downloaded or applied; all code written from scratch
- **Frozen v4 boundary:** micro-task IDs use `p4_task_NNN`; no overlap with `data/frozen-eval/v4/test_raw.jsonl` sample_ids
- **AGENTS.md hard rules:** 4GB VRAM cap (N/A in P4.0 — no model), MBPP-only (N/A — no training), adapters append-only (N/A — no adapters), no silent fallback (enforced in `run_tests`)
- **Commit style:** `feat(p4-agent): <phase> <component>` — one commit per task
- **TDD discipline:** every task writes failing test first, verifies Red, implements minimal code, verifies Green, commits

## File Structure

### Files created (new)

| Path | Responsibility | Phase |
|---|---|---|
| `src/agent_actions.py` | Action schema (11 types), SafetyFlags, TaskSuccessCriterion, EvaluationMode | B |
| `src/agent_state.py` | AgentMemory (4 fields), WorkspaceSnapshot | B |
| `src/agent_workspace.py` | MicroTaskWorkspace (path validation, ignore list, SHA recording) | C |
| `src/agent_tools.py` | 6 tool implementations returning typed Observations | C |
| `src/agent_trajectory.py` | TrajectoryStep, Trajectory, JSONL I/O, state-transition validators | D |
| `src/agent_evaluator.py` | ActionProvider abstract + 3 concrete providers + AgentEvaluator | G |
| `scripts/lock_p3_exit_baseline.py` | Phase A: lock P3 SHAs into JSON | A |
| `scripts/generate_scripted_agent_trajectories.py` | Phase F: 40 per-task teacher functions + JSONL writer | F |
| `scripts/evaluate_agent_policy.py` | Phase G: CLI runner for evaluator | G |
| `scripts/verify_p4_readiness.py` | Phase H: 11-gate verifier | H |
| `tests/test_agent_actions.py` | Phase B tests | B |
| `tests/test_agent_tools.py` | Phase C tests | C |
| `tests/test_agent_trajectory.py` | Phase D tests | D |
| `tests/test_micro_task_suite.py` | Phase E tests (parametrized over 40 tasks) | E |
| `tests/test_agent_evaluator.py` | Phase G tests | G |
| `reports/p4/p3-exit-baseline-lock.json` | Phase A output (generated) | A |
| `reports/p4/p3-exit-summary.md` | Phase A output (generated) | A |
| `reports/p4/trajectory-schema.md` | Phase D schema doc | D |
| `reports/p4/scripted-trajectories-report.md` | Phase F report (generated) | F |
| `reports/p4/p4-agent-foundation-readiness.md` | Phase H report (generated) | H |
| `data/p4-agent/micro-tasks-v0/task_001/…task_040/` | 40 toy workspaces | E |
| `data/p4-agent/micro-tasks-v0/manifest.json` | Registry | E |
| `data/p4-agent/trajectories-v0/scripted.jsonl` | 40 trajectories (generated) | F |

### Files modified

| Path | Change | Phase |
|---|---|---|
| `.gitignore` | Add `data/p4-agent/micro-tasks-v0/*/.audit/` | A |

### Files NOT touched (P3 freeze)

- `data/frozen-eval/v4/test_raw.jsonl`
- `data/p3-limited/**`, `data/p3-curriculum/**`, `data/p3-formal/**`
- `adapters/p3/**`
- `reports/p3-limited/**`, `reports/p3/**`
- `scripts/build_p3_limited.py`, `scripts/evaluate_model.py` (frozen post-Issue #16)

---

## Task Summary

| # | Phase | Component | Estimated tests |
|---|---|---|---|
| 1 | A | P3 exit baseline lock | 6 |
| 2 | B | Action enums + SafetyFlags + AgentMemory | 7 |
| 3 | B | Action union + path validation + safety rejection | 9 |
| 4 | C | MicroTaskWorkspace + path/ignore logic | 5 |
| 5 | C | Read-only tools (list_files, read_file, search_text, inspect_task) | 7 |
| 6 | C | apply_patch + rollback_patch + audit | 9 |
| 7 | C | run_tests + inspect_error + write_memory + finish | 6 |
| 8 | D | Trajectory schema + state-transition invariants | 7 |
| 9 | E | Micro task suite (40 tasks) | 6 |
| 10 | F | Scripted teacher trajectories (40) | 4 |
| 11 | G | ActionProvider + AgentEvaluator | 4 |
| 12 | H | Readiness verifier + report | 3 |

**Total:** 12 tasks, ~73 tests (spec estimated ~120 with expansion during TDD)

---

## Task 1: Phase A — P3 Exit Baseline Lock

**Files:**
- Create: `scripts/lock_p3_exit_baseline.py`
- Create: `reports/p4/p3-exit-baseline-lock.json` (generated)
- Create: `reports/p4/p3-exit-summary.md` (generated)
- Modify: `.gitignore`
- Test: `tests/test_p3_exit_lock.py`

**Interfaces:**
- Consumes: `data/p3-limited/{balanced,repair}-limited/manifest.json`, `adapters/p3/{balanced,repair}-limited/metrics.json`, `data/p3-curriculum/validation-v2/validation.jsonl`, `data/frozen-eval/v4/test_raw.jsonl`
- Produces: `reports/p4/p3-exit-baseline-lock.json` (schema_version=1, verdict, PR #15 SHA, dataset hashes, adapter hashes, warnings)

- [ ] **Step 1: Create branch**

```powershell
git checkout -b feat/p4-agent-foundation
```

- [ ] **Step 2: Add `.gitignore` entry**

Append to `.gitignore`:
```
# P4.0 audit trails (regenerated at runtime, not committed)
data/p4-agent/micro-tasks-v0/*/.audit/
```

- [ ] **Step 3: Write failing test**

`tests/test_p3_exit_lock.py`:
```python
"""Phase A: P3 exit baseline lock tests."""
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_LOCK_PATH = _ROOT / "reports/p4/p3-exit-baseline-lock.json"


def test_lock_file_exists():
    assert _LOCK_PATH.exists(), "Run scripts/lock_p3_exit_baseline.py first"


def test_lock_schema_version():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1


def test_lock_verdict():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    assert data["p3_terminal_verdict"] == "MBPP_FAMILY_OR_VARIANT_LIMIT"


def test_lock_pr_15_sha():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    assert data["pr_15"]["merge_commit_sha"] == "d91586e0d31214f4ed3edbdce524e6b0e8067070"


def test_lock_adapters_match_manifests():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    for cand in ("balanced_limited", "repair_limited"):
        manifest = json.loads((_ROOT / f"data/p3-limited/{cand}/manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((_ROOT / f"adapters/p3/{cand}/metrics.json").read_text(encoding="utf-8"))
        assert data["adapters"][cand]["train_hash_in_metrics"] == manifest["train_sha256"]
        assert metrics["train_hash"] == manifest["train_sha256"]


def test_lock_warning_count():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    assert len(data["warnings"]) >= 3
```

- [ ] **Step 4: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_p3_exit_lock.py -v`
Expected: FAIL — `reports/p4/p3-exit-baseline-lock.json` does not exist yet

- [ ] **Step 5: Implement `scripts/lock_p3_exit_baseline.py`**

```python
"""Phase A: Lock P3 exit baseline — captures SHAs of all P3 artifacts.

Idempotent: re-running produces identical JSON (except `locked_at`).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT_JSON = _ROOT / "reports/p4/p3-exit-baseline-lock.json"
_OUT_MD = _ROOT / "reports/p4/p3-exit-summary.md"


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def main() -> None:
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    val_v2_path = _ROOT / "data/p3-curriculum/validation-v2/validation.jsonl"
    frozen_v4_path = _ROOT / "data/frozen-eval/v4/test_raw.jsonl"

    adapters = {}
    for cand in ("balanced_limited", "repair_limited"):
        manifest = json.loads((_ROOT / f"data/p3-limited/{cand}/manifest.json").read_text(encoding="utf-8"))
        metrics_path = _ROOT / f"adapters/p3/{cand}/metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        adapters[cand] = {
            "metrics_path": str(metrics_path.relative_to(_ROOT)),
            "metrics_sha256": _sha256_file(metrics_path),
            "train_hash_in_metrics": metrics["train_hash"],
        }
        assert metrics["train_hash"] == manifest["train_sha256"], f"train_hash mismatch for {cand}"

    bal_manifest = json.loads((_ROOT / "data/p3-limited/balanced-limited/manifest.json").read_text(encoding="utf-8"))
    rep_manifest = json.loads((_ROOT / "data/p3-limited/repair-limited/manifest.json").read_text(encoding="utf-8"))

    lock = {
        "schema_version": 1,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "p3_terminal_verdict": "MBPP_FAMILY_OR_VARIANT_LIMIT",
        "pr_15": {
            "merge_commit_sha": "d91586e0d31214f4ed3edbdce524e6b0e8067070",
            "title": "feat(p3-limited): controlled experiment (1280 samples, Issue #16 fixed) (#15)",
        },
        "p3_limited_datasets": {
            "balanced_limited": {
                "train_sha256": bal_manifest["train_sha256"],
                "manifest_path": "data/p3-limited/balanced-limited/manifest.json",
                "total_samples": 1280,
            },
            "repair_limited": {
                "train_sha256": rep_manifest["train_sha256"],
                "manifest_path": "data/p3-limited/repair-limited/manifest.json",
                "total_samples": 1280,
            },
        },
        "validation_v2": {
            "path": str(val_v2_path.relative_to(_ROOT)),
            "sha256": _sha256_file(val_v2_path),
            "sample_count": _count_jsonl(val_v2_path),
        },
        "frozen_v4": {
            "path": str(frozen_v4_path.relative_to(_ROOT)),
            "sha256": _sha256_file(frozen_v4_path),
            "sample_count": _count_jsonl(frozen_v4_path),
        },
        "adapters": adapters,
        "warnings": [
            "P3-Limited is NOT a formal capability claim — controlled comparison only.",
            "Best honest pass@1 = 0.61% (Repair-Limited); 2300-sample formal threshold unreachable.",
            "No model weights committed; adapter_model.safetensors excluded by .gitignore.",
        ],
    }

    _OUT_JSON.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# P3 Exit Baseline Summary

- **Verdict:** {lock['p3_terminal_verdict']}
- **PR #15 merge:** `{lock['pr_15']['merge_commit_sha']}`
- **Locked at:** {lock['locked_at']}

## Locked artifacts

| Artifact | SHA-256 (first 16) | Samples |
|---|---|---|
| validation-v2 | `{lock['validation_v2']['sha256'][:16]}...` | {lock['validation_v2']['sample_count']} |
| frozen-v4 | `{lock['frozen_v4']['sha256'][:16]}...` | {lock['frozen_v4']['sample_count']} |
| balanced-limited train | `{lock['p3_limited_datasets']['balanced_limited']['train_sha256'][:16]}...` | 1280 |
| repair-limited train | `{lock['p3_limited_datasets']['repair_limited']['train_sha256'][:16]}...` | 1280 |

## Warnings

{chr(10).join(f'- {w}' for w in lock['warnings'])}

## Next phase

P4.0 Agentic Coder Foundation begins. P3 artifacts are frozen.
"""
    _OUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {_OUT_JSON}")
    print(f"Wrote {_OUT_MD}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the script to generate the lock**

Run: `py -3.11 scripts/lock_p3_exit_baseline.py`
Expected: prints two file paths, no errors

- [ ] **Step 7: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_p3_exit_lock.py -v`
Expected: PASS — all 6 tests

- [ ] **Step 8: Commit**

```powershell
git add scripts/lock_p3_exit_baseline.py reports/p4/ tests/test_p3_exit_lock.py .gitignore
git commit -m "feat(p4-agent): Phase A — P3 exit baseline lock"
```

---

## Task 2: Phase B — Action Enums + SafetyFlags + AgentMemory

**Files:**
- Create: `src/agent_actions.py`
- Create: `src/agent_state.py`
- Create: `tests/test_agent_actions.py`

**Interfaces:**
- Consumes: Pydantic v2
- Produces: `ActionType` (11 values), `TaskSuccessCriterion` (3 values), `EvaluationMode` (2 values), `SafetyFlags` (frozen, 5 fields), `AgentMemory` (4 fields)

- [ ] **Step 1: Write failing test**

`tests/test_agent_actions.py`:
```python
"""Phase B: Action schema tests — Part 1: enums and core schemas."""
import pytest
from pydantic import ValidationError

from src.agent_actions import ActionType, SafetyFlags, TaskSuccessCriterion, EvaluationMode
from src.agent_state import AgentMemory


def test_action_type_has_11_values():
    assert len(ActionType) == 11
    assert ActionType.rollback_patch == "rollback_patch"
    assert ActionType.finish == "finish"


def test_task_success_criterion_values():
    assert TaskSuccessCriterion.TEST_PASS == "test_pass"
    assert TaskSuccessCriterion.IDENTIFY_BUG == "identify_bug"
    assert TaskSuccessCriterion.PATCH_APPLIED == "patch_applied"


def test_evaluation_mode_values():
    assert EvaluationMode.REPLAY == "replay"
    assert EvaluationMode.AGENT_RUN == "agent_run"


def test_safety_flags_frozen():
    flags = SafetyFlags(
        modifies_workspace=False, executes_code=False,
        network_required=False, reads_sensitive_path=False, is_terminal=False,
    )
    with pytest.raises(ValidationError):
        flags.modifies_workspace = True


def test_agent_memory_defaults():
    mem = AgentMemory()
    assert mem.notes == ""
    assert mem.hypothesis == ""
    assert mem.failed_attempts == []
    assert mem.last_test_summary == ""


def test_agent_memory_failed_attempts_list():
    mem = AgentMemory(failed_attempts=["tried x", "tried y"])
    assert len(mem.failed_attempts) == 2


def test_safety_flags_network_field_exists():
    flags = SafetyFlags(
        modifies_workspace=False, executes_code=False,
        network_required=True, reads_sensitive_path=False, is_terminal=False,
    )
    assert flags.network_required is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_agent_actions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent_actions'`

- [ ] **Step 3: Implement `src/agent_actions.py`**

```python
"""P4.0 Agent Action Schema — enums and SafetyFlags.

See docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §4.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class ActionType(str, Enum):
    """11 action types (Issue #17 + rollback_patch)."""
    list_files = "list_files"
    read_file = "read_file"
    search_text = "search_text"
    inspect_task = "inspect_task"
    propose_patch = "propose_patch"
    apply_patch = "apply_patch"
    rollback_patch = "rollback_patch"
    run_tests = "run_tests"
    inspect_error = "inspect_error"
    write_memory = "write_memory"
    finish = "finish"


class TaskSuccessCriterion(str, Enum):
    """How a task is judged successful (user fix #1)."""
    TEST_PASS = "test_pass"
    IDENTIFY_BUG = "identify_bug"
    PATCH_APPLIED = "patch_applied"


class EvaluationMode(str, Enum):
    """Evaluator mode (user fix #2)."""
    REPLAY = "replay"
    AGENT_RUN = "agent_run"


class SafetyFlags(BaseModel):
    """Fixed 5-field safety descriptor.

    P4.0 hard-reject: any action with network_required=True or
    reads_sensitive_path=True is rejected at action-validation time.
    """
    model_config = ConfigDict(frozen=True)

    modifies_workspace: bool
    executes_code: bool
    network_required: bool
    reads_sensitive_path: bool
    is_terminal: bool
```

- [ ] **Step 4: Implement `src/agent_state.py`**

```python
"""P4.0 Agent State — AgentMemory schema.

See docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §4.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentMemory(BaseModel):
    """Fixed 4-field agent memory (user-confirmed)."""
    notes: str = ""
    hypothesis: str = ""
    failed_attempts: list[str] = Field(default_factory=list)
    last_test_summary: str = ""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_actions.py -v`
Expected: PASS — all 7 tests

- [ ] **Step 6: Commit**

```powershell
git add src/agent_actions.py src/agent_state.py tests/test_agent_actions.py
git commit -m "feat(p4-agent): Phase B — action enums, SafetyFlags, AgentMemory"
```

---

## Task 3: Phase B — Action Union + Path Validation + Safety Rejection

**Files:**
- Modify: `src/agent_actions.py` (add Action union + path validator)
- Modify: `tests/test_agent_actions.py` (add action + path tests)

**Interfaces:**
- Consumes: `ActionType`, `SafetyFlags` from Task 2
- Produces: `ActionBase`, 11 concrete action classes, `Action` discriminated union, `validate_path()`, `PathValidationError`, `P4ForbiddenActionError`

- [ ] **Step 1: Write failing tests for path validation and actions**

Append to `tests/test_agent_actions.py`:
```python
import pytest
from src.agent_actions import (
    ActionBase, ListFilesAction, ReadFileAction, ApplyPatchAction,
    RollbackPatchAction, FinishAction, validate_path, PathValidationError,
    P4ForbiddenActionError,
)


def test_validate_path_rejects_absolute():
    with pytest.raises(PathValidationError, match="absolute"):
        validate_path("/etc/passwd")


def test_validate_path_rejects_parent_traversal():
    with pytest.raises(PathValidationError, match="parent traversal"):
        validate_path("../secret")


def test_validate_path_rejects_unc():
    with pytest.raises(PathValidationError, match="UNC"):
        validate_path(r"\\server\share")


def test_validate_path_rejects_url():
    with pytest.raises(PathValidationError, match="URL"):
        validate_path("http://evil.com/x")


def test_validate_path_rejects_secret_basename():
    for name in [".env", ".git", "credentials.json", "secret.key", "token"]:
        with pytest.raises(PathValidationError, match="sensitive"):
            validate_path(name)


def test_validate_path_accepts_normal():
    assert validate_path("src/foo.py") == "src/foo.py"
    assert validate_path("solution.py") == "solution.py"


def test_action_with_network_required_rejected():
    with pytest.raises(P4ForbiddenActionError, match="network_required"):
        ApplyPatchAction(
            action_id="a1", action_type=ActionType.apply_patch,
            reason_short="patch", expected_observation="patch_result",
            safety_flags=SafetyFlags(
                modifies_workspace=True, executes_code=False,
                network_required=True, reads_sensitive_path=False, is_terminal=False,
            ),
            arguments={"file_path": "solution.py", "old_text": "a", "new_text": "b"},
        )


def test_action_with_reads_sensitive_rejected():
    with pytest.raises(P4ForbiddenActionError, match="reads_sensitive_path"):
        ReadFileAction(
            action_id="a1", action_type=ActionType.read_file,
            reason_short="read", expected_observation="file_content",
            safety_flags=SafetyFlags(
                modifies_workspace=False, executes_code=False,
                network_required=False, reads_sensitive_path=True, is_terminal=False,
            ),
            arguments={"path": ".env"},
        )


def test_list_files_action_serialization():
    a = ListFilesAction(
        action_id="a1", action_type=ActionType.list_files,
        reason_short="list", expected_observation="file_list",
        safety_flags=SafetyFlags(
            modifies_workspace=False, executes_code=False,
            network_required=False, reads_sensitive_path=False, is_terminal=False,
        ),
        arguments={},
    )
    j = a.model_dump_json()
    a2 = ListFilesAction.model_validate_json(j)
    assert a2.action_id == "a1"


def test_finish_action_with_success_criterion():
    a = FinishAction(
        action_id="a1", action_type=ActionType.finish,
        reason_short="done", expected_observation="finish",
        safety_flags=SafetyFlags(
            modifies_workspace=False, executes_code=False,
            network_required=False, reads_sensitive_path=False, is_terminal=True,
        ),
        arguments={
            "success_criterion": TaskSuccessCriterion.TEST_PASS,
            "tests_passed": True, "identification_verified": False, "summary": "fixed",
        },
    )
    assert a.arguments["success_criterion"] == TaskSuccessCriterion.TEST_PASS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_agent_actions.py -v -k "validate_path or action_with or list_files_action or finish_action_with"`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement Action union + path validator**

Append to `src/agent_actions.py`:
```python
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


_SECRET_BASENAMES = {
    ".env", ".env.local", ".git", ".ssh",
    "credentials", "credentials.json",
    "secrets", "secret.key", "token",
}
_SECRET_PATTERNS = [re.compile(r"\.pem$"), re.compile(r"\.key$")]
_URL_SCHEMES = ("http://", "https://", "ftp://", "file://")


class PathValidationError(ValueError):
    """Raised when a path violates P4.0 workspace path rules."""


class P4ForbiddenActionError(ValueError):
    """Raised when an action violates P4.0 hard safety rules."""


def validate_path(rel_path: str) -> str:
    """Validate a workspace-relative path. Returns normalized path on success."""
    if not rel_path or not isinstance(rel_path, str):
        raise PathValidationError("empty path")
    if rel_path.startswith("/") or (len(rel_path) >= 2 and rel_path[1] == ":"):
        raise PathValidationError(f"absolute path not allowed: {rel_path}")
    if rel_path.startswith("\\\\"):
        raise PathValidationError(f"UNC path not allowed: {rel_path}")
    for scheme in _URL_SCHEMES:
        if rel_path.lower().startswith(scheme):
            raise PathValidationError(f"URL not allowed: {rel_path}")
    parts = rel_path.replace("\\", "/").split("/")
    if ".." in parts:
        raise PathValidationError(f"parent traversal not allowed: {rel_path}")
    basename = parts[-1].lower()
    if basename in _SECRET_BASENAMES:
        raise PathValidationError(f"sensitive path not allowed: {rel_path}")
    for pat in _SECRET_PATTERNS:
        if pat.search(basename):
            raise PathValidationError(f"sensitive path not allowed: {rel_path}")
    return rel_path


class ActionBase(BaseModel):
    """Base for all actions."""
    model_config = ConfigDict(use_enum_values=True)

    action_id: str
    action_type: ActionType
    reason_short: str = Field(max_length=120)
    expected_observation: str
    safety_flags: SafetyFlags

    @model_validator(mode="after")
    def _reject_forbidden_safety(self):
        if self.safety_flags.network_required:
            raise P4ForbiddenActionError(
                f"network_required=True is forbidden in P4.0 (action {self.action_id})"
            )
        if self.safety_flags.reads_sensitive_path:
            raise P4ForbiddenActionError(
                f"reads_sensitive_path=True is forbidden in P4.0 (action {self.action_id})"
            )
        return self


class ListFilesArgs(BaseModel):
    pattern: str | None = None


class ReadFileArgs(BaseModel):
    path: str
    start_line: int | None = None
    end_line: int | None = None

    @model_validator(mode="after")
    def _validate(self):
        validate_path(self.path)
        return self


class SearchTextArgs(BaseModel):
    query: str
    file_glob: str | None = None
    max_results: int = 20


class InspectTaskArgs(BaseModel):
    pass


class ProposePatchArgs(BaseModel):
    file_path: str
    old_text: str
    new_text: str

    @model_validator(mode="after")
    def _validate(self):
        validate_path(self.file_path)
        if not self.old_text:
            raise ValueError("old_text must be non-empty")
        return self


class ApplyPatchArgs(BaseModel):
    file_path: str
    old_text: str
    new_text: str
    expected_before_sha256: str | None = None

    @model_validator(mode="after")
    def _validate(self):
        validate_path(self.file_path)
        if not self.old_text:
            raise ValueError("old_text must be non-empty")
        return self


class RollbackPatchArgs(BaseModel):
    action_id: str


class RunTestsArgs(BaseModel):
    test_path: str | None = None
    timeout_s: float = 10.0


class InspectErrorArgs(BaseModel):
    error_source: Literal["last_test", "last_patch"]


class WriteMemoryArgs(BaseModel):
    memory: "AgentMemory"


class FinishArgs(BaseModel):
    success_criterion: TaskSuccessCriterion
    tests_passed: bool
    identification_verified: bool
    summary: str = Field(max_length=500)


class ListFilesAction(ActionBase):
    action_type: Literal[ActionType.list_files] = ActionType.list_files
    arguments: ListFilesArgs = Field(default_factory=ListFilesArgs)


class ReadFileAction(ActionBase):
    action_type: Literal[ActionType.read_file] = ActionType.read_file
    arguments: ReadFileArgs


class SearchTextAction(ActionBase):
    action_type: Literal[ActionType.search_text] = ActionType.search_text
    arguments: SearchTextArgs


class InspectTaskAction(ActionBase):
    action_type: Literal[ActionType.inspect_task] = ActionType.inspect_task
    arguments: InspectTaskArgs = Field(default_factory=InspectTaskArgs)


class ProposePatchAction(ActionBase):
    action_type: Literal[ActionType.propose_patch] = ActionType.propose_patch
    arguments: ProposePatchArgs


class ApplyPatchAction(ActionBase):
    action_type: Literal[ActionType.apply_patch] = ActionType.apply_patch
    arguments: ApplyPatchArgs


class RollbackPatchAction(ActionBase):
    action_type: Literal[ActionType.rollback_patch] = ActionType.rollback_patch
    arguments: RollbackPatchArgs


class RunTestsAction(ActionBase):
    action_type: Literal[ActionType.run_tests] = ActionType.run_tests
    arguments: RunTestsArgs = Field(default_factory=RunTestsArgs)


class InspectErrorAction(ActionBase):
    action_type: Literal[ActionType.inspect_error] = ActionType.inspect_error
    arguments: InspectErrorArgs


class WriteMemoryAction(ActionBase):
    action_type: Literal[ActionType.write_memory] = ActionType.write_memory
    arguments: WriteMemoryArgs


class FinishAction(ActionBase):
    action_type: Literal[ActionType.finish] = ActionType.finish
    arguments: FinishArgs


Action = Annotated[
    Union[
        ListFilesAction, ReadFileAction, SearchTextAction, InspectTaskAction,
        ProposePatchAction, ApplyPatchAction, RollbackPatchAction,
        RunTestsAction, InspectErrorAction, WriteMemoryAction, FinishAction,
    ],
    Field(discriminator="action_type"),
]


WriteMemoryArgs.model_rebuild()
```

Note: Need to import `AgentMemory` at top of file or use forward ref. Add to top imports:
```python
from src.agent_state import AgentMemory
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_actions.py -v`
Expected: PASS — all tests (Task 2 + Task 3)

- [ ] **Step 5: Commit**

```powershell
git add src/agent_actions.py tests/test_agent_actions.py
git commit -m "feat(p4-agent): Phase B — Action union with path validation and safety rejection"
```

---

## Tasks 4-12: Continued in next file section

Due to plan length, tasks 4-12 (Phases C-H) follow the same TDD pattern and are summarized below. Full step-by-step detail for each is in the spec §5-§10 and will be expanded by subagents during execution.

### Task 4: Phase C — MicroTaskWorkspace
- Create `src/agent_workspace.py` with `MicroTaskWorkspace.from_task()`, `resolve_path()`, `list_files()`, `file_sha256()`
- 5 tests: copy-to-temp, ignore list, path validation, absolute path return, SHA stability

### Task 5: Phase C — Read-only tools
- Create `src/agent_tools.py` with `tool_list_files`, `tool_read_file`, `tool_search_text`, `tool_inspect_task`
- 7 tests covering each tool + edge cases (binary, line range, max_results, README parsing)

### Task 6: Phase C — apply_patch + rollback_patch + audit
- Extend `src/agent_tools.py` with `tool_apply_patch`, `tool_propose_patch`, `tool_rollback_patch`
- 9 tests: success, not-found, ambiguous, SHA mismatch, audit record, propose-doesn't-modify, rollback success, unknown action_id, double-rollback

### Task 7: Phase C — run_tests + inspect_error + write_memory + finish
- Extend `src/agent_tools.py` with remaining 4 tools
- 6 tests: failing test, passing test, timeout, inspect_error, write_memory, finish

### Task 8: Phase D — Trajectory schema
- Create `src/agent_trajectory.py` with `TrajectoryStep`, `Trajectory`, `load_trajectory`, `save_trajectory`, `is_mutating_action`
- 7 tests: step_index monotonic, must-end-with-finish, success-requires-verified, memory chain, round-trip, tool_distribution, mutating action check

### Task 9: Phase E — 40 micro-tasks
- Create `scripts/build_micro_tasks.py` (generator with 40 task definitions)
- Run script to generate `data/p4-agent/micro-tasks-v0/task_001-040/` + `manifest.json`
- 6 tests: manifest exists, 40 tasks, dirs exist, baseline fails, post-patch passes, no frozen v4 overlap

### Task 10: Phase F — Scripted trajectories
- Create `scripts/generate_scripted_agent_trajectories.py` with `_build_trajectory()` and `_build_locate_trajectory()`
- Run script to generate `data/p4-agent/trajectories-v0/scripted.jsonl`
- 4 tests: file exists, 40 trajectories, all success_label=True, pytest replay passes

### Task 11: Phase G — Evaluator
- Create `src/agent_evaluator.py` with `ActionProvider`, `ScriptedActionProvider`, `ReplayActionProvider`, `CorruptedActionProvider`, `AgentEvaluator`
- Create `scripts/evaluate_agent_policy.py`
- 4 tests: replay provider, replay success, corrupted injection, all metrics present

### Task 12: Phase H — Readiness verifier
- Create `scripts/verify_p4_readiness.py` with 11 gates
- Run to generate `reports/p4/p4-agent-foundation-readiness.md`
- 3 tests: report exists, verdict is GO, all 11 gates listed

---

## Self-Review

**1. Spec coverage check:**

| Spec section | Covered by task(s) |
|---|---|
| §1 Purpose, non-goals, security note | Global Constraints |
| §2 Architecture | File Structure + all tasks |
| §3 Phase A | Task 1 |
| §4 Phase B (11 actions, SafetyFlags, AgentMemory, TaskSuccessCriterion, EvaluationMode, path validation, apply_patch semantics) | Tasks 2, 3 |
| §5 Phase C (6 tools, rollback_patch, network hard-fail) | Tasks 4, 5, 6, 7 |
| §6 Phase D (trajectory schema, state-transition invariants) | Task 8 |
| §7 Phase E (40 tasks, 8 types, manifest) | Task 9 |
| §8 Phase F (scripted teacher, per-task functions) | Task 10 |
| §9 Phase G (ActionProvider, 3 providers, 8 metrics, EvaluationMode) | Task 11 |
| §10 Phase H (11 readiness gates, verdict) | Task 12 |
| §11 Testing strategy | All tasks (TDD) |
| §12 Git delivery | Global Constraints + commit steps |
| §13 Reproduction | Each task has run commands |
| §14 Risk mitigation | Path validation (Task 3), network hard-fail (Task 7), supply-chain rule (Global) |
| §15 P4.1 warning | Global Constraints |

**Gaps:** None identified. All spec sections covered.

**2. Placeholder scan:**
- Tasks 4-12 are summarized rather than full step-by-step. This is intentional for plan length — the spec provides full detail and subagents will expand during execution. Each summarized task has clear file paths, interfaces, and test counts.
- No "TBD", "TODO", "implement later" in detailed tasks (1-3).

**3. Type consistency:**
- `ActionType` — 11 values, consistent across Tasks 2, 3, 8, 10, 11
- `TaskSuccessCriterion` — 3 values, used in Task 3 (FinishArgs), Task 10 (trajectory builder), Task 11 (evaluator)
- `EvaluationMode` — 2 values, used in Task 11
- `SafetyFlags` — 5 fields, consistent across Tasks 2, 3
- `AgentMemory` — 4 fields (notes, hypothesis, failed_attempts, last_test_summary), consistent across Tasks 2, 3, 8, 10, 11
- `validate_path` — used in Tasks 3, 4, 5, 6
- `MicroTaskWorkspace` — used in Tasks 4, 5, 6, 7, 9, 10, 11
- `Trajectory` / `TrajectoryStep` — used in Tasks 8, 10, 11
- `ActionProvider` / `ReplayActionProvider` / `CorruptedActionProvider` — used in Task 11

No type mismatches found.
