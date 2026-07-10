# P4.0 Agentic Coder Foundation — Design

- **Issue**: [#17 — P4.0：Agentic Coder Foundation](https://github.com/Leonardo0402/Qwen0.6B-Experiment-Trail/issues/17)
- **Branch**: `feat/p4-agent-foundation`
- **Delivery**: single PR covering phases A–H
- **Spec date**: 2026-07-08
- **Status**: design approved by user on 2026-07-08

## 1. Purpose and Scope

P3 established that the MBPP-family data frontier is exhausted for Qwen3-0.6B
single-shot training (verdict `MBPP_FAMILY_OR_VARIANT_LIMIT`, PR #15 merged).
P4.0 begins the transition from single-shot code generation toward a
**constrained agentic coder** that can observe a workspace, choose a typed
action, call a limited audited tool, and loop until success or stop.

P4.0 delivers **only the foundation**: schemas, safe tool layer, trajectory
format, a 40-task micro suite, scripted teacher trajectories, and a replay
evaluator. **No model training, no model execution, no weight commits.**

### Hard non-goals (Issue #17)

- No full agent training; no SFT / policy training in this phase
- No HumanEval / LeetCode / SWE-bench or any external dataset
- No migration to Qwen3-1.7B or any other base model
- No arbitrary shell / PowerShell / Bash action
- No Git push, network, credential, or project-external file access
- No modification of Frozen Eval v4 or P3 adapters/reports
- No capability claim from loss alone

### Confirmed design decisions (user-approved 2026-07-08)

| # | Decision | Choice |
|---|---|---|
| 1 | Delivery granularity | Single PR on `feat/p4-agent-foundation` |
| 2 | Scripted teacher form | Per-task independent script functions |
| 3 | Micro-task count | 40 tasks (8 types × 5 tasks each) |
| 4 | `apply_patch` format | Custom structured patch with `expected_before_sha256` |
| 5 | Workspace isolation | Persistent task dir (read-only) + runtime temp copy |
| 6 | Phase G model stub | No model load; `ActionProvider` abstraction with `ScriptedActionProvider` / `ReplayActionProvider` / `CorruptedActionProvider` |
| 7 | `safety_flags` schema | Fixed 5-field schema; `network_required` and `reads_sensitive_path` are hard-reject in P4.0 |
| 8 | `memory` schema | Fixed 4-field schema: `notes`, `hypothesis`, `failed_attempts`, `last_test_summary` |

### Security note (supply-chain warning)

Issue #17 received a comment from an unverified account `depucobose87` attaching
`p4_baseline_fix.zip`. This is treated as a potential supply-chain attack.
**No file from Issue/PR comments will be downloaded, inspected, or applied.**
All P4.0 code is written from scratch by the agent under TDD discipline. This
constraint is documented in the readiness report (Phase H).

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   P4.0 Agent Foundation                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  src/agent_actions.py    ← Phase B: Action schema (Pydantic)│
│    Action (typed union)                                    │
│    SafetyFlags (fixed 5-field)                             │
│                                                             │
│  src/agent_state.py      ← Phase B: AgentMemory schema      │
│    AgentMemory (fixed 4-field)                             │
│    WorkspaceSnapshot                                        │
│                                                             │
│  src/agent_workspace.py  ← Phase C: Workspace manager       │
│    MicroTaskWorkspace (persistent dir + temp copy)         │
│    path validation, ignore-list, SHA recording              │
│                                                             │
│  src/agent_tools.py      ← Phase C: Safe tool layer         │
│    list_files, read_file, search_text, apply_patch,         │
│    run_tests, inspect_error (all return typed Observation)  │
│                                                             │
│  src/agent_trajectory.py ← Phase D: Trajectory schema       │
│    TrajectoryStep, Trajectory (JSONL format)                │
│                                                             │
│  src/agent_evaluator.py  ← Phase G: Replay evaluator        │
│    ActionProvider (abstract)                                │
│    ScriptedActionProvider / ReplayActionProvider            │
│    CorruptedActionProvider (for negative tests)             │
│    AgentEvaluator (replay + metric computation)             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  data/p4-agent/                                             │
│    micro-tasks-v0/        ← Phase E: 40 toy workspaces      │
│      task_001/ … task_040/                                  │
│        README.md, solution.py, test_*.py, manifest.json     │
│      manifest.json (registry with per-task SHA)             │
│    trajectories-v0/                                        │
│      scripted.jsonl       ← Phase F: 40 teacher trajectories│
│                                                             │
│  scripts/                                                   │
│    generate_scripted_agent_trajectories.py ← Phase F        │
│    evaluate_agent_policy.py                ← Phase G        │
│    verify_p4_readiness.py                  ← Phase H        │
│    lock_p3_exit_baseline.py                ← Phase A        │
│                                                             │
│  reports/p4/                                                │
│    p3-exit-baseline-lock.json ← Phase A                    │
│    p3-exit-summary.md         ← Phase A                    │
│    trajectory-schema.md       ← Phase D                    │
│    scripted-trajectories-report.md ← Phase F               │
│    p4-agent-foundation-readiness.md ← Phase H              │
│                                                             │
│  tests/                                                     │
│    test_agent_actions.py    ← Phase B                      │
│    test_agent_tools.py      ← Phase C                      │
│    test_agent_trajectory.py ← Phase D                      │
│    test_micro_task_suite.py ← Phase E                      │
│    test_agent_evaluator.py  ← Phase G                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data flow

```
MicroTaskWorkspace (persistent, read-only)
  │
  ├──(copy at start)──► temp_workspace/
  │                       │
  │                       ├── list_files ──► FileListObservation
  │                       ├── read_file  ──► FileContentObservation
  │                       ├── search_text──► SearchObservation
  │                       ├── apply_patch ──► PatchObservation (+ audit record)
  │                       ├── run_tests  ──► TestObservation
  │                       └── inspect_error──► ErrorObservation
  │
  └──(SHA recorded before & after patch)
```

## 3. Phase A — P3 Exit Baseline Lock

### Deliverables

- `reports/p4/p3-exit-baseline-lock.json`
- `reports/p4/p3-exit-summary.md`
- `scripts/lock_p3_exit_baseline.py` (deterministic generator)

### `p3-exit-baseline-lock.json` schema

```json
{
  "schema_version": 1,
  "locked_at": "2026-07-08T...",
  "p3_terminal_verdict": "MBPP_FAMILY_OR_VARIANT_LIMIT",
  "pr_15": {
    "merge_commit_sha": "d91586e0d31214f4ed3edbdce524e6b0e8067070",
    "title": "feat(p3-limited): controlled experiment — balanced vs repair-biased training (1280 samples, Issue #16 fixed) (#15)"
  },
  "p3_limited_datasets": {
    "balanced_limited": {
      "train_sha256": "2c671ed8d6df8f87198c96760cca08f2991861fe96e1a5950a137e85aaffc06c",
      "manifest_path": "data/p3-limited/balanced-limited/manifest.json",
      "total_samples": 1280
    },
    "repair_limited": {
      "train_sha256": "f3518acf10661bd5ca25a7051ec4088f09b0c199b5eab0bff63f92578ef57ecd",
      "manifest_path": "data/p3-limited/repair-limited/manifest.json",
      "total_samples": 1280
    }
  },
  "validation_v2": {
    "path": "data/p3-curriculum/validation-v2/validation.jsonl",
    "sha256": "<computed at lock time>",
    "sample_count": 180
  },
  "frozen_v4": {
    "path": "data/frozen-eval/v4/test_raw.jsonl",
    "sha256": "<computed at lock time>",
    "sample_count": 465
  },
  "adapters": {
    "balanced_limited": {
      "metrics_path": "adapters/p3/balanced-limited/metrics.json",
      "metrics_sha256": "<computed>",
      "train_hash_in_metrics": "2c671ed8..."
    },
    "repair_limited": {
      "metrics_path": "adapters/p3/repair-limited/metrics.json",
      "metrics_sha256": "<computed>",
      "train_hash_in_metrics": "f3518acf..."
    }
  },
  "warnings": [
    "P3-Limited is NOT a formal capability claim — controlled comparison only.",
    "Best honest pass@1 = 0.61% (Repair-Limited); 2300-sample formal threshold unreachable.",
    "No model weights committed; adapter_model.safetensors excluded by .gitignore."
  ]
}
```

### Acceptance

- Report cites exact file paths and SHA-256 hashes (computed at lock time, not hardcoded)
- No model weights committed
- Existing P3 artifacts remain unchanged (verified by SHA comparison post-lock)
- `scripts/lock_p3_exit_baseline.py` is idempotent — re-running produces identical JSON

## 4. Phase B — Agent Action Schema v0

### Deliverables

- `src/agent_actions.py`
- `src/agent_state.py`
- `tests/test_agent_actions.py`

### Action type enum (11 types — Issue #17 + `rollback_patch`)

```python
class ActionType(str, Enum):
    list_files = "list_files"
    read_file = "read_file"
    search_text = "search_text"
    inspect_task = "inspect_task"
    propose_patch = "propose_patch"
    apply_patch = "apply_patch"
    rollback_patch = "rollback_patch"   # transactional rollback (user fix #5)
    run_tests = "run_tests"
    inspect_error = "inspect_error"
    write_memory = "write_memory"
    finish = "finish"
```

### Task success criterion (user fix #1)

Different task types have different success conditions. The `finish` action
must declare which criterion it claims to satisfy, and the evaluator verifies
that the criterion was actually met.

```python
class TaskSuccessCriterion(str, Enum):
    TEST_PASS = "test_pass"            # finish.tests_passed=True required
    IDENTIFY_BUG = "identify_bug"      # finish.identification_verified=True required
    PATCH_APPLIED = "patch_applied"    # at least one successful apply_patch required
```

- `locate_failing_function` tasks → `IDENTIFY_BUG`
- All other 7 task types → `TEST_PASS`
- `PATCH_APPLIED` is available but not used by any P4.0 task type (reserved for
  future "patch only, no test" tasks)

### Evaluation mode (user fix #2)

The evaluator distinguishes replay (scripted) vs. real agent runs. The two
modes track different metric sets because a scripted trajectory is known-good
and a model agent explores.

```python
class EvaluationMode(str, Enum):
    REPLAY = "replay"        # scripted trajectory replayed through real tools
    AGENT_RUN = "agent_run"  # model/self-driven exploration (P4.1+)
```

**Replay mode metrics** (P4.0): trajectory correctness, tool correctness,
schema correctness, replay success rate.

**Agent mode metrics** (P4.1+): task success, step efficiency, recovery rate,
tool error recovery.

P4.0 evaluator runs in `REPLAY` mode only. The `AGENT_RUN` metric set is
defined in the schema but not computed in P4.0.

### SafetyFlags (fixed 5-field, user-confirmed)

```python
class SafetyFlags(BaseModel):
    model_config = ConfigDict(frozen=True)

    modifies_workspace: bool   # writes files (apply_patch)
    executes_code: bool        # runs code (run_tests)
    network_required: bool     # needs network — P4.0 always reject
    reads_sensitive_path: bool # reads .git/secrets/credentials — P4.0 always reject
    is_terminal: bool          # ends agent loop (finish only)
```

**P4.0 hard reject**: any action with `network_required=True` or
`reads_sensitive_path=True` is rejected at validation time, before tool dispatch.

### Action base + typed union

```python
class ActionBase(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    action_id: str             # UUID4 or monotonic id
    action_type: ActionType
    reason_short: str          # ≤120 chars, human-readable
    expected_observation: str  # e.g. "file_list", "test_result"
    safety_flags: SafetyFlags

# Each concrete action extends ActionBase and adds `arguments`
class ListFilesAction(ActionBase):
    action_type: Literal[ActionType.list_files] = ActionType.list_files
    arguments: ListFilesArgs  # empty / optional glob pattern

class ReadFileAction(ActionBase):
    action_type: Literal[ActionType.read_file] = ActionType.read_file
    arguments: ReadFileArgs   # path, start_line?, end_line?

# ... one per ActionType ...
```

### Per-action arguments

| Action | Arguments |
|---|---|
| `list_files` | `{pattern?: str}` |
| `read_file` | `{path: str, start_line?: int, end_line?: int}` |
| `search_text` | `{query: str, file_glob?: str, max_results?: int}` |
| `inspect_task` | `{}` (reads task README from workspace) |
| `propose_patch` | `{file_path: str, old_text: str, new_text: str}` |
| `apply_patch` | `{file_path: str, old_text: str, new_text: str, expected_before_sha256?: str}` |
| `rollback_patch` | `{action_id: str}` — reverts a prior `apply_patch` by its audit `action_id` |
| `run_tests` | `{test_path?: str, timeout_s?: float}` |
| `inspect_error` | `{error_source: "last_test" \| "last_patch"}` |
| `write_memory` | `{memory: AgentMemory}` |
| `finish` | `{success_criterion: TaskSuccessCriterion, tests_passed: bool, identification_verified: bool, summary: str}` |

### `finish` arguments clarified (user fix #1)

- `success_criterion`: declares which criterion this finish claims
- `tests_passed`: True iff the last `run_tests` call passed (evaluator-enforced)
- `identification_verified`: True iff the agent has run `inspect_task` +
  `read_file` + `inspect_error` at least once (evaluator-enforced for
  `IDENTIFY_BUG` criterion)
- `summary`: free-text, ≤500 chars

Evaluator rejects `finish` if the declared `success_criterion` is not met:
- `TEST_PASS` requires `tests_passed=True`
- `IDENTIFY_BUG` requires `identification_verified=True`
- `PATCH_APPLIED` requires ≥1 successful `apply_patch` in the trajectory

### Path validation rules (apply to all path-bearing arguments)

- Must be workspace-relative (no leading `/`, no drive letter)
- No parent traversal: reject any segment equal to `..`
- No absolute paths, no UNC paths, no `~` expansion
- No URL-like values (`http://`, `https://`, `ftp://`)
- No path that resolves outside workspace root after normalization
- No secret-like path names: reject if basename matches
  `{.env, .git, .ssh, credentials, secrets, *.key, *.pem, token}`

### `apply_patch` semantics (user-confirmed with enhancement)

1. `file_path` must be workspace-relative and pass path validation
2. `old_text` must occur **exactly once** in the current file content
   - 0 matches → `PatchError("old_text not found")`, file unchanged
   - ≥2 matches → `PatchError("old_text ambiguous: N matches")`, file unchanged
3. If `expected_before_sha256` is provided:
   - compute SHA-256 of current file content
   - if mismatch → `PatchError("expected_before_sha256 mismatch")`, file unchanged
4. Compute `before_sha256` (current content)
5. Apply patch: replace `old_text` with `new_text`
6. Compute `after_sha256` (new content)
7. Write new content to file
8. Emit audit record: `{file_path, before_sha256, after_sha256, patch_applied_at}`
9. Return `PatchObservation` with before/after SHA

**Failed patches must not modify files** — steps 2-3 reject before step 5.

### AgentMemory (fixed 4-field, user-confirmed)

```python
class AgentMemory(BaseModel):
    notes: str                          # free-text working notes
    hypothesis: str                     # current hypothesis about the bug
    failed_attempts: list[str]          # list of tried-and-failed approaches
    last_test_summary: str              # summary of most recent test run output
```

### Acceptance tests (Phase B)

- Schema round-trip: action → JSON → action equality
- Invalid path tests: absolute, `..`, UNC, URL, secret-like → all reject
- Forbidden tool tests: `network_required=True` and `reads_sensitive_path=True`
  actions raise at validation
- `apply_patch` argument validation: missing field, empty `old_text`,
  non-unique `old_text` (tested at tool layer, but schema rejects empty)
- Serialization compatibility: all 10 action types serialize/deserialize cleanly

## 5. Phase C — Safe Tool Layer v0

### Deliverables

- `src/agent_tools.py`
- `src/agent_workspace.py`
- `tests/test_agent_tools.py`

### `MicroTaskWorkspace` (Phase C workspace manager)

```python
class MicroTaskWorkspace:
    def __init__(self, persistent_dir: Path, workspace_root: Path):
        self.persistent_dir = persistent_dir   # read-only source
        self.workspace_root = workspace_root   # temp working copy

    @classmethod
    def from_task(cls, task_dir: Path) -> "MicroTaskWorkspace":
        """Copy task_dir to a fresh tempfile.mkdtemp() and return workspace."""
        ...

    def resolve_path(self, rel_path: str) -> Path:
        """Validate and resolve workspace-relative path. Raises on violation."""
        ...

    def list_files(self, pattern: str | None = None) -> list[str]: ...
    def read_file(self, path: str, start_line: int | None, end_line: int | None) -> str: ...
    def search_text(self, query: str, file_glob: str | None, max_results: int) -> list[SearchHit]: ...
    def apply_patch(self, file_path: str, old_text: str, new_text: str, expected_before_sha256: str | None) -> PatchResult: ...
    def run_tests(self, test_path: str | None, timeout_s: float) -> TestResult: ...
    def file_sha256(self, rel_path: str) -> str: ...
```

### Ignore list (list_files / read_file / search_text)

```
.git, __pycache__, .pytest_cache, .mypy_cache,
*.pyc, *.pyo,
.env, .env.local, credentials.json, *.key, *.pem,
adapter_model.safetensors, *.bin, *.pt, *.ckpt
```

### Tool: `list_files`

- Lists only files under `workspace_root`
- Respects ignore list
- Optional glob pattern (default `**/*.py` plus `*.md`, `*.json`, `*.txt`)
- Returns `FileListObservation(paths: list[str])`

### Tool: `read_file`

- Workspace-relative path only
- Size limit: 64 KB (reject larger)
- Text-only: reject if content contains NUL byte or fails UTF-8 decode
- Line-range support: `start_line` / `end_line` (1-indexed, inclusive)
- Returns `FileContentObservation(path, content, line_count, truncated: bool)`

### Tool: `search_text`

- Literal substring search (no regex)
- Bounded result count: default 20, max 100
- Searches text files only (skip binary by NUL-byte detection)
- Optional `file_glob` filter
- Returns `SearchObservation(hits: list[SearchHit])` where
  `SearchHit = {file_path, line_number, line_text}`

### Tool: `propose_patch`

- Same arguments as `apply_patch` but **does not modify the file**
- Validates that `old_text` occurs exactly once (same rule as `apply_patch`)
- Computes what `before_sha256` and `after_sha256` would be
- Returns `PatchProposalObservation(file_path, before_sha256, after_sha256, would_succeed: bool, error: str | None)`
- Purpose: lets the agent (or evaluator) check a patch before committing it

### Tool: `apply_patch`

Implements the semantics in §4 "apply_patch semantics".

- Backup before write: `file.bak.<sha256[:8]>` created in workspace
- Reject binary files (NUL byte detection)
- Reject patch outside workspace (path validation)
- Produce before/after SHA-256
- Audit record appended to `workspace_root/.audit/patches.jsonl`
- **Every successful `apply_patch` records `action_id` in the audit line so it
  can be rolled back**

Audit record format (extended with `action_id`):
```json
{"timestamp": "...", "action_id": "...", "file_path": "...", "before_sha256": "...", "after_sha256": "...", "success": true, "rolled_back": false}
```

Returns `PatchObservation(file_path, before_sha256, after_sha256, backup_path, action_id, success: bool, error: str | None)`.

### Tool: `rollback_patch` (user fix #5)

Reverts a prior successful `apply_patch` by its `action_id`. Lets the agent
recover from a patch that applied successfully but was the wrong fix.

**Semantics:**
1. Look up `action_id` in `workspace_root/.audit/patches.jsonl`
2. If not found, or if the patch was not successful, or if it was already
   rolled back → `RollbackError("action_id not reversible")`
3. Swap file content back: current content has `after_sha256`; restore the
   backup so file content has `before_sha256` again
4. Mark the audit record `rolled_back=true`
5. Write a new audit line for the rollback itself

Returns `PatchObservation(file_path, before_sha256=current_after, after_sha256=original_before, backup_path=None, action_id=new_rollback_action_id, success: bool, error: str | None)`.

**Use case**: agent applies a patch → `run_tests` fails → `rollback_patch`
→ new hypothesis → `apply_patch` again with a different fix.

### Tool: `run_tests`

- Python pytest only
- Reuses `src/sandbox.py::run_pytest` for isolation (temp dir, timeout, output cap)
- Default timeout: 10s; max: 30s
- Output cap: 8 KB stdout + 8 KB stderr
- **Network hard-disabled (user fix #3): `network_required=False` is a hard
  parameter. If the runtime cannot guarantee network isolation, raise
  `ToolUnavailableError("network isolation unavailable")` instead of
  executing with a silent fallback.** Implementation: detect availability of
  network isolation via OS-level mechanisms (on Linux: check
  `/proc/self/ns/net` namespace isolation OR set `unshare -n` if available;
  on Windows CI: assume not available and require explicit
  `P4_ALLOW_NETWORK=0` env var + pytest `-p no:cacheprovider`).
  **AGENTS.md rule: no silent fallback.** If isolation cannot be guaranteed,
  the tool fails loudly.
- Test path: workspace-relative, must match `test_*.py` or `*_test.py`
- Returns `TestObservation(passed, num_collected, num_passed, num_failed, timed_out, stdout, stderr, duration_s)`

### Tool: `inspect_error`

- `error_source="last_test"`: returns last `TestObservation` stderr
- `error_source="last_patch"`: returns last `PatchObservation` error
- Returns `ErrorObservation(source, content)`

### Tool: `inspect_task`

- Reads `README.md` from workspace root
- README.md must follow this format (enforced by `inspect_task`):
  ```markdown
  # Task: <task_id>

  ## Goal
  <one-sentence description of what the agent must accomplish>

  ## Constraints
  - <constraint 1, e.g. "do not modify test_*.py files">
  - <constraint 2>

  ## Hints
  - <optional hint, e.g. "the bug is in the add() function">
  ```
  Parser extracts `Goal`, `Constraints` (list), `Hints` (list) from these sections.
  Missing `## Goal` section → `TaskError("README.md missing Goal section")`.
- Returns `TaskObservation(goal: str, constraints: list[str], hints: list[str])`

### Tool: `write_memory`

- Validates `AgentMemory` schema
- Stores in evaluator state (does not write to workspace)
- Returns `MemoryObservation(memory_before, memory_after)`

### Tool: `finish`

- Must include `tests_passed: bool` and `summary: str`
- `tests_passed=True` requires that the last `run_tests` call passed
  (enforced in evaluator, not tool layer)
- Returns `FinishObservation(tests_passed, summary)`

### Audit records

Every mutating tool (`apply_patch`) writes a JSONL line to
`workspace_root/.audit/patches.jsonl`:

```json
{"timestamp": "...", "action_id": "...", "file_path": "...", "before_sha256": "...", "after_sha256": "...", "success": true}
```

### Acceptance tests (Phase C)

- All tool outputs are typed observations (Pydantic models)
- Every mutating tool creates an audit record
- Failed patches do not modify files (content equality test)
- Timeout tested (sleep test file)
- Output cap tested (print-loop test file)
- Path validation: all 6 violation classes tested
- Ignore list tested (`.git/`, `__pycache__/`, `*.pyc` filtered)
- `apply_patch` non-unique `old_text` rejected, file unchanged
- `apply_patch` `expected_before_sha256` mismatch rejected, file unchanged

## 6. Phase D — Agent Trajectory Schema v0

### Deliverables

- `src/agent_trajectory.py`
- `data/p4-agent/tasks-smoke/` (1 smoke task for schema tests)
- `reports/p4/trajectory-schema.md`

### Trajectory JSONL format (one JSON object per step)

```json
{
  "trajectory_id": "traj_001",
  "task_id": "task_001",
  "workspace_id": "micro-tasks-v0/task_001",
  "step_index": 0,
  "goal": "Fix the failing test in solution.py",
  "state_summary": "initial state, tests failing",
  "observation": {"kind": "task", "goal": "...", "constraints": [...]},
  "action": {"action_id": "...", "action_type": "list_files", "arguments": {...}, ...},
  "result": {"kind": "file_list", "paths": [...]},
  "memory_before": {"notes": "", "hypothesis": "", "failed_attempts": [], "last_test_summary": ""},
  "memory_after": {"notes": "workspace has 3 files", "hypothesis": "", "failed_attempts": [], "last_test_summary": ""},
  "success_label": false,
  "source": "scripted",
  "verified": true
}
```

### Schema invariants (enforced by `TrajectoryStep` validator)

- `step_index` monotonically increasing from 0
- `observation.kind` and `result.kind` must be consistent with `action.action_type`
  (e.g. `list_files` action → `result.kind == "file_list"`)
- `memory_before` of step N == `memory_after` of step N-1 (for N ≥ 1)
- Terminal step must have `action.action_type == "finish"`
- `success_label=True` requires `verified=True` and terminal `finish.tests_passed=True`

### `Trajectory` model

```python
class Trajectory(BaseModel):
    trajectory_id: str
    task_id: str
    workspace_id: str
    goal: str
    steps: list[TrajectoryStep]
    final_success: bool
    final_verified: bool
    source: Literal["human", "teacher_model", "scripted", "model_self_run"]
    action_count: int
    tool_distribution: dict[str, int]  # action_type -> count
```

### Acceptance tests (Phase D)

- Validate full trajectory order (step_index monotonic)
- Reject missing observations
- Reject action/result kind mismatch
- Reject trajectory without terminal `finish`
- Reject `success_label=True` without `verified=True`
- Reject `memory_before`/`memory_after` chain break
- Round-trip: trajectory → JSONL → trajectory equality

## 7. Phase E — Micro Task Suite v0

### Deliverables

- `data/p4-agent/micro-tasks-v0/` directory
- 40 task directories: `task_001/` … `task_040/`
- `data/p4-agent/micro-tasks-v0/manifest.json` (registry)
- `tests/test_micro_task_suite.py`

### Task type allocation (8 types × 5 tasks = 40)

| Type | Description | task_XXX range |
|---|---|---|
| 1. locate failing function | Identify which function has the bug (no fix needed, finish after identify) | 001–005 |
| 2. one-line bug fix | Single-line correction (typo, off-by-one, wrong operator) | 006–010 |
| 3. add missing boundary check | Add `if x < 0: raise ValueError(...)` or similar | 011–015 |
| 4. update small helper | Modify a helper to match updated spec in README | 016–020 |
| 5. repair after pytest failure | Fix code based on pytest error message | 021–025 |
| 6. avoid editing tests | Bug is in `solution.py`; test file must not be modified | 026–030 |
| 7. recover from failed patch | First patch attempt is wrong; must inspect error and retry | 031–035 |
| 8. finish only after tests pass | Multi-step: patch → test fails → patch → test passes → finish | 036–040 |

### Per-task directory structure

```
task_001/
  README.md          # task description, goal, constraints
  solution.py        # code with a bug (baseline fails pytest)
  test_solution.py   # pytest tests (must not be modified by agent)
  expected_patch.json # reference patch: {file_path, old_text, new_text}
  manifest.json      # {task_id, task_type, difficulty, sha256 of each file}
```

### `manifest.json` (registry)

```json
{
  "schema_version": 1,
  "suite_name": "micro-tasks-v0",
  "total_tasks": 40,
  "task_types": ["locate_failing_function", "one_line_fix", ...],
  "tasks": [
    {
      "task_id": "task_001",
      "task_type": "locate_failing_function",
      "difficulty": 1,
      "path": "task_001",
      "readme_sha256": "...",
      "solution_sha256": "...",
      "test_sha256": "...",
      "expected_patch_sha256": "...",
      "baseline_test_passes": false,
      "post_patch_test_passes": true
    }
  ]
}
```

### Acceptance tests (Phase E)

- No overlap with Frozen v4 sample IDs / family IDs (verified by comparing
  `data/frozen-eval/v4/test_raw.jsonl` sample_ids — micro-tasks use
  `p4_task_NNN` IDs, no MBPP families)
- Pytest baseline fails before patch (each task verified)
- Pytest passes after reference patch (each task verified)
- Reference patch is minimal (single `old_text`/`new_text` per task)
- Per-task SHA recorded in manifest
- `test_micro_task_suite.py` iterates all 40 tasks and verifies the above

## 8. Phase F — Scripted Teacher Trajectories v0

### Deliverables

- `scripts/generate_scripted_agent_trajectories.py`
- `data/p4-agent/trajectories-v0/scripted.jsonl`
- `reports/p4/scripted-trajectories-report.md`

### Scripted policy form (user-confirmed: per-task independent scripts)

Each task gets a dedicated Python function in
`scripts/generate_scripted_agent_trajectories.py`:

```python
def _trajectory_task_001(ws: MicroTaskWorkspace, mem: AgentMemory) -> list[TrajectoryStep]:
    """Locate failing function: list → read → run_tests → inspect_error → write_memory → finish."""
    steps = []
    # Step 0: inspect_task
    steps.append(_make_step(0, "inspect_task", ws, mem, ...))
    # Step 1: list_files
    steps.append(_make_step(1, "list_files", ws, mem, ...))
    # ... continue pattern ...
    return steps
```

### Minimum trajectory pattern (Issue #17)

```
inspect_task → list_files → read_file → run_tests → inspect_error
→ propose_patch → apply_patch → run_tests → write_memory → finish
```

Variations per task type:
- **locate failing function**: stops after `inspect_error + write_memory + finish`
  (no patch needed, just identify)
- **one-line fix**: full pattern, single patch
- **add boundary check**: full pattern, single patch
- **update helper**: full pattern, single patch
- **repair after pytest**: full pattern, single patch
- **avoid editing tests**: includes `read_file(test_*.py)` to confirm tests,
  patches only `solution.py`
- **recover from failed patch**: two `apply_patch` calls (first fails,
  `inspect_error`, second succeeds)
- **finish only after tests pass**: two patch-test cycles

### Generation script

```python
TRAJECTORY_FUNCTIONS = {
    "task_001": _trajectory_task_001,
    # ... 40 entries ...
}

def main():
    trajectories = []
    for task_id, fn in TRAJECTORY_FUNCTIONS.items():
        task_dir = ROOT / "data/p4-agent/micro-tasks-v0" / task_id
        ws = MicroTaskWorkspace.from_task(task_dir)
        mem = AgentMemory(notes="", hypothesis="", failed_attempts=[], last_test_summary="")
        steps = fn(ws, mem)
        traj = Trajectory(
            trajectory_id=f"traj_{task_id}",
            task_id=task_id,
            workspace_id=f"micro-tasks-v0/{task_id}",
            goal=ws.read_task_goal(),
            steps=steps,
            final_success=steps[-1].action.arguments.get("tests_passed", False),
            final_verified=True,
            source="scripted",
            action_count=len(steps),
            tool_distribution=_compute_dist(steps),
        )
        trajectories.append(traj)

    # Write JSONL
    out_path = ROOT / "data/p4-agent/trajectories-v0/scripted.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            for step in traj.steps:
                f.write(step.model_dump_json() + "\n")

    # Report
    ...
```

### Acceptance tests (Phase F)

- 40 successful trajectories (one per task)
- Each trajectory verified by pytest (replay patch → run tests → pass)
- No failed trajectory marked success
- Action count and tool distribution reported per trajectory and in aggregate
- Trajectory schema validation passes (all 40 trajectories valid)
- `scripted-trajectories-report.md` includes:
  - total trajectories: 40
  - total steps: N
  - mean steps per trajectory: N/40
  - tool distribution table
  - per-task-type breakdown

## 9. Phase G — Agent Evaluation Harness v0

### Deliverables

- `src/agent_evaluator.py`
- `scripts/evaluate_agent_policy.py`
- `tests/test_agent_evaluator.py`

### `ActionProvider` abstraction (user-confirmed)

```python
class ActionProvider(ABC):
    @abstractmethod
    def next_action(self, state: AgentState) -> Action: ...

class ScriptedActionProvider(ActionProvider):
    """Replays actions from a pre-computed trajectory."""
    def __init__(self, trajectory: Trajectory): ...

class ReplayActionProvider(ActionProvider):
    """Alias for ScriptedActionProvider; semantic name for eval."""
    ...

class CorruptedActionProvider(ActionProvider):
    """Injects intentional corruption at step N (for negative tests)."""
    def __init__(self, base: Trajectory, corruption_step: int, corruption: Corruption): ...

class ModelActionProvider(ActionProvider):
    """Reserved for P4.1 — loads Qwen3-0.6B and generates actions."""
    raise NotImplementedError("P4.1")
```

### `AgentEvaluator`

```python
class AgentEvaluator:
    def __init__(self, workspace: MicroTaskWorkspace, provider: ActionProvider, max_steps: int = 20):
        ...

    def run(self) -> EvalResult:
        """Replay the provider's actions through the real tool layer."""
        ...

    def metrics(self) -> dict[str, float]:
        """Return all 8 Issue #17 metrics."""
        ...
```

### Metrics (Issue #17)

| Metric | Type | Description |
|---|---|---|
| `task_success_rate` | float | fraction of tasks where `finish.tests_passed=True` |
| `action_validity_rate` | float | fraction of actions that passed schema validation |
| `tool_error_rate` | float | fraction of tool calls that returned an error |
| `patch_success_rate` | float | fraction of `apply_patch` calls that succeeded |
| `tests_pass_rate` | float | fraction of `run_tests` calls that passed |
| `forbidden_action_count` | int | count of actions rejected for safety violations |
| `max_step_exceeded_count` | int | count of tasks that hit `max_steps` |
| `finish_without_tests_count` | int | count of `finish` actions with `tests_passed=False` |

### `scripts/evaluate_agent_policy.py`

```python
def main():
    tasks = load_micro_task_suite()
    trajectories = load_scripted_trajectories()
    all_results = []

    for traj in trajectories:
        task = tasks[traj.task_id]
        ws = MicroTaskWorkspace.from_task(task.path)
        provider = ReplayActionProvider(traj)
        evaluator = AgentEvaluator(ws, provider, max_steps=20)
        result = evaluator.run()
        all_results.append(result)

    # Compute aggregate metrics
    metrics = aggregate_metrics(all_results)

    # Write report
    report = {
        "eval_hash": "...",
        "config": {"max_steps": 20, "timeout_s": 10},
        "metrics": metrics,
        "per_task": [...],
    }
    write_json("reports/p4/agent-eval-report.json", report)
```

### Acceptance tests (Phase G)

- Scripted trajectories achieve 100% `task_success_rate` on replay
- Intentionally corrupted trajectories fail with precise reasons:
  - Wrong `action_type` → `action_validity_rate < 1.0`
  - Invalid path → `forbidden_action_count > 0`
  - Wrong patch → `patch_success_rate < 1.0`
  - Skip `run_tests` before `finish` → `finish_without_tests_count > 0`
  - Exceed `max_steps` → `max_step_exceeded_count > 0`
- Evaluator report includes SHA and config

## 10. Phase H — P4 Readiness Report

### Deliverables

- `reports/p4/p4-agent-foundation-readiness.md`
- `scripts/verify_p4_readiness.py`

### 11 readiness gates (Issue #17 + user fix #6)

1. P3 exit baseline locked (Phase A report exists with valid SHAs)
2. Action schema tests pass (`test_agent_actions.py`)
3. Tool layer safety tests pass (`test_agent_tools.py`)
4. Trajectory schema tests pass (`test_agent_trajectory.py`)
5. Micro task suite verified (`test_micro_task_suite.py` — 40 tasks, baseline fail, post-patch pass)
6. Scripted trajectories verified (40 trajectories, all schema-valid, all pytest-verified)
7. Evaluator replay success = 100% (scripted trajectories replayed through real tools)
8. Corrupted trajectory tests fail as expected (each corruption type detected)
9. No unrestricted shell / network / Git push action exists (grep source for forbidden patterns)
10. CI green
11. **State Transition Consistency (user fix #6)**: for every trajectory step,
    verify that:
    - the action is schema-valid
    - the observation came from a real tool call (not fabricated)
    - if `memory_after != memory_before`, there is a preceding observation
      that justifies the memory update
    - if `workspace_sha_after != workspace_sha_before`, the step contains an
      `apply_patch` or `rollback_patch` action (no other action may change
      workspace SHA)
    - if the step is terminal `finish`, the declared `success_criterion` was met

    This gate prevents the model from learning "hallucinated state
    transitions" in future P4.1+ training data.

### Verdict

```
GO_FOR_P4_AGENT_SFT_DATA   # all 10 gates pass
FIX_FIRST                  # any gate fails
STOP                       # fundamental design flaw discovered
```

**P4.0 target**: `GO_FOR_P4_AGENT_SFT_DATA`

### `verify_p4_readiness.py`

Runs all 10 gates, prints verdict, writes
`reports/p4/p4-agent-foundation-readiness.md` with the Final Output Format
specified in Issue #17.

## 11. Testing Strategy

### TDD discipline (per `test-driven-development` skill)

Every module follows Red-Green-Refactor:
1. Write failing test for the smallest unit of behavior
2. Verify test fails (Red)
3. Write minimal code to pass (Green)
4. Refactor if needed, verify tests still pass

### Test groups (Issue #17 + user fixes)

| Test file | Phase | Estimated test count |
|---|---|---|
| `test_agent_actions.py` | B | ~30 (adds TaskSuccessCriterion, EvaluationMode, rollback_patch action) |
| `test_agent_tools.py` | C | ~40 (adds rollback_patch tool, network hard-fail, ToolUnavailableError) |
| `test_agent_trajectory.py` | D | ~20 (adds state transition invariants, success_criterion chain) |
| `test_micro_task_suite.py` | E | ~10 (parameterized over 40 tasks) |
| `test_agent_evaluator.py` | G | ~20 (adds EvaluationMode.REPLAY vs AGENT_RUN, state transition gate) |

**Total**: ~120 tests

### CI

All tests are non-GPU (no model load in P4.0). Existing
`.github/workflows/ci-tests.yml` runs `python -m pytest tests/ -v` —
new tests are automatically included.

## 12. Git Delivery

### Branch

```
feat/p4-agent-foundation  (off main @ d91586e)
```

### Commit strategy

Commits are grouped by phase for reviewability:
```
A: lock_p3_exit_baseline + report
B: agent_actions + agent_state + tests
C: agent_tools + agent_workspace + tests
D: agent_trajectory + tests + smoke task
E: 40 micro-tasks + manifest + tests
F: generate_scripted_agent_trajectories + scripted.jsonl + report
G: agent_evaluator + evaluate_agent_policy + tests
H: verify_p4_readiness + readiness report
```

### .gitignore additions

```
# P4.0 audit trails (regenerated at runtime, not committed)
data/p4-agent/micro-tasks-v0/*/.audit/
```

### PR

- Title: `feat(p4-agent): P4.0 Agentic Coder Foundation — schemas, tools, trajectories, evaluator (Issue #17)`
- Body: references this design doc, lists all 8 phases with checkmarks
- `Closes #17` in commit message or PR body

### Non-committed artifacts

- Model weights (none in P4.0)
- `adapter_model.safetensors` (already gitignored)
- `.audit/` directories (regenerated at runtime)
- Temp workspace copies (in `tempfile.gettempdir()`, auto-cleaned)

## 13. Reproduction

```powershell
# Phase A: lock P3 baseline
py -3.11 scripts/lock_p3_exit_baseline.py

# Phase B-D: run schema tests
py -3.11 -m pytest tests/test_agent_actions.py tests/test_agent_tools.py tests/test_agent_trajectory.py -v

# Phase E: verify micro-task suite
py -3.11 -m pytest tests/test_micro_task_suite.py -v

# Phase F: generate scripted trajectories
py -3.11 scripts/generate_scripted_agent_trajectories.py

# Phase G: evaluate (replay scripted trajectories)
py -3.11 scripts/evaluate_agent_policy.py

# Phase H: verify readiness
py -3.11 scripts/verify_p4_readiness.py

# Full test suite
py -3.11 -m pytest tests/ -v --tb=short
```

## 14. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Path traversal / path escape | Strict path validation in `MicroTaskWorkspace.resolve_path()`; 6 violation classes tested |
| `apply_patch` ambiguity | Require unique `old_text` match; reject on 0 or ≥2 matches; file unchanged on reject |
| Test execution escapes sandbox | Reuse `src/sandbox.py` temp-dir + timeout + output cap; no project tree access |
| Supply-chain attack via Issue comments | Documented in design; no zip/exe from comments downloaded; all code written from scratch |
| Scope creep into model training | Hard non-goals in §1; `ModelActionProvider` raises `NotImplementedError("P4.1")` |
| P3 artifact corruption | Phase A locks SHAs; post-implementation verify compares SHAs unchanged |
| CI timeout from 40 tasks × pytest | Each micro-task pytest is <2s; 40 × 2s + overhead < 2min; well under 20min CI limit |
| Trajectory schema drift | `TrajectoryStep` Pydantic validator enforces invariants; round-trip tests |

## 15. Next Step (P4.1, out of scope)

After `GO_FOR_P4_AGENT_SFT_DATA` verdict:

1. Build supervised action-policy dataset from 40 scripted trajectories
2. Implement `ModelActionProvider` with Qwen3-0.6B
3. Decide: SFT on Qwen3-0.6B vs. teacher-distilled trajectories vs. Qwen3-1.7B migration
4. Train and evaluate agent policy

**This design does not authorize any P4.1 work.**

### Critical warning: 40 trajectories are NOT training data (user fix #4)

The 40 scripted trajectories produced in Phase F are **foundation
verification artifacts only**. They are suitable for:
- validating the trajectory schema
- validating the tool loop
- validating the evaluator replay path

They are **NOT suitable as an Agent SFT dataset** because:
- 40 tasks is too small (typical SFT needs 1k–10k examples)
- 8 task types is too narrow a distribution
- scripted teacher policies have extreme bias (one fixed path per task)
- training directly on 40 trajectories would overfit the trajectory format
  rather than learn general agent behavior

**Correct P4 roadmap (user-confirmed 2026-07-08):**
```
P4.0  environment + evaluator (this spec)
  ↓
P4.1  large-scale scripted + teacher-model trajectories (1000+)
  ↓
P4.2  ModelActionProvider + Qwen3-0.6B action generation
  ↓
P4.3  Agent SFT + self-repair loop
```

**Do NOT skip P4.1.** Training on P4.0's 40 trajectories alone is explicitly
forbidden by this design.
