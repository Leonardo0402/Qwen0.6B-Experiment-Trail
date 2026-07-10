# Task 5 Report — Phase D: Corruption test expansion (all 5 CorruptionType values)

**Status:** DONE

**Commit SHA:** `439c0977a159c704bcb13ceb17140fa42c2bb9c0`

**BASE (T4 HEAD):** `50ec3af0c75a8cf147d07c2933bcfea59222c325`

---

## 1. What was implemented

### `tests/test_agent_evaluator.py` (+117 lines, appended after `test_search_text_dispatched`)

Appended one section comment, one helper, and five regression tests covering all
5 `CorruptionType` values. No other files were modified.

- **`_run_corrupted(monkeypatch, corruption_type, step_index=2)`** — shared helper
  that loads the first success trajectory, builds a `MicroTaskWorkspace`, runs the
  evaluator with a `CorruptedActionProvider`, and returns the `EvalResult`. Cleans
  up the workspace in a `finally`.
- **`test_corruption_wrong_action_type`** — `WRONG_ACTION_TYPE` at the default
  step; asserts no crash, metrics present, `steps_executed > 0`.
- **`test_corruption_invalid_path`** — dynamically finds the first step whose
  action arguments expose `path` or `file_path` (avoids the no-op when
  `step_index=2` misses the right action type). Asserts `len(result.errors) > 0`
  and a path/invalid/sensitive keyword in at least one error message.
- **`test_corruption_wrong_patch`** — finds the first `apply_patch` step
  (strengthened from the brief's `apply_patch OR propose_patch` — see Corrections
  below) and asserts `tool_error_rate > 0 OR patch_success_rate < 1`.
- **`test_corruption_skip_tests_before_finish`** — dynamically finds the first
  `run_tests` step so the corruption converts it to `finish` and has a real
  effect. Strengthens the brief's always-true assertion to
  `finish_without_tests_count >= 1` AND `steps_executed < len(traj.steps)`.
- **`test_corruption_exceed_max_steps`** — `step_index=0`, `max_steps=5`;
  asserts `result.max_steps_hit` and `max_step_exceeded_count >= 1`.

### `src/agent_evaluator.py`

**Not modified.** All five tests pass against the existing P4.0 infrastructure.
The `max_step_exceeded_count` metric was already computed at line 428
(`"max_step_exceeded_count": 1 if max_steps_hit else 0`), and the loop already
sets `max_steps_hit = True` after the for-loop exhausts `max_steps` without a
`finish` (line 392). No evaluator bugs were found.

---

## 2. TDD evidence

### RED (tests not yet defined — collection error)

Command:
```
py -3.11 -m pytest tests/test_agent_evaluator.py::test_corruption_wrong_action_type tests/test_agent_evaluator.py::test_corruption_invalid_path tests/test_agent_evaluator.py::test_corruption_wrong_patch tests/test_agent_evaluator.py::test_corruption_skip_tests_before_finish tests/test_agent_evaluator.py::test_corruption_exceed_max_steps -v -p no:warnings
```

Output (exit code 4):
```
ERROR: not found: E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py::test_corruption_wrong_action_type
(no match in any of [<Module test_agent_evaluator.py>])
ERROR: not found: E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py::test_corruption_invalid_path
(no match in any of [<Module test_agent_evaluator.py>])
ERROR: not found: E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py::test_corruption_wrong_patch
(no match in any of [<Module test_agent_evaluator.py>])
ERROR: not found: E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py::test_corruption_skip_tests_before_finish
(no match in any of [<Module test_agent_evaluator.py>])
ERROR: not found: E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py::test_corruption_exceed_max_steps
(no match in any of [<Module test_agent_evaluator.py>])
collected 0 items
============================ no tests ran in 0.62s ============================
```

### First GREEN attempt (1 genuine failure found → fixed)

After appending the brief's code verbatim (minus the duplicate import), the
`test_corruption_wrong_patch` test FAILED because the brief's search finds
`propose_patch` (step 5) before `apply_patch` (step 6). Corrupting
`propose_patch` is a no-op for metrics: `tool_propose_patch` does not raise, and
`propose_patch` is not counted in `total_patches`, so `tool_error_rate=0.0` and
`patch_success_rate=1.0` (from the uncorrupted `apply_patch` at step 6).

Failure output (exit code 1, abbreviated):
```
tests\test_agent_evaluator.py ..F..                                      [100%]
FAILED tests/test_agent_evaluator.py::test_corruption_wrong_patch
    assert (0.0 > 0 or 1.0 < 1)
1 failed, 4 passed in 7.87s
```

### Fix applied

Changed `test_corruption_wrong_patch` to search for `apply_patch` specifically
(matching the existing `test_corrupted_injection` pattern) so the corruption
produces a real, measurable patch failure. `propose_patch` corruption is
documented in a comment as a metric no-op. This is the strengthening the task
instructions asked for ("Where the corruption actually has an effect,
strengthen the assertion to verify the effect").

### GREEN (all 5 pass)

Command:
```
py -3.11 -m pytest tests/test_agent_evaluator.py::test_corruption_wrong_action_type tests/test_agent_evaluator.py::test_corruption_invalid_path tests/test_agent_evaluator.py::test_corruption_wrong_patch tests/test_agent_evaluator.py::test_corruption_skip_tests_before_finish tests/test_agent_evaluator.py::test_corruption_exceed_max_steps -v -p no:warnings
```

Output (exit code 0):
```
collected 5 items
tests\test_agent_evaluator.py .....                                      [100%]
============================== 5 passed in 7.19s ==============================
```

---

## 3. Broader regression check

Command:
```
py -3.11 -m pytest tests/test_agent_evaluator.py tests/test_agent_tools.py tests/test_agent_actions.py -p no:warnings -q --timeout=120
```

Output (exit code 0):
```
E:\agent\Qwen\qwen3-code-lab\src\agent_tools.py:495: PytestCollectionWarning: cannot collect test class 'TestObservation' because it has a __init__ constructor (from: tests/test_agent_tools.py)
  class TestObservation(BaseModel):
....................................................................     [100%]
============================== 68 passed in 8.16s ==============================
```

68 tests passed, 0 failed, 0 errors. The `PytestCollectionWarning` about
`TestObservation` is pre-existing (it's a pydantic model in
`src/agent_tools.py`, not a test class) and unrelated to this task.

---

## 4. Diffstat

```
439c0977a159c704bcb13ceb17140fa42c2bb9c0
 tests/test_agent_evaluator.py | 117 ++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 117 insertions(+)
```

Only `tests/test_agent_evaluator.py` was staged and committed. No
`src/agent_evaluator.py` changes. No `.superpowers/sdd/` docs staged.

---

## 5. Corrections applied

1. **Duplicate import NOT added.** The brief's Step 1 began with
   `from src.agent_evaluator import CorruptedActionProvider, Corruption, CorruptionType`.
   This is a duplicate of the existing import block at lines 14-18 of the test
   file. Per the pre-flight correction, I did NOT add this line — the names were
   already in scope. Only the section comment, the `_run_corrupted` helper, and
   the 5 test functions were appended.

2. **`SKIP_TESTS_BEFORE_FINISH` step_index adjustment.** The brief's default
   `step_index=2` targets `read_file` in the loaded trajectory (`task_006`),
   which is a no-op for `SKIP_TESTS_BEFORE_FINISH` (the corruption only fires on
   `run_tests` actions). I added a search loop to find the first `run_tests`
   step (step 3 in this trajectory) and pass it as `step_index`. Without this,
   the trajectory would complete normally with `finish_without_tests_count=0`
   and the strengthened assertion would fail. Also strengthened the assertion
   from `"finish_without_tests_count" in result.metrics` (always true) to
   `finish_without_tests_count >= 1 AND steps_executed < len(traj.steps)`.

3. **`INVALID_PATH` step_index adjustment.** The brief's default `step_index=2`
   happens to target `read_file` (which has a `path` argument) in this
   trajectory, so it works. However, per the task guidance, I added a dynamic
   search for the first step whose arguments expose `path` or `file_path`, with
   a `pytest.skip` fallback if none exists. This makes the test robust to
   trajectory changes. For `task_006` it resolves to step 2 (`read_file`).

4. **`WRONG_PATCH` search narrowed to `apply_patch`.** The brief searched for
   `apply_patch OR propose_patch` and found `propose_patch` first (step 5).
   Corrupting `propose_patch` is a metric no-op (`tool_propose_patch` returns an
   observation without raising, and `propose_patch` isn't counted in
   `total_patches`), so the brief's assertion `tool_error_rate > 0 OR
   patch_success_rate < 1` failed (both were `0.0` and `1.0` respectively). I
   narrowed the search to `apply_patch` specifically (step 6 in this
   trajectory), where the corruption causes a real patch failure
   (`tool_apply_patch` returns `obs.success=False` → `tool_errors += 1`,
   `successful_patches` stays 0 → `patch_success_rate = 0.0 < 1`). A comment
   documents why `propose_patch` was excluded.

5. **`WRONG_ACTION_TYPE` and `EXCEED_MAX_STEPS`** — used the brief's code as-is.
   Both produced the expected behavior on the first run. No adjustments needed.

6. **`src/agent_evaluator.py` NOT modified.** All metric infrastructure
   (`max_step_exceeded_count`, `max_steps_hit`, `finish_without_tests_count`)
   already existed from P4.0. No evaluator bugs were found.

---

## 6. Self-review findings

### Correctness

- All 5 `CorruptionType` enum values are exercised: `WRONG_ACTION_TYPE`,
  `INVALID_PATH`, `WRONG_PATCH`, `SKIP_TESTS_BEFORE_FINISH`, `EXCEED_MAX_STEPS`.
- Each test uses `monkeypatch.setenv("P4_ALLOW_NETWORK", "0")` (via the helper
  or inline) to prevent network calls during replay.
- Each test cleans up its `MicroTaskWorkspace` in a `finally` block — no
  workspace leaks.
- The trajectory used (`task_006`, 10 steps) has the action diversity needed
  for all 5 corruption types: `read_file` (path), `run_tests`, `apply_patch`,
  `finish`.
- The strengthened `SKIP_TESTS_BEFORE_FINISH` assertion verifies both the
  metric effect (`finish_without_tests_count >= 1`) and the structural effect
  (`steps_executed < len(traj.steps)` — the trajectory finishes early at step 4
  instead of step 10).
- The `INVALID_PATH` assertion passes because the evaluator wraps the
  `PathValidationError` as `"step N: invalid action: parent traversal not
  allowed: ../etc/passwd"`, which contains the substring `"invalid"`.

### Style

- Matches existing test file conventions: `monkeypatch` fixture, `_load_first_success_trajectory()`
  helper, `MicroTaskWorkspace.from_task`, `try/finally` cleanup, inline
  assertion messages with `f"expected ..., got ..."`.
- No emojis. No docstrings added to existing code. No incidental refactors.
- Comments explain the *why* of non-obvious choices (why `apply_patch` not
  `propose_patch`, why dynamic step search for `SKIP_TESTS`/`INVALID_PATH`).

### Surgical-changes check

- Only `tests/test_agent_evaluator.py` was modified (+117 lines, all appended).
- No changes to `src/agent_evaluator.py`, `src/agent_trajectory.py`, or any
  other source file.
- Every changed line traces to the task brief's 5 test functions + 1 helper.
- The `WRONG_PATCH` narrowing and `SKIP_TESTS`/`INVALID_PATH` dynamic search are
  explicitly authorized by the task instructions ("If the default
  `step_index=2` doesn't target the right action type ... find the correct step
  index" and "strengthen the assertion to verify the effect").
- Only `tests/test_agent_evaluator.py` was staged for commit (verified via
  `git diff --stat 50ec3af HEAD` showing a single file).

---

## 7. One-line test summary

All 5 new corruption regression tests pass (plus 63 pre-existing tests in the
broader suite = 68 passed, 0 failed); no evaluator changes were needed.
