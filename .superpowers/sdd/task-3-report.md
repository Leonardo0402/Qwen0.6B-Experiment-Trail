# Task 3 Report — Phase B: 11-action allowlist + unknown hard-fail + search_text/rollback_patch dispatch

**Status:** DONE_WITH_CONCERNS
**Commit SHA:** `41a0d05`
**Branch:** `feat/p4-1-model-action-provider`
**Date:** 2026-07-09

---

## What was implemented

### `src/agent_evaluator.py`

1. **Imports** (`src/agent_evaluator.py:25-34`): Added `tool_rollback_patch` and `tool_search_text` to the existing `from src.agent_tools import (...)` block. Kept alphabetical ordering within the block.

2. **`_ALLOWED_ACTION_TYPES` constant** (`src/agent_evaluator.py:37-44`): Module-level `frozenset` of exactly 11 action type strings:
   ```python
   _ALLOWED_ACTION_TYPES = frozenset({
       "list_files", "read_file", "search_text", "inspect_task",
       "propose_patch", "apply_patch", "rollback_patch", "run_tests",
       "inspect_error", "write_memory", "finish",
   })
   ```

3. **`search_text` dispatch branch** (`src/agent_evaluator.py:344-346`): Placed before the `finish` branch (per brief Step 3.3). Uses `action.arguments.query` (NOT `pattern` — per Correction 1, `SearchTextArgs` defines `query: str`):
   ```python
   elif at == "search_text":
       total_tools += 1
       tool_search_text(self._ws, action.arguments.query)
   ```

4. **`rollback_patch` dispatch branch** (`src/agent_evaluator.py:347-349`): Uses `action.arguments.action_id` (matches `RollbackPatchArgs.action_id: str`):
   ```python
   elif at == "rollback_patch":
       total_tools += 1
       tool_rollback_patch(self._ws, action.arguments.action_id)
   ```

5. **`else` defensive guard** (`src/agent_evaluator.py:374-383`): Placed after the `finish` branch (per brief Step 3.4). Increments `forbidden_count` and appends an error message — does NOT raise (per Correction 4; raising would be caught by the surrounding `try/except Exception` and mis-recorded as `tool_error`):
   ```python
   else:
       forbidden_count += 1
       errors.append(
           f"step {step}: unknown action type (not in 11-action "
           f"allowlist): {at}"
       )
   ```

### `tests/test_agent_evaluator.py`

Appended 3 tests at the end of the file (lines 377-453):

1. **`test_allowed_action_types_has_exactly_11`** (lines 382-389): Asserts `_ALLOWED_ACTION_TYPES` equals the expected set of 11 names and has length 11. (Per brief, unmodified.)

2. **`test_unknown_action_type_recorded_as_forbidden`** (lines 392-416): Replaces the brief's `test_unknown_action_type_raises` (per Correction 2). Uses `unittest.mock.patch.object(type(finish), 'model_validate', return_value=finish)` to bypass the evaluator's re-validation at line 273, then verifies the `else` branch records the unknown `"shell_exec"` action_type as forbidden (`forbidden_action_count >= 1`) and appends to `errors`.

3. **`test_search_text_dispatched`** (lines 419-453): Deviates from the brief's test body (see "5th brief bug" below). Uses a spy on `tool_search_text` (via `monkeypatch.setattr`) to verify the dispatch actually calls the tool, rather than checking `result.metrics["total_tools"]` (which doesn't exist in the metrics dict).

---

## TDD evidence

### RED (before implementation)

Command:
```
py -3.11 -m pytest tests/test_agent_evaluator.py::test_allowed_action_types_has_exactly_11 tests/test_agent_evaluator.py::test_unknown_action_type_recorded_as_forbidden tests/test_agent_evaluator.py::test_search_text_dispatched -v -p no:warnings
```

Output (exit code 4 — collection error):
```
ERROR: found no collectors for E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py::test_allowed_action_types_has_exactly_11
ERROR: found no collectors for E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py::test_unknown_action_type_recorded_as_forbidden
ERROR: found no collectors for E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py::test_search_text_dispatched
...
ImportError while importing test module 'E:\agent\Qwen\qwen3-code-lab\tests\test_agent_evaluator.py'.
...
tests\test_agent_evaluator.py:379: in <module>
    from src.agent_evaluator import _ALLOWED_ACTION_TYPES  # noqa: E402
E   ImportError: cannot import name '_ALLOWED_ACTION_TYPES' from 'src.agent_evaluator'
============================== 1 error in 1.95s ===============================
```

This matches the brief's expected RED state: "`_ALLOWED_ACTION_TYPES` doesn't exist; search_text not dispatched; unknown action silently no-ops."

### GREEN (after implementation)

Command:
```
py -3.11 -m pytest tests/test_agent_evaluator.py -v -p no:warnings
```

Output (exit code 0):
```
collected 15 items
tests\test_agent_evaluator.py ...............
======================== 15 passed in 94.28s (0:01:34) ========================
```

All 15 tests in the file pass — the 12 pre-existing tests plus the 3 new ones. No regressions.

### Broader regression check

Command:
```
py -3.11 -m pytest tests/test_agent_evaluator.py tests/test_agent_actions.py tests/test_agent_tools.py tests/test_agent_trajectory.py -p no:warnings -q --timeout=60
```

Output (exit code 0):
```
....................................................................     [100%]
```

68 tests pass across the 4 agent test files. No regressions in the touched module surface.

(Note: a separate `py -3.11 -m pytest tests/` run surfaced an unrelated Windows-specific `shutil.rmtree` file-in-use failure in `tests/test_build_frozen_v3_samples.py::test_qualified_family_passes_all_gates` — this is pre-existing and not caused by this task's changes.)

---

## Files changed (diffstat)

```
 src/agent_evaluator.py        | 28 +++++++++++++++
 tests/test_agent_evaluator.py | 79 +++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 107 insertions(+)
```

Only the two files specified in the brief were modified. Unrelated sdd docs in the working tree (`progress.md`, `task-*-brief.md`, `task-*-report.md`) were intentionally left unstaged.

---

## Corrections applied (from task description)

| # | Correction | How applied |
|---|---|---|
| 1 | `SearchTextArgs` uses `query`, not `pattern` | Test uses `SearchTextArgs(query="def")`; dispatch uses `action.arguments.query` (`src/agent_evaluator.py:346`) |
| 2 | `test_unknown_action_type_raises` cannot reach `else` as written | Replaced with `test_unknown_action_type_recorded_as_forbidden` using `mock.patch.object(type(finish), 'model_validate', return_value=finish)` to bypass re-validation |
| 3 | Remove dead `_UnknownAction` class | Not included |
| 4 | Follow the PLAN, do NOT raise in `else` | `else` branch only does `forbidden_count += 1; errors.append(...)` — no raise (`src/agent_evaluator.py:374-383`) |

---

## 5th brief bug discovered (not in corrections list)

**Bug:** The brief's `test_search_text_dispatched` asserts `result.metrics["total_tools"] >= 1`, but `total_tools` is an internal counter in `AgentEvaluator.run()` — it is NOT exposed in the `EvalResult.metrics` dict. The metrics dict only contains the 8 keys enumerated in `test_all_metrics_present` (`task_success_rate`, `action_validity_rate`, `tool_error_rate`, `patch_success_rate`, `tests_pass_rate`, `forbidden_action_count`, `max_step_exceeded_count`, `finish_without_tests_count`). Running the brief's test verbatim raises `KeyError: 'total_tools'`.

**Two options considered:**
1. Add `total_tools` to the metrics dict — rejected: would break `test_all_metrics_present` which asserts the metrics dict has exactly the 8 expected keys.
2. Replace the assertion with a spy on `tool_search_text` — chosen: directly verifies the dispatch actually calls the tool, which is the brief's intent ("search_text action must produce a real tool call").

**Fix applied:** `test_search_text_dispatched` now uses `monkeypatch.setattr(evaluator_mod, "tool_search_text", spy)` where `spy` increments a call counter and delegates to the original. Assertion is `call_count["n"] >= 1`. This is a stronger test than the brief's: it directly verifies dispatch rather than inferring it from a counter.

---

## Self-review findings

### Correctness

- ✅ `_ALLOWED_ACTION_TYPES` is a `frozenset` with exactly 11 members, matching `ActionType` enum at `src/agent_actions.py:16-28`.
- ✅ `search_text` branch reads `action.arguments.query` — matches `SearchTextArgs.query` at `src/agent_actions.py:145` and `tool_search_text(workspace, query, ...)` signature at `src/agent_tools.py:106`.
- ✅ `rollback_patch` branch reads `action.arguments.action_id` — matches `RollbackPatchArgs.action_id` at `src/agent_actions.py:181` and `tool_rollback_patch(workspace, action_id)` signature at `src/agent_tools.py:385`.
- ✅ `else` branch is a defensive guard — unreachable with valid Pydantic actions because every `Action` subclass has a `Literal[ActionType.xxx]` `action_type` field matching one of the 11 types.
- ✅ `else` branch does NOT raise (per Correction 4) — it only increments `forbidden_count` and appends to `errors`. The surrounding `try/except Exception` would otherwise catch a raise and mis-record it as `tool_error`.
- ✅ Both new branches are placed before `finish` (which returns early); `else` is placed after `finish`. The control flow is: dispatch → if finish, return; if unknown, record forbidden and continue to next step.

### Style

- ✅ Imports added in alphabetical order within the existing `from src.agent_tools import (...)` block.
- ✅ Branch style matches existing branches (e.g., `elif at == "list_files":` etc.): `total_tools += 1` first, then the tool call.
- ✅ No emojis, no docstring additions to existing code, no incidental refactors.

### Surgical changes

- ✅ Every changed line traces directly to the brief's Step 3 (allowlist constant, two dispatch branches, else guard) or to the test additions in Step 1.
- ✅ No changes to `_make_result`, no changes to the metrics dict, no changes to other dispatch branches, no changes to `EvalResult` schema.

---

## Issues or concerns

1. **`test_unknown_action_type_recorded_as_forbidden` uses `unittest.mock.patch`** to bypass the evaluator's re-validation (`action.__class__.model_validate(action.model_dump())` at line 273). This is necessary because Pydantic's `Literal["finish"]` field on `FinishAction` would otherwise reject the overridden `"shell_exec"` value before reaching dispatch. The mock approach is slightly fragile (it patches `FinishAction.model_validate` for the duration of the `evaluator.run()` call), but it's the cleanest way to exercise the defensive `else` guard. The brief's Correction 2 explicitly suggests this approach and offers a fallback ("simplify to just test the allowlist constant"); I chose the mock approach because it actually exercises the `else` branch's behavior.

2. **`rollback_patch` dispatch is not directly tested.** The brief only asked for 3 tests (`test_allowed_action_types_has_exactly_11`, `test_unknown_action_type_recorded_as_forbidden`, `test_search_text_dispatched`), and a `rollback_patch` dispatch test was not among them. The `rollback_patch` branch is verified indirectly via `test_allowed_action_types_has_exactly_11` (which confirms "rollback_patch" is in the allowlist) and via the fact that the import and branch were added with the correct signature. If direct coverage is desired, a follow-up test could spy on `tool_rollback_patch` similarly to `test_search_text_dispatched`.

3. **5th brief bug** (described above): the brief's `test_search_text_dispatched` assertion `result.metrics["total_tools"]` is incorrect. I replaced it with a spy-based assertion. This is a deviation from the brief's literal test text, made necessary by the metrics dict not exposing `total_tools`.

4. **Pre-existing unrelated test failure** in `tests/test_build_frozen_v3_samples.py::test_qualified_family_passes_all_gates` — a Windows `shutil.rmtree` file-in-use issue, not caused by this task's changes.

---

## One-line test summary

15/15 tests in `tests/test_agent_evaluator.py` pass (3 new + 12 pre-existing); 68/68 across the 4 agent test files; commit `41a0d05`.
