# Task 6 Report — Phase E: ModelActionProvider prompt builder + JSON extraction + diagnostics

## 1. Status

DONE

## 2. Commit SHA

`37c4ef2410b9eb4b0cc15a45087b4e04fa0b23dd`

- Branch: `feat/p4-1-model-action-provider`
- BASE (T5 HEAD before work): `439c097`
- Parent: `439c0977a159c704bcb13ceb17140fa42c2bb9c0`

## 3. What was implemented

### Created: `src/agent_model_provider.py` (246 lines, new file)
- Module docstring + `from __future__ import annotations` per brief verbatim.
- `ModelStepDiagnostics(BaseModel)` — 9 diagnostic fields per brief (raw_output, json_parse_ok, schema_valid, safety_valid, action_type_valid, arguments_valid, repair_attempted, repair_success, latency_ms).
- `SentinelAction(BaseModel)` — invalid-action marker with `is_invalid: bool = True`, `reason: str = ""`, `action_type` property returning `"invalid"`, and `safety_flags` property returning a non-forbidden `SafetyFlags` (all 5 fields False).
- `build_prompt(state, task_description, last_observation) -> str` — assembles Task ID, Step, Task, optional Notes/Hypothesis (read from `state.memory`), optional last observation, the 11-action allowlist, and a "single JSON object" instruction.
- `_JSON_FENCE_RE` / `_BARE_JSON_RE` module-level compiled regexes.
- `extract_json(raw) -> str | None` — fenced-JSON first, then bare-JSON fallback, else None.
- `repair_json(raw) -> str` — format-only repair: strips markdown fences, removes trailing commas before `}`/`]`, best-effort brace balancing. Does NOT alter `action_type` values (verified by `test_repair_json_does_not_choose_action_type`).
- `ModelActionProvider(ActionProvider)` — lazy-loads Qwen3-0.6B on first `next_action` (GPU only; non-GPU tests mock `_generate`). `next_action` runs extract → direct validate → repair → validate; returns `SentinelAction` on any parse/schema failure. Records a `ModelStepDiagnostics` per call. `reset()` clears diagnostics; `diagnostics` property returns a copy.
- `_validate_action(data) -> Action | None` — uses the top-level `Action` import (Correction 3 applied: brief's inner `from src.agent_actions import Action` removed).

### Created: `tests/test_agent_model_provider.py` (58 lines, new file)
- 7 tests verbatim from brief: `test_build_prompt_produces_nonempty_string`, `test_extract_json_finds_first_json_block`, `test_extract_json_returns_none_on_no_json`, `test_repair_json_strips_markdown_fences`, `test_repair_json_removes_trailing_commas`, `test_repair_json_does_not_choose_action_type`, `test_sentinel_action_marks_invalid`.

### Modified: `src/agent_evaluator.py` (+15 / −3 lines)
- Added `invalid_action_count = 0` to the metric counters block in `AgentEvaluator.run()` (alongside `finish_without_tests = 0`).
- Added SentinelAction dispatch check after `total_actions += 1` and BEFORE the re-validation `try` block:
  ```python
  if hasattr(action, 'is_invalid') and getattr(action, 'is_invalid', False):
      invalid_action_count += 1
      errors.append(f"step {step}: invalid action (sentinel: {getattr(action, 'reason', 'unknown')})")
      continue
  ```
- Added `invalid_action_count: int` parameter to `_make_result` (placed after `finish_without_tests` and before `max_steps_hit` per Correction 2).
- Added `"invalid_action_count": invalid_action_count,` to the metrics dict inside `_make_result` (9th metric).
- Updated BOTH `_make_result` call sites to pass `invalid_action_count` positionally (finish branch and max-steps branch).

### Modified: `tests/test_agent_evaluator.py` (+28 lines)
- Correction 1 applied: `test_all_metrics_present` updated — docstring "all 8 metrics" → "all 9 metrics", `"invalid_action_count"` added to `expected_keys` set, `"invalid_action_count": 0` added to the manual `EvalResult` metrics dict.
- Appended `test_sentinel_action_counted_as_invalid_not_forbidden` (Task 6 SentinelAction test from brief): builds a `_FixedProvider([sentinel, finish])`, runs the evaluator, asserts `invalid_action_count >= 1` AND `forbidden_action_count == 0`.

## 4. TDD evidence

### RED — tests written first, module did not exist

Command:
```
py -3.11 -m pytest tests/test_agent_model_provider.py "tests/test_agent_evaluator.py::test_sentinel_action_counted_as_invalid_not_forbidden" "tests/test_agent_evaluator.py::test_all_metrics_present" -v -p no:warnings
```

Output (collection error — module missing):
```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 2 items / 1 error

=================================== ERRORS ====================================
_____________ ERROR collecting tests/test_agent_model_provider.py _____________
ImportError while importing test module 'E:\agent\Qwen\qwen3-code-lab\tests\test_agent_model_provider.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\20385\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_agent_model_provider.py:2: in <module>
    from src.agent_model_provider import (
E   ModuleNotFoundError: No module named 'src.agent_model_provider'
=========================== short test summary info ===========================
ERROR tests/test_agent_model_provider.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.79s ===============================
```

Follow-up RED run (evaluator tests only, after the collection error above interrupted the first run):
```
py -3.11 -m pytest "tests/test_agent_evaluator.py::test_sentinel_action_counted_as_invalid_not_forbidden" "tests/test_agent_evaluator.py::test_all_metrics_present" -v -p no:warnings
```
Output:
```
collected 2 items
tests\test_agent_evaluator.py F.                                         [100%]

____________ test_sentinel_action_counted_as_invalid_not_forbidden ____________
    def test_sentinel_action_counted_as_invalid_not_forbidden(monkeypatch):
        ...
>           from src.agent_model_provider import SentinelAction
E           ModuleNotFoundError: No module named 'src.agent_model_provider'

tests\test_agent_evaluator.py:584: ModuleNotFoundError
=========================== short test summary info ===========================
FAILED tests/test_agent_evaluator.py::test_sentinel_action_counted_as_invalid_not_forbidden - ModuleNotFoundError: No module named 'src.agent_model_provider'
========================= 1 failed, 1 passed in 0.59s =========================
```

Note: `test_all_metrics_present` passed in RED because it is a freshness update (Correction 1 explicitly notes it does not run the evaluator — it constructs a manual `EvalResult` with the 9 keys and checks the set matches). The new SentinelAction test was the RED failure driving the implementation.

### GREEN — module + evaluator changes implemented, all tests pass

Command:
```
py -3.11 -m pytest tests/test_agent_model_provider.py tests/test_agent_evaluator.py -v -p no:warnings
```
Output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 28 items

tests\test_agent_model_provider.py .......                               [ 25%]
tests\test_agent_evaluator.py .....................                      [100%]

============================== 28 passed in 19.72s ==============================
```

28 tests pass: 7 new model_provider tests + 21 evaluator tests (20 pre-existing + 1 new SentinelAction test).

## 5. Broader regression check

Command:
```
py -3.11 -m pytest tests/test_agent_model_provider.py tests/test_agent_evaluator.py tests/test_agent_tools.py tests/test_agent_actions.py -p no:warnings -q --timeout=120
```
Output:
```
E:\agent\Qwen\qwen3-code-lab\src\agent_tools.py:495: PytestCollectionWarning: cannot collect test class 'TestObservation' because it has a __init__ constructor (from: tests/test_agent_tools.py)
  class TestObservation(BaseModel):
........................................................................ [ 94%]
....                                                                     [100%]

============================= 76 passed in 23.39s =============================
```

Exit code 0. 76 tests pass. The `PytestCollectionWarning` is pre-existing (the `TestObservation` Pydantic model in `src/agent_tools.py` has a `Test*` name and an `__init__`, so pytest cannot collect it as a test class — this is unrelated to T6 and present before this commit).

## 6. Diffstat

Command: `git diff --stat 439c097 HEAD`
```
 src/agent_evaluator.py             |  15 ++-
 src/agent_model_provider.py        | 246 +++++++++++++++++++++++++++++++++++++
 tests/test_agent_evaluator.py      |  28 ++++-
 tests/test_agent_model_provider.py |  58 +++++++++
 4 files changed, 344 insertions(+), 3 deletions(-)
```

## 7. Corrections applied

All 3 pre-flight corrections confirmed:

- **Correction 1 — `test_all_metrics_present` updated.** Docstring changed from "all 8 metrics" to "all 9 metrics"; `"invalid_action_count"` added to the `expected_keys` set (now 9 keys); `"invalid_action_count": 0` added to the manual `EvalResult` construction. (`tests/test_agent_evaluator.py:104-131`.)

- **Correction 2 — `_make_result` signature + call sites.** `invalid_action_count: int` added as a parameter to `_make_result` (placed after `finish_without_tests: int` and before `max_steps_hit: bool`, matching the existing ordering style). `"invalid_action_count": invalid_action_count,` added to the metrics dict inside `_make_result`. BOTH call sites updated to pass `invalid_action_count` positionally: the finish branch (`src/agent_evaluator.py:372-381`) and the max-steps branch (`src/agent_evaluator.py:401-410`).

- **Correction 3 — `_validate_action` redundant import removed.** The brief's `from src.agent_actions import Action` inside `_validate_action` was NOT written; the top-level `from src.agent_actions import Action, SafetyFlags` (line 17 of the new module) is used instead.

## 8. Self-review findings

### Correctness
- `SentinelAction` is a `BaseModel` (not an `Action`), but the evaluator's dispatch loop accepts it because `_FixedProvider.next_action` returns whatever is in its list and the evaluator checks `hasattr(action, 'is_invalid')` BEFORE attempting `action.__class__.model_validate(...)` (which would fail on a SentinelAction since it has no `action_id`/`safety_flags` fields that `ActionBase` requires — but the check `continue`s before that line is reached).
- The SentinelAction check is placed AFTER `total_actions += 1` (so sentinel actions count toward total_actions, keeping `action_validity_rate` denominator honest) and BEFORE the re-validation `try` block (so SentinelAction never reaches `valid_actions += 1`).
- `invalid_action_count` is incremented (not `forbidden_count`) — verified by `test_sentinel_action_counted_as_invalid_not_forbidden` which asserts both `invalid_action_count >= 1` AND `forbidden_action_count == 0`.
- `build_prompt` reads `state.memory.notes` and `state.memory.hypothesis` — both fields exist on `AgentMemory` (`src/agent_state.py:12-13`).
- `SentinelAction.safety_flags` returns all-False `SafetyFlags` — matches the 5-field schema (`modifies_workspace`, `executes_code`, `network_required`, `reads_sensitive_path`, `is_terminal`).
- `repair_json` does NOT alter `action_type` values — explicitly verified by `test_repair_json_does_not_choose_action_type` (input `{"action_type": "???"}` → output still contains `"???"`).
- No circular import: `src.agent_model_provider` imports from `src.agent_evaluator` (for `AgentState`, `ActionProvider`); `src.agent_evaluator` does NOT import from `src.agent_model_provider`.
- The `except (json.JSONDecodeError, Exception)` clauses are redundant (Exception already covers JSONDecodeError) but match the brief verbatim — left as-is per surgical-changes principle.

### Style
- No emojis anywhere in the diff.
- No docstring additions to existing code beyond what the brief specifies (only the new module's classes/functions have docstrings, copied from the brief).
- No incidental refactors.
- `_make_result` parameter ordering matches the existing style (positional ints first, then bools, with `finish_claim_mismatch: bool = False` last as the only defaulted param).
- Comment style for the SentinelAction check (`# Handle SentinelAction (from ModelActionProvider) — invalid, not forbidden`) matches the brief verbatim, including the em-dash.

### Surgical-changes check
- Created exactly 2 new files: `src/agent_model_provider.py`, `tests/test_agent_model_provider.py`.
- Modified exactly 2 existing files: `src/agent_evaluator.py`, `tests/test_agent_evaluator.py`.
- Did NOT touch `src/agent_trajectory.py` (frozen, P4.0 schema).
- Staged exactly the 4 specified files; `.superpowers/sdd/` docs were NOT staged.
- Every changed line traces directly to a brief requirement or a pre-flight correction:
  - evaluator: `invalid_action_count = 0` counter (Step 4.1), SentinelAction check (Step 4.2), `_make_result` param + metric (Step 4.3 / Correction 2), both call sites (Correction 2).
  - test_agent_evaluator: `test_all_metrics_present` update (Correction 1), new SentinelAction test (brief Step 1).
  - test_agent_model_provider: 7 tests verbatim from brief Step 1.
  - agent_model_provider: module verbatim from brief Step 3 with Correction 3 applied.

### Pre-existing dead code noted (not deleted)
`src/agent_evaluator.py:216-225` still contains a stub `class ModelActionProvider(ActionProvider)` that raises `NotImplementedError("ModelActionProvider is P4.1")`. The real `ModelActionProvider` now lives in `src/agent_model_provider.py`. The stub is in a different module, so there is no name conflict and no test imports the stub from `agent_evaluator`. Per the surgical-changes rule, this pre-existing dead code is mentioned here but NOT deleted (the brief did not ask for its removal).

## 9. One-line test summary

All 76 tests pass (7 new model_provider + 21 evaluator + 30 tools + 18 actions) in 23.39s; T6 implementation GREEN with no regressions.
