# Task 4 Report — Phase C: inspect_error returns stdout+stderr capped 8KB

## 1. Status

DONE

## 2. Commit SHA

`50ec3af` — `feat(p4-1): Phase C — inspect_error returns stdout+stderr capped 8KB`

- Branch: `feat/p4-1-model-action-provider`
- BASE (T3 HEAD before this task): `41a0d05`
- Files staged: ONLY `src/agent_tools.py` and `tests/test_agent_tools.py`

## 3. What was implemented

**`src/agent_tools.py`** — `tool_inspect_error`, `last_test` branch (lines 608-613):
- Replaced `content=last_test_observation.stderr` (which was empty for pytest failures since tracebacks go to stdout) with `content = (stdout + "\n" + stderr)[:8192]`.
- The `last_patch` branch and the `else` (invalid source) branch were NOT touched.
- No docstring changes, no refactors.

**`tests/test_agent_tools.py`** — appended 2 new tests after `test_finish`:
- `test_inspect_error_returns_stdout_on_test_failure` — verifies a stdout-only failure (empty stderr) now surfaces content containing `AssertionError`.
- `test_inspect_error_caps_at_8kb` — verifies a 10000-char stdout is truncated to exactly 8192 chars.
- The existing `test_inspect_error_last_test` (line 480) was left unmodified per surgical-changes constraint. Its docstring ("returns the test stderr") is now slightly inaccurate (content is stdout+stderr), but its assertion still holds because it places `AssertionError` in `stderr` with `stdout=""`, and the new code concatenates both. Left alone as instructed.

## 4. TDD evidence

### RED — tests fail for the right reason (before implementation)

Command:
```
py -3.11 -m pytest tests/test_agent_tools.py::test_inspect_error_returns_stdout_on_test_failure tests/test_agent_tools.py::test_inspect_error_caps_at_8kb -v -p no:warnings
```

Output (exit code 1):
```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 2 items

tests\test_agent_tools.py FF                                             [100%]

================================== FAILURES ===================================
______________ test_inspect_error_returns_stdout_on_test_failure ______________
...
>       assert result.content != "", "inspect_error returned empty content for stdout-only failure"
E       AssertionError: inspect_error returned empty content for stdout-only failure
E       assert '' != ''
E        +  where '' = ErrorObservation(source='last_test', content='').content
_______________________ test_inspect_error_caps_at_8kb ________________________
...
>       assert len(result.content) == 8192, \
            f"expected 8192 chars (8KB cap), got {len(result.content)}"
E       AssertionError: expected 8192 chars (8KB cap), got 0
E       assert 0 == 8192
E        +  where 0 = len('')
E        +    where '' = ErrorObservation(source='last_test', content='').content
=========================== short test summary info ===========================
FAILED tests/test_agent_tools.py::test_inspect_error_returns_stdout_on_test_failure
FAILED tests/test_agent_tools.py::test_inspect_error_caps_at_8kb
============================== 2 failed in 0.76s ==============================
```

Both failures are exactly the expected reason: current implementation returns `stderr` (empty) so `content=''`. The cap test fails because no cap is applied to an empty string.

### GREEN — all tests pass (after implementation)

Command:
```
py -3.11 -m pytest tests/test_agent_tools.py -v -p no:warnings
```

Output (exit code 0):
```
collected 30 items

tests\test_agent_tools.py ..............................                 [100%]

============================= 30 passed in 6.11s ==============================
```

All 30 tests in `tests/test_agent_tools.py` pass (28 pre-existing + 2 new).

## 5. Broader regression check

Command:
```
py -3.11 -m pytest tests/test_agent_tools.py tests/test_agent_evaluator.py tests/test_agent_actions.py -p no:warnings -q --timeout=60
```

Output (exit code 0):
```
E:\agent\Qwen\qwen3-code-lab\src\agent_tools.py:495: PytestCollectionWarning: cannot collect test class 'TestObservation' because it has a __init__ constructor (from: tests/test_agent_tools.py)
  class TestObservation(BaseModel):
...............................................................          [100%]
63 passed in 22.58s
```

63/63 tests pass across the three suites. No regressions. (The `PytestCollectionWarning` about `TestObservation` is pre-existing — pytest tries to collect the pydantic model class as a test because its name starts with `Test`; it is unrelated to this task and not introduced by these changes.)

## 6. Diffstat

```
 src/agent_tools.py        |  7 +++----
 tests/test_agent_tools.py | 39 +++++++++++++++++++++++++++++++++++++++
 2 files changed, 42 insertions(+), 4 deletions(-)
```

## 7. Corrections applied

Confirmed: the brief's literal `TestObservation(...)` construction (using non-existent fields `returncode` and `summary`, omitting `num_collected` / `num_passed` / `num_failed` / `timed_out` / `duration_s`) was NOT used. Both new tests use the corrected schema per the pre-flight correction:

```python
test_obs = TestObservation(
    passed=False, num_collected=1, num_passed=0, num_failed=1,
    timed_out=False,
    stdout=...,  # per-test intent
    stderr="",
    duration_s=0.1,
)
```

This matches the verified schema at `src/agent_tools.py:495-504` and the pattern used by the existing `test_inspect_error_last_test` at `tests/test_agent_tools.py:482-486`.

## 8. Self-review findings

**Correctness:**
- The `last_test` branch now returns `stdout + "\n" + stderr` capped at 8192 chars. This matches the brief's implementation spec verbatim.
- The `"\n"` separator preserves readability when both streams are non-empty; when `stderr=""` (the pytest-failure case this task targets), the result is `stdout + "\n"` truncated to 8192 — content is non-empty and contains the traceback. Verified by `test_inspect_error_returns_stdout_on_test_failure`.
- The 8192 cap is `raw[:8192]` (Python slice) which yields exactly 8192 chars when `len(raw) >= 8192`. Verified by `test_inspect_error_caps_at_8kb` (10000-char stdout -> exactly 8192).
- The `last_patch` branch is unchanged; the invalid-`error_source` `else` branch is unchanged.

**Style:**
- Implementation matches the surrounding 4-space indentation and the existing `ErrorObservation(source=..., content=...)` construction style used in the `last_patch` branch (line 619).
- Tests match the existing test style in the file (docstring, `test_obs = TestObservation(...)` construction, `tool_inspect_error(...)` call with keyword args).
- No emojis, no docstring additions to existing code, no incidental refactors.

**Surgical-changes check:**
- `src/agent_tools.py`: only the `last_test` branch body changed (3 lines replaced by 3 lines; net -1 line due to collapsing the multi-line `ErrorObservation(...)` into a single-line call). The function signature, docstring, `last_patch` branch, and `else` branch are byte-identical to before.
- `tests/test_agent_tools.py`: only appended 2 tests + a section comment at the end. No existing test modified. The existing `test_inspect_error_last_test` was left untouched as instructed.
- `src/agent_trajectory.py` (frozen P4.0 schema) was not touched.
- Only the 2 intended files were staged; `.superpowers/sdd/` docs and untracked artifacts were excluded from the commit.
- Temp files created during output capture (`_t4_out.txt`, `_t4_regress.txt`, `_t4_regress2.txt`) were deleted after the commit; they are not present in the working tree and were never staged.

## 9. One-line test summary

30/30 tests in tests/test_agent_tools.py pass (63/63 across the broader regression suite).
