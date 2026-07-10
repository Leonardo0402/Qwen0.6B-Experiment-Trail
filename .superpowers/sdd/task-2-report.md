# Task 2 Report — Phase B: TEST_PASS replay-authoritative + finish_claim_mismatch

**Branch:** `feat/p4-1-model-action-provider`
**Base HEAD:** `824a5c2` (T1)
**Commit:** `f3a9a7e` — `feat(p4-1): Phase B — TEST_PASS replay-authoritative + finish_claim_mismatch`

## What I Implemented

Closes the TEST_PASS trust gap in `AgentEvaluator`. Previously, `EvalResult.success` for a TEST_PASS criterion used `finish.tests_passed` (the model's self-claim). It now uses `passed_tests > 0` (actual replay result). A new `finish_claim_mismatch: bool` field on `EvalResult` records when claim ≠ replay.

### Changes to `src/agent_evaluator.py`

1. **`EvalResult` model** — added field:
   ```python
   finish_claim_mismatch: bool = False
   ```

2. **`AgentEvaluator.run()`** — initialized `finish_claim_mismatch = False` at the top of the method, then changed the TEST_PASS branch in the `finish` dispatch from:
   ```python
   if fa.success_criterion == TaskSuccessCriterion.TEST_PASS:
       success = fa.tests_passed
   ```
   to:
   ```python
   if fa.success_criterion == TaskSuccessCriterion.TEST_PASS:
       replay_passed = passed_tests > 0
       success = replay_passed
       finish_claim_mismatch = (fa.tests_passed != replay_passed)
   ```
   Both `_make_result` call sites (finish path + max_steps path) now pass `finish_claim_mismatch=finish_claim_mismatch`.

3. **`_make_result`** — added parameter `finish_claim_mismatch: bool = False` (default keeps the signature backward-compatible) and threaded it into the returned `EvalResult`.

For non-TEST_PASS criteria (IDENTIFY_BUG, PATCH_APPLIED) and the max_steps-hit path, `finish_claim_mismatch` stays `False` (no claim to compare against in those branches — the brief only defines mismatch for TEST_PASS).

### Changes to `tests/test_agent_evaluator.py`

Appended 3 new tests under a `# --- Task 2: ...` section, using the existing helpers `_load_first_success_trajectory`, `TASKS_DIR`, `MicroTaskWorkspace`, `_FixedProvider`. Verbatim from the brief.

## TDD Evidence

### RED — before implementation

Command:
```
py -3.11 -m pytest tests/test_agent_evaluator.py::test_test_pass_success_uses_replay_not_claim tests/test_agent_evaluator.py::test_test_pass_mismatch_claimed_fail_actual_pass tests/test_agent_evaluator.py::test_test_pass_no_mismatch_when_claim_matches_replay -v -p no:warnings
```

Result: **3 failed in 6.15s**. Relevant failures:

- `test_test_pass_success_uses_replay_not_claim` — `AssertionError: expected success=False (replay has 0 passed, claim says True)`. `assert not True` where `EvalResult(... success=True ...).success`. **Expected:** success was still derived from `fa.tests_passed` (claim=True).
- `test_test_pass_mismatch_claimed_fail_actual_pass` — `AssertionError: expected success=True (replay passed_tests>0 is authoritative)`. `assert False` where `EvalResult(... success=False ...).success`. **Expected:** claim=False was being used.
- `test_test_pass_no_mismatch_when_claim_matches_replay` — `AttributeError: 'EvalResult' object has no attribute 'finish_claim_mismatch'`. **Expected:** field not yet added.

All three failures are the intended pre-implementation state.

### GREEN — after implementation

Command (full file, including the 9 pre-existing tests):
```
py -3.11 -m pytest tests/test_agent_evaluator.py -v -p no:warnings
```

Result: **12 passed in 17.35s** (9 pre-existing + 3 new). No regressions.

Pre-implementation baseline (9 pre-existing tests alone): **9 passed in 11.63s** — confirmed clean before starting.

## Files Changed

- `src/agent_evaluator.py` — `EvalResult` model (+1 field), `run()` (+1 local var, TEST_PASS branch rewritten, 2 call sites updated), `_make_result` (+1 param, +1 field in returned model)
- `tests/test_agent_evaluator.py` — +3 tests appended under a new `# --- Task 2: ...` section

Diffstat: `2 files changed, 81 insertions(+), 1 deletion(-)`.

## Self-Review Findings

- **Surgical:** every changed line in `src/agent_evaluator.py` traces directly to a brief step. No adjacent formatting/style changes, no rename of pre-existing symbols.
- **Backward compatibility:** `_make_result`'s new `finish_claim_mismatch` param has a default (`False`), so any external caller that doesn't pass it still compiles. No external callers exist outside the two in `run()` (verified via grep — only `src/agent_evaluator.py` and `tests/test_agent_evaluator.py` reference `_make_result` or `EvalResult` in code; other matches are docs/plans).
- **Default-safe:** `EvalResult.finish_claim_mismatch` defaults to `False`, so the existing `test_all_metrics_present` test (which constructs an `EvalResult` without the field) continues to pass — verified.
- **Field semantics for non-TEST_PASS criteria:** the brief only defines mismatch for TEST_PASS. For IDENTIFY_BUG / PATCH_APPLIED / max_steps-hit, `finish_claim_mismatch` stays `False`. This is the minimal interpretation; if a future task wants mismatch detection for other criteria it can extend this without breaking current behavior.
- **Boolean comparison safety:** `fa.tests_passed != replay_passed` compares two `bool` values — well-defined, no truthiness ambiguity.

## Issues or Concerns

None. Implementation matches the brief exactly; all tests pass; no downstream callers affected.
