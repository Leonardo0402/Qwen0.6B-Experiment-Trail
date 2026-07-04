# Task F2 Report — Issue #10 Fix 3 + Fix 4

**Status:** DONE

## Summary

Implemented two merged fixes for the P3 Capability Expansion v2 (Issue #10):

- **Fix 3** — Tightened `check_early_stop` trigger 2: full validation must now
  show a confirmed drop (`full_history[-1] < full_history[-2] - probe_min_delta`)
  before early-stopping on probe patience drops. Previously any non-empty
  `full_history` would confirm the stop.
- **Fix 4** — Unified the metrics schema: introduced
  `METRICS_SCHEMA_VERSION = "1.0.0"`, a `BASELINE_TO_METRICS_KEY_MAP`, and the
  `normalize_baseline_key` helper so `check_hard_constraint` compares a single
  canonical `pass_at_1` key instead of ad-hoc `codegen_pass1` vs `pass_at_1`.
  Added a `schema_version` field to the baseline-lock JSON and a non-failing
  warning when the baseline schema version mismatches the metrics schema.

## Modified Files

1. `src/metrics.py` — Added `METRICS_SCHEMA_VERSION`, `BASELINE_TO_METRICS_KEY_MAP`,
   and `normalize_baseline_key()` helper (Issue #10 Fix 4).
2. `src/p3_checkpoint_evaluator.py` — Tightened `check_early_stop` trigger 2
   (Fix 3); updated `check_hard_constraint` to use `normalize_baseline_key` +
   added `schema_version` mismatch warning (Fix 4); extended the
   `src.metrics` import.
3. `tests/test_p3_checkpoint_evaluator.py` — Added 9 new tests
   (`TestCheckEarlyStopFix3` x5, `TestMetricsSchemaFix4` x4); updated the
   existing `test_early_stop_probe_patience` Case 2 to require two full
   validations with a drop (matches new tightened logic).
4. `reports/p3/p3-baseline-lock.json` — Added top-level
   `"schema_version": "1.0.0"` (no existing fields removed; per-model
   `codegen_pass1` retained for backward compatibility).

## Test Results

```
python -m pytest tests/test_p3_checkpoint_evaluator.py tests/test_metrics.py -v
============================= 73 passed in 2.73s ==============================
```

- `tests/test_p3_checkpoint_evaluator.py`: **26 passed / 26 total**
  (17 pre-existing + 9 new; the updated `test_early_stop_probe_patience`
  Case 2 still passes under the new logic)
- `tests/test_metrics.py`: **47 passed / 47 total** (no regressions —
  `summarize()` output keys unchanged; new constants/function are additive)

TDD cycle followed:
- RED: `ImportError: cannot import name 'METRICS_SCHEMA_VERSION' from
  'src.metrics'` confirmed the feature was missing before implementation.
- GREEN: All 26 tests in `test_p3_checkpoint_evaluator.py` pass after
  implementing Fix 3 + Fix 4.

## Commit

- **SHA (full):** `462c8d8af458b62c08505cdd94555bd5de0e6319`
- **SHA (short):** `462c8d8`
- **Message:** `fix(p3): tighten check_early_stop trigger 2 + unify metrics schema (Issue #10 Fix 3+4)`
- **Files in commit:** 4 files changed, 225 insertions(+), 14 deletions(-)

## Concerns

1. **Staging scope.** The task instructions specified `git add -A`, but the
   working directory contained many unrelated modifications from parallel
   Fix tasks (task briefs, `data/family-registry.json`,
   `scripts/_debug_specific.py`, etc.). Per CLAUDE.md ("prefer adding specific
   files by name rather than using 'git add -A'"), I staged only the four
   Fix 3+4 files. The other parallel work remains unstaged for the respective
   implementers to commit.
2. **schema_version warning semantics.** The `check_hard_constraint` return
   contract is `(passed, violations)`. To honor "记录到 violations 但不 FAIL",
   I split internally into `hard_violations` (drop_pct — causes FAIL) and
   `warnings` (schema_version mismatch — does not), then concatenate them
   for the returned `violations` list. `passed` is computed only from
   `hard_violations`. This keeps the public 2-tuple signature stable while
   making schema mismatches observable in the output.
3. **No behavioral change to `summarize()`.** `METRICS_SCHEMA_VERSION` is a
   module-level constant only; it is NOT injected into `summarize()` output,
   so the existing `test_metrics.py::TestSummarize::test_returns_all_expected_keys`
   exact-key assertion still passes.
4. **Backward compatibility.** The baseline-lock JSON retains
   `codegen_pass1` in every model's `historical_held_out_metrics`
   (untouched). `normalize_baseline_key` returns a copy, so the original
   baseline dict is never mutated.
