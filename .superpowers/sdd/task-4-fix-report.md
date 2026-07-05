# Task 4 Fix Report -- hidden<3 from REJECT to WARNING

**Branch:** `feat/p3-capability-expansion-v2`
**Base commit:** `3dce2ce` (Task 4 original)
**Status:** COMPLETE
**Date:** 2026-07-04

## Problem

`scripts/verify_imported_mbpp.py` had a hard rejection for
`hidden_tests.count("assert ") >= 3` (constant `_MIN_HIDDEN_ASSERTS = 3`).
This rejected ALL 964 MBPP samples because MBPP ships ~3 tests per task and
the importer (Task 3) splits them as 2 public + 1 hidden, so the hidden
field has only 1 assert < 3.

## Fix Applied (user-approved)

Changed `hidden < 3` from REJECT to WARNING:

1. Samples with `hidden < 3` now go to `verified/{split}.jsonl` (NOT
   rejected) and a warning is recorded.
2. The manifest gains a new `warnings` field (a summary dict) tracking
   how many verified samples have `hidden_assertion_count < 3`:
   `{"low_hidden_count": <int>}`.
3. `public >= 2` remains a HARD rejection (unchanged).
4. The hard `hidden >= 3` enforcement will happen at Frozen v3 build
   time (Task 8), not at import verification.

### Files Modified

- `scripts/verify_imported_mbpp.py`
  - `check_sample()` now returns a 3-tuple `(passed, reason, warning)`.
    The hidden-assertion check no longer returns `(False, ...)`; instead
    it builds a non-empty `warning` string when `n_hidden < 3`. The
    public, repair, and execution-feedback checks remain hard rejects.
  - `verify_split()` returns `(verified, rejected, warnings_count)` and
    counts samples that triggered the low-hidden warning.
  - `update_manifest_with_verified()` accepts a new required `warnings`
    kwarg (a summary dict) and persists it as `manifest["warnings"]`.
  - `main()` passes `warnings={"low_hidden_count": warnings_count}` to
    the manifest updater and prints a warnings summary line.

- `tests/test_import_mbpp_p3.py`
  - Renamed `test_rejected_low_hidden_count` to
    `test_low_hidden_count_warning_not_reject`. New assertions: sample
    passes verification, lands in `verified/` (NOT `rejected/`), and
    `warnings_count == 1`.
  - All `check_sample` callers updated to unpack the 3-tuple.
  - `verify_split` callers updated to unpack the 3-tuple.
  - `test_manifest_updated_with_verified_fields` updated to pass
    `warnings={"low_hidden_count": 0}` and assert the field is persisted.

### Files NOT Modified (per constraints)

- `scripts/import_mbpp.py` -- Task 3 importer is correct.
- `src/` -- no schema or validator changes.
- `scripts/generate_source_audit.py` -- audit already reads from
  manifests, which now carry the `warnings` field; no audit changes
  needed (it does not surface warnings in the report; that was not
  requested by the brief).

### Data Re-generated

- `data/external/mbpp/verified/{test,validation,train}.jsonl` -- now
  populated (was empty/non-existent before).
- `data/external/mbpp/rejected/{test,train}.jsonl` -- rewritten with
  only the legitimate hard-rejections (public-test failures).
- `data/external/mbpp/rejected/validation.jsonl` -- DELETED (the new
  validation run had 0 rejections; the script does not write the
  rejected file when empty, so the previous stale file was removed to
  keep state consistent with the manifest's `rejected_count: 0`).
- `data/external/mbpp/manifest.{test,validation,train}.json` -- updated
  with new `verified_count`, `rejected_count`, `verified_sha256`,
  `rejected_sha256`, `verified_at`, and the new `warnings` summary.
- `reports/p3/mbpp-source-audit.json` -- regenerated.

## Verification Results

### Before fix (commit 3dce2ce)

| split      | n_in | verified | rejected | warnings |
|------------|------|----------|----------|----------|
| test       | 500  | 0        | 500      | n/a      |
| validation | 90   | 0        | 90       | n/a      |
| train      | 374  | 0        | 374      | n/a      |
| **total**  | 964  | **0**    | **964**  | n/a      |

All 964 rejected with reason `hidden assertions 1 < 3`.
Conclusion: `INFEASIBLE`.

### After fix

| split      | n_in | verified | rejected | warnings |
|------------|------|----------|----------|----------|
| test       | 500  | 494      | 6        | 494      |
| validation | 90   | 90       | 0        | 90       |
| train      | 374  | 371      | 3        | 371      |
| **total**  | 964  | **955**  | **9**    | **955**  |

- 955 verified (was 0) -- 99.1% of samples now pass.
- 9 rejected -- all legitimate hard-rejections (public-test pytest
  failures, NOT low-hidden-count).  Reason strings look like
  `public tests: 1 failure(s); stdout=...`.
- 955 warnings -- every verified sample has `hidden_assertion_count < 3`
  (expected, since MBPP ships ~3 tests and the importer splits 2/1).
- Conclusion: `LIKELY_FEASIBLE` (test verified=494 >= 240 threshold).
- `new_families_available`: 584 (test 494 + validation 90).

## Test Summary

```
$ python -m pytest tests/test_import_mbpp_p3.py tests/test_generate_source_audit.py -v
============================= 42 passed in 9.07s =============================
```

All 42 tests pass (11 in `test_import_mbpp_p3.py`, 31 in
`test_generate_source_audit.py`).

## Concerns

1. **All 955 verified samples carry the low-hidden warning.** This is
   expected -- MBPP ships ~3 tests per task and the importer splits
   them as 2 public + 1 hidden, so every sample has `hidden_count = 1`.
   The hard `hidden >= 3` enforcement will happen at Frozen v3 build
   time (Task 8).  If Frozen v3 strictly enforces `hidden >= 3`, ALL
   955 samples will be filtered out and the dataset will become
   INFEASIBLE again.  The team should decide whether to (a) relax the
   Frozen v3 threshold for MBPP, (b) re-split MBPP tests as 1 public +
   2 hidden, or (c) supplement MBPP hidden tests with synthetic
   assertions.  This is out of scope for the current fix.

2. **`rejected/validation.jsonl` was deleted.** The verify script does
   not write the rejected file when there are 0 rejections (pre-existing
   behavior).  The previous stale file (from the all-rejected run) was
   removed manually so the on-disk state matches the manifest's
   `rejected_count: 0` for validation.  This is a one-time cleanup;
   future runs that produce 0 rejections will simply not write the
   file, so no recurring cleanup is needed.

3. **No changes to `generate_source_audit.py`.** The audit reads only
   `verified_count` / `rejected_count` from the manifests, which are
   correct.  The new `warnings` field is not surfaced in the audit
   report -- the brief did not request that, and adding it would be
   out of scope for this fix.
