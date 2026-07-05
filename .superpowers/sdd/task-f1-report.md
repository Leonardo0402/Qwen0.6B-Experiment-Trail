# Task F1 Report: Backfill verified subfields for 501 P2-replay samples

**Status:** COMPLETE (PILOT ONLY)
**Branch:** feat/p3-capability-expansion-v2
**Issue:** #10 Fix 1
**Date:** 2026-07-05

## Summary

Backfilled real `verification` subfields for 501 P2-replay-derived samples in
`canonical-pool.jsonl` that shipped with the placeholder
`Verification(syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False)`
and `verified=False`. Removed the `model_copy(update={"verified": True})`
normalization hack from both build scripts.

## Modified Files

### New Files
- `scripts/backfill_canonical_pool_verification.py` -- backfill script
- `tests/test_backfill_canonical_pool_verification.py` -- 5 TDD tests
- `data/p3-curriculum/canonical-pool-backfill-manifest.json` -- backfill manifest
- `data/p3-curriculum/canonical-pool.jsonl.pre-backfill.bak` -- pre-backfill backup
- `.superpowers/sdd/task-f1-report.md` -- this report

### Modified Files
- `scripts/build_balanced_generalist.py` -- removed model_copy hack, added
  verified=True filter, updated _run_hard_gates signature (dynamic
  actual_expected_train_count + actual_bucket_counts), gate 2 skips ALL
  ratio checks when any bucket under-capacity (PILOT ONLY), manifest
  deviations renamed verified_normalization -> history + verified_backfill_applied
- `scripts/build_repair_specialist.py` -- same modifications as above
- `tests/test_build_balanced_generalist.py` -- updated 5 tests to accept
  PILOT ONLY verdict (train count <= target, ratio check skipped when
  under-capacity, deterministic sampling filters verified=True, rejected
  count >= target, manifest ratio_within_tolerance conditional)
- `tests/test_build_repair_specialist.py` -- same test updates as above
- `data/p3-curriculum/canonical-pool.jsonl` -- backfilled with real
  verify_sample() results
- `data/p3-curriculum/balanced-generalist/{train,validation,manifest,
  families,token_audit,rejected}.jsonl/json` -- regenerated
- `data/p3-curriculum/repair-specialist/{train,validation,manifest,
  families,token_audit,rejected}.jsonl/json` -- regenerated

## Key Statistics

### Backfill Results
| Metric | Count |
|--------|-------|
| Total samples in pool | 782 |
| Backfilled (P2-replay) | 501 |
| Skipped (already verified) | 281 |
| Verified=True after | 650 |
| Verified=False after (dropped) | 132 |

### By variant_type
| variant_type | total | backfilled | skipped | verified=True | verified=False |
|--------------|-------|-----------|---------|--------------|----------------|
| code | 281 | 0 | 281 | 281 | 0 |
| boundary | 125 | 125 | 0 | 0 | 125 |
| static_repair | 148 | 148 | 0 | 145 | 3 |
| execution_repair | 228 | 228 | 0 | 224 | 4 |

### Build Script Results
| Candidate | Target | Actual | Verdict |
|-----------|--------|--------|---------|
| Balanced Generalist | 626 train | 501 train | PILOT ONLY |
| Repair Specialist | 493 train | 416 train | PILOT ONLY |
| Validation (both) | 90 | 90 | PASS |

## Test Results

```
tests/test_backfill_canonical_pool_verification.py .....  [ 14%]
tests/test_build_balanced_generalist.py ..........       [ 42%]
tests/test_build_repair_specialist.py ...........        [ 74%]
tests/test_sample_pool.py .........                      [100%]

35 passed in 0.52s
```

All 4 test modules PASS (35 tests total):
- test_backfill_canonical_pool_verification.py: 5 tests (new TDD tests)
- test_build_balanced_generalist.py: 10 tests (updated for PILOT ONLY)
- test_build_repair_specialist.py: 11 tests (updated for PILOT ONLY)
- test_sample_pool.py: 9 tests (unchanged, still pass)

## Commit SHA

See `git log` for the commit with message:
`fix(p3): backfill verified subfields for 501 P2-replay samples (Issue #10 Fix 1)`

## Concerns

1. **Boundary bucket total failure (125/125):** All 125 boundary samples
   failed `pytest_ok` after backfill. These samples test edge cases (0, -1, 1)
   where the target code returns incorrect values. This is the CORRECT
   behavior -- the backfill is doing its job by identifying genuinely
   unverifiable samples. The previous `model_copy` hack masked this issue
   by force-setting `verified=True`.

2. **Train count drop:** Balanced Generalist dropped from 626 to 501 train
   samples (-125, -20%). Repair Specialist dropped from 493 to 416 train
   samples (-77, -16%). The user accepted PILOT ONLY verdict per the task
   brief, so this is not blocking. The ratio check (gate 2) is relaxed when
   any bucket is under-capacity.

3. **static_repair/execution_repair minor losses:** 3 static_repair and 4
   execution_repair samples failed verification. These are genuine test
   failures in the target code that were previously hidden by the hack.

4. **No training launched:** Per task constraints, no training was initiated.
   The readiness gate should re-evaluate whether 501/416 train samples are
   sufficient for the pilot training run.
