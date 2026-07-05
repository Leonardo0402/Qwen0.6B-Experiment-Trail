# P3 Formal Readiness Gate v2 Report

**Generated**: 2026-07-05T18:36:00.081549+00:00
**Branch**: feat/p3-capability-expansion-v2
**Wave**: 5-J (Issue #14)
**Scope**: Formal training data + configs + infrastructure readiness.

## Verdict: PENDING_DATASET_BUILD

## 21 Checks (check6 split into 6a/6b)

| # | Check | Status | Details |
|---|---|---|---|
| 1 | Frozen v4 SHA locked | PASS | sha_lock=4405a5885336960c... |
| 2 | Family isolation (formal train disjoint) | PASS | formal=0 val_v2=45 frozen_v4=100 hist_frozen=109 hist_val=74 wl=replay∩p2(206) |
| 3 | Assistant retention = 100% (formal) | SKIP | SKIP: formal train datasets not built yet |
| 4 | Silent truncation = 0 (formal) | SKIP | SKIP: formal train datasets not built yet |
| 5 | Canary all fail (v4) | PASS | 100/100 verified=False |
| 6 | CPU smoke (mandatory) | PASS | smoke=ok numpy=True sum=49995000 |
| 7 | GPU smoke (deferrable) | PASS | bf16=True smoke=True device=cuda |
| 8 | Formal output dirs don't exist | PASS | 2 paths checked, none exist |
| 9 | CPU CI green | PASS | 105/105 tests pass (rc=0) |
| 10 | P3 baseline lock present | PASS | 3/3 models, all fields present |
| 11 | Formal train capacity (>=2300/candidate) | SKIP | SKIP: formal train datasets not built yet |
| 12 | verified ⟺ verification (formal) | SKIP | SKIP: formal train datasets not built yet |
| 13 | Candidate ratio +/-3pp (formal) | SKIP | SKIP: formal train datasets not built yet |
| 14 | All buckets non-empty (formal) | SKIP | SKIP: formal train datasets not built yet |
| 15 | Composite evaluator complete (5 comp) | PASS | 5 components present, compute_ok |
| 16 | Frozen v4 coverage gate | PASS | fam=100 formal=365 canary=100 code=27.40% bdry=17.81% sr=27.40% er=27.40% |
| 17 | Validation v2 gate | PASS | total=180 variants={'code': 45, 'boundary': 45, 'execution_repair': 45, 'static_repair': 45} sha_match=True |
| 18 | Formal pool SHA lock | PASS | sha=cd72eb63969b6998... samples=162 families=25 |
| 19 | Formal config validity | PASS | 2 configs valid (independent, null adapter, 2 epochs) |
| 20 | Per-family cap enforcement (formal) | SKIP | SKIP: formal train datasets not built yet |
| 21 | Capacity verdict (formal builder) | SKIP | SKIP: formal dataset manifests not built yet |

## SKIP Summary

The following checks were SKIPped (datasets not built yet). 
SKIP does not fail the gate. Run the formal dataset builder, then re-run this gate.

- **Check 3 (Assistant retention = 100% (formal))**: SKIP: formal train datasets not built yet
- **Check 4 (Silent truncation = 0 (formal))**: SKIP: formal train datasets not built yet
- **Check 11 (Formal train capacity (>=2300/candidate))**: SKIP: formal train datasets not built yet
- **Check 12 (verified ⟺ verification (formal))**: SKIP: formal train datasets not built yet
- **Check 13 (Candidate ratio +/-3pp (formal))**: SKIP: formal train datasets not built yet
- **Check 14 (All buckets non-empty (formal))**: SKIP: formal train datasets not built yet
- **Check 20 (Per-family cap enforcement (formal))**: SKIP: formal train datasets not built yet
- **Check 21 (Capacity verdict (formal builder))**: SKIP: formal dataset manifests not built yet

## Verdict Logic

- **GO_FOR_P3_TRAINING**: all mandatory checks PASS AND both candidates' formal capacity >= 2300.
- **MBPP_FAMILY_OR_VARIANT_LIMIT**: all checks PASS but any candidate's formal capacity < 2300 (or capacity verdict is MBPP_FAMILY_OR_VARIANT_LIMIT).
- **FIX_FIRST**: any mandatory check FAILS.
- **PENDING_DATASET_BUILD**: all checks PASS (incl. SKIP) but formal datasets not built yet -- capacity cannot be verified.

## Conclusion

**PENDING_DATASET_BUILD** -- infrastructure checks PASS but formal datasets not built yet.

The formal data/config/infrastructure is ready. Next steps:
1. Run the formal pool builder: `py -3.11 scripts/p3_formal_pool_builder.py`
2. Run the formal dataset builder: `py -3.11 scripts/p3_formal_dataset_builder.py --candidate both`
3. Re-run this gate: `py -3.11 scripts/p3_formal_readiness_gate.py`

After datasets are built, checks 3/4/10/11/12/13/19/20 will run (no longer SKIP) and the verdict
will be determined by the actual data quality and capacity.
