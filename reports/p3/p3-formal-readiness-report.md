# P3 Formal Readiness Gate v2 Report

**Generated**: 2026-07-06T05:59:31.476847+00:00
**Branch**: feat/p3-boundary-repair-pipeline-v3
**Wave**: 5-J (Issue #14)
**Scope**: Formal training data + configs + infrastructure readiness.

## Verdict: MBPP_FAMILY_OR_VARIANT_LIMIT

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
| 9 | Local CPU CI green (subprocess) | PASS | 105/105 tests pass (rc=0) |
| 10 | P3 baseline lock present | PASS | 3/3 models, all fields present |
| 11 | Formal train capacity (>=2300/candidate) | SKIP | SKIP: formal train datasets not built yet |
| 12 | verified ⟺ verification (formal) | SKIP | SKIP: formal train datasets not built yet |
| 13 | Candidate ratio +/-3pp (formal) | SKIP | SKIP: formal train datasets not built yet |
| 14 | All buckets non-empty (formal) | SKIP | SKIP: formal train datasets not built yet |
| 15 | Composite evaluator complete (5 comp) | PASS | 5 components present, compute_ok |
| 16 | Frozen v4 coverage gate | PASS | fam=100 formal=365 canary=100 code=27.40% bdry=17.81% sr=27.40% er=27.40% |
| 17 | Validation v2 gate | PASS | total=180 variants={'code': 45, 'boundary': 45, 'execution_repair': 45, 'static_repair': 45} sha_match=True |
| 18 | Formal pool SHA lock | PASS | sha=c6186afd49df4fa1... samples=2544 families=403 |
| 19 | Formal config validity | PASS | 2 configs valid (independent, null adapter, 2 epochs) |
| 20 | Per-family cap enforcement (formal) | SKIP | SKIP: formal train datasets not built yet |
| 21 | Capacity verdict (formal builder) | PASS | balanced: max=2181 binding=execution_repair; repair: max=1591 binding=execution_repair status=LIMIT src=pool_fallback_lp |

> **Note**: Check 8 runs a local pytest subprocess and reports
> local results only. The GitHub Actions CI status on the PR
> must be verified separately on the PR page.

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

## Verdict Logic

- **GO_FOR_P3_TRAINING**: all mandatory checks PASS AND both candidates' formal capacity >= 2300.
- **MBPP_FAMILY_OR_VARIANT_LIMIT**: all checks PASS but any candidate's formal capacity < 2300 (or capacity verdict is MBPP_FAMILY_OR_VARIANT_LIMIT).
- **FIX_FIRST**: any mandatory check FAILS.
- **PENDING_DATASET_BUILD**: all checks PASS (incl. SKIP) but formal datasets not built yet -- capacity cannot be verified.

## Conclusion

**MBPP_FAMILY_OR_VARIANT_LIMIT** -- all checks PASS but at least one candidate's formal capacity < 2300.

The MBPP family/variant supply is insufficient for formal training at the 2300-sample threshold.
Options:
- Expand the candidate pool (more families or more variants per family).
- Accept PILOT_ONLY training with reduced capacity (results must NOT be reported as formal capability).
- Re-run the formal dataset builder after pool expansion.
