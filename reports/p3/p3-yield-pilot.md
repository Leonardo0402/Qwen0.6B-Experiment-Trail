# P3 Yield Pilot Report (Issue #14 P5)

**Generated**: 2026-07-05T15:43:32.035538+00:00
**Generator**: p3_yield_pilot.py

## 1. Configuration

- Families requested: 25
- Families sampled: 25
- Families with source sample: 25
- Families without source: 0
- Seed: 42
- Total shared train families: 425
- Wilson z-score (90% one-sided): 1.645
- Feasible margin: +10%

## 2. Verdict

### **MBPP_FAMILY_OR_VARIANT_LIMIT**

## 3. Per-Bucket Yield & Extrapolation

| Bucket | Attempts | Verified | Yield | Wilson low 90% | Max/fam | Point@425 | Lower@425 |
|---|---|---|---|---|---|---|---|
| code | 75 | 75 | 1.0000 | 0.9652 | 3 | 1275 | **1230** |
| boundary | 25 | 24 | 0.9600 | 0.8391 | 1 | 408 | **356** |
| static_repair | 50 | 44 | 0.8800 | 0.7843 | 3 | 1122 | **1000** |
| execution_repair | 50 | 44 | 0.8800 | 0.7843 | 3 | 1122 | **1000** |

## 4. Bucket Verdict vs Targets (2500 + 10% margin)

### Balanced Generalist

| Bucket | Target @2500 | Lower@425 | Margin OK? |
|---|---|---|---|
| code | 750 | 1230 | PASS |
| boundary | 500 | 356 | FAIL |
| static_repair | 500 | 1000 | PASS |
| execution_repair | 750 | 1000 | PASS |
| **TOTAL** | 2500 | 3586 | PASS |

### Repair Specialist

| Bucket | Target @2500 | Lower@425 | Margin OK? |
|---|---|---|---|
| code | 375 | 1230 | PASS |
| boundary | 375 | 356 | FAIL |
| static_repair | 750 | 1000 | PASS |
| execution_repair | 1000 | 1000 | FAIL |
| **TOTAL** | 2500 | 3586 | PASS |

## 5. Critical Buckets (below bare target, no margin)

- Balanced critical buckets: ['boundary']
- Repair critical buckets: ['boundary']

## 6. Notes

- Yield rate = verified / attempts (actual generation attempts, not theoretical max).
- Wilson 90% one-sided lower bound on per-attempt success.
- Lower@425 = floor(425 × max_per_family × wilson_lower).
- Boundary bucket uses 1 variant/family (existing generator); max_per_family=1.
- Families without a source sample (not in canonical pool) are excluded from attempts.

