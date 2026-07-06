# P3 Formal Capacity Audit (Issue #14 P5)

**Generated**: 2026-07-05T15:17:43.591604+00:00
**Generator**: p3_formal_capacity_audit.py

## 1. Family Inventory

| Bucket | Count |
|---|---|
| Total in registry | 807 |
| Shared train (new + replay) | 425 |
| &nbsp;&nbsp;p3_train_new | 219 |
| &nbsp;&nbsp;p3_train_replay | 206 |
| p3_validation | 90 |
| validation_v2 | 45 |
| frozen_v4 | 100 |
| Quarantined | 50 |
| Remaining new available | 0 |
| P2 used (any tag) | 374 |

## 2. Theoretical Yield (no verification)

- Variants per family per bucket: **3**
- Buckets: code, boundary, static_repair, execution_repair
- Total variants per family: **12**
- Shared train families: **425**
- Max per bucket: **1275** (families × 3)
- Max total: **5100** (families × 12)

## 3. Current Canonical Pool (reference)

- Path: `data/p3-curriculum/canonical-pool.jsonl`
- Total samples: **782**
- Variant distribution: {'code': 281, 'execution_repair': 228, 'static_repair': 148, 'boundary': 125}
- Bug type distribution: {'none': 406, 'branch_deletion': 57, 'condition_error': 61, 'off_by_one': 35, 'return_value_error': 132, 'initialization_error': 31, 'aggregation_error': 21, 'index_error': 9, 'type_error': 30}

## 4. Capacity Targets (Issue #14 P5-P7)

### Balanced Generalist

| Bucket | @2300 | @2500 | Theoretical max |
|---|---|---|---|
| code | 690 | 750 | 1275 |
| boundary | 460 | 500 | 1275 |
| static_repair | 460 | 500 | 1275 |
| execution_repair | 690 | 750 | 1275 |

### Repair Specialist

| Bucket | @2300 | @2500 | Theoretical max |
|---|---|---|---|
| code | 345 | 375 | 1275 |
| boundary | 345 | 375 | 1275 |
| static_repair | 690 | 750 | 1275 |
| execution_repair | 920 | 1000 | 1275 |

## 5. Theoretical Feasibility (ignoring verification yield)

- Total >= 2300 (balanced/repair): **True**
- Total >= 2500 (balanced/repair): **True**

Per-bucket vs Balanced @2500 targets:

| Bucket | Target @2500 | Theoretical max | Sufficient? |
|---|---|---|---|
| code | 750 | 1275 | PASS |
| boundary | 500 | 1275 | PASS |
| static_repair | 500 | 1275 | PASS |
| execution_repair | 750 | 1275 | PASS |

Per-bucket vs Repair @2500 targets:

| Bucket | Target @2500 | Theoretical max | Sufficient? |
|---|---|---|---|
| code | 375 | 1275 | PASS |
| boundary | 375 | 1275 | PASS |
| static_repair | 750 | 1275 | PASS |
| execution_repair | 1000 | 1275 | PASS |

## 6. Caveats

- Theoretical yield assumes 100% verification pass rate.
- Actual yield is measured by `scripts/p3_yield_pilot.py`.
- Boundary bucket currently produces 1 variant/family via `generate_boundary_variants.py`; reaching 3/family requires additional boundary-test strategies.
- Repair Specialist execution_repair target (920-1000) requires 2.16-2.35 verified variants per family on average (cap = 3/family).

