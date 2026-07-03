# P2 Data Factory Report

Generated: 2026-07-03T02:47:58.039248+00:00

## Data Source

- Dataset: google-research-datasets/mbpp
- Original MBPP samples: 374
- License: CC-BY-4.0

## Pipeline

```
MBPP Raw → Normalize → Verify → Bug Inject → Execute → Confirm → Training Sample
```

## Generated Data Summary

| Stage | Train | Validation | Families |
|---|---:|---:|---:|
| stage1-code | 84 | 34 | 224 |
| stage2-boundary | 280 | 113 | 224 |
| stage3-repair | 560 | 226 | 224 |
| Frozen Eval v2 | 576 | — | 75 |

## Task Type Distribution (Frozen Eval)

| Task Type | Count |
|---|---:|
| code_generation | 140 |
| execution_repair | 218 |
| static_repair | 218 |

## Bug Injection Types

- condition_error (< → <=, > → >=, == → !=)
- off_by_one (range(n) → range(n-1), range(n+1))
- return_value_error (return value → return None)
- index_error (items[0] → items[1])
- initialization_error (total=0 → total=1)
- aggregation_error (min → max, sum → len)
- branch_deletion (delete/reverse if branch)
- type_error (str/int, list/tuple, None mismatch)

## Family Partition

- Train families: 224
- Validation families: 75
- Frozen families: 75
- Train ∩ Validation: 0
- Train ∩ Frozen: 0
- Validation ∩ Frozen: 0

## Verification

- Every bug sample confirmed: original passes, bugged fails, repair passes
- Execution feedback captured with compressed traceback
- Token audit: 100% Assistant retention across all stages
