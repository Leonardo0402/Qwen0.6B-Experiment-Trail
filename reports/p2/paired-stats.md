# P3: Paired Statistics Summary

- Models compared: base, stage2-boundary, stage3-repair, independent-stage3, stage3-v3-antiforget
- Common sample IDs: 120

## Per-pair sample-level comparison

| Pair | N | Win | Loss | Unchanged | Δ rate | McNemar b/c | p (2-sided) | 95% CI |
|------|---|-----|------|-----------|--------|-------------|-------------|--------|
| base → stage2-boundary | 120 | 15 | 15 | 90 | +0.0000 | 15/15 | 1.0000 | [-0.0917, +0.0917] |
| stage2-boundary → stage3-repair | 120 | 7 | 8 | 105 | -0.0083 | 8/7 | 1.0000 | [-0.0750, +0.0583] |
| stage3-repair → independent-stage3 | 120 | 13 | 6 | 101 | +0.0583 | 6/13 | 0.1671 | [-0.0083, +0.1333] |
| independent-stage3 → stage3-v3-antiforget | 120 | 7 | 10 | 103 | -0.0250 | 10/7 | 0.6291 | [-0.0917, +0.0417] |
| base → stage3-repair | 120 | 18 | 19 | 83 | -0.0083 | 19/18 | 1.0000 | [-0.1083, +0.0917] |
| base → independent-stage3 | 120 | 19 | 13 | 88 | +0.0500 | 13/19 | 0.3771 | [-0.0417, +0.1417] |
| base → stage3-v3-antiforget | 120 | 19 | 16 | 85 | +0.0250 | 16/19 | 0.7359 | [-0.0750, +0.1250] |

## Per-pair family-level comparison

| Pair | Families | Gained | Lost | Net | A pass | B pass |
|------|----------|--------|------|-----|--------|--------|
| base → stage2-boundary | 58 | 5 | 4 | +1 | 11 | 12 |
| stage2-boundary → stage3-repair | 58 | 3 | 2 | +1 | 12 | 13 |
| stage3-repair → independent-stage3 | 58 | 3 | 4 | -1 | 13 | 12 |
| independent-stage3 → stage3-v3-antiforget | 58 | 3 | 2 | +1 | 12 | 13 |
| base → stage3-repair | 58 | 7 | 5 | +2 | 11 | 13 |
| base → independent-stage3 | 58 | 7 | 6 | +1 | 11 | 12 |
| base → stage3-v3-antiforget | 58 | 7 | 5 | +2 | 11 | 13 |

## Per bug_type repair success rate

| Bug type | base | stage2-boundary | stage3-repair | independent-stage3 | stage3-v3-antiforget |
|----------|---|---|---|---|---|
| aggregation_error | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) |
| branch_deletion | 9/18 (50.0%) | 8/18 (44.4%) | 8/18 (44.4%) | 8/18 (44.4%) | 9/18 (50.0%) |
| condition_error | 2/7 (28.6%) | 2/7 (28.6%) | 2/7 (28.6%) | 6/7 (85.7%) | 4/7 (57.1%) |
| index_error | 0/1 (0.0%) | 0/1 (0.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) |
| initialization_error | 3/7 (42.9%) | 4/7 (57.1%) | 4/7 (57.1%) | 5/7 (71.4%) | 4/7 (57.1%) |
| off_by_one | 8/13 (61.5%) | 8/13 (61.5%) | 8/13 (61.5%) | 8/13 (61.5%) | 8/13 (61.5%) |
| return_value_error | 15/27 (55.6%) | 16/27 (59.3%) | 14/27 (51.9%) | 16/27 (59.3%) | 14/27 (51.9%) |
| type_error | 4/6 (66.7%) | 5/6 (83.3%) | 5/6 (83.3%) | 5/6 (83.3%) | 4/6 (66.7%) |
