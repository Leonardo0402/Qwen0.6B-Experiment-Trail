# Full-576 Paired Statistics Summary

- Models compared: full576-base, full576-stage2-boundary, full576-stage3-repair, full576-independent-stage3, full576-stage3-v3-antiforget
- Common sample IDs: 576

## Per-pair sample-level comparison

| Pair | N | Win | Loss | Unchanged | Δ rate | McNemar b/c | p (2-sided) | 95% CI |
|------|---|-----|------|-----------|--------|-------------|-------------|--------|
| full576-base → full576-stage2-boundary | 576 | 86 | 40 | 450 | +0.0799 | 40/86 | 0.0001 | [+0.0417, +0.1181] |
| full576-stage2-boundary → full576-stage3-repair | 576 | 35 | 49 | 492 | -0.0243 | 49/35 | 0.1557 | [-0.0556, +0.0069] |
| full576-stage3-repair → full576-independent-stage3 | 576 | 72 | 33 | 471 | +0.0677 | 33/72 | 0.0002 | [+0.0330, +0.1024] |
| full576-independent-stage3 → full576-stage3-v3-antiforget | 576 | 26 | 49 | 501 | -0.0399 | 49/26 | 0.0106 | [-0.0694, -0.0104] |
| full576-base → full576-stage3-repair | 576 | 96 | 64 | 416 | +0.0556 | 64/96 | 0.0140 | [+0.0122, +0.0990] |
| full576-base → full576-independent-stage3 | 576 | 112 | 41 | 423 | +0.1233 | 41/112 | 0.0000 | [+0.0816, +0.1649] |
| full576-base → full576-stage3-v3-antiforget | 576 | 102 | 54 | 420 | +0.0833 | 54/102 | 0.0002 | [+0.0417, +0.1250] |

## Per-pair family-level comparison

| Pair | Families | Gained | Lost | Net | A pass | B pass |
|------|----------|--------|------|-----|--------|--------|
| full576-base → full576-stage2-boundary | 75 | 0 | 0 | +0 | 0 | 0 |
| full576-stage2-boundary → full576-stage3-repair | 75 | 1 | 0 | +1 | 0 | 1 |
| full576-stage3-repair → full576-independent-stage3 | 75 | 1 | 0 | +1 | 1 | 2 |
| full576-independent-stage3 → full576-stage3-v3-antiforget | 75 | 0 | 0 | +0 | 2 | 2 |
| full576-base → full576-stage3-repair | 75 | 1 | 0 | +1 | 0 | 1 |
| full576-base → full576-independent-stage3 | 75 | 2 | 0 | +2 | 0 | 2 |
| full576-base → full576-stage3-v3-antiforget | 75 | 2 | 0 | +2 | 0 | 2 |

## Per bug_type repair success rate

| Bug type | full576-base | full576-stage2-boundary | full576-stage3-repair | full576-independent-stage3 | full576-stage3-v3-antiforget |
|----------|---|---|---|---|---|
| aggregation_error | 5/28 (17.9%) | 6/28 (21.4%) | 8/28 (28.6%) | 9/28 (32.1%) | 8/28 (28.6%) |
| branch_deletion | 18/70 (25.7%) | 29/70 (41.4%) | 29/70 (41.4%) | 33/70 (47.1%) | 28/70 (40.0%) |
| condition_error | 27/62 (43.5%) | 30/62 (48.4%) | 25/62 (40.3%) | 35/62 (56.5%) | 29/62 (46.8%) |
| index_error | 4/16 (25.0%) | 5/16 (31.2%) | 6/16 (37.5%) | 7/16 (43.8%) | 9/16 (56.2%) |
| initialization_error | 19/38 (50.0%) | 27/38 (71.1%) | 26/38 (68.4%) | 32/38 (84.2%) | 26/38 (68.4%) |
| off_by_one | 23/48 (47.9%) | 31/48 (64.6%) | 32/48 (66.7%) | 34/48 (70.8%) | 33/48 (68.8%) |
| return_value_error | 71/138 (51.4%) | 81/138 (58.7%) | 72/138 (52.2%) | 80/138 (58.0%) | 76/138 (55.1%) |
| type_error | 18/36 (50.0%) | 25/36 (69.4%) | 26/36 (72.2%) | 28/36 (77.8%) | 25/36 (69.4%) |
