# P2 Full-576 Comparison Report

Generated: 2026-07-04T02:27:22.748143+00:00

## Evaluation Setup

- Dataset: data/p2-curriculum/frozen-eval-v2/test_raw.jsonl
- Samples: 576
- Families: 75
- Task types: code_generation (140), static_repair (218), execution_repair (218)
- Common sample count (paired-stats): 576

## Overall Metrics

| Model | Pass@1 | Syntax | Repair | Hidden | Format | Timeout | Family Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 16.4% | 99.7% | 42.4% | 45.5% | 100.0% | 0.2% | 0.0% |
| Stage2-v2 | 14.3% | 98.8% | 53.7% | 51.7% | 99.8% | 1.7% | 0.0% |
| Stage3-v2-Continual | 11.4% | 98.1% | 51.4% | 48.8% | 99.7% | 3.0% | 1.3% |
| Stage3-Independent | 15.0% | 98.4% | 59.2% | 57.1% | 99.3% | 1.2% | 2.7% |
| Stage3-v3-Antiforget | 15.7% | 98.6% | 53.7% | 54.0% | 99.7% | 2.4% | 2.7% |

## Per-Task-Type Breakdown

### code_generation

| Model | Total | Passed | Rate | Syntax | Format |
|---|---:|---:|---:|---:|---:|
| Base | 140 | 23 | 16.4% | 100.0% | 100.0% |
| Stage2-v2 | 140 | 20 | 14.3% | 98.6% | 100.0% |
| Stage3-v2-Continual | 140 | 16 | 11.4% | 97.1% | 98.6% |
| Stage3-Independent | 140 | 21 | 15.0% | 95.7% | 97.1% |
| Stage3-v3-Antiforget | 140 | 22 | 15.7% | 97.1% | 98.6% |

### static_repair

| Model | Total | Passed | Rate | Syntax | Format |
|---|---:|---:|---:|---:|---:|
| Base | 218 | 109 | 50.0% | 99.5% | 100.0% |
| Stage2-v2 | 218 | 121 | 55.5% | 98.2% | 100.0% |
| Stage3-v2-Continual | 218 | 114 | 52.3% | 97.2% | 100.0% |
| Stage3-Independent | 218 | 126 | 57.8% | 98.6% | 100.0% |
| Stage3-v3-Antiforget | 218 | 114 | 52.3% | 98.2% | 100.0% |

### execution_repair

| Model | Total | Passed | Rate | Syntax | Format |
|---|---:|---:|---:|---:|---:|
| Base | 218 | 76 | 34.9% | 99.5% | 100.0% |
| Stage2-v2 | 218 | 113 | 51.8% | 99.5% | 99.5% |
| Stage3-v2-Continual | 218 | 110 | 50.5% | 99.5% | 100.0% |
| Stage3-Independent | 218 | 132 | 60.6% | 100.0% | 100.0% |
| Stage3-v3-Antiforget | 218 | 120 | 55.0% | 100.0% | 100.0% |

## Family-Level Pass

| Model | Families Passed | Total Families | Rate |
|---|---:|---:|---:|
| Base | 0 | 75 | 0.0% |
| Stage2-v2 | 0 | 75 | 0.0% |
| Stage3-v2-Continual | 1 | 75 | 1.3% |
| Stage3-Independent | 2 | 75 | 2.7% |
| Stage3-v3-Antiforget | 2 | 75 | 2.7% |

- Stage3-v2-Continual vs Base family delta: +1

## Paired Statistics Summary

| Pair | N | Win | Loss | Unchanged | Delta | McNemar b/c | p (2-sided) | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full576-base → full576-stage2-boundary | 576 | 86 | 40 | 450 | +8.0% | 40/86 | 0.0001 | [+0.0417, +0.1181] |
| full576-stage2-boundary → full576-stage3-repair | 576 | 35 | 49 | 492 | -2.4% | 49/35 | 0.1557 | [-0.0556, +0.0069] |
| full576-stage3-repair → full576-independent-stage3 | 576 | 72 | 33 | 471 | +6.8% | 33/72 | 0.0002 | [+0.0330, +0.1024] |
| full576-independent-stage3 → full576-stage3-v3-antiforget | 576 | 26 | 49 | 501 | -4.0% | 49/26 | 0.0106 | [-0.0694, -0.0104] |
| full576-base → full576-stage3-repair | 576 | 96 | 64 | 416 | +5.6% | 64/96 | 0.0140 | [+0.0122, +0.0990] |
| full576-base → full576-independent-stage3 | 576 | 112 | 41 | 423 | +12.3% | 41/112 | 0.0000 | [+0.0816, +0.1649] |
| full576-base → full576-stage3-v3-antiforget | 576 | 102 | 54 | 420 | +8.3% | 54/102 | 0.0002 | [+0.0417, +0.1250] |

## Bug-Type Repair Success Rate

| Bug Type | Base | Stage2-v2 | Stage3-v2-Continual | Stage3-Independent | Stage3-v3-Antiforget |
|---|---:|---:|---:|---:|---:|
| aggregation_error | 5/28 (17.9%) | 6/28 (21.4%) | 8/28 (28.6%) | 9/28 (32.1%) | 8/28 (28.6%) |
| branch_deletion | 18/70 (25.7%) | 29/70 (41.4%) | 29/70 (41.4%) | 33/70 (47.1%) | 28/70 (40.0%) |
| condition_error | 27/62 (43.5%) | 30/62 (48.4%) | 25/62 (40.3%) | 35/62 (56.5%) | 29/62 (46.8%) |
| index_error | 4/16 (25.0%) | 5/16 (31.2%) | 6/16 (37.5%) | 7/16 (43.8%) | 9/16 (56.2%) |
| initialization_error | 19/38 (50.0%) | 27/38 (71.1%) | 26/38 (68.4%) | 32/38 (84.2%) | 26/38 (68.4%) |
| off_by_one | 23/48 (47.9%) | 31/48 (64.6%) | 32/48 (66.7%) | 34/48 (70.8%) | 33/48 (68.8%) |
| return_value_error | 71/138 (51.4%) | 81/138 (58.7%) | 72/138 (52.2%) | 80/138 (58.0%) | 76/138 (55.1%) |
| type_error | 18/36 (50.0%) | 25/36 (69.4%) | 26/36 (72.2%) | 28/36 (77.8%) | 25/36 (69.4%) |

## Router Feasibility Summary

| Model/Router | Type | Overall | Family | CodeGen | StaticRepair | ExecRepair | Lift vs Best |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | single_model | 36.1% | 0.0% | 16.4% | 50.0% | 34.9% | -0.1233 |
| Stage2-v2 | single_model | 44.1% | 0.0% | 14.3% | 55.5% | 51.8% | -0.0434 |
| Stage3-v2-Continual | single_model | 41.7% | 1.3% | 11.4% | 52.3% | 50.5% | -0.0677 |
| Stage3-Independent | single_model | 48.4% | 2.7% | 15.0% | 57.8% | 60.6% | +0.0000 |
| Stage3-v3-Antiforget | single_model | 44.4% | 2.7% | 15.7% | 52.3% | 55.0% | -0.0399 |
| Best Single | best_single | 48.4% | 2.7% | 15.0% | 57.8% | 60.6% | +0.0000 |
| Oracle Router | router | 61.6% | 2.7% | 24.3% | 73.9% | 73.4% | +0.1319 |
| Metadata Router | router | 48.8% | 1.3% | 16.4% | 57.8% | 60.6% | +0.0035 |
| Deployable Router | router | 48.8% | 1.3% | 16.4% | 57.8% | 60.6% | +0.0035 |

## P3 Decision Gate

**Verdict: SIGNAL**

Oracle lift (13.2pp) >= 5pp (routing potential exists), but Deployable Router lift (0.3pp) or significance (McNemar p=0.7905, CI=[-0.0087, +0.0156]) does not meet the GO threshold — observable signals alone cannot capture the potential.

### Gate Criteria

| Criterion | Value | Threshold | Met |
|-----------|-------|-----------|-----|
| Oracle lift | 13.2pp | >= 5.0pp | YES |
| Deployable lift | 0.3pp | >= 5.0pp | NO |
| McNemar p | 0.7905 | < 0.05 | NO |
| 95% CI | [-0.0087, +0.0156] | lower > 0 | NO |
