# P2 Router Feasibility Analysis

- Models loaded: Base, Stage2-v2, Stage3-v2-Continual, Stage3-Independent, Stage3-v3-Antiforget
- Common sample count: 576
- Sample passes iff `public_passed AND hidden_passed`.
- Family passes iff ALL its samples pass.

> **Note:** Routing maps are currently determined on the full eval set
> (train-on-test). A future change will split validation/eval subsets.

## Comparison Table

| Model / Router | Type | Overall | Family | CodeGen | StaticRepair | ExecRepair | Lift vs Best |
|----------------|------|---------|--------|---------|--------------|------------|--------------|
| Base | single_model | 36.1% | 0.0% | 16.4% | 50.0% | 34.9% | -0.1233 |
| Stage2-v2 | single_model | 44.1% | 0.0% | 14.3% | 55.5% | 51.8% | -0.0434 |
| Stage3-v2-Continual | single_model | 41.7% | 1.3% | 11.4% | 52.3% | 50.5% | -0.0677 |
| Stage3-Independent | single_model | 48.4% | 2.7% | 15.0% | 57.8% | 60.6% | +0.0000 |
| Stage3-v3-Antiforget | single_model | 44.4% | 2.7% | 15.7% | 52.3% | 55.0% | -0.0399 |
| Best Single | best_single | 48.4% | 2.7% | 15.0% | 57.8% | 60.6% | +0.0000 |
| Oracle Router | router | 61.6% | 2.7% | 24.3% | 73.9% | 73.4% | +0.1319 |
| Metadata Router | router | 48.8% | 1.3% | 16.4% | 57.8% | 60.6% | +0.0035 |
| Deployable Router | router | 48.8% | 1.3% | 16.4% | 57.8% | 60.6% | +0.0035 |

## 1. Best Single Model

- **Model:** Stage3-Independent (`full576-independent-stage3`)
- Overall pass: 48.4%
- Family pass: 2.7%

### Per-task-type pass rates

| Task Type | Total | Passed | Rate |
|-----------|-------|--------|------|
| code_generation | 140 | 21 | 15.0% |
| static_repair | 218 | 126 | 57.8% |
| execution_repair | 218 | 132 | 60.6% |

### All single models

| Model | Overall | Family | CodeGen | StaticRepair | ExecRepair |
|-------|---------|--------|---------|--------------|------------|
| Base | 36.1% | 0.0% | 16.4% | 50.0% | 34.9% |
| Stage2-v2 | 44.1% | 0.0% | 14.3% | 55.5% | 51.8% |
| Stage3-v2-Continual | 41.7% | 1.3% | 11.4% | 52.3% | 50.5% |
| Stage3-Independent | 48.4% | 2.7% | 15.0% | 57.8% | 60.6% |
| Stage3-v3-Antiforget | 44.4% | 2.7% | 15.7% | 52.3% | 55.0% |

## 2. Oracle Router (Upper Bound)

Oracle passes a sample if ANY model passes it.

- Overall pass: 61.6%
- Family pass: 2.7%
- Lift vs Best Single: +0.1319

| Task Type | Total | Passed | Rate |
|-----------|-------|--------|------|
| code_generation | 140 | 34 | 24.3% |
| static_repair | 218 | 161 | 73.9% |
| execution_repair | 218 | 160 | 73.4% |

## 3. Metadata Router (route by `task_type`)

### Routing map

| Task Type | Routed Model | Pass Rate (selection) |
|-----------|-------------|----------------------|
| code_generation | Base | 16.4% |
| static_repair | Stage3-Independent | 57.8% |
| execution_repair | Stage3-Independent | 60.6% |

- Overall pass: 48.8%
- Family pass: 1.3%
- Lift vs Best Single: +0.0035

## 4. Deployable Deterministic Router (observable signals only)

### Routing rules

| Observable Signal | Category | Candidate | Routed Model |
|------------------|----------|-----------|--------------|
| no broken_code, no execution_feedback | code_generation | Generalist (Best Single on code_gen) | Base |
| broken_code present, no execution_feedback | static_repair | Static Repair Specialist | Stage3-Independent |
| broken_code + execution_feedback | execution_repair | Repair Specialist | Stage3-Independent |

- Overall pass: 48.8%
- Family pass: 1.3%
- Lift vs Best Single: +0.0035

## Appendix: Bug Type Distribution (extracted from sample_id)

| Bug Type | Count |
|----------|-------|
| (none) | 140 |
| aggregation_error | 28 |
| branch_deletion | 70 |
| condition_error | 62 |
| index_error | 16 |
| initialization_error | 38 |
| off_by_one | 48 |
| return_value_error | 138 |
| type_error | 36 |

## Methodology Notes

- **Best Single:** model with highest overall pass rate.
- **Oracle Router:** upper bound; passes if ANY model passes the sample.
- **Metadata Router:** uses `task_type` metadata to pick best model per task_type.
- **Deployable Router:** uses only deployment-observable signals:
  - `broken_code` presence (inferred from task_type being static_repair or execution_repair).
  - `execution_feedback` presence (inferred from task_type being execution_repair).
  - Request category (generation vs repair).
- **No leakage:** Deployable Router does not use hidden tests, gold answers, or gold bug_type.
- **Limitation:** routing maps currently determined on full eval set (train-on-test).

## P3 Decision Gate

**Verdict: SIGNAL**

Oracle lift (13.2pp) >= 5pp (routing potential exists), but Deployable Router lift (0.3pp) or significance (McNemar p=0.7905, CI=[-0.0087, +0.0156]) does not meet the GO threshold — observable signals alone cannot capture the potential.

### Gate Criteria

| Criterion | Value | Threshold | Met |
|-----------|-------|-----------|-----|
| Oracle lift vs Best Single | 13.2pp | >= 5.0pp | YES |
| Deployable lift vs Best Single | 0.3pp | >= 5.0pp | NO |
| Deployable McNemar p (2-sided) | 0.7905 | < 0.05 | NO |
| Deployable 95% CI | [-0.0087, +0.0156] | lower > 0 | NO |
| Deployable b/c (McNemar) | 6/8 | — | — |
| Common samples | 576 | — | — |
