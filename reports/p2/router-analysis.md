# P2 Router Feasibility Analysis

- Models loaded: Base, Stage2-v2, Stage3-v2-Continual, Stage3-Independent, Stage3-v3-Antiforget
- Common sample count: 234
- Sample passes iff `public_passed AND hidden_passed`.
- Family passes iff ALL its samples pass.

> **Methodology:** Router policy is frozen in `router-policy-v1.json` (fit on
> 45 selection families, evaluated here on the held-out 30 eval families —
> family-disjoint). No selection-eval leakage.

## Selection / Eval Split

- Policy version: v1
- Selection families: 45 (342 samples) — used to fit routing maps
- Eval families: 30 (234 samples) — held out, used here for evaluation
- Selection ∩ Eval: empty (verified)
- Dataset SHA256: 748ea3a467876ef6...1d897df3 (verified across all loaded evals)

## Comparison Table

| Model / Router | Type | Overall | Family | CodeGen | StaticRepair | ExecRepair | Lift vs Best |
|----------------|------|---------|--------|---------|--------------|------------|--------------|
| Base | single_model | 36.3% | 0.0% | 15.5% | 46.6% | 39.8% | -0.1325 |
| Stage2-v2 | single_model | 50.0% | 0.0% | 15.5% | 60.2% | 62.5% | +0.0043 |
| Stage3-v2-Continual | single_model | 45.3% | 0.0% | 10.3% | 56.8% | 56.8% | -0.0427 |
| Stage3-Independent | single_model | 49.6% | 3.3% | 15.5% | 61.4% | 60.2% | +0.0000 |
| Stage3-v3-Antiforget | single_model | 46.6% | 3.3% | 17.2% | 55.7% | 56.8% | -0.0299 |
| Best Single | best_single | 49.6% | 3.3% | 15.5% | 61.4% | 60.2% | +0.0000 |
| Oracle Router | router | 64.5% | 3.3% | 25.9% | 80.7% | 73.9% | +0.1496 |
| Metadata Router | router | 49.6% | 3.3% | 15.5% | 61.4% | 60.2% | +0.0000 |
| Deployable Router | router | 49.6% | 3.3% | 15.5% | 61.4% | 60.2% | +0.0000 |

## 1. Best Single Model

- **Model:** Stage3-Independent (`full576-independent-stage3`)
- Overall pass: 49.6%
- Family pass: 3.3%
- **Source:** frozen policy v1 (selected on held-out selection subset)

### Per-task-type pass rates

| Task Type | Total | Passed | Rate |
|-----------|-------|--------|------|
| code_generation | 58 | 9 | 15.5% |
| static_repair | 88 | 54 | 61.4% |
| execution_repair | 88 | 53 | 60.2% |

### All single models

| Model | Overall | Family | CodeGen | StaticRepair | ExecRepair |
|-------|---------|--------|---------|--------------|------------|
| Base | 36.3% | 0.0% | 15.5% | 46.6% | 39.8% |
| Stage2-v2 | 50.0% | 0.0% | 15.5% | 60.2% | 62.5% |
| Stage3-v2-Continual | 45.3% | 0.0% | 10.3% | 56.8% | 56.8% |
| Stage3-Independent | 49.6% | 3.3% | 15.5% | 61.4% | 60.2% |
| Stage3-v3-Antiforget | 46.6% | 3.3% | 17.2% | 55.7% | 56.8% |

## 2. Oracle Router (Upper Bound)

Oracle passes a sample if ANY model passes it.

- Overall pass: 64.5%
- Family pass: 3.3%
- Lift vs Best Single: +0.1496

| Task Type | Total | Passed | Rate |
|-----------|-------|--------|------|
| code_generation | 58 | 15 | 25.9% |
| static_repair | 88 | 71 | 80.7% |
| execution_repair | 88 | 65 | 73.9% |

## 3. Metadata Router (route by `task_type`)

### Routing map

| Task Type | Routed Model |
|-----------|-------------|
| code_generation | Base |
| static_repair | Stage3-Independent |
| execution_repair | Stage3-Independent |

- Overall pass (eval subset): 49.6%
- Family pass: 3.3%
- Selection pass rate (overall, frozen): 48.2%
- Lift vs Best Single: +0.0000
- **Source:** frozen policy v1 (selected on held-out selection subset)

## 4. Deployable Deterministic Router (observable signals only)

### Routing rules

| Observable Signal | Category | Candidate | Routed Model |
|------------------|----------|-----------|--------------|
| no broken_code, no execution_feedback | code_generation | Generalist (Best Single on code_gen) | Base |
| broken_code present, no execution_feedback | static_repair | Static Repair Specialist | Stage3-Independent |
| broken_code + execution_feedback | execution_repair | Repair Specialist | Stage3-Independent |

- Overall pass: 49.6%
- Family pass: 3.3%
- Selection pass rate (overall, frozen): 48.2%
- Lift vs Best Single: +0.0000
- **Source:** frozen policy v1 (selected on held-out selection subset)

## Appendix: Bug Type Distribution (extracted from sample_id)

| Bug Type | Count |
|----------|-------|
| (none) | 58 |
| aggregation_error | 10 |
| branch_deletion | 32 |
| condition_error | 30 |
| index_error | 2 |
| initialization_error | 18 |
| off_by_one | 14 |
| return_value_error | 54 |
| type_error | 16 |

## Methodology Notes

- **Best Single:** model with highest overall pass rate.
- **Oracle Router:** upper bound; passes if ANY model passes the sample.
- **Metadata Router:** uses `task_type` metadata to pick best model per task_type.
- **Deployable Router:** uses only deployment-observable signals:
  - `broken_code` presence (inferred from task_type being static_repair or execution_repair).
  - `execution_feedback` presence (inferred from task_type being execution_repair).
  - Request category (generation vs repair).
- **No leakage:** Deployable Router does not use hidden tests, gold answers, or gold bug_type.
- **Methodology:** routing maps loaded from frozen policy artifact (`router-policy-v1.json`). Selection and eval subsets are family-disjoint. No selection-eval leakage.

## P3 Decision Gate

**Verdict: SIGNAL**

Oracle lift (15.0pp) >= 5pp (routing potential exists), but Deployable Router lift (0.0pp) or significance (McNemar p=1.0000, CI=[-0.0256, +0.0256]) does not meet the GO threshold — observable signals alone cannot capture the potential.

### Gate Criteria

| Criterion | Value | Threshold | Met |
|-----------|-------|-----------|-----|
| Oracle lift vs Best Single | 15.0pp | >= 5.0pp | YES |
| Deployable lift vs Best Single | 0.0pp | >= 5.0pp | NO |
| Deployable McNemar p (2-sided) | 1.0000 | < 0.05 | NO |
| Deployable 95% CI | [-0.0256, +0.0256] | lower > 0 | NO |
| Deployable b/c (McNemar) | 4/4 | — | — |
| Common samples | 234 | — | — |
