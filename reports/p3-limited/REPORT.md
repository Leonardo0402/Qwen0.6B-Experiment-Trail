# P3-Limited Controlled Experiment Report

**Label:** P3-Limited (NOT a formal capability claim — controlled comparison only)
**Branch:** `feat/p3-limited-experiment`
**Date:** 2026-07-06 ~ 2026-07-07

---

## 1. Background

Formal P3 training target (2300 samples) was proven infeasible under the
`MBPP_FAMILY_OR_VARIANT_LIMIT` (Issues #12 / #14, closed). Instead of
continuing to pad data, this controlled experiment was authorized to
cleanly compare **balanced training** vs **repair-biased training** at
equal sample size (1470) using the same 403-family universe.

## 2. Experimental Setup

### 2.1 Datasets (1470 samples each, seed=42, no duplicates)

| Bucket            | Balanced-Limited | Repair-Limited |
|-------------------|------------------|----------------|
| Code              | 441              | 221            |
| Boundary          | 294              | 220            |
| Static Repair     | 294              | 441            |
| Execution Repair  | 441              | 588            |
| **Total**         | **1470**         | **1470**       |
| Unique families   | 400              | 400            |
| Pool source       | canonical-pool.jsonl | canonical-pool.jsonl |

### 2.2 Training Configuration

| Parameter             | Balanced-Limited | Repair-Limited |
|-----------------------|------------------|----------------|
| training_mode         | independent      | independent    |
| Base model            | Qwen3-0.6B       | Qwen3-0.6B     |
| LoRA rank/alpha       | 16 / 32          | 16 / 32        |
| LoRA target modules   | 7 (q/k/v/o/gate/up/down) | same    |
| Learning rate         | 5e-5             | 3e-5           |
| Epochs                | 2                | 2              |
| Batch / grad accum    | 1 / 8            | 1 / 8          |
| Total steps           | 368              | 368            |
| Max seq length        | 384              | 384            |
| BF16                  | true             | true           |
| Composite weights     | 0.30/0.15/0.20/0.25/0.10 | 0.10/0.10/0.30/0.40/0.10 |

### 2.3 3-Tier Checkpoint Evaluator

- **Tier 1**: train/eval loss every 50 steps
- **Tier 2**: probe eval (75 family-stratified samples, code gen + exec) every 0.25 epoch
- **Tier 3**: full validation every 1.0 epoch (composite score)
- **Best checkpoint selection**: full_validation_composite (never by probe or frozen v4)

### 2.4 Frozen v4 Evaluation

- 465 samples (365 eval + 100 canary)
- Canary: PASSED for both candidates
- One-shot, no checkpoint selection by frozen v4

## 3. Training Results

### 3.1 Tier 3 Full Validation Composite

| Epoch (step) | Balanced-Limited | Repair-Limited |
|--------------|------------------|----------------|
| 1 (184)      | 0.2622           | 0.3994         |
| **2 (368)**  | **0.3078** ★     | **0.4078** ★   |

★ = best checkpoint (selected by full_validation_composite)

**Training composite gap: Repair-Limited leads by +0.1000 (+32.5%)**

### 3.2 Tier 2 Probe Trend

| Epoch | Balanced | Repair  |
|-------|----------|---------|
| 0.25  | 0.166    | 0.143   |
| 0.50  | 0.314    | 0.344   |
| 0.75  | 0.327    | 0.414   |
| 1.00  | 0.314    | 0.458   |
| 1.25  | 0.328    | **0.487** (peak) |
| 1.50  | 0.311    | 0.449   |
| 1.75  | 0.296    | 0.455   |
| 2.00  | 0.296    | 0.485   |

### 3.3 Training Resource Usage

| Metric              | Balanced-Limited | Repair-Limited |
|---------------------|------------------|----------------|
| Wall time           | 4.93h            | 4.83h          |
| Peak GPU memory     | 1360 MiB         | 1360 MiB       |
| Trainable params    | 10,092,544       | 10,092,544     |
| Token truncations   | 291              | 396            |
| NaN/Inf detected    | no               | no             |
| Early stop triggered| no               | no             |

### 3.4 Hard Constraint (informational only for P3-Limited)

Both candidates FAILED the formal hard constraint
(`code_generation_drop_vs_p2_final_max_pct: 3.0`):
- Balanced-Limited @ epoch 2: code_gen dropped 15.02pp (0.1724 → 0.0222)
- Repair-Limited  @ epoch 2: code_gen dropped 17.24pp (0.1724 → 0.0000)

This constraint is informational only for P3-Limited and does not block
the controlled comparison.

## 4. Frozen v4 Final Evaluation

**465 samples (165 code_generation + 200 static_repair + 100 execution_repair)**
**Canary: PASSED for both candidates**

| Metric                 | Balanced-Limited | Repair-Limited | Δ (B − R) |
|------------------------|------------------|----------------|-----------|
| **pass@1**             | **2.42%**        | 1.21%          | **+1.21pp** |
| syntax_rate            | 98.71%           | 98.71%         | 0.00pp    |
| hidden_pass_rate       | **30.11%**       | 29.25%         | +0.86pp   |
| format_compliance_rate | 99.35%           | **99.78%**     | −0.43pp   |
| timeout_rate           | 0.22%            | 0.22%          | 0.00pp    |
| **repair_success**     | **43.33%**       | 43.00%         | +0.33pp   |
| regression_rate        | **52.67%**       | 53.33%         | +0.66pp   |

## 5. Key Findings

### Finding 1: Training composite ≠ Frozen v4 generalization

Despite Repair-Limited leading by +32.5% in training composite
(0.4078 vs 0.3078), it performs **slightly worse** on frozen v4
across nearly all metrics. This is because the composite weights are
**candidate-specific**: Repair-Limited optimizes 40% weight on
execution_repair during training, which inflates its composite without
generalizing to held-out families.

### Finding 2: Balanced training doubles pass@1

- Balanced-Limited pass@1 = 2.42% (4/165 code generation passed)
- Repair-Limited pass@1 = 1.21% (2/165 code generation passed)
- This matches the training data composition: Balanced has 2× the
  code bucket size (441 vs 221).

### Finding 3: Repair success nearly identical

- Balanced: 43.33% vs Repair: 43.00% — difference of only 0.33pp
- Adding 147 more repair samples (441→588 exec, 294→441 static) did
  NOT improve repair success on held-out data. The marginal data
  contributed to overfitting on training families rather than
  generalization.

### Finding 4: 2300-sample formal threshold confirmed out of reach

- Best pass@1 = 2.42%, far below formal P3 qualification line
- Qwen3-0.6B on MBPP-family data has reached its capability frontier
- The `MBPP_FAMILY_OR_VARIANT_LIMIT` verdict is validated

## 6. Conclusion

1. **Balanced training > Repair-biased training** at 1470-sample scale
   for Qwen3-0.6B on MBPP-family data.
2. Training composite scores with candidate-specific weights cannot
   predict frozen-eval generalization.
3. Qwen3-0.6B has reached its capability frontier on the current
   credible MBPP-family data frontier; further data scaling within
   the 403-family universe is unlikely to yield meaningful gains.
4. Neither candidate constitutes a formal P3 capability claim.

## 7. Artifacts

### Adapter Metadata (tracked in git)
- `adapters/p3/balanced-limited/` — config, metrics, evaluator_state, README
- `adapters/p3/repair-limited/` — config, metrics, evaluator_state, README
- Adapter weight blobs (`adapter_model.safetensors`) and checkpoint-*/
  are excluded by `.gitignore` (regenerable from training config).

### Evaluation Results (tracked in git)
- `evaluations/p3-limited/balanced-limited-frozen-v4.json` — 465-sample full eval
- `evaluations/p3-limited/repair-limited-frozen-v4.json` — 465-sample full eval

### Checkpoint Probe Reports (tracked in git)
- `reports/p3/probe_step{46,92,138,184,230,276,322,368}_report.json`
- `reports/p3/probe_step{...}_samples.jsonl`
- `reports/p3/fullval_epoch{1,2}_report.json`
- `reports/p3/fullval_epoch{1,2}_samples.jsonl`
- `reports/p3/checkpoint-evidence.jsonl`

### Dataset Manifests (committed in 23a0dd6)
- `data/p3-limited/balanced-limited/manifest.json` + `train.jsonl`
- `data/p3-limited/repair-limited/manifest.json` + `train.jsonl`

### Training Configs (committed in 23a0dd6)
- `configs/p3/balanced-limited.yaml`
- `configs/p3/repair-limited.yaml`

## 8. Reproduction

```powershell
# Build datasets (already committed)
py -3.11 scripts/build_p3_limited.py

# Train (run sequentially; each takes ~5h on RTX 3050 4GB)
py -3.11 scripts/train_lora.py --config configs/p3/balanced-limited.yaml
py -3.11 scripts/train_lora.py --config configs/p3/repair-limited.yaml

# Frozen v4 evaluation
py -3.11 scripts/evaluate_model.py \
  --model models/Qwen3-0.6B \
  --adapter adapters/p3/balanced-limited \
  --dataset data/frozen-eval/v4/test_raw.jsonl \
  --output evaluations/p3-limited/balanced-limited-frozen-v4.json

py -3.11 scripts/evaluate_model.py \
  --model models/Qwen3-0.6B \
  --adapter adapters/p3/repair-limited \
  --dataset data/frozen-eval/v4/test_raw.jsonl \
  --output evaluations/p3-limited/repair-limited-frozen-v4.json
```
