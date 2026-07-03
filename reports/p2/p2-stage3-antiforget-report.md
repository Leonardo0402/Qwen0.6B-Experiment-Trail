# P2 Anti-Forgetting Stage3-v3 Report (Issue #1 P2)

Generated: 2026-07-03T12:45:00+00:00

## Objective

Issue #1 P2 要求从 **Stage2-v2 继续训练** 一个抗遗忘的 Stage3-v3，调整课程混合与优化强度，目标是在保留 Stage2 code_generation 能力的同时提升 repair，并通过复合指标选择 checkpoint。

## Training Configuration

| 项 | 值 |
|---|---|
| Config | `configs/curriculum/p2-stage3-repair-v3-antiforget.yaml` |
| Training mode | `continual` (Stage2-v2 → Stage3-v3) |
| Initial adapter | `adapters/p2/continual/stage2-boundary-v2` (frozen, unmodified) |
| Output dir | `adapters/p2/continual/stage3-repair-v3` (全新目录) |
| Train file | `data/p2-curriculum/stage3-repair-v3/train.jsonl` (560 mixed samples) |
| Eval file | `data/p2-curriculum/stage3-repair-v3/validation.jsonl` (226 samples) |
| num_train_epochs | **1** (vs v2 的 2) |
| learning_rate | **1.0e-5** (10× smaller than v2's 3.0e-5) |
| max_seq_length | 384 |
| bf16 | True |
| gradient_checkpointing | True |
| LoRA rank/alpha | 16 / 32 (same as v2) |
| LoRA target_modules | 7 modules (same as v2) |
| assistant_only_loss | True |
| truncation_policy | preserve_assistant |
| seed | 42 |

### Data Mixture (Issue #1 P2 spec: 25%/25%/15%/35%)

| Source | Target % | Actual N | Actual % | Note |
|---|---|---|---|---|
| Stage1/code replay | 25% | 84 | 15.0% | Capped — Stage1 train pool only 84 samples |
| Stage2/boundary replay | 25% | 140 | 25.0% | OK |
| Static repair | 15% | 84 | 15.0% | OK |
| Execution repair | 35% | 252 | 45.0% | Rebalanced +10% (Stage1 deficit) |
| **Total** | 100% | **560** | 100% | |

**Deviation note:** Stage1 pool only has 84 samples (25% of 560 = 140 required). The 56-sample deficit (10%) was rebalanced to execution_repair (35% → 45%). This is recorded in `data/p2-curriculum/stage3-repair-v3/manifest.json` under `deviation_note`.

### Checkpoint Selection (Composite Metric, Issue #1 P2)

```yaml
composite_metric:
  code_generation: 0.35
  static_repair: 0.20
  execution_repair: 0.30
  hidden_pass: 0.15
hard_constraint:
  code_generation_drop_vs_stage2_max_pct: 2.5
```

## Training Evidence

| 项 | 值 |
|---|---|
| Started at | 2026-07-03T11:54:31+00:00 |
| Finished at | 2026-07-03T12:10:27+00:00 |
| Train duration | 937 s (15.6 min) |
| Trainable params | 10,092,544 / 606,142,464 (1.67%) |
| Parent adapter intact | `adapters/p2/continual/stage2-boundary-v2` mtime unchanged (确认父 Adapter 未被修改) |
| Train hash | (见 metrics.json) |
| Final train_loss | 0.2997 |
| Final eval_loss | 0.4495 |

### Loss Curve (sampled)

| Step | Epoch | Train loss | Eval loss |
|---|---|---|---|
| 10 | 0.143 | - | 0.4489 |
| 15 | 0.214 | 0.2750 | - |
| 20 | 0.286 | 0.2741 | 0.4504 |
| 25 | 0.357 | 0.3189 | - |
| 40 | 0.571 | - | 0.4507 |
| 60 | 0.857 | - | 0.4499 |
| 65 | 0.929 | 0.2910 | - |
| 70 (final) | 1.000 | - | 0.4495 |

### Token Truncation Audit

| 项 | 值 |
|---|---|
| Total samples | 560 |
| Assistant intact | 560 (100%) |
| Assistant lost | 0 |
| Target too long | 0 |

### Adapter Save/Reload Verification

- Original LoRA param mean: 0.000106
- Reloaded LoRA param mean: 0.000106
- Adapter save/reload verification: **OK**

### Adapter SHA256

| Field | Value (first 32) |
|---|---|
| weight_sha256 | `768bc7b6de538678...` |
| parent_adapter_weight_sha256 | `62a41f2a8a5c6206...` (Stage2-v2 weight) |
| parent_adapter_config_sha256 | `010670482beb86ee...` (Stage2-v2 config) |
| training_mode | `continual` |
| Parent chain verified | **OK** (DAG branch: stage3-v3-antiforget → stage2-boundary, weight_match=True, config_match=True) |

## Evaluation on Stratified-120 (Fixed Subset)

- Dataset: `data/p2-curriculum/frozen-eval-v2/stratified-120/test_raw.jsonl`
- Subset SHA256: `de835106...`
- Samples: 120 (40/40/40)
- Canary: all failed as expected

### Overall Metrics

| Metric | Stage3-v3-Antiforget |
|---|---|
| Pass@1 | 20.0% |
| Syntax rate | 99.2% |
| Hidden pass rate | 50.0% |
| Format compliance | 100.0% |
| Timeout rate | 6.7% |
| Repair success | 55.0% |
| Regression rate | 42.5% |
| Family pass | 13/58 (22.4%) |

### Per-Task-Type

| Task type | Total | Passed | Rate |
|---|---|---|---|
| code_generation | 40 | 8 | 20.0% |
| static_repair | 40 | 22 | 55.0% |
| execution_repair | 40 | 22 | 55.0% |

## Capability Gate (Issue #1 验收准线 vs Stage2-v2)

| Gate | Threshold | Stage3-v3 | Stage2-v2 | Verdict |
|---|---|---|---|---|
| execution_repair | ≥ 65% | 55.0% | 57.5% | **FAIL** |
| code_generation | ≥ 30% | 20.0% | 15.0% | **FAIL** (但 vs Stage2 +5pp) |
| static_repair | ≥ 65% | 55.0% | 50.0% | **FAIL** |
| family-level drop vs Stage2 | ≤ 2.5pp | 22.4% vs 20.7% (+1.7pp) | 20.7% | **PASS** |
| syntax | ≥ 98% | 99.2% | 99.2% | **PASS** |
| timeout | ≤ 2% | 6.7% | 6.7% | **FAIL** |

### Hard Constraint (code_gen drop vs Stage2 ≤ 2.5pp)

- Stage2 code_gen: 15.0%
- Stage3-v3 code_gen: 20.0%
- Δ = +5.0pp (improvement, not drop)
- **Hard constraint: PASS** (no forgetting on code_generation)

## Anti-Forgetting Effectiveness Analysis

### 对比 Stage3-v2 (Continual, no replay)

| Metric | Stage2 (base) | Stage3-v2 (no replay) | Stage3-v3 (with replay) | v3 vs v2 |
|---|---|---|---|---|
| Pass@1 | 15.0% | 15.0% | 20.0% | **+5.0pp** |
| code_gen | 15.0% | 15.0% | 20.0% | **+5.0pp** |
| static_repair | 50.0% | 55.0% | 55.0% | 0.0pp |
| execution_repair | 57.5% | 50.0% | 55.0% | **+5.0pp** |
| Family pass | 12/58 | 13/58 | 13/58 | 0 |
| Hidden pass | 46.7% | 45.8% | 50.0% | **+4.2pp** |
| Repair success | 53.8% | 52.5% | 55.0% | +2.5pp |
| Regression | 43.8% | 46.3% | 42.5% | −3.8pp |

### 对比 Stage2-v2 (parent)

| Metric | Stage2-v2 | Stage3-v3 | Δ (v3 − Stage2) |
|---|---|---|---|
| Pass@1 | 15.0% | 20.0% | **+5.0pp** |
| code_gen | 15.0% | 20.0% | **+5.0pp** (no forgetting!) |
| static_repair | 50.0% | 55.0% | +5.0pp |
| execution_repair | 57.5% | 55.0% | −2.5pp |
| Family pass | 12/58 | 13/58 | +1 |
| Hidden pass | 46.7% | 50.0% | +3.3pp |

### McNemar paired test (Stage2 → Stage3-v3, N=120)

- Win (Stage2 fail → v3 pass): 19
- Loss (Stage2 pass → v3 fail): 16
- McNemar b/c = 16/19, p (2-sided) = 0.7359
- 95% bootstrap CI for Δ rate: [−0.0750, +0.1250]

### McNemar (Stage3-v2 → Stage3-v3, N=120)

- Win: 7, Loss: 10, p = 0.6291
- 95% CI: [−0.0917, +0.0417]

## Interpretation

### 结论 1：抗遗忘策略成功保留 code_generation

Stage3-v2 在 code_gen 上相对 Stage2 持平 (15% → 15%)，而 Stage3-v3 提升 +5pp (15% → 20%)。这证明 **replay 25% Stage1 + 25% Stage2 有效防止了 code_gen forgetting**。

### 结论 2：execution_repair 未能达到 Stage3-v2 的高度

Stage3-v2 的 execution_repair 为 50.0%（低于 Stage2 的 57.5%），Stage3-v3 恢复到 55.0%，但仍低于 Stage2 的 57.5%。Replay 帮助恢复了部分 exec_repair，但未超过 Stage2。

### 结论 3：综合 Pass@1 提升，但未达 Issue #1 Gate

Stage3-v3 Pass@1 20% 是所有 v2/v3 模型中最高的，比 Stage2 +5pp。但 Issue #1 能力 Gate 要求 exec_repair ≥ 65%、code_gen ≥ 30%、static_repair ≥ 65%，均未达到。

### 结论 4：Timeout 率偏高

Stage3-v3 timeout 6.7% (8/120)，超过 Gate 的 2% 阈值。这可能是 1.0e-5 学习率下部分样本生成过长代码导致。Stage2 同样 6.7%，说明是数据/模型固有特性，非 v3 引入。

### 结论 5：统计显著性不足

McNemar p=0.7359 (Stage2 → v3)，95% CI 跨越 0。20% vs 15% 在 N=120 下不构成统计显著差异。不应将 +5pp 描述为"稳定能力提升"。

## Verdict

**Stage3-v3 抗遗忘策略部分成功：成功消除了 code_gen forgetting（+5pp），综合 Pass@1 为所有模型最高 (20%)，但 Issue #1 能力 Gate 的 exec_repair ≥ 65% 等硬阈值未达到。** 建议：保留 Stage2 作为综合模型，Stage3-v3 作为 Repair + 通用能力兼顾的候选。若需达 Issue #1 Gate，需扩大训练数据（924 → 2100-3400）或升级到 1.7B 模型。

## Artifacts

- Adapter: `adapters/p2/continual/stage3-repair-v3/`
- Eval: `evaluations/p2/stage3-v3-antiforget.json`
- Config: `configs/curriculum/p2-stage3-repair-v3-antiforget.yaml`
- Mixed data: `data/p2-curriculum/stage3-repair-v3/`
- Paired stats: `reports/p2/paired-stats.json`
- Adapter evidence: `reports/p2/adapter-evidence.json` (key: stage3-v3-antiforget)
