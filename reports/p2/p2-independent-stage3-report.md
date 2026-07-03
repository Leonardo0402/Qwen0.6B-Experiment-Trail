# P2 Independent Stage3 Report (Issue #1 P1)

Generated: 2026-07-03T12:45:00+00:00

## Objective

Issue #1 P1 要求训练一个 **Independent Stage3**（从 Base 独立训练，不加载 Stage1/Stage2 Adapter），用于区分 Stage3 Continual 的提升究竟来自 "Repair 数据本身" 还是 "Continual Curriculum"。Independent 与 Continual Stage3 使用完全一致的数据、LoRA、精度、序列长度、epoch、学习率和 seed，唯一差异是不加载父 Adapter。

## Training Configuration

| 项 | 值 |
|---|---|
| Config | `configs/curriculum/p2-stage3-repair-independent.yaml` |
| Training mode | `independent` (从 Base，no parent adapter) |
| Model | `models/Qwen3-0.6B` |
| Train file | `data/p2-curriculum/stage3-repair/train.jsonl` (560 samples) |
| Eval file | `data/p2-curriculum/stage3-repair/validation.jsonl` (226 samples) |
| Output dir | `adapters/p2/independent/stage3-repair-v2` (全新目录，未覆盖任何 v2 Adapter) |
| max_seq_length | 384 |
| num_train_epochs | 2 |
| learning_rate | 3.0e-5 (cosine) |
| warmup_ratio | 0.03 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 8 |
| bf16 | True (BF16 supported validated) |
| gradient_checkpointing | True |
| LoRA rank/alpha | 16 / 32 |
| LoRA target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| assistant_only_loss | True |
| truncation_policy | preserve_assistant |
| seed | 42 |

## Training Evidence

| 项 | 值 |
|---|---|
| Started at | 2026-07-03T10:57:39+00:00 |
| Finished at | 2026-07-03T11:25:28+00:00 |
| Train duration | 1661 s (27.7 min) |
| Trainable params | 10,092,544 / 606,142,464 (1.67%) |
| Peak GPU | 1376 MiB (RTX 3050 4GB) |
| Train hash | d5413247371c4da7... |
| Eval hash | a17e4198196b7212... |

### Token Truncation Audit

| 项 | 值 |
|---|---|
| Total samples | 560 |
| Not truncated | 429 |
| Truncated | 131 |
| Assistant intact | 560 (100%) |
| Assistant partial | 0 |
| Assistant lost | 0 |
| Target too long | 0 |

### Loss Curve (sampled)

| Step | Epoch | Train loss | Eval loss |
|---|---|---|---|
| 25 | 0.357 | 0.5542 | - |
| 30 | 0.429 | 0.4909 | 0.5262 |
| 50 | 0.714 | - | 0.4758 |
| 85 | 1.214 | 0.3441 | - |
| 120 | 1.714 | 0.3884 | 0.4378 |
| 140 (final) | 2.000 | - | 0.4379 |

Final train_loss=0.4657，eval_loss 从 0.59 降至 0.44，收敛正常。

### Adapter Save/Reload Verification

- Original LoRA param mean: 0.000110
- Reloaded LoRA param mean: 0.000110
- Adapter save/reload verification: **OK**

### Adapter SHA256

| Field | Value (first 32) |
|---|---|
| weight_sha256 | `65e5c1f0aacff3d7...` |
| config_sha256 | (见 adapter-evidence.json) |
| parent_adapter_weight_sha256 | `None` (independent mode, root node) |
| parent_adapter_config_sha256 | `None` |
| training_mode | `independent` |
| Parent chain verified | **OK** (DAG root node, no parent — verified via DAG lookup) |

## Evaluation on Stratified-120 (Fixed Subset)

- Dataset: `data/p2-curriculum/frozen-eval-v2/stratified-120/test_raw.jsonl`
- Subset SHA256: `de835106...`
- Samples: 120 (40 code_generation + 40 static_repair + 40 execution_repair)
- Families: 58 unique
- Canary: all canaries failed as expected (harness trustworthy)

### Overall Metrics

| Metric | Independent Stage3 |
|---|---|
| Pass@1 | 15.0% |
| Syntax rate | 99.2% |
| Hidden pass rate | 52.5% |
| Format compliance | 100.0% |
| Timeout rate | 3.3% |
| Repair success | 61.3% |
| Regression rate | 37.5% |
| Family pass | 12/58 (20.7%) |

### Per-Task-Type

| Task type | Total | Passed | Rate |
|---|---|---|---|
| code_generation | 40 | 6 | 15.0% |
| static_repair | 40 | 22 | 55.0% |
| execution_repair | 40 | 24 | 60.0% |

### Per-Bug-Type Repair Success

| Bug type | Independent Stage3 |
|---|---|
| aggregation_error | 0/1 (0.0%) |
| branch_deletion | 8/18 (44.4%) |
| condition_error | 6/7 (85.7%) |
| index_error | 1/1 (100.0%) |
| initialization_error | 5/7 (71.4%) |
| off_by_one | 8/13 (61.5%) |
| return_value_error | 16/27 (59.3%) |
| type_error | 5/6 (83.3%) |

## Comparison vs Continual Stage3-v2 (Same Data, Same Eval Set)

| Metric | Independent Stage3 | Continual Stage3-v2 | Δ (Indep − Cont) |
|---|---|---|---|
| Pass@1 | 15.0% | 15.0% | 0.0 |
| Syntax | 99.2% | 98.3% | +0.8 |
| Repair success | 61.3% | 52.5% | **+8.8** |
| Regression | 37.5% | 46.3% | **−8.8** |
| Hidden pass | 52.5% | 45.8% | **+6.7** |
| Family pass | 12/58 | 13/58 | −1 |
| code_gen | 15.0% | 15.0% | 0.0 |
| static_repair | 55.0% | 55.0% | 0.0 |
| execution_repair | 60.0% | 50.0% | **+10.0** |

### McNemar paired test (Continual → Independent, N=120)

- Win (Cont fail → Indep pass): 13
- Loss (Cont pass → Indep fail): 6
- McNemar b/c = 6/13, p (2-sided) = 0.1671
- 95% bootstrap CI for Δ rate: [−0.0083, +0.1333]

## Interpretation

### 结论 1：Continual Curriculum 对 Stage3 的 net 提升 在统计上不显著

Continual Stage3-v2 vs Independent Stage3 在 Pass@1 上完全相同 (15.0% vs 15.0%)。McNemar p=0.1671 > 0.05，95% CI 跨越 0，无法拒绝 H0。

### 结论 2：Independent Stage3 在 Repair 维度更强

Independent 在 execution_repair (+10pp) 和 repair_success (+8.8pp) 上明显优于 Continual，hidden_pass 也 +6.7pp。说明 Continual Curriculum 中 Stage1/Stage2 的累积反而对 Repair 产生了轻微干扰（forgetting side-effect）。

### 结论 3：Continual 在 family-level 略有优势

Continual Stage3 通过 13 个 family，Independent 通过 12 个。但 net 差异仅 1 个 family，不构成稳定结论。

### 结论 4：当前没有证据证明 Continual 相比 Independent 有显著额外收益

对比 Base → Independent Stage3：
- Pass@1: 20.0% → 15.0% (−5pp)
- repair_success: 51.3% → 61.3% (+10pp)
- execution_repair: 47.5% → 60.0% (+12.5pp)
- static_repair: 55.0% → 55.0% (0pp)
- code_gen: 20.0% → 15.0% (−5pp)

Repair 数据本身带来了 execution_repair +12.5pp 的提升，但代价是 code_gen −5pp（数据偏置导致的 capability shift，与是否 Continual 无关）。

## Verdict

**Independent Stage3 在 Repair 维度上优于 Continual Stage3-v2，但 Pass@1 持平，差异未达统计显著。** Continual Curriculum 的主要价值在于 family-level 稍多 1 个，但这不足以证明 Continual 优于 Independent。当前没有证据证明 Continual 相比 Independent 有显著额外收益（McNemar p=0.1671，95% CI 跨越 0）。

## Artifacts

- Adapter: `adapters/p2/independent/stage3-repair-v2/`
- Eval: `evaluations/p2/independent-stage3.json`
- Config: `configs/curriculum/p2-stage3-repair-independent.yaml`
- Paired stats: `reports/p2/paired-stats.json` (pair: stage3-repair → independent-stage3)
- Adapter evidence: `reports/p2/adapter-evidence.json` (key: independent-stage3)
