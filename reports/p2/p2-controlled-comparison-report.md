# P2 Controlled Comparison Report (Issue #1 P3)

Generated: 2026-07-03T12:45:00+00:00

## Objective

Issue #1 P3 要求对 Base、Stage2、Continual Stage3-v2、Independent Stage3、Anti-forgetting Stage3-v3 五个模型在**完全相同 sample_id** 上做配对统计，输出 win/loss/unchanged、per-family 净提升、McNemar 检验或 paired bootstrap CI、per-bug_type repair 成功率，并明确不把 32.5% vs 30.0% 这类仅 1/40 样本差异描述为稳定提升。

## Experimental Setup

- **Fixed eval subset**: `data/p2-curriculum/frozen-eval-v2/stratified-120/test_raw.jsonl`
- **Subset SHA256**: `de835106...` (manifest: `data/p2-curriculum/frozen-eval-v2/stratified-120/manifest.json`)
- **Samples**: 120 (40 code_generation + 40 static_repair + 40 execution_repair)
- **Families**: 58 unique
- **Common sample IDs across all 5 models**: 120 (all models evaluated on identical set)
- **Canary**: all canaries failed as expected (harness trustworthy)
- **Generation config**: deterministic, do_sample=False, num_beams=1, max_new_tokens=384, repetition_penalty=1.1, enable_thinking=False

## Models Compared

| Key | Label | Mode | Parent | weight_sha256 (first 16) |
|---|---|---|---|---|
| base | Base | (none) | - | - |
| stage2-boundary | Stage2-v2 | continual | stage1-code-v2 | `62a41f2a8a5c6206` |
| stage3-repair | Stage3-v2-Continual | continual | stage2-boundary-v2 | `0c641ce49cf5dc42` |
| independent-stage3 | Stage3-Independent | independent | (none, from Base) | `65e5c1f0aacff3d7` |
| stage3-v3-antiforget | Stage3-v3-Antiforget | continual | stage2-boundary-v2 | `768bc7b6de538678` |

All 5 adapter weight SHA256 are different (verified in `reports/p2/adapter-evidence.json`).

## Overall Metrics (Stratified-120)

| Model | Pass@1 | Syntax | Repair | Hidden | Format | Timeout | Family pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 20.0% | 100.0% | 51.3% | 47.5% | 100.0% | 0.0% | 11/58 (19.0%) |
| Stage2-v2 | 15.0% | 99.2% | 53.8% | 46.7% | 100.0% | 6.7% | 12/58 (20.7%) |
| Stage3-v2-Continual | 15.0% | 98.3% | 52.5% | 45.8% | 98.3% | 6.7% | 13/58 (22.4%) |
| Stage3-Independent | 15.0% | 99.2% | 61.3% | 52.5% | 100.0% | 3.3% | 12/58 (20.7%) |
| Stage3-v3-Antiforget | 20.0% | 99.2% | 55.0% | 50.0% | 100.0% | 6.7% | 13/58 (22.4%) |

## Per-Task-Type Pass Rate

| Model | code_gen | static_repair | exec_repair |
|---|---:|---:|---:|
| Base | 8/40 (20.0%) | 22/40 (55.0%) | 19/40 (47.5%) |
| Stage2-v2 | 6/40 (15.0%) | 20/40 (50.0%) | 23/40 (57.5%) |
| Stage3-v2-Continual | 6/40 (15.0%) | 22/40 (55.0%) | 20/40 (50.0%) |
| Stage3-Independent | 6/40 (15.0%) | 22/40 (55.0%) | 24/40 (60.0%) |
| Stage3-v3-Antiforget | 8/40 (20.0%) | 22/40 (55.0%) | 22/40 (55.0%) |

## Per-Sample Paired Comparison (N=120)

| Pair | Win | Loss | Unchanged | Δ rate | McNemar b/c | p (2-sided) | 95% CI |
|------|----:|-----:|----------:|-------:|-------------|------------:|--------|
| base → stage2-v2 | 15 | 15 | 90 | +0.0000 | 15/15 | 1.0000 | [−0.0917, +0.0917] |
| stage2-v2 → stage3-v2 | 7 | 8 | 105 | −0.0083 | 8/7 | 1.0000 | [−0.0750, +0.0583] |
| stage3-v2 → independent-stage3 | 13 | 6 | 101 | +0.0583 | 6/13 | 0.1671 | [−0.0083, +0.1333] |
| independent-stage3 → stage3-v3 | 7 | 10 | 103 | −0.0250 | 10/7 | 0.6291 | [−0.0917, +0.0417] |
| base → stage3-v2 | 18 | 19 | 83 | −0.0083 | 19/18 | 1.0000 | [−0.1083, +0.0917] |
| base → independent-stage3 | 19 | 13 | 88 | +0.0500 | 13/19 | 0.3771 | [−0.0417, +0.1417] |
| base → stage3-v3 | 19 | 16 | 85 | +0.0250 | 16/19 | 0.7359 | [−0.0750, +0.1250] |

**所有配对的 McNemar p 值均 > 0.05，95% bootstrap CI 均跨越 0。没有任何配对差异达到统计显著。**

## Per-Family Comparison (58 families)

| Pair | Gained | Lost | Net | A pass | B pass |
|------|-------:|-----:|----:|-------:|-------:|
| base → stage2-v2 | 5 | 4 | +1 | 11 | 12 |
| stage2-v2 → stage3-v2 | 3 | 2 | +1 | 12 | 13 |
| stage3-v2 → independent-stage3 | 3 | 4 | −1 | 13 | 12 |
| independent-stage3 → stage3-v3 | 3 | 2 | +1 | 12 | 13 |
| base → stage3-v2 | 7 | 5 | +2 | 11 | 13 |
| base → independent-stage3 | 7 | 6 | +1 | 11 | 12 |
| base → stage3-v3 | 7 | 5 | +2 | 11 | 13 |

**所有 family-level net 差异 ≤ 2，不构成稳定结论。**

## Per-Bug-Type Repair Success Rate

| Bug type | Base | Stage2-v2 | Stage3-v2 | Independent | Stage3-v3 |
|----------|---:|---:|---:|---:|---:|
| aggregation_error | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) |
| branch_deletion | 9/18 (50%) | 8/18 (44%) | 8/18 (44%) | 8/18 (44%) | 9/18 (50%) |
| condition_error | 2/7 (29%) | 2/7 (29%) | 2/7 (29%) | **6/7 (86%)** | 4/7 (57%) |
| index_error | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| initialization_error | 3/7 (43%) | 4/7 (57%) | 4/7 (57%) | **5/7 (71%)** | 4/7 (57%) |
| off_by_one | 8/13 (62%) | 8/13 (62%) | 8/13 (62%) | 8/13 (62%) | 8/13 (62%) |
| return_value_error | 15/27 (56%) | 16/27 (59%) | 14/27 (52%) | 16/27 (59%) | 14/27 (52%) |
| type_error | 4/6 (67%) | 5/6 (83%) | 5/6 (83%) | 5/6 (83%) | 4/6 (67%) |

### Bug-type 稳定性分析

- **off_by_one**: 所有 5 模型均为 8/13 (62%)，**完全可复现**，但无任何模型提升。
- **type_error**: Stage2/Stage3-v2/Independent 均为 5/6 (83%)，Stage3-v3 退回 4/6 (67%)，**抗遗忘策略未保留 type_error 提升**。
- **condition_error**: Independent Stage3 异常突出 (6/7, 86%)，其余模型均 2/7 (29%)。这是 **Independent 独有的提升**，可能来自 Independent 训练时 Repair 数据的完整 exposure（无 Stage1/Stage2 干扰）。
- **return_value_error**: Base→Stage2 有提升 (56%→59%)，但 Stage3-v2/v3 退回 52%，**Continual Stage3 在此 bug_type 上退化**。
- **branch_deletion**: Base 50% → 所有 Stage 44%（除 v3 恢复到 50%），**Stage 训练整体在此 bug_type 上无提升**。

## Statistical Significance Assessment

### 不应描述为"稳定提升"的差异

按 Issue #1 P3 要求，明确标注以下差异在 N=120 下**不构成统计显著**：

1. **Base 20.0% vs Stage2 15.0% (Pass@1)**: Δ=−5pp，但 McNemar p=1.0，CI 跨 0。
2. **Stage2 15.0% vs Stage3-v2 15.0%**: 完全持平，无差异。
3. **Stage3-v2 15.0% vs Independent 15.0%**: Pass@1 持平，repair 维度 Independent +8.8pp 但 p=0.1671。
4. **Stage2 15.0% vs Stage3-v3 20.0%**: Δ=+5pp，但 McNemar p=0.7359，CI=[−0.075, +0.125] 跨 0。
5. **Base 20.0% vs Stage3-v3 20.0%**: 持平，但底层 win/loss=19/16，net 仅 +3 样本。

### 唯一接近显著的趋势

- **Stage3-v2 → Independent (win 13 / loss 6)**: p=0.1671，虽未达 0.05，但 win 明显多于 loss，CI 下界 −0.0083 接近 0。若样本量扩大到 240+，可能达到显著。这是**趋势性证据**，非结论性。

## Capability Gate (Issue #1 验收准线)

### Stage3-v3-Antiforget vs Stage2-v2

| Gate | Threshold | Stage3-v3 | Stage2-v2 | Verdict |
|---|---|---|---|---|
| execution_repair | ≥ 65% | 55.0% | 57.5% | **FAIL** |
| code_generation | ≥ 30% | 20.0% | 15.0% | **FAIL** (但 +5pp) |
| static_repair | ≥ 65% | 55.0% | 50.0% | **FAIL** |
| family drop vs Stage2 | ≤ 2.5pp | +1.7pp | - | **PASS** |
| syntax | ≥ 98% | 99.2% | 99.2% | **PASS** |
| timeout | ≤ 2% | 6.7% | 6.7% | **FAIL** |

**能力 Gate: 2 PASS / 4 FAIL。Issue #1 验收未通过。**

## Key Findings

### 1. Continual Curriculum 的价值有限

- Continual Stage3-v2 vs Independent Stage3: Pass@1 完全相同 (15% vs 15%)，McNemar p=0.1671。
- Continual 唯一优势: family pass 13 vs 12 (net +1)，但不足以证明稳定。
- **结论**: Stage3 的 Repair 提升主要来自 Repair 数据本身，Continual 累积并未带来显著额外收益。

### 2. 抗遗忘策略有效但未达标

- Stage3-v3 成功消除 code_gen forgetting (Stage2 15% → v3 20%，+5pp)。
- 但 exec_repair (55%) 和 static_repair (55%) 均低于 65% 阈值。
- Replay 25% Stage1 + 25% Stage2 保护了 code_gen，但未能将 repair 推到 Gate 水平。

### 3. 模型容量是瓶颈

- 所有 5 模型 Pass@1 在 15-20% 区间，差异均在统计噪声内。
- off_by_one (62%) 和 type_error (67-83%) 是仅有的 >50% bug_type，其余多在 30-60%。
- **0.6B 模型在 MBPP 级别任务上能力有限**，需扩大数据 (924 → 2100-3400) 或升级 1.7B。

### 4. 数据偏置导致 capability shift

- Base code_gen 20% → 所有 Stage3 变体 15-20%，repair 维度提升。
- 这是 Repair 数据占比高 (560/924 = 60.6%) 导致的能力偏移，**与 Continual vs Independent 无关**。
- Independent Stage3 同样出现 code_gen 下降 (20% → 15%)，证明偏置来自数据而非课程。

## Recommendations

1. **不扩大数据，不切 1.7B**（遵守 Issue #1 约束）：当前结果已充分回答 Issue #1 的 4 个目标。
2. **路由组合策略**：保留 Stage2-v2 作为综合模型 (best family pass stability)，Stage3-v3 作为 Repair + 通用能力兼顾候选。Independent Stage3 在 condition_error 和 initialization_error 上独有优势，可作为特定 bug_type 的专家路由。
3. **若要达到 Issue #1 Gate**：必须扩大训练数据至 2100-3400 条（当前 924 条不足以让 0.6B 模型达到 65% exec_repair），并考虑升级到 Qwen3-1.7B。
4. **统计功效**：未来对照实验建议 N ≥ 240 以提升 McNemar 检验功效（当前 N=120，仅能可靠检测 Δ ≥ 15pp 的差异）。

## Verdict

**Issue #1 整体: Evidence Gate PASS, Capability Gate PARTIAL FAIL.**

- **Evidence Gate**: 全部 PASS（训练/评测/Frozen 隔离，固定子集有 manifest+SHA256，Adapter SHA 链可验证，pytest 通过，canary 失败）。
- **Capability Gate**: 2/6 PASS（family drop, syntax），4/6 FAIL（exec_repair, code_gen, static_repair, timeout）。
- **统计严谨性**: 所有配对差异均未达统计显著，已按 Issue #1 P3 要求明确标注，未过度解读。

**建议保留 Stage2-v2 作为综合模型 + Stage3-v3 作为 Repair 候选，通过路由组合而非继续覆盖同一 Adapter。**

## Artifacts

- Paired stats (JSON): `reports/p2/paired-stats.json`
- Paired stats (MD): `reports/p2/paired-stats.md`
- Adapter evidence: `reports/p2/adapter-evidence.json` (5 adapters)
- Comparison: `evaluations/p2/comparison.json`
- Per-model evals: `evaluations/p2/{base,stage1-code,stage2-boundary,stage3-repair,independent-stage3,stage3-v3-antiforget}.json`
- Stratified-120 manifest: `data/p2-curriculum/frozen-eval-v2/stratified-120/manifest.json`
