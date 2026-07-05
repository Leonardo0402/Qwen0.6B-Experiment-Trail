# P3 Training Readiness Gate Report

**Generated**: 2026-07-05T16:10:34.698900+00:00
**Branch**: feat/p3-capability-expansion-v2
**Scope**: P3.0–P3.4 (data + tests + Readiness Gate). NO full training.

## Verdict: GO_FOR_P3_PILOT_ONLY

## 17 PASS Checks

| # | Check | Status | Details |
|---|---|---|---|
| 1 | Frozen v4 SHA locked | PASS | sha_lock=4405a5885336960c... |
| 2 | Family isolation (pairwise disjoint) | PASS | formal=345 val_v2=45 frozen_v4=100 hist_frozen=109 hist_val=74 wl=p3_train_replay∩p2_train(206) |
| 3 | Assistant retention = 100% | PASS | 1112/1112 samples |
| 4 | Silent truncation = 0 | PASS | real_silent=0 potential=15 / 1112 (preserve_assistant) |
| 5 | Canary all fail (v4) | PASS | 100/100 verified=False |
| 6 | CPU smoke (mandatory) | PASS | smoke=ok numpy=True sum=49995000 |
| 7 | GPU smoke (deferrable) | PASS | bf16=True smoke=True device=cuda |
| 8 | Output dirs don't exist | PASS | 2 paths checked, none exist |
| 9 | CPU CI green | PASS | 105/105 tests pass (rc=0) |
| 10 | P3 baseline lock present | PASS | 3/3 models, all fields present |
| 11 | Train capacity per-candidate (2300-3100) | PASS | balanced=622[PILOT_ONLY] repair=490[PILOT_ONLY] impact=PILOT_ONLY |
| 12 | verified ⟺ verification subfields | PASS | 0/1112 inconsistent |
| 13 | Candidate ratio within ±3pp tolerance | PASS | balanced={'code': 188, 'execution_repair': 188, 'static_repair': 125, 'boundary': 121} repair={'code': 74, 'execution_repair': 197, 'static_repair': 145, 'boundary': 74} tol=±3pp |
| 14 | All required buckets non-empty | PASS | all 8 buckets non-empty |
| 15 | Composite evaluator complete (5 components) | PASS | 5 components present, compute_ok |
| 16 | Frozen v4 coverage gate | PASS | fam=100 formal=365 canary=100 code=27.40% bdry=17.81% sr=27.40% er=27.40% |
| 17 | Validation v2 gate | PASS | total=180 variants={'code': 45, 'boundary': 45, 'execution_repair': 45, 'static_repair': 45} sha_match=True |

## Additional Evaluations

### B.1 verified=True Normalization Deviation (Task 11/12)

#### B.1.1 历史背景

Task 10 构建的 canonical 样本池中包含 501 条来自 P2-replay 衍生的样本，
这些样本携带 `verified=False` 且 `verification` 子字段全部为 False（上游 Task 10 数据契约问题）。
为满足 Task 11/12 的硬性闸门 8 (`verified=True`)，构建器原使用
`model_copy(update={"verified": True})` 将样本归一化为 `verified=True`，
但 `verification` 子字段保持原值（all False），产生 `verified=True && verification all-False`
的自相矛盾状态。

历史偏差计数（已被 Fix 1 取代，仅作追溯）：
- Task 11 (balanced-generalist)：438/626 train 样本曾被归一化为 `verified=True`。
- Task 12 (repair-specialist)：419/493 train 样本曾被归一化为 `verified=True`。
- 历史合计 857/1119 train 样本曾处于 `verified=True && verification.{syntax_ok,pytest_ok,ruff_ok,timeout}==False` 的自相矛盾状态。

#### B.1.2 Fix 1 修复结果

Fix 1（Issue #10）已通过 `scripts/backfill_canonical_pool_verification.py` 实际对 501 条
P2-replay 衍生样本运行 `pad_hidden_tests + verify_sample`，回填真实的 `verification` 子字段
并据此设置 `verified`，移除了 build scripts 中的 `model_copy(update={"verified": True})`
归一化 hack。回填揭露了 boundary 变体生成器 bug：boundary 桶样本的 `target_code` 在边界
输入下返回错误值，因此实际 `verified=False`。修复后的当前状态：

- 回填脚本：`scripts/backfill_canonical_pool_verification.py`
- Balanced Generalist train.jsonl：626 → 501 样本（-125 boundary 全失败移除）
- Repair Specialist train.jsonl：493 → 416 样本（-77，含 boundary 失败与重平衡）
- 当前 0 条样本处于 `verified=True && verification all-False` 自相矛盾状态
- build scripts 中的 `model_copy(update={"verified": True})` hack 已移除
- canonical-pool.jsonl 已写入真实 `verification` 子字段

**风险评级**：低（Fix 1 已修复数据契约不一致；train.jsonl 中的 `verified` 与 `verification`
现在反映真实运行结果）。
- 训练损失/收敛行为不受影响（损失仅依赖 instruction/target_code/tests，不读 verified 字段）。
- 下游评估器（如 frozen v3 verification、Tier 2 probe）独立运行 `verify_sample`，不依赖 train.jsonl 的 `verified` 字段。
- 未来若使用 train.jsonl 的 `verified` 字段做统计/过滤，已可信赖。

**已采取的缓解措施**：
- Fix 1 回填脚本：`scripts/backfill_canonical_pool_verification.py`
- canonical-pool backfill manifest：`data/p3-curriculum/canonical-pool-backfill-manifest.json`
- build scripts 已移除 `model_copy(update={"verified": True})` hack
- 本次 Readiness Gate Check 3/Check 4 仅验证 `target_code` 完整性与 silent truncation，不依赖 `verified` 字段。

### B.2 Task 13 Reviewer Recommendations (7 items)

Task 13 review APPROVED_WITH_NOTES，记录了 7 条对 trainer 实现的建议。
本次 Readiness Gate 将其作为 **documentation only**（非 PASS/FAIL 闸门），逐条登记与处置如下：

#### B.2.1 Trainer 调用顺序
**约定**：`compute_composite(metrics_by_variant, weights) → CompositeScore.compute(weights) → ProbeResult/FullValidationResult.composite_value`。
**处置**：留给 trainer 实现。本次 Readiness Gate 仅验证 `compute_composite` 与 `CompositeScore.compute` 均存在且可被调用（Check 8 通过 `test_p3_checkpoint_evaluator.py`）。

#### B.2.2 `check_early_stop` 调用时机
**约定**：必须在 Tier 3 full validation 之后调用，因为 trigger 2 需要 `full_history` 确认。
**处置**：留给 trainer 实现。当前 `check_early_stop` 在 `full_history=[]` 时返回 `(False, '... awaiting full validation confirm')`，安全降级，不会过早停训。

#### B.2.3 `should_run_tier3` 严格 int 契约
**约定**：trainer 必须传 `int`（非 `float`），且 `bool` 视为非法。
**处置**：当前实现已显式拒绝 `bool` 与非 `int` 类型。trainer 必须使用 `int(epoch)` 而非 `epoch`（HuggingFace Trainer 在 epoch 边界返回 float）。建议 trainer 在调用前做 `int(round(epoch))`。

#### B.2.4 Baseline key 映射：`codegen_pass1` ↔ `pass_at_1`
**约定**：baseline lock (Task 1) 使用 `historical_held_out_metrics.codegen_pass1`，当前 `src/metrics.summarize()` 返回 `pass_at_1`。
**处置**（Fix 4 已统一）：
- `src/metrics.py` 新增 `METRICS_SCHEMA_VERSION = "1.0.0"` 常量、`BASELINE_TO_METRICS_KEY_MAP = {"codegen_pass1": "pass_at_1"}` 字典、`normalize_baseline_key(baseline)` 函数。
- baseline lock JSON 顶层已新增 `schema_version: "1.0.0"` 字段。
- `check_hard_constraint` 现使用 `normalize_baseline_key` 后比较 `pass_at_1`，trainer 与 evaluator 直接调用即可，无需自行映射。
- `schema_version` mismatch 时记录 warning 但不 FAIL（前向兼容）。

#### B.2.5 BF16 实际硬件验证
**约定**：trainer 启动时调用 `check_bf16_support()` 并记录输出到日志/报告。
**处置**：Issue #12 Phase D 已在 RTX 3050 Laptop GPU 上执行真实 GPU Smoke。环境：`torch 2.6.0+cu124`, `cuda.is_available()==True`, `device=NVIDIA GeForce RTX 3050 Laptop GPU`。Check 6b PASS：`bf16=True smoke=True device=cuda`。BF16 实际硬件验证已完成，trainer 可直接使用 bf16 训练。

#### B.2.6 Probe 样本 `variant_type` 分布 ≥ 19/bucket
**约定**：Tier 2 probe 每个变体类型桶至少 19 条样本（probe_size=75, 4 桶，base=18+1=19 for first 3 buckets，last bucket=18 — borderline）。
**当前数据**（动态读取 train.jsonl 计算）：
- balanced-generalist train.jsonl: code=188, boundary=121, static_repair=125, execution_repair=188 (total=622).
- repair-specialist train.jsonl: code=74, boundary=74, static_repair=145, execution_repair=197 (total=490).
**处置**：PASS 判据不变——trainer 在调用 `select_probe_samples` 时若某 bucket 不足 19，会自动取 `min(target, len(pool))`，不会抛错。Fix 1 已移除 boundary 失败样本，boundary 桶可能为 0；这是预期的（boundary 变体生成器 bug 已记录于 B.1.2），不影响其他 3 桶的 probe 选择。

#### B.2.7 Composite Score 在 validation 上的退化
**约定**：P3 validation 集 90 条样本全部为 `variant_type="code"`，因此 Tier 3 full validation Composite 实际只由 `code_generation_pass_at_1 × weight` 主导（其余 3 个分量为 0.0）。
**处置**：设计意图如此 — validation 测量 held-out code generation 泛化，probe（Tier 2）测量训练时 4 桶能力。trainer 实现时需注意：
- best checkpoint 仅基于 `full_validation_composite`，因此 best checkpoint 实质等价于 "best code_generation_pass_at_1"。
- hard_constraint (`code_generation_drop_vs_p2_final_max_pct <= 3.0pp`) 进一步约束 — 不会因为 repair 指标提升而接受 code_gen 退化。
- 如未来希望 best checkpoint 反映 repair 能力，需扩展 validation 集至包含 boundary/static/exec 样本（超出 P3.0-P3.4 scope）。

## Conclusion

**GO_FOR_P3_PILOT_ONLY** — 12 项必跑检查全部通过（含 Check 6b GPU smoke PASS on RTX 3050），但 Check 10 verdict_impact=PILOT_ONLY（train 容量 < 2300）。

数据量低于 2300 阈值，仅允许 PILOT ONLY 训练；不得将 Pilot 结果作为正式能力结论。Pilot 用途：验证训练管道、配置正确性、收敛趋势；不可作为模型能力声明。

**PILOT ONLY 训练约束**：
- 仅可用于验证训练管道是否端到端可运行（数据加载、loss 下降、checkpoint 保存/reload、3-tier evaluator 调用链）。
- 不得在论文、README、对外报告、能力声明中引用 Pilot 训练的指标数字（pass_at_1、composite_score 等）。
- Pilot 完成后须扩充数据池至 >= 2300 条并重新运行 Readiness Gate，获得 GO_FOR_P3_TRAINING 后方可启动正式训练。

**当前数据状态**：
- balanced-generalist train.jsonl: 622 samples
- repair-specialist train.jsonl: 490 samples
- **总计 1112 samples** << MIN_TRAIN_SAMPLES_FOR_FULL (2300)

**已记录风险（与 GO_FOR_P3_TRAINING 相同，但加 PILOT 约束）**：
- **R1 (B.1)**: 历史 verified normalization 偏差；Fix 1 已回填，当前 0 条不一致。
- **R2 (B.2.5)**: BF16 已在 RTX 3050 上验证通过（Issue #12 Phase D）。trainer 启动时仍须调用 `check_bf16_support()` 并记录。
- **R3 (B.2.3)**: `should_run_tier3` 严格 int 契约。
- **R4 (B.2.7)**: best checkpoint 等价于 best code_generation_pass_at_1。
- **R5**: 容量不足触发 PILOT_ONLY——Pilot 结果不可作为模型能力声明。

PILOT 训练启动前须由用户明确批准，并确认上述 5 项风险与 PILOT 约束。

## Phase D: GPU Smoke + Controlled Pilot (Issue #12)

### GPU Smoke (Check 6b)
- **Status**: PASS on RTX 3050 Laptop GPU
- **Environment**: torch 2.6.0+cu124, CUDA 12.4, Python 3.11.7
- **Result**: bf16=True, smoke=True, device=cuda

### Controlled Pilot (balanced-generalist)
- **Config**: `configs/p3/balanced-generalist-pilot.yaml`
- **Mode**: continual (initial_adapter = P2 stage3-repair-v3)
- **Steps**: 20/20 (0.25 epoch, well within 50-step cap)
- **Duration**: 146s (2.4 min)
- **Train loss**: 0.8375 (smoke) → 0.4041 (final)
- **Eval loss**: 0.5935
- **Peak GPU**: 1350 MiB / 4096 MiB
- **Token audit**: 622 samples, 0 assistant lost, 0 target too long
- **Adapter save/reload**: VERIFIED OK
- **Parent adapter (P2 final)**: intact
- **Output**: `adapters/p3/balanced-generalist-pilot/`

**Pilot 结论**: 训练管道端到端可运行（数据加载→loss 下降→checkpoint 保存/reload）。
Pilot 结果不可作为模型能力声明（GO_FOR_P3_PILOT_ONLY 约束）。

## Session End

P3.0–P3.4 scope complete. Issue #12 Phase D Pilot completed (balanced-generalist, 0.25 epoch).
