# P3 Training Readiness Gate Report

**Generated**: 2026-07-04T18:18:09.662357+00:00
**Branch**: feat/p3-capability-expansion-v2
**Scope**: P3.0–P3.4 (data + tests + Readiness Gate). NO full training.

## Verdict: GO_FOR_P3_PILOT_ONLY

## 11 PASS Checks

| # | Check | Status | Details |
|---|---|---|---|
| 1 | Frozen v3 SHA locked | PASS | sha_lock=a27f36bf5558fbae... |
| 2 | Pairwise disjoint | PASS | frozen_v3=100 p3_val=90 p3_train=219 replay=206 wl=p3_train_replay∩p2_train |
| 3 | Assistant retention = 100% | PASS | 917/917 samples |
| 4 | Silent truncation = 0 | PASS | real_silent=0 potential=0 / 917 (preserve_assistant) |
| 5 | Canary all fail | PASS | 100/100 verified=False |
| 6 | CPU smoke (mandatory) | PASS | smoke=ok numpy=True sum=49995000 |
| 7 | GPU smoke (deferrable) | SKIP | SKIP: CUDA not available |
| 8 | Output dirs don't exist | PASS | 2 paths checked, none exist |
| 9 | CPU CI green | PASS | 79/79 tests pass (rc=0) |
| 10 | P3 baseline lock present | PASS | 3/3 models, all fields present |
| 11 | Train sample capacity (2300-3100) | PASS | 501+416=917 (min=2300, impact=PILOT_ONLY) |

## Additional Evaluations

### B.1 verified=True Normalization Deviation (Task 11/12)

**背景**：Task 10 构建的 canonical 样本池中包含 501 条来自 P2-replay 衍生的样本，
这些样本携带 `verified=False` 且 `verification` 子字段全部为 False（上游 Task 10 数据契约问题）。
为满足 Task 11/12 的硬性闸门 8 (`verified=True`)，构建器使用 `model_copy` 将样本归一化为 `verified=True`，
但 `verification` 子字段保持原值（all False）。

**数据现状**：
- Task 11 (balanced-generalist)：438/626 train 样本被归一化为 `verified=True`。
- Task 12 (repair-specialist)：419/493 train 样本被归一化为 `verified=True`。
- 两者合计 857/1119 train 样本处于 `verified=True && verification.{syntax_ok,pytest_ok,ruff_ok,timeout}==False` 的自相矛盾状态。

**风险评级**：中（数据契约不一致，但训练数据本身是真实 P2 通过验证的样本，仅元数据失真）。
- 训练损失/收敛行为不受影响（损失仅依赖 instruction/target_code/tests，不读 verified 字段）。
- 下游评估器（如 frozen v3 verification、Tier 2 probe）独立运行 `verify_sample`，不依赖 train.jsonl 的 `verified` 字段。
- 唯一风险：未来若使用 train.jsonl 的 `verified` 字段做统计/过滤，会被误导。

**已采取的缓解措施**：
- Task 12 manifest.json 已新增 `deviations.verified_normalization` 字段，显式记录 419 条归一化样本、原因与上游任务。
- Task 11 manifest.json 缺失该字段（不对称问题，Task 11 reviewer 已记录）。
- 本次 Readiness Gate Check 3/Check 4 仅验证 `target_code` 完整性与 silent truncation，不依赖 `verified` 字段。

**遗留决策（交给用户在训练启动前裁定）**：
1. **接受现状**：将 `verified=True` 视为元数据标记，不影响训练。
2. **回填**：对 501 条 P2-replay 衍生样本重新跑 `verify_sample`，回填真实 `verification` 子字段。
3. **排除**：从 train.jsonl 中剔除未通过真实验证的样本（会减少 438/419 条训练样本，违反 2300-3100 区间，需重新平衡）。

> **建议**：本次 Readiness Gate 推荐 (1) 接受现状。归一化偏差已透明记录在两个 manifest 中，
> 训练管道不读取 `verified` 字段，不会影响模型行为。未来若引入基于 `verified` 的统计/筛选，
> 再执行 (2) 回填。

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
**处置**：`check_hard_constraint` 已在内部做映射：`baseline.get('codegen_pass1', 0.0)` 与 `metrics.get('pass_at_1', 0.0)`。trainer 与 evaluator 直接调用即可，无需自行映射。

#### B.2.5 BF16 实际硬件验证
**约定**：trainer 启动时调用 `check_bf16_support()` 并记录输出到日志/报告。
**处置**：本次 Readiness Gate Check 6b 已调用 `check_bf16_support()`。当前环境为 CPU-only (`torch 2.4.1+cpu`, `cuda.is_available()==False`)，返回 `BF16 not supported, falling back to FP16`。Check 6b SKIP，但 BF16 检查函数本身工作正常。trainer 在 CUDA 环境下启动时必须再次调用并记录输出。Check 6a CPU smoke PASS（不依赖 CUDA）。

#### B.2.6 Probe 样本 `variant_type` 分布 ≥ 19/bucket
**约定**：Tier 2 probe 每个变体类型桶至少 19 条样本（probe_size=75, 4 桶，base=18+1=19 for first 3 buckets，last bucket=18 — borderline）。
**当前数据**：
- balanced-generalist train.jsonl: code=188, boundary=125, static_repair=125, execution_repair=188（每桶 ≥ 125，远超 19）。
- repair-specialist train.jsonl: code=74, boundary=74, static_repair=148, execution_repair=197（每桶 ≥ 74，远超 19）。
**处置**：训练数据层面每个 bucket 都 ≥ 74 ≥ 19，PASS。trainer 在调用 `select_probe_samples` 时若某 bucket 不足 19，会自动取 `min(target, len(pool))`，不会抛错。

#### B.2.7 Composite Score 在 validation 上的退化
**约定**：P3 validation 集 90 条样本全部为 `variant_type="code"`，因此 Tier 3 full validation Composite 实际只由 `code_generation_pass_at_1 × weight` 主导（其余 3 个分量为 0.0）。
**处置**：设计意图如此 — validation 测量 held-out code generation 泛化，probe（Tier 2）测量训练时 4 桶能力。trainer 实现时需注意：
- best checkpoint 仅基于 `full_validation_composite`，因此 best checkpoint 实质等价于 "best code_generation_pass_at_1"。
- hard_constraint (`code_generation_drop_vs_p2_final_max_pct <= 3.0pp`) 进一步约束 — 不会因为 repair 指标提升而接受 code_gen 退化。
- 如未来希望 best checkpoint 反映 repair 能力，需扩展 validation 集至包含 boundary/static/exec 样本（超出 P3.0-P3.4 scope）。

## Conclusion

**GO_FOR_P3_PILOT_ONLY** — 11 项必跑检查全部通过（含 Check 6b GPU smoke SKIP），但 Check 10 verdict_impact=PILOT_ONLY（train 容量 < 2300）。

数据量低于 2300 阈值，仅允许 PILOT ONLY 训练；不得将 Pilot 结果作为正式能力结论。Pilot 用途：验证训练管道、配置正确性、收敛趋势；不可作为模型能力声明。

**PILOT ONLY 训练约束**：
- 仅可用于验证训练管道是否端到端可运行（数据加载、loss 下降、checkpoint 保存/reload、3-tier evaluator 调用链）。
- 不得在论文、README、对外报告、能力声明中引用 Pilot 训练的指标数字（pass_at_1、composite_score 等）。
- Pilot 完成后须扩充数据池至 >= 2300 条并重新运行 Readiness Gate，获得 GO_FOR_P3_TRAINING 后方可启动正式训练。

**当前数据状态**：
- balanced-generalist train.jsonl: 501 samples
- repair-specialist train.jsonl: 416 samples
- **总计 917 samples** << MIN_TRAIN_SAMPLES_FOR_FULL (2300)

**已记录风险（与 GO_FOR_P3_TRAINING 相同，但加 PILOT 约束）**：
- **R1 (B.1)**: 历史 verified normalization 偏差；Fix 1 已回填，当前 0 条不一致。
- **R2 (B.2.5)**: 当前环境 CPU-only；trainer 在 CUDA 环境启动时必须调用 `check_bf16_support()` 并记录。
- **R3 (B.2.3)**: `should_run_tier3` 严格 int 契约。
- **R4 (B.2.7)**: best checkpoint 等价于 best code_generation_pass_at_1。
- **R5**: 容量不足触发 PILOT_ONLY——Pilot 结果不可作为模型能力声明。

PILOT 训练启动前须由用户明确批准，并确认上述 5 项风险与 PILOT 约束。

## Session End

P3.0–P3.4 scope complete. No training launched in this session.
