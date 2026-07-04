# Task F4 Report — Update Readiness Report Docs (Issue #10)

**Status**: DONE
**Branch**: feat/p3-capability-expansion-v2
**Commit SHA**: `891b30f`
**Date**: 2026-07-05

## Goal

修复 `scripts/p3_readiness_gate.py` 中 `render_report()` 函数的过时 documentation，
使其反映 Fix 1（回填 501 条 P2-replay 样本）与 Fix 4（统一 metrics schema）后的实际状态。

## Changes

### 1. 新增 helper 函数 `_count_variant_types(path)`（line 119-139）

读取 JSONL 文件并按 `variant_type` 字段统计每个桶的样本数。文件不存在时返回空 dict。
为 B.2.6 提供动态数据来源。

### 2. 重写 B.1 section（"B.1 verified=True Normalization Deviation"）

拆分为两个子节：
- **B.1.1 历史背景**：保留 Task 11/12 的 438/626、419/493、857/1119 偏差叙述，
  明确标注为 "历史偏差计数（已被 Fix 1 取代，仅作追溯）"。
- **B.1.2 Fix 1 修复结果**：新增子节，描述：
  - 回填脚本 `scripts/backfill_canonical_pool_verification.py` 已对 501 条样本
    实际运行 `pad_hidden_tests + verify_sample`
  - 回填揭露 boundary 变体生成器 bug（125/125 boundary 样本实际 `verified=False`）
  - Balanced Generalist train 从 626 → 501（-125 boundary）
  - Repair Specialist train 从 493 → 416（-77，含 boundary 失败与重平衡）
  - 当前 0 条 `verified=True && verification all-False` 不一致样本
  - build scripts 中的 `model_copy(update={"verified": True})` hack 已移除
  - 风险评级从"中"降为"低"

### 3. 重写 B.2.4 section（"Baseline key 映射"）

将旧的 "check_hard_constraint 已在内部做映射" 改为 Fix 4 描述：
- `src/metrics.py` 新增 `METRICS_SCHEMA_VERSION = "1.0.0"` 常量
- `BASELINE_TO_METRICS_KEY_MAP = {"codegen_pass1": "pass_at_1"}` 字典
- `normalize_baseline_key(baseline)` 函数
- baseline lock JSON 顶层已新增 `schema_version: "1.0.0"` 字段
- `check_hard_constraint` 现使用 `normalize_baseline_key` 后比较 `pass_at_1`
- `schema_version` mismatch 时记录 warning 但不 FAIL（前向兼容）

### 4. 重写 B.2.6 section（"Probe 样本 variant_type 分布"）

- 改为使用 `_count_variant_types(BALANCED_TRAIN_PATH)` 与
  `_count_variant_types(REPAIR_TRAIN_PATH)` 动态计算每个桶的样本数
- 输出格式：`code=N, boundary=N, static_repair=N, execution_repair=N (total=N)`
- 当某桶样本数 < 19 时记录 warning（informational），但 PASS 判据不变
- 处置说明保持 `min(target, len(pool))` 行为可接受

## Verification

### Verdict（保持不变）

```
Verdict: GO_FOR_P3_PILOT_ONLY
```

11 项检查结果不变，verdict_impact=PILOT_ONLY（train 容量 917 < 2300）。
**未改动 verdict 计算逻辑**。

### B.1 当前数据（节选）

```
历史偏差计数（已被 Fix 1 取代，仅作追溯）：
- Task 11 (balanced-generalist)：438/626 train 样本曾被归一化为 verified=True。
- Task 12 (repair-specialist)：419/493 train 样本曾被归一化为 verified=True。

Fix 1 修复结果：
- Balanced Generalist train.jsonl：626 → 501 样本（-125 boundary 全失败移除）
- Repair Specialist train.jsonl：493 → 416 样本（-77，含 boundary 失败与重平衡）
- 当前 0 条样本处于 verified=True && verification all-False 自相矛盾状态
```

### B.2.6 动态计数（运行时实际值）

```
balanced-generalist train.jsonl: code=188, boundary=0, static_repair=125, execution_repair=188 (total=501).
repair-specialist train.jsonl: code=74, boundary=0, static_repair=145, execution_repair=197 (total=416).
Warnings（informational，不影响 PASS 判据）：
- balanced-generalist `boundary` bucket has 0 < 19 samples
- repair-specialist `boundary` bucket has 0 < 19 samples
```

boundary 桶为 0 是预期的（Fix 1 已移除所有失败 boundary 样本）。

## 测试结果

```
tests/test_p3_readiness_gate.py .....................                    [ 44%]
tests/test_p3_checkpoint_evaluator.py ..........................         [100%]
47 passed, 1 warning in 16.13s
```

1 warning 为 pre-existing `PytestUnknownMarkWarning: Unknown pytest.mark.slow`，
与本任务无关。

## 修改的文件列表

- `scripts/p3_readiness_gate.py` — 新增 helper + 重写 B.1/B.2.4/B.2.6
- `reports/p3/p3-training-readiness-report.md` — 重新生成的 report

## Constraints Honored

- ✅ 未改动 verdict 计算逻辑（仅改 documentation）
- ✅ 未破坏现有测试（47/47 passed）
- ✅ 未启用训练（NO training launched）
- ✅ Python 3.8.10 兼容（使用 `from __future__ import annotations`，dict 类型注解用字符串引用）
- ✅ B.1 历史背景保留（明确标注为 "历史背景"）；当前状态用 "Fix 1 修复结果" 子节描述
