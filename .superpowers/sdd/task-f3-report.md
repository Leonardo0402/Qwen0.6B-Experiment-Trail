# Task F3 Report: Readiness Gate 综合 Fix (2+5+6)

## 状态

**DONE**

## 修改的文件

| 文件 | 改动 |
|------|------|
| `scripts/p3_readiness_gate.py` | 拆分 check6 → check6a/6b；新增 check10_train_capacity；compute_verdict 三态；render_report PILOT_ONLY 分支；CHECK_NAMES 11 项 |
| `tests/test_p3_readiness_gate.py` | 新增 10 个 Fix 2/5/6 测试；更新 check3/check4 容量断言（1119 → 917，Fix 1 影响） |
| `reports/p3/p3-training-readiness-report.md` | 重新生成；verdict 由 GO_FOR_P3_TRAINING 改为 GO_FOR_P3_PILOT_ONLY |

## 测试结果

- `tests/test_p3_readiness_gate.py`: **21/21 PASS**（17s）
- `tests/test_p3_checkpoint_evaluator.py + test_p3_readiness_gate.py` 联合回归: **47/47 PASS**（17.5s）

新增测试覆盖：
- `test_check6a_cpu_smoke_pass_when_python_healthy`
- `test_check6a_cpu_smoke_fail_on_exception`（mock builtins.sum 抛异常）
- `test_check6a_skip_not_allowed`
- `test_check6b_gpu_smoke_skip_when_no_cuda`
- `test_check6b_gpu_smoke_pass_when_cuda_available_and_smoke_ok`（mock torch.cuda）
- `test_check10_capacity_pass_when_above_min`（tmp_path，total=2400）
- `test_check10_capacity_warn_when_below_min`（tmp_path，total=900）
- `test_check10_capacity_fail_when_zero`（tmp_path，total=0）
- `test_compute_verdict_full_when_no_fail_no_capacity_warning`
- `test_compute_verdict_pilot_when_capacity_warning`
- `test_compute_verdict_fix_first_when_any_fail`

## 实际 Gate 运行结果

```
==============================================================================
P3 Readiness Gate — 11 PASS Checks (SESSION END)
==============================================================================

[1/11]  Frozen v3 SHA locked            -> PASS  sha_lock=a27f36bf5558fbae...
[2/11]  Pairwise disjoint                -> PASS  frozen_v3=100 p3_val=90 p3_train=219 replay=206
[3/11]  Assistant retention = 100%      -> PASS  917/917 samples
[4/11]  Silent truncation = 0           -> PASS  real_silent=0 potential=0 / 917
[5/11]  Canary all fail                 -> PASS  100/100 verified=False
[6/11]  CPU smoke (mandatory)            -> PASS  smoke=ok numpy=True sum=49995000
[7/11]  GPU smoke (deferrable)           -> SKIP  CUDA not available
[8/11]  Output dirs don't exist          -> PASS  2 paths checked, none exist
[9/11]  CPU CI green                     -> PASS  79/79 tests pass (rc=0)
[10/11] P3 baseline lock present        -> PASS  3/3 models, all fields present
[11/11] Train sample capacity (2300-3100) -> PASS 501+416=917 (min=2300, impact=PILOT_ONLY)

Verdict: GO_FOR_P3_PILOT_ONLY
```

退出码: 0（PILOT_ONLY 允许进入下一步）

## Commit SHA

`263978fae9676263e0e2105a8ab74bff74081bff`

## 与任务约束对照

| 约束 | 满足 |
|------|------|
| Verdict 必须基于实际数据计算（非硬编码） | ✓ compute_verdict 从 check10 details.verdict_impact 推导 |
| Verdict 必须是 GO_FOR_P3_PILOT_ONLY（917 < 2300） | ✓ |
| Check 10 应显示 PILOT_ONLY impact | ✓ "501+416=917 (min=2300, impact=PILOT_ONLY)" |
| Check 6a 应 PASS（CPU smoke） | ✓ smoke=ok numpy=True sum=49995000 |
| Check 6b 应 SKIP（CPU-only 环境） | ✓ SKIP: CUDA not available |
| 所有原 PASS 的 check 应继续 PASS | ✓ Check 1/2/3/4/5/7/8/9 全部 PASS |
| SKIP 必须含 skipped:True 与 reason 字段 | ✓ 6b details = {skipped:True, reason:"CUDA not available", torch_version:...} |
| Python 3.8.10 兼容（from __future__ import annotations） | ✓ |
| 不修改 src/p3_checkpoint_evaluator.py | ✓ |
| 不修改 build scripts 或 canonical-pool.jsonl | ✓ |
| 不启用训练 | ✓ 未触发任何训练调用 |

## Concerns

1. **B.2.6 / B.1 中的历史数据未更新**（已知，超出本 Fix 范围）：
   - B.2.6 仍显示 `balanced: code=188, boundary=125, static_repair=125, execution_repair=188`，但 Fix 1 已移除 boundary 样本，当前实际是 `code=188, execution_repair=188, static_repair=125`（无 boundary，总 501）。
   - B.1 仍显示历史 `438/626, 419/493` 计数（Task 11/12 时期），未补记 Fix 1 回填后的状态（当前 0 条 verified 不一致）。
   - 这两处在 `render_report()` 的硬编码字符串里，不在 Fix 2/5/6 任务范围内。建议后续 Fix 7 单独清理这两处历史文档。

2. **PILOT_ONLY 训练约束已显式写入报告 Conclusion section**，包含：
   - 不得在论文/README/对外报告/能力声明中引用 Pilot 指标
   - Pilot 完成后须扩充数据池至 >= 2300 并重跑 Gate 获得 GO_FOR_P3_TRAINING 后方可正式训练
   - 当前数据状态 (501/416/917 << 2300) 已显式列出

3. **测试中 `test_check6b_gpu_smoke_pass_when_cuda_available_and_smoke_ok`** 使用 `unittest.mock` 模拟 torch.cuda.is_available()=True，未真实跑 GPU forward/backward。这是 mock 测试的固有限制；真实 GPU smoke 在 CUDA 环境首次运行时仍需手动验证（与原 check6_gpu_smoke 一致）。
