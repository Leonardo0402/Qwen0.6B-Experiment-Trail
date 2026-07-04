# P3 Capability Expansion v2 — SDD Progress Ledger

Branch: feat/p3-capability-expansion-v2
Started: 2026-07-04
Plan: .superpowers/sdd/p3-plan.md
Scope: P3.0–P3.4 (data + tests + Readiness Gate). NO full training.

## Tasks

- Task 1: Lock Historical Baseline — COMPLETE (commit 48614af, review APPROVED, 26/26 tests pass) [v2.1: compliant, no rework]
- Task 2: Extend Sample Schema — COMPLETE (commit 5b88a6e, review APPROVED, 48/48 tests pass) [v2.1: compliant, no rework]
- Task 3: Fix import_mbpp.py + New verify_imported_mbpp.py — COMPLETE (commit 621be8d, pad-then-verify implemented, 33 tests pass) [v2.1 A2 compliant]
- Task 4: Run Import + Verify + Source Audit — COMPLETE (commit c9bb78a; pad-then-verify re-run: 714/964 verified, 433 new families, LIKELY_FEASIBLE) [v2.1 A2 compliant]
- Task 5: Cross-Split Semantic Dedup Audit — COMPLETE (v2.1 redo; 714 samples, 50 quarantined families, unresolved=0, PASS)
- Task 6: Build Family Registry — COMPLETE (v2.1 redo; 807 families, 374 P2, 50 quarantine, 409 new-available)
- Task 7: Frozen v3 Candidate Reservation — COMPLETE (v2.1 redo; source_pool=348, 120 candidates reserved, 10/10 tests pass)
- Task 8: Build Frozen v3 Samples + Verify + Freeze — COMPLETE (v2.1 redo; 120/120 qualified, freeze_100, 100 frozen, 300 samples, 12/12 tests pass, sha_lock verified) [review APPROVED_WITH_NOTES: stale Concerns #2/#3 removed, Deviation #1 updated for shared module extraction; all binding reqs PASS]
- Task 9: P3 Validation + Train Family Partition (pairwise disjoint + P2 replay whitelist) — COMPLETE (commit 697a585; validation=90, train_new=219, train_replay=206, total=425; 11/11 tests pass; pairwise disjoint PASS; review APPROVED_WITH_NOTES) [brief note: p2_train∩quarantine=18 not 26, replay=206 not 198; implementer correctly used dynamic count; downstream tasks use actual values 206/425]
- Task 10: Canonical Verified Sample Pool — COMPLETE (commit c13dd02; 782 samples, 408/425 families; code=281/boundary=125/static=148/exec=228; 274 dupes removed, 7 capped; 16/16 tests pass; all hard gates PASS; review APPROVED_WITH_NOTES) [pool yield 782 < 2300-3100 but non-blocker per A7; Task 11 max 625 samples, Task 12 max 493 samples via sub-sampling]
- Task 11: Balanced Generalist Train Data (30/20/20/30) — COMPLETE (commit 120fc98; 626 train + 90 val; 10/10 tests pass; all hard gates pass; review APPROVED_WITH_NOTES) [deviation: 438/626 train samples normalized verified=True via model_copy; upstream Task 10 issue — P2 replay variants have verification subfields all False; MUST be evaluated at Task 14 Readiness Gate]
- Task 12: Repair Specialist Train Data (15/15/30/40) — COMPLETE (commit 59a128c; 493 train + 90 val; 11/11 tests pass; all hard gates pass; review APPROVED — no notes) [deviations.verified_normalization field added to manifest.json (419 samples normalized, addresses Task 11 review Concern #2); validation SHA byte-identical to Task 11 (Global Constraint #18 confirmed); token means rounded to 4 decimals]
- Task 13: Training Config + 3-Tier Checkpoint Evaluator — COMPLETE (commit eb475d5; 2 YAMLs + evaluator module + 17/17 tests pass; all 12 hard gates pass; review APPROVED_WITH_NOTES) [concerns: compute_composite signature deviates (justified), early_stop confirm semantics loose, should_run_tier3 strict int, baseline key mapping codegen_pass1↔pass_at_1; 7 recommendations for Task 14 documented]
- Task 14: Readiness Gate Report — COMPLETE (commit 86d879b; verdict GO_FOR_P3_TRAINING; 9/9 checks PASS [8 PASS + 1 SKIP]; 11/11 tests pass; report at reports/p3/p3-training-readiness-report.md) [documented: 857/1119 train samples verified=True normalization (Task 11:438 + Task 12:419); 5 risks R1-R5; 7 Task 13 recommendations documented; SESSION END — no training launched]

## Issue #10 Fixes (6 items, 4 commits)

- **Fix 1**: verified 数据一致性回填 — COMPLETE (commit 2003abd; 35 tests pass; backfill_canonical_pool_verification.py 新建；canonical-pool.jsonl 中 501 P2-replay 样本回填真实 verification 子字段；build scripts 移除 model_copy hack；揭露 boundary 变体 125/125 全失败；Balanced train 626→501, Repair train 493→416, total 917 << 2300 阈值)
- **Fix 3+4**: Composite Score check_early_stop 收紧 + Metrics Schema 统一 — COMPLETE (commit 462c8d8; 26+73 tests pass; trigger 2 现要求 full_history>=2 且比较连续下降；METRICS_SCHEMA_VERSION=1.0.0 + normalize_baseline_key() 函数；baseline lock 添加 schema_version 字段)
- **Fix 2+5+6**: Readiness Gate 综合修复 — COMPLETE (commit 263978f; 21+47 tests pass; Check 6 拆为 6a CPU smoke (必跑) + 6b GPU smoke (可 SKIP)；新增 Check 10 train capacity vs 2300-3100；compute_verdict 三态: GO_FOR_P3_TRAINING / GO_FOR_P3_PILOT_ONLY / FIX_FIRST)
- **Fix docs**: Report documentation 一致性更新 — COMPLETE (commit 891b30f; 47 tests pass; B.1 加 Fix 1 修复结果子节；B.2.4 描述 normalize_baseline_key；B.2.6 动态计算 variant_type 分布)

**最终 Readiness Gate 输出**: Verdict = `GO_FOR_P3_PILOT_ONLY`（11 项检查全 PASS，含 6b SKIP；但 Check 10 verdict_impact=PILOT_ONLY 因 train 总量 917 < 2300）。**未启动训练。**

**关键发现**: Fix 1 回填揭露了 boundary 变体生成器的固有 bug — 125/125 boundary 样本的 target_code 在边界输入（0/-1/1）下返回错误值，全部 verified=False。这意味着 boundary 变体生成器本身需要后续修复（超出 Issue #10 scope）。

## Minor Findings (triaged by final review)

- Task 1 #1: review-diff.txt 为空（19 字节），git diff 重定向问题；reviewer 直接审阅实际文件绕过。后续 task 改用 `git show` 或分步生成 diff 文件。
- Task 1 #2: codegen_pass1 字段名 vs router-analysis.json 的 code_generation_pass（brief 已指定 codegen_pass1，符合）
- Task 1 #3: per-task-type 指标来自 234 样本 eval subset（非全 576），lock 中已用 eval_subset_sample_count/source_file 自证来源
- Task 1 #4: created_at 为固定时间戳（确定性 lock，合理）
- Task 3 #1 (M1): verify_imported_mbpp.py docstring 退出码描述不一致（line 8-10，1 行修复，final review triage）
- Task 3 #2 (M2): rejected 样本 verification 字段为 all-false preset（硬检查先于 verify_sample，合理性能优化，非阻塞）
- Task 3 #3 (M3): test_no_pytest_in_importer 无法捕获 `import src.validators as sv` 写法（实际防御足够）
- Task 3 #4 (M4): 失败标记启发式偏保守（"fail"/"error" 子串匹配，安全方向，可接受）
- Task 4 #1: hidden>=3 硬性检查导致全部 964 样本被拒（plan 缺陷，user-approved fix: 改为 warning，955/964 通过）
- Task 4 #2: 所有 955 样本 hidden_count=1（MBPP 原始数据限制），Task 8 需生成边界测试使 hidden>=3
- Task 5 #1 (M1): `unresolved_count=0` literal hard-coded (consistent with brief binding rule #3, conclusion computed from variable, transparent in report)
- Task 5 #2 (M2): `func_signature` 仅提取 positional args，不含 *args/**kwargs/kwonly/posonly（brief "name + arg names" 语义模糊，implementer 解释合理，high-sim 启发式可接受）
- Task 5 #3 (M3): n-gram bucketing 按 instruction_hash 前 2 hex 字符分组可能产生 false negative（brief 强制要求此 bucketing 策略，非 implementer 偏差）
- Task 5 #4 (M4): 测试运行时 Python 3.8.10 vs pyproject `>=3.10,<3.11`（from __future__ annotations 兼容，10/10 通过，非阻塞）
- Task 6 #1 (M1): `to_path` 不保证 byte-identical round-trip（generated_at 每次保存重算时间戳；families dict round-trip 完美；非阻塞）
- Task 6 #2 (M2): invariant #3 计算 actual_new 但未与任何值断言（按构造为恒真，仅缺自文档化）
- Task 6 #3 (M3): 报告中说 30 个 overlap 全是 p2_train，实际 20 p2_train + 6 p2_validation + 4 p2_frozen_v2（算术结论 30→556 不变，仅 prose 不精确）
- Task 6 #4 (M4): `FamilyRegistry.is_used(unknown_id)` raises KeyError（brief API sketch 未指定，合理设计选择）
- Task 6 #5 (M5): `backfill_quarantine` 防御性创建 stub entry 用于未在 registry 的 quarantine family（实际运行未触发，所有 58 个已在 registry）
- Task 7 #1 (M1): source pool filter 省略条件 #4 (frozen_v3_candidate NOT in usage) — brief 自相矛盾，implementer 遵循 idempotency section PREFERRED 方案（不排除已 claimed family），首次运行输出与字面 spec 一致
- Task 7 #2 (M2): registry 写入非原子（先写 registry 后跑 post-claim assertion 7-8；pre-claim assertion 1-6 已验证候选有效，实际运行成功，风险低）
- Task 7 #3 (M3): post-claim assertion #7 仅检查 set 相等，未验证每个候选 usage 列表恰好为 ["frozen_v3_candidate"]（defense-in-depth，非 exploitable）

## Notes

- Audit report: .superpowers/sdd/mbpp-family-audit.md (P2 family usage, confirms train split exhausted)
- PR #8 (router holdout fix) merged to main at 42a489c before this branch started
- All 14 tasks must complete before Readiness Gate verdict
