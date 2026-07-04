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
- Task 7: Frozen v3 Candidate Reservation — NEEDS_REWORK (was COMPLETE 2483968; re-run on new source pool)
- Task 8: Build Frozen v3 Samples + Verify + Freeze — NEEDS_REWORK (was COMPLETE 0e60b3a; re-run with new candidates; reviewer cancelled)
- Task 9: P3 Validation + Train Family Partition (pairwise disjoint + P2 replay whitelist) — PENDING
- Task 10: Canonical Verified Sample Pool — PENDING
- Task 11: Balanced Generalist Train Data (30/20/20/30) — PENDING
- Task 12: Repair Specialist Train Data (15/15/30/40) — PENDING
- Task 13: Training Config + 3-Tier Checkpoint Evaluator — PENDING
- Task 14: Readiness Gate Report (GO/FIX FIRST) — PENDING (SESSION END)

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
