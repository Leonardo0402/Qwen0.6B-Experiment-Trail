# Frozen Eval Set (v1)

本目录为 P1 阶段冻结评测集，**一经冻结不得修改**。

## 摘要

- 版本: `v1`
- 样本数: **51**
- 独立 family 数: **9**
- SHA256 (`test_raw.jsonl`): `91c801da58c1c886c1b863fe1c832ee65444f316b7294941be25dafd27421697`
- 冻结时间 (UTC): 2026-06-27T14:50:25.768847+00:00

## task_type 分布

| task_type | 数量 |
|---|---|
| code_generation | 9 |
| static_repair | 21 |
| execution_repair | 21 |

## difficulty 分布

| difficulty | 数量 |
|---|---|
| 0 | 13 |
| 1 | 7 |
| 2 | 10 |
| 3 | 21 |

## family 列表

- `fam_l0_abs_value` (7 条)
- `fam_l0_is_even` (5 条)
- `fam_l0_is_positive` (5 条)
- `fam_l0_multiply` (5 条)
- `fam_l1_count_char` (7 条)
- `fam_l1_zip_sum` (5 条)
- `fam_l2_find_duplicates` (7 条)
- `fam_l2_merge_sorted` (5 条)
- `fam_l2_rotate_list` (5 条)

## 数据来源

- `data/splits/test_raw.jsonl` — 原 36 条冻结测试样本 (6 family)，逐行原样保留以保证向后兼容。
- `data/verified/code_gen.jsonl` — code_generation 样本池。
- `data/verified/repairs.jsonl` — static_repair / execution_repair 样本池。

## 防泄漏保证

- **严禁接入外部数据集**；仅使用项目现有数据池。
- **严禁** family_id 同时出现在训练集与冻结评测集。训练 family 通过`data/curriculum/*/families.json`（如存在）与 `data/splits/train.jsonl` 的 target_code 反查联合检测并排除。
- 已排除的训练 family 数: 20

## 冻结约束

- **禁止**后续训练使用本目录任何样本。
- **禁止**根据模型在本集上的表现修改题目、target_code 或测试用例。
- 如需扩展或修订，**必须创建新版本目录**（如 `v2`），不得原地修改 `v1`。

## 目标达成情况

- P1 目标: >= 12 个独立 family_id
- 实际: 9 个 family_id
- **未达标**: 数据池仅含 29 个 family，其中 20 个已进入训练，可用的未训练 family 不足 12。如实报告，未编造任何 family。

  - test_raw 原有 family: 6
  - 新增未训练 family: 3
  - 合计: 9
