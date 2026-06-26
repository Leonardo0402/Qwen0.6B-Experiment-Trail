# Qwen3-0.6B Python Code Recovery Lab

在 RTX 3050 Laptop 4GB（实际可用 ~3GB）上，用 LoRA + 课程学习强化 Qwen3-0.6B 的
**Python 代码生成 / 静态 Bug 修复 / 执行反馈修复**能力的训练与评测实验平台。

> 配套文档：`../docx/Qwen3-0.6B_Code_Training_Development_Spec.md`（开发规范，已适配本机）
> 实施计划：`../docx/Qwen3-0.6B_Implementation_Plan.md`（M0~M5）

## 环境

- Conda env：`qwen3-code-lab`（python=3.10），独立于本机现有环境
- PyTorch：cu124 为主、cu121 备用（驱动 566.36 / CUDA 12.7）
- HuggingFace 缓存：重定向到 `./.hf_cache`（避免占用 C: 系统盘）

## 核心原则

Claude 负责生成与组织数据，**编译器 / pytest / 静态检查器负责判定数据是否合格**。
样本合格只认 `py_compile` / `pytest` / `ruff` / `mypy` / Hypothesis，不认语言判断。

## 目录

见 spec §4。`scripts/` 命令行入口，`src/` 复用模块，`tests/` 单元测试，
`data/` 数据流水线产物，`adapters/` 各阶段 LoRA，`evaluations/` 评测与报告。

## 快速开始

```powershell
conda activate qwen3-code-lab
python scripts/check_environment.py
```
