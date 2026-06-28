# Qwen3-Code-Lab 训练问题日志

**项目**: Qwen3-0.6B LoRA 代码生成微调  
**GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)  
**记录时间**: 2026-06-27  
**状态**: 持续更新中

---

## 问题汇总

| # | 问题类型 | 严重程度 | 状态 | 影响阶段 |
|---|---------|---------|------|---------|
| 1 | Qwen3 Thinking 模式 | 高 | ✅ 已解决 | 推理/评测 |
| 2 | 旧 Adapter 配置不匹配 | 高 | ✅ 已解决 | 训练启动 |
| 3 | CUDA 内存访问违规 | 高 | ⚠️ 部分解决 | Boundary 训练 |
| 4 | Python 语法错误 | 低 | ✅ 已解决 | 数据生成 |
| 5 | CLI 参数错误 | 中 | ✅ 已解决 | 训练恢复 |

---

## 详细问题记录

### 问题 #1: Qwen3 Thinking 模式导致输出错误

**发现时间**: 2026-06-27  
**严重程度**: 高  
**状态**: ✅ 已解决  
**影响范围**: 模型推理和评测阶段

#### 问题描述
模型在推理时输出 `<think>...</think>` 标签内的思考过程文本，而非预期的代码生成结果。

#### 根本原因
Qwen3 模型默认启用 thinking 模式，会在输出中包含推理过程。

#### 解决方案
在 `scripts/evaluate_model.py` 中进行以下修改：

1. 在 `apply_chat_template` 调用时添加 `enable_thinking=False` 参数
2. 在 prompt 末尾追加 `/no_think\n` 指令

```python
# 修改前
text = tokenizer.apply_chat_template(messages, tokenize=False)

# 修改后
text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,
    enable_thinking=False
)
text += "/no_think\n"
```

#### 验证结果
- ✅ 模型正确输出代码，不再包含思考过程
- ✅ Pass@1 评测结果正常

---

### 问题 #2: 旧 Adapter 配置与新训练配置不匹配

**发现时间**: 2026-06-27  
**严重程度**: 高  
**状态**: ✅ 已解决  
**影响范围**: Boundary 训练启动

#### 问题描述
训练 Boundary 阶段时崩溃，错误信息显示 LoRA adapter 配置冲突。

#### 错误信息
```
RuntimeError: LoRA adapter configuration mismatch:
- Existing adapter: rank=8, target_modules=[q_proj, v_proj]
- New config: rank=32, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

#### 根本原因
输出目录 `adapters/code-lora-v3-boundary` 中已存在旧的 adapter 文件（来自早期实验），其配置（rank=8, 仅 2 个模块）与新的训练配置（rank=32, 7 个模块）不兼容。

#### 解决方案
1. 删除旧的 adapter 目录：
   ```powershell
   Remove-Item -Recurse -Force adapters\code-lora-v3-boundary
   ```

2. 重新创建空目录并放置新的配置文件

#### 预防措施
- 每次训练前检查输出目录是否为空
- 使用版本号区分不同实验（v1, v2, v3）

---

### 问题 #3: CUDA 内存访问违规 (ACCESS_VIOLATION)

**发现时间**: 2026-06-27  
**严重程度**: 高  
**状态**: ⚠️ 部分解决（Easy 阶段通过，Boundary 阶段仍不稳定）  
**影响范围**: Boundary 阶段训练

#### 问题描述
Boundary 阶段训练在运行约 71% 进度时崩溃，出现 Windows CUDA 内存访问违规错误。

#### 错误信息
```
Windows Error: ACCESS_VIOLATION
CUDA error: an illegal memory access was encountered
```

#### 尝试的解决方案

##### 尝试 1: 清理 GPU 内存
```python
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
```
**结果**: ❌ 无效

##### 尝试 2: 降低 LoRA rank
```yaml
# 从 rank=32 降低到 rank=16
lora:
  rank: 16
  alpha: 32
```
**结果**: ❌ 仍然崩溃

##### 尝试 3: 减少 target_modules
```yaml
# 仅保留 q_proj 和 v_proj
target_modules:
  - q_proj
  - v_proj
```
**结果**: ❌ 仍然崩溃

##### 尝试 4: 降低 max_seq_length
```yaml
# 从 512 降低到 256
max_seq_length: 256
```
**结果**: ⚠️ 部分有效，训练进度更长但仍崩溃

##### 尝试 5: 修改精度设置
```yaml
# 从 bf16 切换到 fp16
fp16: true
bf16: false
```
**结果**: ⚠️ 部分有效

##### 尝试 6: 禁用 gradient_checkpointing
```yaml
gradient_checkpointing: false
use_cache: true
```
**结果**: ⚠️ 训练进度达到 71%，但仍崩溃

##### 尝试 7: 从 checkpoint 恢复
```powershell
# 删除损坏的 checkpoint，让脚本自动从最新的恢复
Remove-Item -Recurse -Force adapters\code-lora-v3-boundary\checkpoint-*
```
**结果**: ❌ 无可用 checkpoint

#### 当前配置 (train_boundary.yaml)
```yaml
max_seq_length: 256
num_train_epochs: 10
fp16: true
bf16: false
gradient_checkpointing: false
use_cache: true

lora:
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

#### 可能的原因分析
1. **Windows CUDA 驱动问题**: Windows 平台上的 CUDA 驱动可能存在内存管理 bug
2. **GPU 内存碎片化**: 长时间训练导致 GPU 内存碎片化
3. **数据问题**: Boundary 阶段的训练数据可能包含异常样本
4. **PyTorch/CUDA 版本兼容性**: 当前环境可能存在版本不兼容

#### 待尝试方案
- [ ] 使用 WSL2 代替 Windows 原生环境
- [ ] 进一步降低 batch_size 和 max_seq_length
- [ ] 检查并清洗 Boundary 训练数据
- [ ] 升级/降级 PyTorch 和 CUDA 版本
- [ ] 使用 gradient_checkpointing（虽然之前禁用，但可能有助于内存管理）

---

### 问题 #4: Python 语法错误

**发现时间**: 2026-06-27  
**严重程度**: 低  
**状态**: ✅ 已解决  
**影响范围**: 数据生成脚本

#### 问题描述
修改 `scripts/mutate_code.py` 时引入语法错误。

#### 错误信息
```
SyntaxError: invalid syntax
File "scripts/mutate_code.py", line XX
    max_per_sample: int = 5,5,
                          ^
```

#### 根本原因
编辑文件时误输入重复的逗号：`max_per_sample: int = 5,5,`

#### 解决方案
修正为正确的语法：
```python
# 错误
max_per_sample: int = 5,5,

# 正确
max_per_sample: int = 5,
```

#### 验证结果
- ✅ 脚本正常运行
- ✅ 成功生成 162 个修复样本（从原来的 3 个增加到 5 个 per sample）

---

### 问题 #5: CLI 参数错误

**发现时间**: 2026-06-27  
**严重程度**: 中  
**状态**: ✅ 已解决  
**影响范围**: 训练恢复流程

#### 问题描述
尝试使用 `--resume_from_checkpoint` 参数恢复训练时失败。

#### 错误信息
```
error: unrecognized arguments: --resume_from_checkpoint adapters/code-lora-v3-boundary/checkpoint-50
```

#### 根本原因
`scripts/train_lora.py` 不支持 `--resume_from_checkpoint` CLI 参数。脚本内部实现了自动从最新 checkpoint 恢复的逻辑。

#### 解决方案
直接运行训练脚本，无需额外参数：
```powershell
# 错误方式
python scripts/train_lora.py --config configs/train_boundary.yaml --resume_from_checkpoint ...

# 正确方式
python scripts/train_lora.py --config configs/train_boundary.yaml
```

脚本会自动检测 `output_dir` 中的最新 checkpoint 并恢复训练。

#### 代码实现参考
```python
# scripts/train_lora.py 中的自动恢复逻辑
checkpoint_dir = Path(output_dir) / "checkpoint-latest"
if checkpoint_dir.exists():
    print(f"Resuming from checkpoint: {checkpoint_dir}")
    # ... 恢复逻辑
```

---

## 训练进度总结

### 已完成的训练阶段

#### 1. v3-easy 阶段 ✅
- **配置**: rank=32, 7 modules, 10 epochs, max_seq_length=512
- **训练数据**: 70% L0 + 30% L1
- **训练结果**: 
  - Checkpoints: 50, 100, 150, 170
  - Pass@1: 66.7%
  - Repair success: 80.0%
- **状态**: 成功完成

#### 2. v3-boundary 阶段 ⚠️
- **配置**: rank=32, 7 modules, 10 epochs, max_seq_length=256
- **训练数据**: 20% L0 + 30% L1 + 50% L2
- **训练结果**: 崩溃（CUDA ACCESS_VIOLATION）
- **当前状态**: 调试中

#### 3. v3-repair 阶段 ❌
- **状态**: 未开始
- **依赖**: 需要 Boundary 阶段完成

---

## 环境信息

### 硬件
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
- **VRAM**: 4096 MiB
- **系统**: Windows

### 软件
- **Python**: Anaconda 环境 (qwen3-code-lab)
- **PyTorch**: (需确认版本)
- **CUDA**: (需确认版本)

### 关键依赖
```
transformers
peft
datasets
torch
```

---

## 下一步计划

1. **解决 Boundary 训练崩溃问题**
   - 尝试在 WSL2 环境中运行
   - 进一步降低内存占用
   - 检查训练数据质量

2. **完成三阶段训练**
   - v3-boundary (当前)
   - v3-repair (待开始)

3. **最终评测**
   - 对比 Baseline vs v3-easy vs v3-boundary vs v3-repair
   - 记录 Pass@1, Repair success, 推理延迟等指标

---

## 附录: 关键文件路径

- **训练配置**: `configs/train_easy.yaml`, `configs/train_boundary.yaml`, `configs/train_repair.yaml`
- **训练脚本**: `scripts/train_lora.py`
- **评测脚本**: `scripts/evaluate_model.py`
- **数据生成**: `scripts/mutate_code.py`, `scripts/generate_tasks.py`
- **模型输出**: `adapters/code-lora-v3-*`
- **训练日志**: `train_boundary.log`

---

**最后更新**: 2026-06-27  
**维护者**: AI Assistant
