# P4.1b Protocol Ablation 开发启动 Prompt

## 会话角色

你是 **开发会话（Development Session）**，负责实现 P4.1b 协议 ablation 实验的所有代码。

## 项目背景

### 项目概述
- **项目名称**: Qwen0.6B-Experiment-Trail
- **训练目标**: 在 RTX 3050 (4GB VRAM) 上训练 Qwen3-0.6B 作为 Constrained Local Repair Agent
- **当前阶段**: P4.1 → P4.1b → P4.2（训练准备）
- **工作目录**: `E:\agent\Qwen\qwen3-code-lab`

### P4.1 当前状态
- ✅ 本地完成已声明（2026-07-10）
- ⚠️ 远端验证待定
- ⚠️ 存在数据集计数不匹配问题（1315 总轨迹，920+130+220=1270，45 条未说明）
- ⚠️ Model smoke 仅 5+5 样本（LIMITED_SMOKE，非完整 40 任务评估）

### P4.1b 定位
P4.1b 是 P4.1（基础 harness）和 P4.2（训练准备）之间的**协议 ablation 实验阶段**。

**目标**: 评估三种 trajectory 输出协议（JSON / Tag / DSL），选择最优格式用于后续训练。

**为什么需要 P4.1b**:
- P4.1 使用 JSON 协议，但未验证是否为最优选择
- 协议格式影响：parseability、fidelity、token efficiency、LLM compatibility
- 在 P4.2 冻结训练配置前，必须完成协议 ablation

## 开发任务

### 核心交付物

#### 1. 三种协议格式实现
```
src/protocols/
├── __init__.py
├── base.py              # Protocol 基类
├── json_protocol.py     # JSON structured 格式
├── tag_protocol.py      # Tag hybrid 格式
└── dsl_protocol.py      # DSL domain-specific 格式
```

**JSON 协议**（已存在于 P4.1，需重构为独立模块）:
```json
{
  "task_id": "mbpp_123",
  "strategy": "targeted_fix",
  "steps": [
    {
      "action": "edit",
      "file": "solution.py",
      "line_start": 10,
      "line_end": 15,
      "content": "def fixed_function():\n    return 42"
    }
  ],
  "confidence": 0.85
}
```

**Tag 协议**:
```
<TASK_ID>mbpp_123</TASK_ID>
<STRATEGY>targeted_fix</STRATEGY>
<STEP>
  <ACTION>edit</ACTION>
  <FILE>solution.py</FILE>
  <LINES>10-15</LINES>
  <CONTENT>
def fixed_function():
    return 42
  </CONTENT>
</STEP>
<CONFIDENCE>0.85</CONFIDENCE>
```

**DSL 协议**:
```
task: mbpp_123
strategy: targeted_fix
edit solution.py:10-15
  def fixed_function():
      return 42
confidence: 0.85
```

#### 2. 协议转换层
```
src/protocols/converter.py
```

**功能**:
- `convert_trajectory(source: Trajectory, from_proto: str, to_proto: str) -> str`
- 支持任意两种协议之间的双向转换
- 转换过程必须无损（fidelity = 100%）

#### 3. 协议评估框架
```
src/protocols/evaluator.py
```

**评估维度**（每个维度 0-1 分）:

1. **Parseability**（可解析性）
   - 解析成功率（%）
   - 解析错误率（syntax errors, schema violations）
   - 解析速度（trajectories/second）

2. **Fidelity**（保真度）
   - 转换后信息完整性（%）
   - 字段丢失率
   - 语义等价性验证

3. **Token Efficiency**（Token 效率）
   - 平均 token 数/trajectory
   - Token 压缩比（vs JSON baseline）
   - 信息密度（bits/token）

4. **LLM Compatibility**（LLM 兼容性）
   - LLM 生成成功率（%）
   - 格式遵循度（schema adherence）
   - 幻觉率（hallucination rate）

**评估方法**:
- Parseability: 对每种协议生成 100 个样本，统计解析成功率
- Fidelity: 对同一 trajectory 进行 10 次协议转换，验证信息完整性
- Token Efficiency: 统计 100 个 trajectory 的平均 token 数（使用 tiktoken）
- LLM Compatibility: 使用 Qwen3-0.6B 生成 50 个 trajectory，统计格式遵循度

#### 4. Ablation 实验脚本
```
scripts/run_protocol_ablation.py
```

**功能**:
- 从 P4.1 的 1315 条 trajectory 中采样 100 条（分层采样：920 train + 130 validation + 220 test）
- 对每条 trajectory 进行三种协议格式转换
- 运行四维评估框架
- 生成对比报告

**输出**:
```
artifacts/p4_1b_protocol_ablation/
├── evaluation_report.json
├── comparison_table.md
├── token_efficiency.csv
├── parseability_results.json
├── fidelity_results.json
└── llm_compatibility_results.json
```

#### 5. 协议选择决策
```
docs/proposals/p4_1b_protocol_selection.md
```

**内容**:
- 三种协议的评估结果对比
- 加权评分（建议权重：Parseability 30%, Fidelity 30%, Token Efficiency 20%, LLM Compatibility 20%）
- 最终推荐协议及理由
- 迁移计划（如果选择非 JSON 协议，如何迁移 P4.1 harness）

### 验收标准

#### 功能验收
- [ ] 三种协议格式均可正确解析和生成
- [ ] 协议转换层支持任意双向转换，fidelity = 100%
- [ ] 评估框架可输出四维评分
- [ ] Ablation 脚本可端到端运行，生成完整报告

#### 质量验收
- [ ] 所有新代码通过 `pytest tests/test_p4_1b_*.py`
- [ ] 测试覆盖率 > 90%（使用 `pytest --cov=src/protocols`）
- [ ] 无 linting 错误（`flake8 src/protocols/`）
- [ ] 类型注解完整（`mypy src/protocols/`）

#### 文档验收
- [ ] 每种协议格式有完整的 schema 文档
- [ ] 评估框架有使用说明
- [ ] 协议选择决策文档已提交

#### 兼容性验收
- [ ] 不破坏 P4.1 现有功能（`pytest tests/test_p4_*.py` 全部通过）
- [ ] 向后兼容：P4.1 的 JSON 协议仍可正常使用
- [ ] 新增协议模块可独立使用，不依赖 P4.1 其他组件

## 技术约束

### 硬件约束
- **GPU**: RTX 3050 (4GB VRAM)
- **内存**: 16GB RAM
- **存储**: 确保 artifacts 目录有足够空间（预计 ~500MB）

### 软件约束
- **Python**: 3.10+
- **依赖**: 仅使用项目已有依赖（见 `requirements.txt`），不引入新依赖
- **Token 计数**: 使用 `tiktoken`（如未安装，使用 `transformers` 的 tokenizer）

### 代码规范
- **命名**: snake_case（函数/变量），PascalCase（类），UPPER_SNAKE_CASE（常量）
- **注释**: 所有公共函数/类必须有 docstring（Google style）
- **类型注解**: 所有函数签名必须有类型注解
- **错误处理**: 使用显式异常，不使用 bare `except:`

### 测试规范
- **TDD**: 先写测试，再写实现
- **测试文件命名**: `test_p4_1b_*.py`
- **测试覆盖率**: 使用 `pytest-cov`，目标 > 90%
- **测试数据**: 使用 fixture，不硬编码

## 开发流程

### 1. 创建 Feature Branch
```bash
git checkout -b feature/p4-1b-protocol-ablation
```

### 2. 实现顺序（建议）
```
Phase 1: 协议格式实现（TDD）
  1.1 编写 json_protocol 测试 → 实现 → 验证
  1.2 编写 tag_protocol 测试 → 实现 → 验证
  1.3 编写 dsl_protocol 测试 → 实现 → 验证

Phase 2: 协议转换层（TDD）
  2.1 编写 converter 测试 → 实现 → 验证

Phase 3: 评估框架（TDD）
  3.1 编写 evaluator 测试 → 实现 → 验证

Phase 4: Ablation 脚本
  4.1 实现采样逻辑
  4.2 实现评估流程
  4.3 生成报告

Phase 5: 协议选择决策
  5.1 运行 ablation 实验
  5.2 分析结果
  5.3 撰写决策文档
```

### 3. 提交规范
```bash
git commit -m "feat(p4-1b): implement JSON protocol format"
git commit -m "feat(p4-1b): add protocol converter layer"
git commit -m "feat(p4-1b): add evaluation framework"
git commit -m "feat(p4-1b): run protocol ablation experiment"
git commit -m "docs(p4-1b): add protocol selection decision"
```

### 4. 验证门
完成所有开发后，运行：
```bash
# 1. 运行所有 P4.1b 测试
pytest tests/test_p4_1b_*.py -v

# 2. 运行 P4.1 回归测试
pytest tests/test_p4_*.py -v

# 3. 检查测试覆盖率
pytest --cov=src/protocols --cov-report=term-missing

# 4. Linting
flake8 src/protocols/

# 5. 类型检查
mypy src/protocols/
```

**通过标准**: 所有测试通过，覆盖率 > 90%，无 linting 错误，无类型错误。

## 关键依赖

### P4.1 现有模块
- `src/agent_trajectory.py`: Trajectory 数据结构
- `src/agent_model_provider.py`: ModelActionProvider（生成 trajectory）
- `src/training_data.py`: SFT 数据构建器

### 外部依赖
- `datasets`: 加载 MBPP 数据集
- `tiktoken` 或 `transformers`: Token 计数
- `pytest`, `pytest-cov`: 测试框架
- `flake8`, `mypy`: 代码质量工具

## 输出交付物

### 代码交付
- [ ] `src/protocols/` 目录（包含所有协议实现）
- [ ] `scripts/run_protocol_ablation.py`
- [ ] `tests/test_p4_1b_*.py`（所有测试文件）

### 文档交付
- [ ] `docs/proposals/p4_1b_protocol_selection.md`
- [ ] `artifacts/p4_1b_protocol_ablation/`（所有评估结果）

### 验证交付
- [ ] 所有测试通过的截图/日志
- [ ] 测试覆盖率报告
- [ ] Linting 和类型检查报告

## 风险与缓解

### 风险 1: LLM Compatibility 评估需要 GPU
**缓解**: 如果 GPU 不可用，使用 CPU 推理（速度较慢但可行），或仅评估前三个维度。

### 风险 2: DSL 协议可能难以解析
**缓解**: 如果 DSL 解析成功率 < 90%，在决策文档中明确标注风险，建议不采用。

### 风险 3: 协议转换可能丢失信息
**缓解**: 在转换层添加严格的 schema 验证，确保 fidelity = 100%。如果无法达到 100%，在决策文档中说明。

## 下一步行动

1. **确认理解**: 阅读本 prompt 后，确认你对任务的理解。
2. **创建分支**: 创建 `feature/p4-1b-protocol-ablation` 分支。
3. **开始 Phase 1**: 从 JSON 协议开始，按 TDD 流程实现。
4. **定期检查点**: 每完成一个 Phase，运行验证门，确保质量。

## 参考文档

- **Proposal**: `docs/proposals/p4_1b_protocol_harness_ablation.md`
- **Roadmap**: `docs/roadmaps/Qwen0.6B-Experiment-Trail_P4-Roadmap_2026-07-10.md`
- **P4.1 Spec**: `docs/superpowers/specs/2026-07-09-p4-1-model-action-provider-and-sft-data.md`
- **P4.1 Plan**: `docs/superpowers/plans/2026-07-09-p4-1-model-action-provider-and-sft-data.md`
- **Workflow**: `docs/workflow/Qwen0.6B-Trae-Multi-Session-Workflow.md`

## 联系与反馈

如果在开发过程中遇到阻断性问题：
1. 记录问题到 `docs/issues/p4_1b_blockers.md`
2. 停止当前任务
3. 通知决策会话（Decision Session）进行评审

---

**开发启动时间**: 2026-07-10
**预计完成时间**: 根据实际进度（不设定硬性 deadline）
**验证门**: 所有测试通过 + 覆盖率 > 90% + 无 linting 错误
