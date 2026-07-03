"""Check P2 completion status against original spec."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

print("=" * 70)
print("P2 完成情况对照检查")
print("=" * 70)

# 1. 数据规模检查
print("\n[1] 数据规模")
total_train = 0
for stage in ['stage1-code', 'stage2-boundary', 'stage3-repair']:
    m = json.load(open(_ROOT / f'data/p2-curriculum/{stage}/manifest.json'))
    tc = m.get('sample_counts', {})
    t = tc.get('train', 0)
    v = tc.get('validation', 0)
    f = len(m.get('train_families', []))
    total_train += t
    print(f"  {stage}: train={t} val={v} families={f}")
print(f"  训练样本总计: {total_train}")
fm = json.load(open(_ROOT / 'data/p2-curriculum/frozen-eval-v2/manifest.json'))
# frozen-eval-v2 manifest uses test_sha256 / test_families (P0-2 fix);
# never reads train fields.
fe_samples = fm.get('sample_counts', {}).get('test', 0)
fe_fams = len(fm.get('test_families', []))
print(f"  frozen-eval: samples={fe_samples} families={fe_fams}")
print(f"  frozen-eval test_sha256: {fm.get('test_sha256', '?')[:32]}...")

# 2. 任务类型分布
print("\n[2] 任务类型分布")
tt = fm.get('task_type_mix', {})
for t, c in sorted(tt.items()):
    print(f"  {t}: {c}")

# 3. 报告检查
print("\n[3] 报告清单 (目标6个)")
reports = [
    'p2-data-factory-report.md',
    'p2-training-readiness-report.md',
    'p2-stage1-code-report.md',
    'p2-stage2-boundary-report.md',
    'p2-stage3-repair-report.md',
    'p2-final-comparison-report.md',
]
for r in reports:
    p = _ROOT / 'reports' / 'p2' / r
    status = "OK" if p.exists() else "MISSING"
    print(f"  {r}: {status}")

# 4. Adapter检查
print("\n[4] Adapter检查 (v2)")
for stage in ['stage1-code-v2', 'stage2-boundary-v2', 'stage3-repair-v2']:
    p = _ROOT / 'adapters' / 'p2' / 'continual' / stage / 'adapter_model.safetensors'
    print(f"  {stage}: {'OK' if p.exists() else 'MISSING'}")

# 5. 评测结果
print("\n[5] 评测结果")
for name in ['base', 'stage1-code', 'stage2-boundary', 'stage3-repair']:
    p = _ROOT / 'evaluations' / 'p2' / f'{name}.json'
    if p.exists():
        d = json.load(open(p))
        m = d['metrics']
        print(f"  {name}: Pass@1={m['pass_at_1']:.3f} Syntax={m['syntax_rate']:.3f}")
    else:
        print(f"  {name}: MISSING")

# 6. Independent Stage3 (可选,用于对比)
print("\n[6] Independent Stage3 Adapter (可选对比)")
p = _ROOT / 'adapters' / 'p2' / 'independent' / 'stage3-repair-v2'
print(f"  {'OK' if p.exists() else '未训练 (规范中为可选项)'}")

# 7. 评测器Bug修复状态
print("\n[7] 评测器Bug修复状态")
print("  问题: MBPP测试代码是裸assert,无'from solution import',导致NameError")
print("  修复: src/sandbox.py添加_normalize_test_code自动归一化裸assert")
print("  状态: 已修复,需重跑评测验证Pass@1提升")
