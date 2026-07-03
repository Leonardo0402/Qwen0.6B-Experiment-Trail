"""Show evaluation summary for all P2 models."""
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

models = [
    ("base", "Base Qwen3-0.6B"),
    ("stage1-code", "Stage 1 Code"),
    ("stage2-boundary", "Stage 2 Boundary"),
    ("stage3-repair", "Stage 3 Repair"),
]

print(f"{'Model':<25} {'Pass@1':>8} {'Syntax':>8} {'Repair':>8} {'Hidden':>8} {'Format':>8} {'Timeout':>8}")
print("-" * 85)

for name, label in models:
    f = _ROOT / "evaluations" / "p2" / f"{name}.json"
    if f.exists():
        d = json.load(open(f))
        m = d["metrics"]
        print(f"{label:<25} {m['pass_at_1']:>8.3f} {m['syntax_rate']:>8.3f} {m['repair_success_rate']:>8.3f} {m['hidden_pass_rate']:>8.3f} {m['format_compliance_rate']:>8.3f} {m['timeout_rate']:>8.3f}")
    else:
        print(f"{label:<25} {'PENDING':>8}")
