"""Show v2 evaluation metrics summary."""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1] if len(sys.argv) > 1 else "base"
p = _ROOT / "evaluations" / "p2" / f"{name}.json"
if not p.exists():
    print(f"{name}: file not found")
    sys.exit(1)

d = json.load(open(p, encoding="utf-8"))
m = d["metrics"]
outcomes = d.get("outcomes", [])
passed = sum(1 for o in outcomes if o.get("public_passed") and o.get("hidden_passed"))
print(f"=== {name} v2 (n={len(outcomes)}) ===")
print(f"  Pass@1: {m['pass_at_1']:.3f}  ({passed}/{len(outcomes)} passed)")
print(f"  Syntax: {m['syntax_rate']:.3f}")
print(f"  Repair: {m['repair_success_rate']:.3f}")
print(f"  Hidden: {m['hidden_pass_rate']:.3f}")

# Per task_type
from collections import Counter
by_type = Counter()
pass_by_type = Counter()
for o in outcomes:
    t = o.get("task_type", "unknown")
    by_type[t] += 1
    if o.get("public_passed") and o.get("hidden_passed"):
        pass_by_type[t] += 1
print("\n  Per task_type:")
for t in sorted(by_type):
    print(f"    {t}: {pass_by_type[t]}/{by_type[t]} = {100*pass_by_type[t]/max(1,by_type[t]):.1f}%")
