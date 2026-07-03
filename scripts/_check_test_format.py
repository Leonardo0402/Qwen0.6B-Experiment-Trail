"""Check test format examples - both raw and with-import."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
p = _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2" / "test_raw.jsonl"

samples = []
for line in p.open(encoding="utf-8"):
    if line.strip():
        samples.append(json.loads(line))

# Show one raw (no import) and one with-import
print("=== RAW (no from solution) example ===")
for s in samples:
    pt = s.get("public_tests") or ""
    if "from solution" not in pt and "def test" not in pt:
        print(f"sample_id={s['sample_id']}")
        print(f"public_tests:")
        print(pt)
        break

print("\n=== WITH 'from solution' example ===")
for s in samples:
    pt = s.get("public_tests") or ""
    if "from solution" in pt:
        print(f"sample_id={s['sample_id']}")
        print(f"public_tests:")
        print(pt)
        break

print("\n=== WITH 'def test' example ===")
for s in samples:
    pt = s.get("public_tests") or ""
    if "def test" in pt:
        print(f"sample_id={s['sample_id']}")
        print(f"public_tests:")
        print(pt)
        break
