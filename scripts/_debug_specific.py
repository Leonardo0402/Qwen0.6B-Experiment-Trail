"""Debug specific Stage2 sample - count_char looks correct but fails."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Load frozen eval
frozen_path = _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2" / "test_raw.jsonl"
frozen = {}
for line in frozen_path.open(encoding="utf-8"):
    if line.strip():
        s = json.loads(line)
        frozen[s["sample_id"]] = s

# Load Stage2 eval
d = json.load(open(_ROOT / "evaluations" / "p2" / "stage2-boundary.json", encoding="utf-8"))
for o in d["outcomes"]:
    if o["sample_id"] in ("mbpp_666", "mbpp_678", "mbpp_611", "mbpp_635"):
        sid = o["sample_id"]
        fs = frozen.get(sid, {})
        print(f"\n=== {sid} (task_type={o['task_type']}) ===")
        print(f"  INSTRUCTION: {fs.get('instruction', '')[:300]}")
        print(f"  PUBLIC_TESTS:\n    {(fs.get('public_tests') or '')[:400]}")
        print(f"  HIDDEN_TESTS:\n    {(fs.get('hidden_tests') or '')[:400]}")
        print(f"  GENERATED (extracted_code):")
        gen = o.get("extracted_code") or ""
        for line in gen.split("\n"):
            print(f"    {line}")
        print(f"  public_passed: {o.get('public_passed')}, hidden_passed: {o.get('hidden_passed')}")
        if o.get("error"):
            print(f"  error: {o['error']}")
