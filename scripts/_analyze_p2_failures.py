"""Analyze v2 P2 evaluation failures - identify root cause of 0% Pass@1."""
import json
import re
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Load frozen-eval-v2 test_raw.jsonl to get target_code
frozen_path = _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2" / "test_raw.jsonl"
target_by_id = {}
if frozen_path.exists():
    for line in frozen_path.open(encoding="utf-8"):
        if line.strip():
            s = json.loads(line)
            target_by_id[s.get("sample_id")] = s.get("target_code") or ""

for name in ["base", "stage1-code", "stage2-boundary", "stage3-repair"]:
    p = _ROOT / "evaluations" / "p2" / f"{name}.json"
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    d = json.load(open(p, encoding="utf-8"))
    m = d["metrics"]
    outcomes = d.get("outcomes") or []
    print(f"\n=== {name} (n={len(outcomes)}) ===")
    print(f"  Pass@1={m['pass_at_1']:.3f}  Syntax={m['syntax_rate']:.3f}")

    # Function name match check
    correct_func_name = 0
    no_func_in_gen = 0
    no_target = 0
    for o in outcomes:
        gen = o.get("extracted_code") or ""
        sid = o.get("sample_id")
        tgt = target_by_id.get(sid, "")
        if not tgt:
            no_target += 1
            continue
        tgt_match = re.search(r'def\s+(\w+)\s*\(', tgt)
        gen_match = re.search(r'def\s+(\w+)\s*\(', gen)
        if not gen_match:
            no_func_in_gen += 1
            continue
        if tgt_match and gen_match and tgt_match.group(1) == gen_match.group(1):
            correct_func_name += 1
    print(f"  func name match: {correct_func_name}/{len(outcomes)} ({100*correct_func_name/max(1,len(outcomes)):.1f}%)")
    print(f"  no def in gen: {no_func_in_gen}  no_target: {no_target}")

    # Show 2 code_generation failures with correct func name
    code_gen_failures = []
    for o in outcomes:
        if o.get("task_type") != "code_generation" or o.get("public_passed"):
            continue
        gen = o.get("extracted_code") or ""
        sid = o.get("sample_id")
        tgt = target_by_id.get(sid, "")
        if not tgt or "def " not in gen:
            continue
        tgt_match = re.search(r'def\s+(\w+)\s*\(', tgt)
        gen_match = re.search(r'def\s+(\w+)\s*\(', gen)
        if tgt_match and gen_match and tgt_match.group(1) == gen_match.group(1):
            code_gen_failures.append(o)
        if len(code_gen_failures) >= 2:
            break
    if code_gen_failures:
        print(f"  Code-gen failures (correct func name, wrong logic):")
        for o in code_gen_failures:
            sid = o.get("sample_id")
            tgt = target_by_id.get(sid, "")
            gen = (o.get("extracted_code") or "").strip()
            print(f"    [{sid}] family={o.get('family_id')}")
            print(f"      TARGET (first 8 lines):")
            for line in tgt.split("\n")[:8]:
                print(f"        {line}")
            print(f"      GENERATED (first 8 lines):")
            for line in gen.split("\n")[:8]:
                print(f"        {line}")
