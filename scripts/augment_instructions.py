"""Augment all P2 data with function signatures in instructions.

Reads all JSONL files in data/p2-curriculum/ and updates instruction
fields by appending function signatures extracted from target_code.
Also updates the frozen-eval-v2 test_raw.jsonl.

This fixes the 0% Pass@1 issue: MBPP instructions lack function names,
so the model cannot infer exact function names expected by tests.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.extract_function_signature import augment_instruction


def process_jsonl(path: Path) -> tuple[int, int]:
    """Process a JSONL file, augmenting instructions.

    Returns (total_samples, augmented_count).
    """
    if not path.exists():
        return 0, 0

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    updated = []
    augmented = 0

    for line in lines:
        if not line.strip():
            continue
        s = json.loads(line)
        old_instr = s.get("instruction", "")
        target = s.get("target_code", "")
        new_instr = augment_instruction(old_instr, target)
        if new_instr != old_instr:
            s["instruction"] = new_instr
            augmented += 1
        updated.append(json.dumps(s, ensure_ascii=False))

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(updated) + "\n")

    return len(updated), augmented


def main() -> None:
    base = _ROOT / "data" / "p2-curriculum"

    files = [
        base / "stage1-code" / "train.jsonl",
        base / "stage1-code" / "validation.jsonl",
        base / "stage2-boundary" / "train.jsonl",
        base / "stage2-boundary" / "validation.jsonl",
        base / "stage3-repair" / "train.jsonl",
        base / "stage3-repair" / "validation.jsonl",
        # frozen-eval-v2/test_raw.jsonl is augmented in place; never use a
        # "train.jsonl" inside frozen-eval-v2 (deleted in P0-2 fix).
        base / "frozen-eval-v2" / "test_raw.jsonl",
    ]

    print("Augmenting instructions with function signatures...")
    print("=" * 60)
    total_aug = 0
    total_all = 0
    for f in files:
        if not f.exists():
            print(f"  SKIP (not found): {f.relative_to(_ROOT)}")
            continue
        total, aug = process_jsonl(f)
        total_all += total
        total_aug += aug
        print(f"  {f.relative_to(_ROOT)}: {total} samples, {aug} augmented")
    print("=" * 60)
    print(f"Total: {total_all} samples, {total_aug} augmented")

    # Verify augmentation
    print("\nVerification (first 2 samples from frozen-eval):")
    test = base / "frozen-eval-v2" / "test_raw.jsonl"
    with test.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            s = json.loads(line)
            print(f"\n  sample_id: {s['sample_id']}")
            print(f"  instruction: {s['instruction'][:200]}")


if __name__ == "__main__":
    main()
