"""P0-5 fix: fixate the 120-sample stratified evaluation subset.

Previously, scripts/evaluate_model.py --max-samples 120 re-sampled on every
run, so the 120 samples used to evaluate Base / Stage1 / Stage2 / Stage3 were
NOT guaranteed to be the same set. Issue #1 requires fixing the subset with:
- explicit sample_id / family_id / task_type lists
- seed
- SHA256 of the subset file
- a manifest

This script:
1. Reads frozen-eval-v2/test_raw.jsonl
2. Stratified-samples 40 code_generation + 40 static_repair + 40 execution_repair
   using seed=42 (matching earlier runs for reproducibility)
3. Writes data/p2-curriculum/frozen-eval-v2/stratified-120/test_raw.jsonl
4. Writes data/p2-curriculum/frozen-eval-v2/stratified-120/manifest.json
   with sample_id / family_id / task_type / seed / SHA256

The eval scripts must be updated to read this fixed subset instead of
re-sampling.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FE_DIR = _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2"
TEST_FILE = FE_DIR / "test_raw.jsonl"
OUT_DIR = FE_DIR / "stratified-120"
OUT_FILE = OUT_DIR / "test_raw.jsonl"
OUT_MANIFEST = OUT_DIR / "manifest.json"

SEED = 42
PER_TYPE = 40
TARGET_TYPES = ("code_generation", "static_repair", "execution_repair")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not TEST_FILE.exists():
        print(f"ERROR: {TEST_FILE} not found", file=sys.stderr)
        return 1

    # Load all samples
    all_samples = []
    with TEST_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))
    print(f"Loaded {len(all_samples)} samples from {TEST_FILE.name}")

    # Group by task_type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for s in all_samples:
        by_type[s.get("task_type", "unknown")].append(s)

    # Stratified sample with fixed seed
    rng = random.Random(SEED)
    selected: list[dict] = []
    for t in TARGET_TYPES:
        pool = list(by_type.get(t, []))
        if len(pool) < PER_TYPE:
            print(f"WARN: only {len(pool)} samples of {t} available; "
                  f"using all of them", flush=True)
            chosen = pool
        else:
            chosen = rng.sample(pool, PER_TYPE)
        # Sort by sample_id for stable ordering
        chosen.sort(key=lambda x: x.get("sample_id", ""))
        selected.extend(chosen)
        print(f"  {t}: {len(chosen)} / {len(pool)}")

    # Write subset
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8", newline="\n") as fh:
        for s in selected:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    subset_sha = sha256_file(OUT_FILE)
    sample_ids = [s["sample_id"] for s in selected]
    family_ids = sorted({s["family_id"] for s in selected})
    task_type_counts = {t: sum(1 for s in selected if s.get("task_type") == t)
                         for t in TARGET_TYPES}

    manifest = {
        "purpose": "Fixed 120-sample stratified evaluation subset",
        "source_file": "data/p2-curriculum/frozen-eval-v2/test_raw.jsonl",
        "subset_file": "data/p2-curriculum/frozen-eval-v2/stratified-120/test_raw.jsonl",
        "subset_sha256": subset_sha,
        "seed": SEED,
        "per_task_type": PER_TYPE,
        "total_samples": len(selected),
        "sample_ids": sample_ids,
        "family_ids": family_ids,
        "family_count": len(family_ids),
        "task_type_counts": task_type_counts,
        "stratification_rule": (
            "For each task_type in (code_generation, static_repair, "
            "execution_repair), sample PER_TYPE=40 samples with seed=42 "
            "from frozen-eval-v2/test_raw.jsonl. Sorted by sample_id for "
            "stable ordering. This subset is FROZEN — never re-sample."
        ),
        "integrity_note": (
            "Use this file as the eval dataset for all P2 model comparisons. "
            "Do NOT pass --max-samples to evaluate_model.py for formal "
            "evaluation; pass this file as --dataset instead."
        ),
    }
    with OUT_MANIFEST.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote: {OUT_FILE.relative_to(_ROOT)}")
    print(f"Wrote: {OUT_MANIFEST.relative_to(_ROOT)}")
    print(f"  subset_sha256: {subset_sha[:32]}...")
    print(f"  total_samples: {len(selected)}")
    print(f"  families: {len(family_ids)}")
    print(f"  task_type_counts: {task_type_counts}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
