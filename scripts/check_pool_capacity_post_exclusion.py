"""Check canonical pool capacity after excluding frozen v4 families."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

pool_path = ROOT / "data/p3-formal/canonical-pool.jsonl"
frozen_v4_path = ROOT / "data/frozen-eval/v4/test_raw.jsonl"

def load_jsonl(p):
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

pool = load_jsonl(pool_path)
frozen = load_jsonl(frozen_v4_path)

frozen_sample_ids = {s["sample_id"] for s in frozen}
frozen_family_ids = {s.get("family_id") or s.get("mbpp_family_id") for s in frozen}

print(f"Pool: {len(pool)} samples")
print(f"Frozen v4: {len(frozen_sample_ids)} sample_ids, {len(frozen_family_ids)} family_ids")

# How many pool samples overlap frozen by sample_id
pool_overlap_sample = sum(1 for s in pool if s.get("sample_id") in frozen_sample_ids)
print(f"\nPool overlap by sample_id: {pool_overlap_sample}")

# How many pool samples belong to frozen families
pool_overlap_family = sum(1 for s in pool if (s.get("family_id") or s.get("mbpp_family_id")) in frozen_family_ids)
print(f"Pool overlap by family_id: {pool_overlap_family}")

# Pool after exclusion
pool_after = [s for s in pool if (s.get("family_id") or s.get("mbpp_family_id")) not in frozen_family_ids]
print(f"\nPool after excluding frozen families: {len(pool_after)} samples")

# Bucket counts after exclusion
buckets_after = Counter(s.get("variant_type") for s in pool_after)
print(f"Bucket counts after exclusion: {dict(buckets_after)}")

# Check if we still have enough for both candidates
print(f"\n--- Capacity check ---")
print(f"Balanced-Limited needs: code=441, boundary=294, static=294, exec=441")
print(f"Repair-Limited needs:   code=221, boundary=220, static=441, exec=588")
print(f"Available:              {dict(buckets_after)}")

for bucket, needed in [("code", 441), ("boundary", 294), ("static_repair", 294), ("execution_repair", 441)]:
    avail = buckets_after.get(bucket, 0)
    status = "OK" if avail >= needed else "SHORT"
    print(f"  {bucket}: need {needed} (balanced), have {avail} → {status}")

for bucket, needed in [("code", 221), ("boundary", 220), ("static_repair", 441), ("execution_repair", 588)]:
    avail = buckets_after.get(bucket, 0)
    status = "OK" if avail >= needed else "SHORT"
    print(f"  {bucket}: need {needed} (repair), have {avail} → {status}")

# Family count after exclusion
families_after = set((s.get("family_id") or s.get("mbpp_family_id")) for s in pool_after)
print(f"\nFamilies after exclusion: {len(families_after)}")
