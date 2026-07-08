"""
Verify Issue #16 claims:
1. Frozen v4 sample/family overlap with P3-Limited train files
2. Canary rows included in scored Frozen v4 metrics
3. Provenance hash mismatches
"""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_jsonl_ids(p: Path, id_field: str = "sample_id") -> set:
    ids = set()
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if id_field in rec:
                    ids.add(rec[id_field])
    return ids

def load_jsonl_family_ids(p: Path) -> set:
    fams = set()
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                for k in ("family_id", "mbpp_family_id"):
                    if k in rec:
                        fams.add(rec[k])
                        break
    return fams

print("=" * 70)
print("CLAIM 1: Frozen v4 overlap with P3-Limited train files")
print("=" * 70)

frozen_v4 = ROOT / "data/frozen-eval/v4/test_raw.jsonl"
balanced_train = ROOT / "data/p3-limited/balanced-limited/train.jsonl"
repair_train = ROOT / "data/p3-limited/repair-limited/train.jsonl"

frozen_ids = load_jsonl_ids(frozen_v4)
frozen_fams = load_jsonl_family_ids(frozen_v4)
print(f"Frozen v4: {len(frozen_ids)} sample_ids, {len(frozen_fams)} family_ids")

for name, train in [("balanced-limited", balanced_train), ("repair-limited", repair_train)]:
    train_ids = load_jsonl_ids(train)
    train_fams = load_jsonl_family_ids(train)
    overlap_ids = train_ids & frozen_ids
    overlap_fams = train_fams & frozen_fams
    print(f"\n{name}:")
    print(f"  train sample_ids: {len(train_ids)}")
    print(f"  train family_ids: {len(train_fams)}")
    print(f"  OVERLAP sample_ids: {len(overlap_ids)}")
    print(f"  OVERLAP family_ids: {len(overlap_fams)}")
    if overlap_ids:
        print(f"  first 5 overlapping sample_ids: {sorted(overlap_ids)[:5]}")

print("\n" + "=" * 70)
print("CLAIM 2: Canary rows in scored Frozen v4 metrics")
print("=" * 70)

for name in ["balanced-limited", "repair-limited"]:
    eval_json = ROOT / f"evaluations/p3-limited/{name}-frozen-v4.json"
    if not eval_json.exists():
        print(f"{name}: eval file missing")
        continue
    with eval_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    m = data.get("metrics", {})
    print(f"\n{name}:")
    print(f"  n_total: {m.get('n_total')}")
    print(f"  n_generation: {m.get('n_generation')}")
    print(f"  n_repair: {m.get('n_repair')}")
    print(f"  canary.passed: {data.get('canary', {}).get('passed')}")
    canary_cases = data.get("canary", {}).get("cases", [])
    print(f"  canary.cases count: {len(canary_cases)}")
    # Check if canary rows are in outcomes
    outcomes = data.get("outcomes", [])
    canary_outcomes = [o for o in outcomes if "canary" in str(o.get("sample_id", ""))]
    print(f"  outcomes total: {len(outcomes)}")
    print(f"  outcomes with 'canary' in sample_id: {len(canary_outcomes)}")

print("\n" + "=" * 70)
print("CLAIM 3: Provenance hash mismatches")
print("=" * 70)

# Committed validation-v2 file
val_v2 = ROOT / "data/p3-curriculum/validation-v2/validation.jsonl"
if val_v2.exists():
    print(f"\nCommitted validation-v2/validation.jsonl sha256: {sha256_file(val_v2)[:16]}...")
else:
    print(f"\nvalidation-v2/validation.jsonl: MISSING")

# Committed frozen v4
print(f"Committed frozen-eval/v4/test_raw.jsonl sha256: {sha256_file(frozen_v4)[:16]}...")

# Adapter metrics recorded hashes
for name in ["balanced-limited", "repair-limited"]:
    metrics_p = ROOT / f"adapters/p3/{name}/metrics.json"
    if not metrics_p.exists():
        print(f"\n{name}: metrics.json missing")
        continue
    with metrics_p.open("r", encoding="utf-8") as fh:
        m = json.load(fh)
    print(f"\n{name} metrics.json:")
    for k in ("train_hash", "eval_hash", "dataset_hash", "validation_hash"):
        if k in m:
            print(f"  {k}: {m[k][:16]}...")
    # dump full keys
    print(f"  all keys: {list(m.keys())}")

# Frozen eval JSON recorded hashes
for name in ["balanced-limited", "repair-limited"]:
    eval_json = ROOT / f"evaluations/p3-limited/{name}-frozen-v4.json"
    if not eval_json.exists():
        continue
    with eval_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"\n{name} frozen-v4 eval JSON:")
    for k in ("dataset_sha256", "dataset_hash", "eval_hash"):
        if k in data:
            print(f"  {k}: {data[k][:16]}...")
    print(f"  all top-level keys: {list(data.keys())}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
