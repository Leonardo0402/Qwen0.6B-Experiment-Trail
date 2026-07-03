"""P0-2 fix: clean frozen-eval-v2 directory and manifest.

This script:
1. Deletes frozen-eval-v2/train.jsonl and validation.jsonl
   (legacy files that duplicate test_raw.jsonl / are empty).
2. Rewrites frozen-eval-v2/manifest.json to use test_sha256 / test_families
   instead of train_sha256 / train_families (frozen-eval is NOT a training
   set — calling its families "train_families" caused audit_p2_dataset.py
   to count 576 frozen samples as training data, inflating totals to 1500).
3. Records the test_raw.jsonl SHA256 properly.

Run once after deleting the legacy files. Idempotent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FE_DIR = _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # 1. Delete legacy train.jsonl and validation.jsonl
    for name in ("train.jsonl", "validation.jsonl"):
        p = FE_DIR / name
        if p.exists():
            p.unlink()
            print(f"DELETED: {p.relative_to(_ROOT)}")
        else:
            print(f"already absent: {name}")

    # 2. Load partition to get the frozen family set (the families the
    #    frozen-eval is actually allowed to use).
    partition_path = _ROOT / "data" / "p2-curriculum" / "family-partition.json"
    partition = json.load(partition_path.open(encoding="utf-8"))
    frozen_families = sorted(set(partition.get("frozen_families", [])))

    # 3. Compute test_raw.jsonl SHA256 and sample count
    test_path = FE_DIR / "test_raw.jsonl"
    test_sha = sha256_file(test_path)
    n_samples = sum(1 for _ in test_path.open(encoding="utf-8") if _.strip())
    n_families = len(frozen_families)

    # 4. Build the corrected manifest. The old manifest mixed partition
    #    sets (train_families / validation_families / frozen_families)
    #    which is misleading because frozen-eval only ever uses the
    #    frozen set. Keep partition info under partition_* keys for
    #    traceability, but the top-level fields describe the eval set.
    old = json.load((FE_DIR / "manifest.json").open(encoding="utf-8"))

    manifest = {
        "stage": "frozen-eval-v2",
        "purpose": "FROZEN evaluation set — never used for training",
        "dataset_version": old.get("dataset_version", "p2.2"),
        "family_partition_version": old.get("family_partition_version", "p2"),
        "created_at": old.get("created_at"),
        "seed": old.get("seed", 42),
        # Renamed from train_sha256 -> test_sha256
        "test_sha256": test_sha,
        "test_file": "test_raw.jsonl",
        "sample_counts": {
            "test": n_samples,
            "train": 0,   # frozen-eval has NO train split
            "validation": 0,
        },
        "family_counts": {
            "test": n_families,
            "train": 0,
            "validation": 0,
        },
        # The families the frozen-eval actually uses
        "test_families": frozen_families,
        # Partition traceability (read-only reference)
        "partition_train_families": sorted(set(partition.get("train_families", []))),
        "partition_validation_families": sorted(set(partition.get("validation_families", []))),
        "partition_frozen_families": frozen_families,
        "partition_overlap": {
            "train_validation": partition.get("train_validation_overlap", []),
            "train_frozen": partition.get("train_frozen_overlap", []),
            "validation_frozen": partition.get("validation_frozen_overlap", []),
            "three_way_disjoint": partition.get("three_way_disjoint", False),
        },
        "task_type_mix": old.get("task_type_mix", {}),
        "difficulty_mix": old.get("difficulty_mix", {}),
        "max_seq_length": old.get("max_seq_length", 512),
        "assistant_target_retention_rate": old.get("assistant_target_retention_rate", 1.0),
        "integrity_note": (
            "frozen-eval-v2 contains NO train.jsonl or validation.jsonl. "
            "Audit scripts must read test_raw.jsonl, not train.jsonl."
        ),
    }

    out = FE_DIR / "manifest.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"REWRITTEN: {out.relative_to(_ROOT)}")
    print(f"  test_sha256: {test_sha[:32]}...")
    print(f"  samples (test): {n_samples}")
    print(f"  families (test): {n_families}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
