"""scripts/build_p3_limited.py -- P3-Limited dataset builder (Issue #14 follow-up).

Builds two 1470-sample datasets from the Formal Canonical Pool v2 with exact
central ratios (not at tolerance edges):

  Balanced Limited (30/20/20/30):
    code=441, boundary=294, static_repair=294, execution_repair=441

  Repair Limited (15/15/30/40):
    code=221, boundary=220, static_repair=441, execution_repair=588

Both candidates:
  - Draw from the same 403-family universe (canonical-pool.jsonl)
  - Use deterministic random sampling (seed=42) for reproducibility
  - Have no duplicate sample_ids within each candidate
  - Preserve the original sample format (ChatML messages)

Output:
  data/p3-limited/balanced-limited/train.jsonl
  data/p3-limited/balanced-limited/manifest.json
  data/p3-limited/repair-limited/train.jsonl
  data/p3-limited/repair-limited/manifest.json

Usage:
  py -3.11 scripts/build_p3_limited.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

POOL_PATH = _ROOT / "data" / "p3-formal" / "canonical-pool.jsonl"
POOL_MANIFEST_PATH = _ROOT / "data" / "p3-formal" / "canonical-pool-manifest.json"

OUTPUT_DIR = _ROOT / "data" / "p3-limited"
BALANCED_DIR = OUTPUT_DIR / "balanced-limited"
REPAIR_DIR = OUTPUT_DIR / "repair-limited"

SEED = 42

# Exact central ratios (not at tolerance edges)
CANDIDATES = {
    "balanced-limited": {
        "dir": BALANCED_DIR,
        "ratios": {
            "code": 0.30,
            "boundary": 0.20,
            "static_repair": 0.20,
            "execution_repair": 0.30,
        },
        "counts": {
            "code": 441,
            "boundary": 294,
            "static_repair": 294,
            "execution_repair": 441,
        },
        "lr": 5e-5,
        "composite_weights": {
            "code_generation_pass_at_1": 0.30,
            "boundary_pass_at_1": 0.15,
            "static_repair_success": 0.20,
            "execution_repair_success": 0.25,
            "hidden_pass_rate": 0.10,
        },
    },
    "repair-limited": {
        "dir": REPAIR_DIR,
        "ratios": {
            "code": 0.15,
            "boundary": 0.15,
            "static_repair": 0.30,
            "execution_repair": 0.40,
        },
        "counts": {
            "code": 221,
            "boundary": 220,
            "static_repair": 441,
            "execution_repair": 588,
        },
        "lr": 3e-5,
        "composite_weights": {
            "code_generation_pass_at_1": 0.10,
            "boundary_pass_at_1": 0.10,
            "static_repair_success": 0.30,
            "execution_repair_success": 0.40,
            "hidden_pass_rate": 0.10,
        },
    },
}

BUCKETS = ("code", "boundary", "static_repair", "execution_repair")


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _select_samples(
    pool: list[dict],
    bucket: str,
    count: int,
    rng: random.Random,
) -> list[dict]:
    """Select `count` samples from pool for a given bucket (variant_type)."""
    candidates = [s for s in pool if s.get("variant_type") == bucket]
    if len(candidates) < count:
        raise ValueError(
            f"Not enough samples for bucket={bucket}: "
            f"need {count}, have {len(candidates)}"
        )
    selected = rng.sample(candidates, count)
    return selected


def _build_candidate(
    name: str,
    cfg: dict,
    pool: list[dict],
    pool_sha: str,
) -> dict:
    """Build one P3-Limited candidate dataset."""
    rng = random.Random(SEED)
    selected: list[dict] = []
    seen_ids: set[str] = set()

    for bucket in BUCKETS:
        count = cfg["counts"][bucket]
        samples = _select_samples(pool, bucket, count, rng)
        for s in samples:
            sid = s.get("sample_id", "")
            if sid in seen_ids:
                raise ValueError(f"Duplicate sample_id within candidate: {sid}")
            seen_ids.add(sid)
            selected.append(s)

    total = len(selected)
    expected_total = sum(cfg["counts"].values())
    assert total == expected_total, f"{name}: total={total} != expected={expected_total}"

    # Verify ratios (should be exact since we use exact counts)
    bucket_counts = {}
    for bucket in BUCKETS:
        bc = sum(1 for s in selected if s.get("variant_type") == bucket)
        bucket_counts[bucket] = bc
        ratio = bc / total
        assert abs(ratio - cfg["ratios"][bucket]) < 0.01, (
            f"{name}.{bucket}: ratio={ratio:.4f} != target={cfg['ratios'][bucket]:.4f}"
        )

    # Family coverage
    families = set(s.get("family_id", "") for s in selected)

    # Write train.jsonl
    out_dir = cfg["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"

    with train_path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in selected:
            # Write the sample in ChatML format
            if "messages" in s:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            else:
                # If sample doesn't have messages, write as-is
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    train_sha = _sha256(_read_bytes(train_path))

    # Write manifest
    manifest = {
        "schema_version": 1,
        "candidate": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pool_sha256": pool_sha,
        "pool_source": "data/p3-formal/canonical-pool.jsonl",
        "total_samples": total,
        "bucket_counts": bucket_counts,
        "target_ratios": cfg["ratios"],
        "actual_ratios": {
            b: round(bucket_counts[b] / total, 6) for b in BUCKETS
        },
        "unique_sample_ids": len(seen_ids),
        "unique_families": len(families),
        "seed": SEED,
        "learning_rate": cfg["lr"],
        "composite_weights": cfg["composite_weights"],
        "train_file": str(train_path.relative_to(_ROOT)),
        "train_sha256": train_sha,
        "training_mode": "independent",
        "initial_adapter": None,
        "num_train_epochs": 2,
        "label": "P3-Limited (not formal capability claim)",
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest


def main():
    if not POOL_PATH.exists():
        print(f"ERROR: pool not found: {POOL_PATH}", file=sys.stderr)
        sys.exit(1)

    pool = _read_jsonl(POOL_PATH)
    print(f"Pool: {len(pool)} samples")

    with POOL_MANIFEST_PATH.open(encoding="utf-8") as fh:
        pool_manifest = json.load(fh)
    pool_sha = pool_manifest.get("pool_sha256", "")
    print(f"Pool SHA256: {pool_sha[:16]}...")

    # Pool bucket counts
    pool_buckets = {}
    for s in pool:
        vt = s.get("variant_type", "unknown")
        pool_buckets[vt] = pool_buckets.get(vt, 0) + 1
    print(f"Pool buckets: {pool_buckets}")

    print()

    manifests = {}
    for name, cfg in CANDIDATES.items():
        print(f"Building {name}...")
        manifest = _build_candidate(name, cfg, pool, pool_sha)
        manifests[name] = manifest
        print(f"  Total: {manifest['total_samples']}")
        print(f"  Buckets: {manifest['bucket_counts']}")
        print(f"  Ratios: {manifest['actual_ratios']}")
        print(f"  Families: {manifest['unique_families']}")
        print(f"  Unique IDs: {manifest['unique_sample_ids']}")
        print(f"  Train SHA: {manifest['train_sha256'][:16]}...")
        print()

    # Summary
    print("=" * 60)
    print("P3-Limited Dataset Build Complete")
    print("=" * 60)
    for name, m in manifests.items():
        print(f"\n{name}:")
        print(f"  Total: {m['total_samples']}")
        print(f"  Buckets: {m['bucket_counts']}")
        print(f"  LR: {m['learning_rate']}")
        print(f"  Output: {m['train_file']}")
        print(f"  Label: {m['label']}")

    print(f"\nBoth candidates use the same 403-family universe.")
    print(f"Results must be labeled 'P3-Limited', not formal capability.")


if __name__ == "__main__":
    main()
