"""scripts/build_p3_limited.py -- P3-Limited dataset builder (Issue #16 fix).

Builds two 1280-sample datasets from the Formal Canonical Pool v2 with exact
central ratios (not at tolerance edges), excluding frozen v4 evaluation data
(Issue #16 holdout-boundary fix).

  Balanced Limited (30/20/20/30):
    code=384, boundary=256, static_repair=256, execution_repair=384

  Repair Limited (15/15/30/40):
    code=192, boundary=192, static_repair=384, execution_repair=512

Both candidates:
  - Draw from the canonical pool AFTER excluding any sample whose
    sample_id OR family_id matches frozen v4 evaluation data
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
FROZEN_V4_PATH = _ROOT / "data" / "frozen-eval" / "v4" / "test_raw.jsonl"
FROZEN_V4_REL = "data/frozen-eval/v4/test_raw.jsonl"

OUTPUT_DIR = _ROOT / "data" / "p3-limited"
BALANCED_DIR = OUTPUT_DIR / "balanced-limited"
REPAIR_DIR = OUTPUT_DIR / "repair-limited"

SEED = 42

# Exact central ratios (not at tolerance edges). Both candidates use 1280
# samples (LP-feasible max for Repair after frozen v4 exclusion; Issue #16).
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
            "code": 384,
            "boundary": 256,
            "static_repair": 256,
            "execution_repair": 384,
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
            "code": 192,
            "boundary": 192,
            "static_repair": 384,
            "execution_repair": 512,
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


def _load_frozen_v4() -> tuple[set[str], set[str]]:
    """Return (sample_ids, family_ids) from frozen v4 test_raw.jsonl."""
    samples = _read_jsonl(FROZEN_V4_PATH)
    sids = {s["sample_id"] for s in samples}
    fams = {s["family_id"] for s in samples}
    return sids, fams


def _exclude_frozen(
    pool: list[dict],
    frozen_sids: set[str],
    frozen_fams: set[str],
) -> tuple[list[dict], dict]:
    """Filter pool to exclude any sample whose sample_id OR family_id matches
    frozen v4. Returns (filtered_pool, exclusion_stats).

    exclusion_stats counts pool samples excluded by each match type. A sample
    matching by sample_id is attributed to sample-match (sample_id takes
    precedence over family_id) so the two counts are disjoint.
    """
    kept: list[dict] = []
    excluded_by_sample = 0
    excluded_by_family = 0
    for s in pool:
        sid = s.get("sample_id", "")
        fid = s.get("family_id") or s.get("mbpp_family_id") or ""
        if sid in frozen_sids:
            excluded_by_sample += 1
        elif fid in frozen_fams:
            excluded_by_family += 1
        else:
            kept.append(s)
    stats = {
        "excluded_frozen_sample_count": excluded_by_sample,
        "excluded_frozen_family_count": excluded_by_family,
        "frozen_sample_ids_in_train": 0,  # filled in by _build_candidate
        "frozen_family_ids_in_train": 0,  # filled in by _build_candidate
    }
    return kept, stats


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
    frozen_sids: set[str],
    frozen_fams: set[str],
    exclusion_stats: dict,
) -> dict:
    """Build one P3-Limited candidate dataset.

    `pool` must already have frozen v4 samples excluded. `frozen_sids` and
    `frozen_fams` are used for a post-selection overlap check that proves
    zero contamination. `exclusion_stats` carries the counts of pool samples
    excluded by sample_id vs family_id match (computed once over the pool,
    shared across candidates).
    """
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

    # Post-selection overlap check against frozen v4 (Issue #16)
    train_sids = {s.get("sample_id", "") for s in selected}
    train_fams = {s.get("family_id") or s.get("mbpp_family_id") or "" for s in selected}
    leaked_sids = train_sids & frozen_sids
    leaked_fams = train_fams & frozen_fams
    overlap_check_passed = (not leaked_sids) and (not leaked_fams)

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
        "frozen_exclusion": {
            "frozen_eval_path": FROZEN_V4_REL,
            "excluded_frozen_sample_count": exclusion_stats["excluded_frozen_sample_count"],
            "excluded_frozen_family_count": exclusion_stats["excluded_frozen_family_count"],
            "frozen_sample_ids_in_train": len(leaked_sids),
            "frozen_family_ids_in_train": len(leaked_fams),
            "overlap_check_passed": overlap_check_passed,
        },
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest


def build_all() -> dict:
    """Build both P3-Limited candidates. Returns {candidate_name: manifest}.

    Excludes frozen v4 evaluation data (sample_id OR family_id match) from
    the canonical pool before sampling (Issue #16).
    """
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"pool not found: {POOL_PATH}")

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

    # Frozen v4 exclusion (Issue #16)
    frozen_sids, frozen_fams = _load_frozen_v4()
    pool_filtered, exclusion_stats = _exclude_frozen(pool, frozen_sids, frozen_fams)
    excluded_total = (
        exclusion_stats["excluded_frozen_sample_count"]
        + exclusion_stats["excluded_frozen_family_count"]
    )
    print(
        f"Excluding {excluded_total} pool samples from frozen v4 families "
        f"(by sample_id: {exclusion_stats['excluded_frozen_sample_count']}, "
        f"by family_id: {exclusion_stats['excluded_frozen_family_count']})"
    )

    # Filtered pool bucket counts
    filtered_buckets = {}
    for s in pool_filtered:
        vt = s.get("variant_type", "unknown")
        filtered_buckets[vt] = filtered_buckets.get(vt, 0) + 1
    print(f"Filtered pool buckets: {filtered_buckets}")

    print()

    manifests = {}
    for name, cfg in CANDIDATES.items():
        print(f"Building {name}...")
        manifest = _build_candidate(
            name, cfg, pool_filtered, pool_sha,
            frozen_sids, frozen_fams, exclusion_stats,
        )
        manifests[name] = manifest
        print(f"  Total: {manifest['total_samples']}")
        print(f"  Buckets: {manifest['bucket_counts']}")
        print(f"  Ratios: {manifest['actual_ratios']}")
        print(f"  Families: {manifest['unique_families']}")
        print(f"  Unique IDs: {manifest['unique_sample_ids']}")
        fe = manifest["frozen_exclusion"]
        print(
            f"  Frozen exclusion: sample_count={fe['excluded_frozen_sample_count']}, "
            f"family_count={fe['excluded_frozen_family_count']}, "
            f"overlap_check_passed={fe['overlap_check_passed']}"
        )
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
        fe = m["frozen_exclusion"]
        print(
            f"  Frozen exclusion: overlap_check_passed={fe['overlap_check_passed']} "
            f"(sample_ids_in_train={fe['frozen_sample_ids_in_train']}, "
            f"family_ids_in_train={fe['frozen_family_ids_in_train']})"
        )

    print(f"\nBoth candidates exclude frozen v4 (Issue #16).")
    print(f"Results must be labeled 'P3-Limited', not formal capability.")

    return manifests


def main():
    try:
        build_all()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
