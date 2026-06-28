"""scripts/rebuild_curriculum_v2.py -- P1 curriculum v2 with global family partition.

Fixes Train/Validation family leakage by partitioning families BEFORE any
per-stage sampling.  Every family is assigned to exactly one of {train,
validation, frozen}.  No family_id ever appears in more than one role.

Key changes from v1:
  * Global family-level partition (seed=42, 80/20) instead of per-sample split.
  * Three overlap checks (train∩val, train∩frozen, val∩frozen) must all be ∅.
  * No per-stage test_raw.jsonl (unified frozen eval at data/frozen-eval/v1/).
  * If validation families lack a required difficulty for a stage, validation
    may be empty (recorded in manifest).

Usage
-----
    python scripts/rebuild_curriculum_v2.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
DATASET_VERSION = "v2"
MAX_SEQ_LENGTH = 512
ASSISTANT_TARGET_RETENTION_RATE = 1.0
VAL_FAMILY_FRACTION = 0.20
MIN_VAL_FAMILIES = 3

SOURCE_FILES = [
    "data/verified/code_gen.jsonl",
    "data/verified/repairs.jsonl",
]

FROZEN_EVAL_PATH = "data/frozen-eval/v1/test_raw.jsonl"

OUT_ROOT = _ROOT / "data" / "curriculum-v2"

# Per-stage bucket definitions (source_name, difficulty_levels, fraction).
STAGE_BUCKETS: dict[str, list[tuple[str, list[int], float]]] = {
    "easy": [
        ("easy", [0], 0.70),
        ("easy", [1], 0.30),
    ],
    "boundary": [
        ("easy", [0, 1], 0.30),
        ("boundary", [2], 0.70),
    ],
    "repair": [
        ("easy", [0, 1], 0.15),
        ("boundary", [2], 0.25),
        ("repair", [3], 0.60),
    ],
}

# Target source fractions (for recording in manifest).
STAGE_TARGET_SOURCE_RATIOS: dict[str, dict[str, float]] = {
    "easy": {"easy": 1.0, "boundary": 0.0, "repair": 0.0},
    "boundary": {"easy": 0.30, "boundary": 0.70, "repair": 0.0},
    "repair": {"easy": 0.15, "boundary": 0.25, "repair": 0.60},
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_samples_file(path: Path) -> list[Sample]:
    """Load + validate every line of a JSONL file as a Sample."""
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(Sample.model_validate(data))
    return samples


def load_family_ids(path: Path) -> set[str]:
    """Return the set of family_ids present in a JSONL file."""
    fams: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fams.add(json.loads(line)["family_id"])
    return fams


def write_samples_jsonl(samples: list[Sample], path: Path) -> None:
    """Write samples as Raw Sample JSONL (compact, LF-terminated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------

def build_difficulty_pools(samples: list[Sample]) -> dict[int, list[Sample]]:
    """Group samples by difficulty, each bucket sorted by sample_id."""
    pools: dict[int, list[Sample]] = {}
    for s in samples:
        pools.setdefault(s.difficulty, []).append(s)
    for diff in pools:
        pools[diff].sort(key=lambda s: s.sample_id)
    return pools


def bucket_pool(
    levels: list[int],
    pools: dict[int, list[Sample]],
) -> list[Sample]:
    """Collect the (sorted, deduped) pool across given difficulty levels."""
    seen: set[str] = set()
    out: list[Sample] = []
    for lvl in levels:
        for s in pools.get(lvl, []):
            if s.sample_id not in seen:
                seen.add(s.sample_id)
                out.append(s)
    out.sort(key=lambda s: s.sample_id)
    return out


def calibrate_total(
    buckets_with_pools: list[tuple[float, list[Sample]]],
) -> int:
    """Pick the largest total whose every bucket target is satisfiable."""
    candidates: list[float] = []
    for fraction, pool in buckets_with_pools:
        if fraction > 0.0 and len(pool) > 0:
            candidates.append(len(pool) / fraction)
        elif fraction > 0.0 and len(pool) == 0:
            return 0  # unsatisfiable
    if not candidates:
        return 0
    return int(math.floor(min(candidates)))


def sample_stage_mix(
    buckets: list[tuple[str, list[int], float]],
    pools: dict[int, list[Sample]],
    seed: int = SEED,
) -> tuple[list[Sample], dict]:
    """Assemble the stage mix deterministically from the given pools.

    Returns (chosen_samples, sampling_report).
    """
    buckets_with_pools = [
        (frac, bucket_pool(levels, pools))
        for (_source, levels, frac) in buckets
    ]
    total = calibrate_total(buckets_with_pools)

    rng = random.Random(seed)
    chosen: list[Sample] = []
    report = {
        "calibrated_total": total,
        "buckets": [],
        "shortfalls": {},
    }

    for (source, levels, fraction), (_, pool) in zip(buckets, buckets_with_pools):
        target_count = round(total * fraction) if total > 0 else 0
        if target_count > len(pool):
            report["shortfalls"][f"{source}={levels}"] = target_count - len(pool)
            target_count = len(pool)
        if target_count > 0:
            picked = rng.sample(pool, target_count)
        else:
            picked = []
        chosen.extend(picked)
        report["buckets"].append({
            "source": source,
            "levels": levels,
            "fraction": fraction,
            "pool_size": len(pool),
            "target_count": round(total * fraction) if total > 0 else 0,
            "actual_count": len(picked),
        })

    rng.shuffle(chosen)
    return chosen, report


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def task_type_mix(samples: list[Sample]) -> dict[str, int]:
    out: dict[str, int] = {
        "code_generation": 0,
        "static_repair": 0,
        "execution_repair": 0,
    }
    for s in samples:
        out[s.task_type] = out.get(s.task_type, 0) + 1
    return out


def difficulty_mix(samples: list[Sample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        key = str(s.difficulty)
        out[key] = out.get(key, 0) + 1
    return out


def source_replay_ratios(samples: list[Sample]) -> dict[str, float]:
    """Actual fraction of samples drawn from each curriculum source."""
    counts = {"easy": 0, "boundary": 0, "repair": 0}
    for s in samples:
        if s.difficulty in (0, 1):
            counts["easy"] += 1
        elif s.difficulty == 2:
            counts["boundary"] += 1
        elif s.difficulty == 3:
            counts["repair"] += 1
    total = sum(counts.values())
    if total == 0:
        return {"easy": 0.0, "boundary": 0.0, "repair": 0.0}
    return {k: round(v / total, 6) for k, v in counts.items()}


def family_set(samples: list[Sample]) -> set[str]:
    return {s.family_id for s in samples}


# ---------------------------------------------------------------------------
# Per-stage build
# ---------------------------------------------------------------------------

def build_stage(
    stage: str,
    train_pools: dict[int, list[Sample]],
    val_pools: dict[int, list[Sample]],
    train_families: list[str],
    validation_families: list[str],
    frozen_families: list[str],
) -> dict:
    """Build one stage directory and return its manifest dict."""
    stage_dir = OUT_ROOT / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    buckets = STAGE_BUCKETS[stage]

    # --- Train: sample from train_families pools only ---
    train_samples, train_report = sample_stage_mix(buckets, train_pools, seed=SEED)

    # --- Validation: sample from validation_families pools only ---
    val_samples, val_report = sample_stage_mix(buckets, val_pools, seed=SEED)

    # --- Write outputs ---
    train_path = stage_dir / "train.jsonl"
    val_path = stage_dir / "validation.jsonl"
    write_samples_jsonl(train_samples, train_path)
    write_samples_jsonl(val_samples, val_path)

    # --- Hashes ---
    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path) if val_samples else ""

    # --- families.json for train ---
    train_fams = sorted(family_set(train_samples))
    families_doc = {
        "stage": stage,
        "dataset_version": DATASET_VERSION,
        "family_count": len(train_fams),
        "family_ids": train_fams,
    }
    with (stage_dir / "families.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(families_doc, fh, indent=2, ensure_ascii=False)

    # --- Manifest ---
    val_fams = sorted(family_set(val_samples))

    # Compute actual replay ratios dict
    actual_replay = source_replay_ratios(train_samples)

    # Build replay_ratios in the format expected by the manifest spec
    # For easy: {"easy": 0.7, "boundary": 0.3}
    # For boundary: {"easy": 0.3, "boundary": 0.7}
    # For repair: {"easy": 0.15, "boundary": 0.25, "repair": 0.6}
    replay_ratios_display: dict[str, float] = {}
    for (source, _levels, fraction) in buckets:
        replay_ratios_display[source] = fraction

    manifest = {
        "stage": stage,
        "dataset_version": DATASET_VERSION,
        "family_partition_version": "v2",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": SEED,
        "source_files": SOURCE_FILES,
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "train_families": train_families,
        "validation_families": validation_families,
        "frozen_families": frozen_families,
        "train_validation_overlap": [],
        "train_frozen_overlap": [],
        "validation_frozen_overlap": [],
        "sample_counts": {"train": len(train_samples), "validation": len(val_samples)},
        "family_counts": {"train": len(train_fams), "validation": len(val_fams)},
        "task_type_mix": task_type_mix(train_samples),
        "difficulty_mix": difficulty_mix(train_samples),
        "max_seq_length": MAX_SEQ_LENGTH,
        "assistant_target_retention_rate": ASSISTANT_TARGET_RETENTION_RATE,
        "replay_ratios": replay_ratios_display,
        "train_sampling_report": train_report,
        "validation_sampling_report": val_report,
    }
    with (stage_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    # ========================================================================
    # Step 1: Collect all trainable families
    # ========================================================================
    print("=" * 64)
    print("Step 1: Load verified samples")
    print("=" * 64)

    all_samples: list[Sample] = []
    for rel in SOURCE_FILES:
        p = _ROOT / rel
        if not p.exists():
            print(f"ERROR: source file not found: {p}", file=sys.stderr)
            return 1
        loaded = load_samples_file(p)
        print(f"  loaded {len(loaded):>4} samples from {rel}")
        all_samples.extend(loaded)
    print(f"  total source samples: {len(all_samples)}")

    # ========================================================================
    # Step 2: Identify frozen families
    # ========================================================================
    frozen_path = _ROOT / FROZEN_EVAL_PATH
    if not frozen_path.exists():
        print(f"ERROR: frozen eval not found: {frozen_path}", file=sys.stderr)
        return 1
    frozen_families_set = load_family_ids(frozen_path)
    frozen_families = sorted(frozen_families_set)
    print(f"\n  frozen families: {len(frozen_families)}")
    for f in frozen_families:
        print(f"    - {f}")

    # ========================================================================
    # Step 3: Group remaining samples by family_id
    # ========================================================================
    print("\n" + "=" * 64)
    print("Step 2: Build global family partition")
    print("=" * 64)

    # Group samples by family_id, excluding frozen families
    family_samples: dict[str, list[Sample]] = {}
    for s in all_samples:
        if s.family_id in frozen_families_set:
            continue
        family_samples.setdefault(s.family_id, []).append(s)

    trainable_family_ids = sorted(family_samples.keys())
    print(f"  trainable families: {len(trainable_family_ids)}")
    for fid in trainable_family_ids:
        diffs = sorted(set(s.difficulty for s in family_samples[fid]))
        print(f"    {fid}: {len(family_samples[fid])} samples, difficulties={diffs}")

    # ========================================================================
    # Step 4: Partition families into train / validation
    # ========================================================================
    rng = random.Random(SEED)
    val_count = max(MIN_VAL_FAMILIES, round(len(trainable_family_ids) * VAL_FAMILY_FRACTION))
    # Ensure we don't take more than available
    val_count = min(val_count, len(trainable_family_ids))
    # Ensure train gets at least 1
    if val_count >= len(trainable_family_ids):
        val_count = len(trainable_family_ids) - 1

    shuffled = list(trainable_family_ids)
    rng.shuffle(shuffled)
    validation_families = sorted(shuffled[:val_count])
    train_families = sorted(shuffled[val_count:])

    validation_families_set = set(validation_families)
    train_families_set = set(train_families)

    print(f"\n  train families: {len(train_families)}")
    for f in train_families:
        print(f"    - {f}")
    print(f"\n  validation families: {len(validation_families)}")
    for f in validation_families:
        print(f"    - {f}")

    # Overlap checks
    train_val_overlap = train_families_set & validation_families_set
    train_frozen_overlap = train_families_set & frozen_families_set
    val_frozen_overlap = validation_families_set & frozen_families_set

    print(f"\n  Overlap checks:")
    print(f"    train ∩ validation: {sorted(train_val_overlap) if train_val_overlap else '∅'} {'✓' if not train_val_overlap else '✗ LEAK!'}")
    print(f"    train ∩ frozen:     {sorted(train_frozen_overlap) if train_frozen_overlap else '∅'} {'✓' if not train_frozen_overlap else '✗ LEAK!'}")
    print(f"    validation ∩ frozen: {sorted(val_frozen_overlap) if val_frozen_overlap else '∅'} {'✓' if not val_frozen_overlap else '✗ LEAK!'}")

    if train_val_overlap or train_frozen_overlap or val_frozen_overlap:
        print("\nFATAL: Family overlap detected! Aborting.", file=sys.stderr)
        return 1

    # ========================================================================
    # Step 5: Write family-partition.json
    # ========================================================================
    print("\n" + "=" * 64)
    print("Step 3: Write family-partition.json")
    print("=" * 64)

    partition_doc = {
        "seed": SEED,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "train_families": train_families,
        "validation_families": validation_families,
        "frozen_families": frozen_families,
        "train_validation_overlap": [],
        "train_frozen_overlap": [],
        "validation_frozen_overlap": [],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    partition_path = OUT_ROOT / "family-partition.json"
    with partition_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(partition_doc, fh, indent=2, ensure_ascii=False)
    print(f"  written: {partition_path}")

    # ========================================================================
    # Step 6: Build per-stage pools
    # ========================================================================
    print("\n" + "=" * 64)
    print("Step 4: Build per-stage curriculum data")
    print("=" * 64)

    # Build train and validation sample pools
    train_samples_pool = []
    val_samples_pool = []
    for fid, samples in family_samples.items():
        if fid in train_families_set:
            train_samples_pool.extend(samples)
        elif fid in validation_families_set:
            val_samples_pool.extend(samples)

    train_pools = build_difficulty_pools(train_samples_pool)
    val_pools = build_difficulty_pools(val_samples_pool)

    train_diff_summary = {d: len(v) for d, v in sorted(train_pools.items())}
    val_diff_summary = {d: len(v) for d, v in sorted(val_pools.items())}
    print(f"  train pool difficulty distribution: {train_diff_summary}")
    print(f"  validation pool difficulty distribution: {val_diff_summary}")

    # Build all three stages
    stages = ["easy", "boundary", "repair"]
    stage_manifests: dict[str, dict] = {}

    for stage in stages:
        print(f"\n  --- [{stage}] ---")
        m = build_stage(
            stage,
            train_pools,
            val_pools,
            train_families,
            validation_families,
            frozen_families,
        )
        stage_manifests[stage] = m
        print(f"    train: {m['sample_counts']['train']} samples, "
              f"{m['family_counts']['train']} families, "
              f"difficulty_mix={m['difficulty_mix']}")
        print(f"    validation: {m['sample_counts']['validation']} samples, "
              f"{m['family_counts']['validation']} families")

    # ========================================================================
    # Step 7: Verification
    # ========================================================================
    print("\n" + "=" * 64)
    print("VERIFICATION")
    print("=" * 64)

    all_ok = True

    train_shas: dict[str, str] = {}
    for stage in stages:
        m = stage_manifests[stage]
        train_path = OUT_ROOT / stage / "train.jsonl"
        nonempty = train_path.exists() and train_path.stat().st_size > 0
        train_shas[stage] = m["train_sha256"]

        # Verify train families are subset of train_families
        train_fams = set(json.loads(
            (OUT_ROOT / stage / "families.json").read_text(encoding="utf-8")
        )["family_ids"])
        train_fams_in_train = train_fams.issubset(train_families_set)
        all_ok = all_ok and nonempty and train_fams_in_train

        print(f"  [{stage}] train exists & non-empty: {nonempty}")
        print(f"          train SHA256: {m['train_sha256']}")
        print(f"          train families ⊆ train_families: {train_fams_in_train}")

        # Check val families are subset of validation_families
        val_path = OUT_ROOT / stage / "validation.jsonl"
        if val_path.exists() and val_path.stat().st_size > 0:
            val_fams = family_set(load_samples_file(val_path))
            val_fams_in_val = val_fams.issubset(validation_families_set)
            if not val_fams_in_val:
                print(f"          VAL LEAK: validation families not subset of validation_families!")
                all_ok = False
        else:
            print(f"          validation: empty (no suitable samples in validation families)")

    distinct_shas = len(set(train_shas.values())) == len(train_shas)
    all_ok = all_ok and distinct_shas
    print(f"\n  three-stage train SHA256 all distinct: {distinct_shas}")
    for s, h in train_shas.items():
        print(f"    {s:>9}: {h}")

    print(f"\n  overall verification: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())