"""scripts/build_curriculum_data.py -- P1 spec §7 curriculum replay data builder.

Builds per-stage training data under ``data/curriculum/{easy,boundary,repair}/``
implementing the P1 curriculum-replay ratios:

  Easy      : Level 0 (difficulty=0) 70%, Level 1 (difficulty=1) 30%
  Boundary  : Easy replay (diff 0+1) 30%, Boundary (diff 2) 70%
  Repair    : Easy replay (diff 0+1) 15%, Boundary replay (diff 2) 25%, Repair (diff 3) 60%

For every stage the script writes:
  - train.jsonl        : 85% of the assembled stage mix (Raw Sample JSONL)
  - validation.jsonl   : the remaining 15% of the stage mix (Raw Sample JSONL)
  - test_raw.jsonl     : a byte-copy of the frozen ``data/splits/test_raw.jsonl``
  - manifest.json      : provenance + hashes + actual ratios
  - families.json      : family_ids used in train.jsonl

A top-level ``data/curriculum/manifest.json`` aggregates the three stages.

Guarantees
----------
* Train/test family isolation: every family_id appearing in
  ``data/splits/test_raw.jsonl`` (and ``data/frozen-eval/v1/test_raw.jsonl``
  if present) is excluded from the sampling pool BEFORE any stage is built.
* All output is Raw Sample format; every record round-trips through
  ``Sample.model_validate``.
* Deterministic given ``seed=42`` (``random.Random(42)`` per stage).
* Never modifies anything under ``data/splits/`` or ``data/verified/``.

Usage
-----
    python scripts/build_curriculum_data.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
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
DATASET_VERSION = "p1-v1"
MAX_SEQ_LENGTH = 512
ASSISTANT_TARGET_RETENTION_RATE = 1.0
VAL_FRACTION = 0.15

SOURCE_FILES = [
    "data/verified/code_gen.jsonl",
    "data/verified/repairs.jsonl",
]
TEST_RAW_SOURCE = "data/splits/test_raw.jsonl"
FROZEN_EVAL_SOURCE = "data/frozen-eval/v1/test_raw.jsonl"

OUT_ROOT = _ROOT / "data" / "curriculum"

# Per-stage bucket definitions.
# Each bucket: (source_name, list_of_difficulty_levels, fraction).
# "easy" source = difficulty 0 + 1 (the Easy stage pool).
# "boundary" source = difficulty 2.
# "repair" source = difficulty 3.
#
# The Easy stage splits the easy pool into its two difficulty levels (70/30);
# later stages replay the easy pool as a single bucket.
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
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(Sample.model_validate(data))  # validates; raises on bad data
    return samples


def load_family_ids(path: Path) -> set[str]:
    """Return the set of family_ids present in a JSONL file (Sample format)."""
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
# Pool construction
# ---------------------------------------------------------------------------

def build_difficulty_pools(samples: list[Sample]) -> dict[int, list[Sample]]:
    """Group samples by difficulty, each bucket sorted by sample_id (stable)."""
    pools: dict[int, list[Sample]] = {}
    for s in samples:
        pools.setdefault(s.difficulty, []).append(s)
    for diff in pools:
        pools[diff].sort(key=lambda s: s.sample_id)
    return pools


def bucket_pool(bucket: tuple[str, list[int], float],
                pools: dict[int, list[Sample]]) -> list[Sample]:
    """Collect the (sorted, deduped) pool for a bucket across its levels."""
    seen: set[str] = set()
    out: list[Sample] = []
    for lvl in bucket[1]:
        for s in pools.get(lvl, []):
            if s.sample_id not in seen:
                seen.add(s.sample_id)
                out.append(s)
    # already sorted per-level; re-sort globally for full determinism
    out.sort(key=lambda s: s.sample_id)
    return out


# ---------------------------------------------------------------------------
# Stage sampling
# ---------------------------------------------------------------------------

def calibrate_total(buckets_with_pools: list[tuple[tuple[str, list[int], float], list[Sample]]]) -> int:
    """Pick the largest stage total whose every bucket target is satisfiable.

    For each bucket, the binding constraint is ``len(pool) / fraction``. Taking
    the floor of the minimum keeps every ``round(T * fraction)`` within the
    pool (a defensive cap in the sampler handles any rounding edge case).
    """
    candidates: list[float] = []
    for (_bucket, pool) in buckets_with_pools:
        fraction = _bucket[2]
        if fraction > 0.0:
            candidates.append(len(pool) / fraction)
    if not candidates:
        return 0
    return int(math.floor(min(candidates)))


def sample_stage(stage: str, pools: dict[int, list[Sample]],
                 seed: int = SEED) -> tuple[list[Sample], dict]:
    """Assemble the stage mix deterministically.

    Returns (stage_mix, sampling_report) where sampling_report records target
    vs actual per-bucket counts and the calibrated total.
    """
    buckets = STAGE_BUCKETS[stage]
    buckets_with_pools = [(b, bucket_pool(b, pools)) for b in buckets]
    total = calibrate_total(buckets_with_pools)

    rng = random.Random(seed)
    chosen: list[Sample] = []
    report = {
        "calibrated_total": total,
        "buckets": [],
        "shortfalls": {},
    }

    for (bucket, pool) in buckets_with_pools:
        source, levels, fraction = bucket
        target_count = round(total * fraction) if total > 0 else 0
        # Defensive cap: never request more than the pool holds.
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


def split_validation(stage_mix: list[Sample], val_fraction: float = VAL_FRACTION,
                     seed: int = SEED) -> tuple[list[Sample], list[Sample]]:
    """Slice a deterministic 15% validation subset off the stage mix.

    The stage mix is already shuffled by ``sample_stage`` using the same seed;
    to keep the val/train carve-out independent and reproducible we re-shuffle
    with a fresh RNG seeded from the same value, then take the first ``n_val``
    samples as validation and the rest as train.
    """
    rng = random.Random(seed)
    shuffled = list(stage_mix)
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    n_val = min(max(n_val, 0), len(shuffled))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


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
    """Actual fraction of train samples drawn from each curriculum source.

    Source mapping:
      difficulty 0 or 1 -> "easy"
      difficulty 2      -> "boundary"
      difficulty 3      -> "repair"
    """
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

def build_stage(stage: str, pools: dict[int, list[Sample]],
                excluded_family_sources: list[str],
                test_raw_source: Path) -> dict:
    """Build one stage directory and return its manifest dict."""
    stage_dir = OUT_ROOT / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # 1. Assemble the stage mix.
    stage_mix, sampling_report = sample_stage(stage, pools, seed=SEED)

    # 2. Carve out validation.
    train, val = split_validation(stage_mix, VAL_FRACTION, seed=SEED)

    # 3. Write train.jsonl / validation.jsonl (Raw Sample format).
    train_path = stage_dir / "train.jsonl"
    val_path = stage_dir / "validation.jsonl"
    write_samples_jsonl(train, train_path)
    write_samples_jsonl(val, val_path)

    # 4. Copy the frozen test_raw.jsonl (byte-identical across all stages).
    test_raw_path = stage_dir / "test_raw.jsonl"
    shutil.copyfile(test_raw_source, test_raw_path)

    # 5. Hashes.
    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path)
    test_raw_sha = sha256_file(test_raw_path)

    # 6. families.json for train.
    train_families = sorted(family_set(train))
    families_doc = {
        "stage": stage,
        "dataset_version": DATASET_VERSION,
        "family_count": len(train_families),
        "family_ids": train_families,
    }
    with (stage_dir / "families.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(families_doc, fh, indent=2, ensure_ascii=False)

    # 7. Per-stage manifest.
    test_raw_sample_count = sum(1 for _ in test_raw_path.open(encoding="utf-8") if _.strip())
    test_raw_family_count = len(load_family_ids(test_raw_path))

    manifest = {
        "stage": stage,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": SEED,
        "source_files": SOURCE_FILES,
        "excluded_family_sources": excluded_family_sources,
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "test_raw_sha256": test_raw_sha,
        "sample_counts": {
            "train": len(train),
            "validation": len(val),
            "test_raw": test_raw_sample_count,
        },
        "family_counts": {
            "train": len(train_families),
            "validation": len(family_set(val)),
            "test_raw": test_raw_family_count,
        },
        "task_type_mix": task_type_mix(train),
        "difficulty_mix": difficulty_mix(train),
        "max_seq_length": MAX_SEQ_LENGTH,
        "assistant_target_retention_rate": ASSISTANT_TARGET_RETENTION_RATE,
        "replay_ratios": source_replay_ratios(train),
        "target_replay_ratios": STAGE_TARGET_SOURCE_RATIOS[stage],
        "calibrated_total": sampling_report["calibrated_total"],
        "sampling_report": sampling_report,
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

    # 1. Load source samples (validated).
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

    # 2. Collect excluded family_ids from the frozen test set(s).
    excluded_family_sources: list[str] = []
    excluded_families: set[str] = set()

    test_raw_path = _ROOT / TEST_RAW_SOURCE
    if not test_raw_path.exists():
        print(f"ERROR: frozen test set not found: {test_raw_path}", file=sys.stderr)
        return 1
    excluded_families |= load_family_ids(test_raw_path)
    excluded_family_sources.append(TEST_RAW_SOURCE)

    frozen_eval_path = _ROOT / FROZEN_EVAL_SOURCE
    if frozen_eval_path.exists():
        excluded_families |= load_family_ids(frozen_eval_path)
        excluded_family_sources.append(FROZEN_EVAL_SOURCE)
        print(f"  frozen-eval found, merging exclusion families from {FROZEN_EVAL_SOURCE}")
    else:
        print(f"  frozen-eval not present at {FROZEN_EVAL_SOURCE} (skipping)")

    print(f"  excluded family_ids ({len(excluded_families)}): {sorted(excluded_families)}")

    # 3. Filter pool: drop samples whose family is in the frozen test set.
    pool = [s for s in all_samples if s.family_id not in excluded_families]
    print(f"  train-eligible pool: {len(pool)} samples, "
          f"{len({s.family_id for s in pool})} families")

    # 4. Bucket by difficulty.
    pools = build_difficulty_pools(pool)
    diff_summary = {d: len(v) for d, v in sorted(pools.items())}
    print(f"  difficulty distribution (eligible): {diff_summary}")

    # 5. Build all three stages.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stages = ["easy", "boundary", "repair"]
    stage_manifests: dict[str, dict] = {}
    print("\nBuilding curriculum stages:")
    for stage in stages:
        m = build_stage(stage, pools, excluded_family_sources, test_raw_path)
        stage_manifests[stage] = m
        print(f"  [{stage}] train={m['sample_counts']['train']} "
              f"val={m['sample_counts']['validation']} "
              f"test_raw={m['sample_counts']['test_raw']} "
              f"train_families={m['family_counts']['train']} "
              f"difficulty_mix={m['difficulty_mix']} "
              f"replay={m['replay_ratios']}")

    # 6. Top-level manifest.
    top_manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": SEED,
        "source_files": SOURCE_FILES,
        "excluded_family_sources": excluded_family_sources,
        "excluded_family_count": len(excluded_families),
        "stages": stage_manifests,
    }
    with (OUT_ROOT / "manifest.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(top_manifest, fh, indent=2, ensure_ascii=False)

    # 7. Verification checks.
    print("\n" + "=" * 64)
    print("VERIFICATION")
    print("=" * 64)

    test_families = load_family_ids(test_raw_path)
    all_ok = True

    train_shas: dict[str, str] = {}
    for stage in stages:
        m = stage_manifests[stage]
        train_path = OUT_ROOT / stage / "train.jsonl"
        nonempty = train_path.exists() and train_path.stat().st_size > 0
        train_shas[stage] = m["train_sha256"]
        train_fams = set(json.loads((OUT_ROOT / stage / "families.json").read_text(encoding="utf-8"))["family_ids"])
        overlap = train_fams & test_families
        iso_ok = not overlap
        all_ok = all_ok and nonempty and iso_ok
        print(f"  [{stage}] train exists & non-empty: {nonempty}")
        print(f"          train SHA256: {m['train_sha256']}")
        print(f"          train families: {len(train_fams)} | test_raw families: {len(test_families)}")
        print(f"          train∩test_raw family overlap: {sorted(overlap)}  -> {'OK' if iso_ok else 'LEAK!'}")

    distinct_shas = len(set(train_shas.values())) == len(train_shas)
    all_ok = all_ok and distinct_shas
    print(f"\n  three-stage train SHA256 all distinct: {distinct_shas}")
    for s, h in train_shas.items():
        print(f"    {s:>9}: {h}")

    print("\n  per-stage sample / family counts:")
    for stage in stages:
        m = stage_manifests[stage]
        print(f"    {stage:>9}: train={m['sample_counts']['train']:>3} "
              f"(families={m['family_counts']['train']}), "
              f"val={m['sample_counts']['validation']:>2} "
              f"(families={m['family_counts']['validation']}), "
              f"test_raw={m['sample_counts']['test_raw']}")

    print(f"\n  top-level manifest: {OUT_ROOT / 'manifest.json'}")
    print(f"  overall verification: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
